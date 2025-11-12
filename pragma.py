#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Offline RAG (LangChain + Chroma) using ONLY Ollama (local GGUF models)
• Now ingests MANY file types by converting everything to Markdown with markitdown
• Preserves original chat/retrieval behavior and CLI, but generalized beyond PDFs
================================================================================

Notes
-----
- Fully offline, no network calls. You must have local Ollama models created via `ollama create`.
- Requires the `markitdown` package (pip install markitdown). Some formats may need optional
  native deps (e.g., for PDFs/images, depending on your system). Script will gracefully warn
  and fall back to plain-text read for basic types.
- Keeps the same prompt and chat loop semantics as your previous script.

Supported file types (via markitdown)
-------------------------------------
Common: .pdf, .doc, .docx, .ppt, .pptx, .xls, .xlsx, .csv, .txt, .md, .rtf, .html, .htm, .json, .xml, .epub
Images (basic OCR if supported): .png, .jpg, .jpeg, .tiff, .bmp, .gif
(Anything markitdown supports on your machine will be attempted.)
"""

from __future__ import annotations
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple
import time

# ---- Force OFFLINE + disable telemetry ---------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")     # Chroma telemetry off
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")     # LangChain tracing off
os.environ.setdefault("LANGCHAIN_ENDPOINT", "")            # no endpoint
os.environ.setdefault("LANGCHAIN_API_KEY", "")             # block any tracing

# ---- LangChain / community imports -------------------------------------------
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings  # updated per LangChain 0.3.1 deprecation

# markitdown (soft dependency with graceful fallback)
_MK_AVAILABLE = True
try:
    from markitdown import MarkItDown
except Exception:
    _MK_AVAILABLE = False


def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

# ---- LLM + Embeddings (Ollama only) -----------------------------------------

def make_llm(chat_model: str, temperature: float = 0.2):
    try:
        return ChatOllama(model=chat_model, temperature=temperature)
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Ollama chat model. Ensure `ollama` is running and the model NAME exists locally (no pulling).\n"
            f"Model requested: {chat_model}\nUnderlying error: {e}"
        ) from e


def make_embeddings(embed_model: str):
    try:
        return OllamaEmbeddings(model=embed_model)
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Ollama embedding model. Ensure `ollama` is running and the embedding model NAME exists locally (no pulling).\n"
            f"Model requested: {embed_model}\nUnderlying error: {e}"
        ) from e


# ---- File discovery + Markdown conversion -----------------------------------
SUPPORTED_EXTS = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".txt", ".md", ".rtf",
    ".html", ".htm", ".json", ".xml", ".epub", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif",
}


def iter_files(data_dir: Path, include_exts: Iterable[str] | None = None) -> Iterable[Path]:
    include = {e.lower() for e in (include_exts or SUPPORTED_EXTS)}
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in include:
            yield p


def convert_to_markdown(fp: Path) -> Tuple[str, str]:
    """
    Return (markdown_text, warning_message). warning_message is empty if OK.

    Order of attempts:
      1) markitdown (best structure when extras are installed)
      2) PyMuPDF plain text extraction (for text-based PDFs)
      3) OCR with pytesseract (for scanned PDFs)
      4) Lightweight fallbacks for simple textual formats
    """
    fallback_warn = ""

    # 0) MarkItDown (robust to API differences across versions)
    if _MK_AVAILABLE:
        try:
            md = MarkItDown()
            result = md.convert(str(fp))

            # Some versions expose .text, some .markdown, etc.
            text = ""
            for attr in ("text", "markdown", "text_content", "content"):
                val = getattr(result, attr, None)
                if isinstance(val, str) and val.strip():
                    text = val.strip()
                    break

            if text:
                return text, ""
            else:
                fallback_warn = f"[warn] Converted but empty via markitdown: {fp.name}"
        except Exception as e:
            fallback_warn = f"[warn] markitdown failed for {fp.name}: {e}"
    else:
        fallback_warn = "[warn] markitdown not installed; using fallbacks"

    ext = fp.suffix.lower()

    # 1) PDF -> PyMuPDF text (handles most non-scanned PDFs)
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            pages = []
            with fitz.open(str(fp)) as doc:
                # If encrypted and no password, bail to OCR below
                try:
                    if doc.is_encrypted:  # type: ignore[attr-defined]
                        # attempt empty password; if it still needs pass, skip to OCR
                        try:
                            doc.authenticate("")  # no-op if not needed
                        except Exception:
                            raise RuntimeError("PDF is encrypted and requires a password")
                except Exception:
                    # Older PyMuPDF may not have is_encrypted; continue best-effort
                    pass

                for p in doc:
                    # "text" explicitly requests the plain text extractor
                    pages.append(p.get_text("text") or "")
            txt = "\n\n".join(pages).strip()
            if txt:
                return txt, (fallback_warn + " [used PyMuPDF text]") if fallback_warn else "[used PyMuPDF text]"
        except Exception as e:
            fallback_warn = (fallback_warn + f" [PyMuPDF text error: {e}]").strip()

        # 2) PDF -> OCR (for scanned/image-only PDFs)
        try:
            import fitz
            import pytesseract
            from PIL import Image

            ocr_pages = []
            with fitz.open(str(fp)) as doc:
                # Render ~2x for better OCR accuracy
                zoom = fitz.Matrix(2, 2)
                for p in doc:
                    pm = p.get_pixmap(matrix=zoom)
                    mode = "RGB" if pm.alpha == 0 else "RGBA"
                    img = Image.frombytes(mode, (pm.width, pm.height), pm.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    ocr_pages.append(ocr_text or "")
            ocr_txt = "\n\n".join(ocr_pages).strip()
            if ocr_txt:
                return ocr_txt, (fallback_warn + " [used OCR]") if fallback_warn else "[used OCR]"
        except Exception as e:
            fallback_warn = (fallback_warn + f" [OCR error: {e}]").strip()

    # 3) Simple textual formats (best-effort)
    try:
        if ext in {".txt", ".md"}:
            return fp.read_text(encoding="utf-8", errors="ignore"), fallback_warn
        if ext == ".json":
            import json, pprint
            obj = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
            return "```json\n" + pprint.pformat(obj, width=100) + "\n```", fallback_warn
        if ext in {".xml", ".html", ".htm"}:
            lang = ext.lstrip(".")
            return f"```{lang}\n" + fp.read_text(encoding="utf-8", errors="ignore") + "\n```", fallback_warn
        if ext == ".csv":
            return "```csv\n" + fp.read_text(encoding="utf-8", errors="ignore") + "\n```", fallback_warn
    except Exception as e:
        raise RuntimeError(f"Fallback read failed for {fp.name}: {e}")

    # 4) Could not convert
    raise RuntimeError(
        (fallback_warn + " — ") if fallback_warn else ""
        + "cannot convert this type without optional dependencies. "
          'Install extras: pip install "markitdown[all]"'
    )
# ---- Ingestion / Indexing ----------------------------------------------------

def ingest_any_to_chroma(data_dir: Path, persist_dir: Path, embeddings, collection_name: str,
                         chunk_size: int = 1200, chunk_overlap: int = 150) -> Chroma:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = list(iter_files(data_dir))

    print(f"[info] Found {len(files)} candidate files under {data_dir}")
    for fp in files:
        print("  -", fp)
    if not files:
        raise RuntimeError(f"No supported files found in {data_dir}")

    print(f"Converting {len(files)} file(s) to Markdown…")
    docs: List[Document] = []
    warnings: List[str] = []

    for fp in files:
        try:
            print(f"[conv] {fp.name} ...", end="", flush=True)
            text, warn = convert_to_markdown(fp)
            if warn:
                print(f" WARN ({warn[:120]})", flush=True)
            else:
                print(" ok", flush=True)
            if not text.strip():
                print(f"[skip] empty after conversion: {fp.name}")
                continue
            print(f"[keep] {fp.name} -> {len(text)} chars")
            meta = {"source": fp.name, "path": str(fp), "ext": fp.suffix.lower()}
            docs.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            warnings.append(f"[warn] Skipping {fp.name}: {e}")

    if not docs:
        raise RuntimeError("No documents produced after conversion. See warnings above.")

    # Report non-fatal conversion warnings
    if warnings:
        for w in warnings:
            print(w, file=sys.stderr)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks. Building Chroma index…")

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )
    vectordb.persist()
    print(f"Persisted index to {persist_dir}")
    return vectordb


def wipe_dir(p: Path):
    if not p.exists():
        return
    for sub in sorted(p.rglob("*"), reverse=True):
        try:
            if sub.is_file() or sub.is_symlink():
                sub.unlink(missing_ok=True)
            else:
                sub.rmdir()
        except Exception:
            pass
    try:
        p.rmdir()
    except Exception:
        pass


def load_or_build_chroma(data_dir: Path, persist_dir: Path, embeddings, collection_name: str,
                         rebuild: bool, chunk_size: int, chunk_overlap: int) -> Chroma:
    clear_screen()
    if rebuild:
        print(f"Rebuilding index: clearing {persist_dir} …")
        wipe_dir(persist_dir)

    if persist_dir.exists() and any(persist_dir.iterdir()) and not rebuild:
        print(f"Reusing existing Chroma index at {persist_dir}")
        return Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_name=collection_name,
        )
    else:
        return ingest_any_to_chroma(
            data_dir,
            persist_dir,
            embeddings,
            collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


# ---- RAG chain ---------------------------------------------------------------

def build_rag_chain(llm, retriever):
    system_prompt = (
        """
        You are a conversational retrieval-augmented assistant (RAG model) named PRAGMAtic.

        Your purpose is to help users explore and understand accurate, up-to-date information from the excerpts in the knowledge base. Use pretrained knowledge where appropriate, but default to the provided excerpts.

        Your responses must be:
        - **Grounded** strictly in the retrieved documents or knowledge base.
        - **Clear and concise**, written in natural, user-friendly language.
        - **Action-oriented**, guiding users to the next step (e.g., who to contact, where to go, what form to use).

        Do **not** include internal reasoning, hidden thoughts, or planning text in your output — only the final user-facing answer. 
        Do **not** cite document names or corpus references directly; instead, provide the raw URL or resource name when available.

        If no relevant information is found, politely state that and suggest where the user might look next.
        """
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Question: {input}\n\nUse these relevant excerpts to answer:\n{context}\n"),
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_chain


def print_art():
    art = r"""
|=======================================================================|
|        Personal Retrieval-Augmented Generative Model Architecture     |
|=======================================================================|
                    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                       
                ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⠶⠾⠟⠛⢛⣿⣿⣷⣶⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠀⢀⣴⠟⠋⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀
                ⠀⠀⠀⢀⣤⣶⣾⠋⠀⠀⠀⠀⠀⠀⠸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⣄⠀⠀⠀
                ⠀⠀⣠⡞⠁⣸⡏⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⠇⢻⡇⠉⢷⡄⠀
                ⠀⢠⣿⠷⠛⢻⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠛⠛⠋⠁⠀⣿⡛⠻⢾⣿⠀
                ⠀⠈⠀⠀⣴⠛⢿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡟⢻⣆⠀⠈⠀
                ⠀⠀⠀⣼⢁⣠⡾⢿⡄⠀⠀⠀⠀⣀⣤⣤⣤⣀⠀⠀⠀⠀⣰⡿⢶⣄⢹⡇⠀⠀
                ⠀⠀⠀⣿⠟⠋⣠⡾⢿⣆⠀⢰⣿⣿⣿⣿⣿⣿⣷⡄⠀⣴⡟⢷⡄⠙⢿⣿⠀⠀
                ⠀⠀⠀⠀⠀⢰⡟⠀⣠⡿⠀⣏⠉⣿⣿⣿⣿⡏⢉⡇⠀⢿⣄⠈⣿⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⢸⣇⣾⠏⢀⡄⠹⣿⠉⢻⣿⡟⠙⣿⠇⣠⠀⠻⣧⣼⠃⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠈⠛⠁⠀⣾⣿⣆⡉⠓⠿⠿⠷⠛⣁⣴⣿⡇⠀⠙⠿⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⠋⠀⠀⠀⠀ ⠈⠻⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠏⠀⠀⠀⠀⠀⠀⠀ ⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀

              __________    _____    ________                
        ______\______   \  /  _  \  /  _____/  _____ _____   
        \____ \|       _/ /  /_\  \/   \  ___ /     \\__  \  
        |  |_> >    |   \/    |    \    \_\  \  Y Y  \/ __ \_
        |   __/|____|_  /\____|__  /\______  /__|_|  (____  /
        |__|          \/         \/        \/      \/     \/          
      ⠀⠀        
|=======================================================================|
|=======================================================================|
"""
    print(art+"\n")



# ---- CLI loop ----------------------------------------------------------------
def chat_loop(chain, session_id: str = "default", datadir: str = "./data"):
    clear_screen()
    print(f"==================================================\nSession trained on: {datadir}\nCommands: :q to quit\n==================================================\n")

    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user in {":q", ":quit", ":exit"}:
            break

        conf = {"configurable": {"session_id": session_id}}

        try:
            result = chain.invoke({"input": user}, conf)
            answer = result.get("answer", "(no answer)")

            # 🧹 Remove everything from "<|end_of_document|>" onward
            answer = re.split(r"<\|end_of_document\|>", answer, maxsplit=1)[0].strip()

            # Collect citations
            docs: List = result.get("context", []) or []
            cited = set()
            for d in docs:
                name = Path(d.metadata.get("source", "")).name
                if name:
                    cited.add(f"[{name}]")
            citation_str = " " + " ".join(sorted(cited)) if cited else ""

            print(f"bot › {answer}{citation_str}\n")

        except Exception as e:
            print(f"[error] {e}\n")


# ---- Main --------------------------------------------------------------------

def main():
    
    print_art()
    time.sleep(3)

    ap = argparse.ArgumentParser(description="OFFLINE RAG over mixed files via markitdown + Ollama + Chroma")
    ap.add_argument("--data-dir", default="./data/corpus", help="Directory of files to index (mixed types)")
    ap.add_argument("--persist-dir", default=".chroma", help="Chroma persistence directory")
    ap.add_argument("--collection", default="doc-rag", help="Chroma collection name")

    # Ollama models (both must already exist locally via `ollama create`)
    ap.add_argument("--chat-model", required=True, help="Local Ollama chat model NAME")
    ap.add_argument("--embed-model", required=True, help="Local Ollama embedding model NAME")

    ap.add_argument("--k", type=int, default=4, help="# of chunks to retrieve")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild of vector index")

    ap.add_argument("--chunk-size", type=int, default=1200, help="Chunk size for splitting")
    ap.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap")

    ap.add_argument("--mmr", action="store_true", help="Use MMR retriever")
    ap.add_argument("--mmr-diversity", type=float, default=0.95, help="MMR diversity (0..1)")

    ap.add_argument("--include-exts", nargs="*", default=None,
                    help="Optional list of file extensions to include (e.g. .pdf .docx .html). Defaults to a broad set.")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    persist_dir = Path(args.persist_dir)

    

    # Initialize models (Ollama only)
    embeddings = make_embeddings(embed_model=args.embed_model)
    llm = make_llm(chat_model=args.chat_model)

    time.sleep(1)
    print(f"[info] Using Ollama chat model: {args.chat_model}")
    print(f"[info] Using Ollama embedding model: {args.embed_model}")
    time.sleep(1)
    print(f"[info] Data directory: {data_dir}")
    print(f"[info] Chroma persist directory: {persist_dir}")
    time.sleep(3)
    clear_screen()
    # Narrow file types if requested
    global SUPPORTED_EXTS
    if args.include_exts:
        SUPPORTED_EXTS = {e if e.startswith('.') else f'.{e}' for e in args.include_exts}

    vectordb = load_or_build_chroma(
        data_dir=data_dir,
        persist_dir=persist_dir,
        embeddings=embeddings,
        collection_name=args.collection,
        rebuild=args.rebuild,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Choose retriever type
    if args.mmr:
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": args.k,
                "fetch_k": max(args.k * 4, 20),
                "lambda_mult": args.mmr_diversity,
            },
        )
    else:
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": args.k},
        )

    chain = build_rag_chain(llm, retriever)
    chat_loop(chain,args.data_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal: {e}", file=sys.stderr)
        sys.exit(1)
