# PRAGMAtic — Offline RAG Using Local Ollama (GGUF)

**PRAGMAtic** (`pragma.py`) provides a fully offline Retrieval-Augmented Generation (RAG) pipeline using local GGUF models through Ollama for both embedding and chat generation. It does not depend on external APIs such as Hugging Face or OpenAI and can operate in completely air-gapped environments.

---

## Overview

`pragma.py` builds a local Chroma vector database from PDF files and allows question-answering using a local Ollama model for both embedding and generation.  
It is ideal for secure environments that prohibit network access.

---

## Features

- Uses Ollama for both chat and embeddings.  
- Fully offline execution with no external dependencies.  
- Persistent Chroma vector store for fast querying.  
- Command-line interface with configurable parameters.  
- Conversational memory for context continuity.  

---

## System Requirements

- Python 3.0 or higher  
- Ollama installed and running locally  
- Pre-downloaded GGUF model files for both LLM and embeddings  

**Download Python:** [https://www.python.org/downloads/](https://www.python.org/downloads/)  
**Download Ollama / CLI Install:** [https://ollama.com/download](https://ollama.com/download)  
**Ollama CLI Reference:** [https://github.com/ollama/ollama/blob/main/docs/cli.md](https://github.com/ollama/ollama/blob/main/docs/cli.md)  
**Ollama General Documentation:** [https://ollama.com/docs](https://ollama.com/docs)  
**Hugging Face Docs:** [https://huggingface.co/docs](https://huggingface.co/docs)  
**CC BY-SA 4.0 License:** [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)  

---

## Quick Run Example(s)

```bash
python pragma.py --pdf-dir ./data/pdfs/ccby/stem --persist-dir ./.chroma --collection pdf-rag --chat-model Phi4 --embed-model nomic-embed-text --k 4 --rebuild
```

```bash
python pragma.py --data-dir ./data/pdfs/lib/LMDS/v2 --persist-dir ./.chroma --collection lmds-rag --chat-model Phi4 --embed-model snowflake-arctic-embed --k 8 --chunk-size 1400 --chunk-overlap 300 --mmr --rebuild
```

---

## Setup Instructions

### 1. Prepare Local GGUF Models

Ensure you have GGUF model files for both your chat and embedding models. They can be anywhere on your local filesystem, as long as you reference them correctly when creating Ollama models.

It is recommended to use smaller models for embedding tasks and larger models for chat generation, depending on your hardware capabilities. Additional models are available on the Ollama model hub in GGUF format, as well as from Hugging Face and other sources.

`.gguf` files can be stored anywhere locally and are required for full offline operation.

**If you do not wish to run offline, you can also use Ollama's built-in models without GGUF files. For instructions, skip to section `2B. Use Built-in Ollama Models`.**

---

### 2A. Create Local Ollama Models

Create Ollama models using `ollama create` from your GGUF files.

**Chat model:**  
Create a Modelfile for your chat model using the template in `./modelfiles/modeltemplate.yaml`.  
Modify it to include the correct path to your GGUF file in the `FROM` line.

Then create your Ollama model:

```bash
ollama create CHOSENNAME -f PATH/TO/YOUR/MODELFILE.yaml
```

---

### 2B. Use Built-in Ollama Models

If you prefer to use Ollama’s built-in models instead of local GGUF files, specify the model names directly when running **PRAGMAtic**. Ensure the Ollama daemon is running and that the desired models are available locally.

To pull a built-in model:

```bash
ollama pull MODELNAME
```

You can then use `MODELNAME` directly with the `--chat-model` and `--embed-model` parameters.

For a list of available built-in models, see the [Ollama Model Library](https://ollama.com/library).

---

***In either case, ensure that models have been created or pulled successfully and that the Ollama daemon is running:***

```bash
ollama list
```

---

## Python Environment Setup (REQUIRES INTERNET)

Running **PRAGMAtic** requires several Python packages. It’s recommended to create a virtual environment first (`venv`, `conda`, etc.).

Some dependencies are only available via `pip`. Run the following to install requirements:

```bash
pip install langchain langchain-core langchain-community langchain-classic langchain-ollama chromadb markitdown pypdf pdfminer.six beautifulsoup4 lxml pillow pytesseract ollama
```

---

## Usage

### Basic Command

```bash
python pragma.py --pdf-dir ./path/to/pdfs --persist-dir ./.chroma --collection pdf-rag --chat-model my-llm --embed-model my-embed --k 4
```

### Rebuild Index

Use the `--rebuild` flag when PDFs are modified or a new embedding model is used.

```bash
python pragma.py --pdf-dir ./path/to/pdfs --persist-dir ./.chroma --collection pdf-rag --chat-model my-llm --embed-model my-embed --k 4 --rebuild
```

---

### Command-Line Parameters

| Parameter | Type | Description | Default |
|------------|------|-------------|----------|
| `--pdf-dir` | str | Directory containing PDF files to ingest. | `./data/pdfs/ccby/stem` |
| `--persist-dir` | str | Directory where Chroma DB is stored. | `./.chroma` |
| `--collection` | str | Chroma collection name. | `pdf-rag` |
| `--chat-model` | str | Name of local Ollama chat model created with `ollama create`. | Required |
| `--embed-model` | str | Name of local Ollama embedding model created with `ollama create`. | Required |
| `--k` | int | Number of document chunks to retrieve per query. | `4` |
| `--rebuild` | flag | Force rebuild of vector index. | `False` |
| `--chunk-size` | int | Character size per chunk when splitting PDFs. | `1200` |
| `--chunk-overlap` | int | Character overlap between chunks. | `150` |
| `--mmr` | flag | Use Maximal Marginal Relevance retriever. | `False` |
| `--mmr-diversity` | float | MMR diversity parameter (0–1). | `0.5` |

---

## Troubleshooting

**Error: Dimension mismatch (e.g., 384 vs. 768)**  
Your Chroma index was built with a different embedding dimension. Rebuild the index using `--rebuild`.

**Error: Ollama model not found**  
Ensure you created the model locally using `ollama create` and that the Ollama daemon is running.

**Error: Connection refused or timeout**  
Start the Ollama service using `ollama serve`.

---

## License

Licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

