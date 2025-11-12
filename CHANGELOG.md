🧾 CHANGELOG

All notable changes to the Local RAG Script will be documented in this file.
This project tracks the evolution of the Retrieval-Augmented Generation (RAG) system and its related experiments with local models and datasets.


[v0.8] - Markitdown Support & Enhancements

Update system to first convert data files into Markitdown format for improved parsing.

Support for additional file types beyond .pdf, including .docx and .html.

Optional Tesseract OCR integration for scanned documents as fallback.


[v0.7] - Prompt Refinement & Retrieval Optimization

Refine system prompts for better context utilization.

Optimize retrieval strategies to enhance response relevance.

Alternative model evaluations for improved performance.

[v0.6] - Library Master Dataset Refinement

Developing a streamlined Library Master Dataset for more efficient and accurate retrieval.

Implementing improved data curation and filtering methods to reduce noise.

Refining embedding and vectorization processes for higher precision.

Goals:

Achieve a clean, high-signal dataset.

Improve retrieval quality and contextual responses.

Build a stable foundation for fine-tuning and hybrid RAG strategies.

[v0.5] - Library Master Dataset (Initial Implementation)
Added

Introduced the first version of the Library Master Dataset, integrating multiple document sources into a shared ChromaDB instance.

Issues

Dataset proved overly noisy and redundant, leading to imprecise retrieval.

Lacked effective data cleaning or filtering pipeline.

[v0.4] - Custom GGUF Model Support Added

Support for uploading and running any GGUF model locally.

Expanded compatibility beyond Ollama’s included models.

Improved modular structure for easier model management.

[v0.3] - Offline Model Execution Added

Enabled offline inference using locally cached Ollama models.

Removed internet dependency for RAG operations.

Improved

Increased reliability for air-gapped or offline environments.

Simplified setup and runtime behavior.

[v0.2] - PDF Uploads & Local Vectorization Added

Ability to upload PDFs as local knowledge sources.

Introduced ChromaDB vector store for document embeddings.

Enabled local context retrieval.

Limitations

Depended on Ollama-hosted models.

Vectorization pipeline lacked optimization.

[v0.1] - Initial Version
Overview

Initial prototype using Ollama’s default model library.

Required active internet connection for model execution.

No local data integration or customization options.