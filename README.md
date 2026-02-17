# RAG Pipeline Evaluation System

A production-ready Retrieval-Augmented Generation (RAG) pipeline with quality metrics and confidence scoring for Windows.

## Overview

This system evaluates AI-generated answers by measuring citation accuracy, groundedness, and evidence strength. Documents are chunked, embedded, and indexed with FAISS for efficient retrieval.

**Two versions available:**
- **v1**: Basic evaluation (citation precision, groundedness, evidence strength)
- **v2**: Enhanced evaluation (adds citation recall, confidence scoring, quality gating)

**Supported formats:** PDF, HTML, TXT, Markdown

## Prerequisites
1. Python 3.12.x (exact version: 3.12.0-3.12.9)
2. Ollama v0.X.X with qwen3:1.7b model
   - Install: https://ollama.com/download
   - Pull model: `ollama pull qwen3:1.7b`
   - Run model: `ollama serve`
3. Source documents in `data/raw/` directory

## Quick Start

### Option 1: Basic Pipeline (v1)

```powershell
# One-time setup
.\setup_v1.ps1

# Ask questions
.\.venv\Scripts\Activate.ps1
python rag_pipeline_eval_v1.py query --question "Your question here"

# Re-run batch evaluation
.\evaluate_v1.ps1
```

### Option 2: Enhanced Pipeline (v2)

```powershell
# One-time setup
.\setup_v2.ps1

# Ask questions
.\.venv\Scripts\Activate.ps1
python rag_pipeline_eval_v2.py query --question "Your question here"

# Re-run batch evaluation
.\evaluate_v2.ps1
```
### Setup Scripts

#### `setup_v1.ps1` / `setup_v2.ps1`

**What it does:**
1. ✅ Verifies Python 3.12 is installed
2. ✅ Creates project directories
3. ✅ Creates virtual environment
4. ✅ Installs dependencies from requirements.txt
5. ✅ Verifies FAISS installation
6. ✅ Runs `run_all` (ingests docs, creates chunks, builds FAISS index)
7. ✅ Runs `batch` (evaluates predefined questions)

**When to run:**
- First time setup
- After adding new documents to `data/raw/`
- After modifying chunking or indexing settings

**Run time:** 5-15 minutes (depending on corpus size)

### Evaluation Scripts

#### `evaluate_v1.ps1` / `evaluate_v2.ps1`

**What it does:**
1. ✅ Verifies index exists
2. ✅ Activates virtual environment
3. ✅ Runs batch evaluation questions
4. ✅ Shows results summary

**When to run:**
- After modifying prompts or thresholds
- To re-test questions without rebuilding index
- To compare v1 vs v2 metrics

**Run time:** 1-5 minutes


## How It Works

### 1. Document Ingestion
- Reads documents from `data/raw/`
- Parses PDFs, HTML, and text files
- Extracts text content by page
Note: Claude Opus 4.6 was prompted with the Phase 1 materials to find the sources for the corpus. 
Once found, PDF's were verified and downloaded manually.

### 2. Chunking
- Splits documents into ~500 token chunks
- 100 token overlap between chunks
- Detects section headers for semantic boundaries

### 3. Embedding & Indexing
- Embeds chunks using `all-MiniLM-L6-v2`
- Builds FAISS vector index for fast similarity search
- Stores chunk metadata for retrieval

### 4. Retrieval
- Embeds user question
- Retrieves top-k most similar chunks (default k=3)
- Filters by similarity threshold (default 0.35)

### 5. Generation
- Constructs prompt with retrieved chunks
- LLM generates answer with inline citations
- Each claim must cite chunk ID: `[CHUNK_ID=...]`

### 6. Evaluation

**v1 Metrics:**
- **Citation Precision**: % of citations that are valid (not hallucinated)
- **Groundedness**: % of sentences that have citations
- **Evidence Strength**: Mean similarity of cited chunks

**v2 Additional Metrics:**
- **Citation Recall**: % of retrieved chunks actually cited
- **Retrieval Coverage**: Max similarity score (how good was best match)
- **Overall Confidence**: Weighted composite score (0.0-1.0)
- **Quality Gate**: Rejects/flags answers below confidence threshold

### Status Values
- `ok`: Answer passed all checks
- `low_confidence`: (v2) Answer below confidence threshold
- `below_threshold`: No relevant chunks found
- `hallucinated_ids`: Fabricated citations detected
- `no_citations`: Answer provided but no citations
- `llm_error`: LLM failed to respond

### 7. Logging

**`logs/rag_logs.jsonl`**: Full query details (question, answer, chunks, prompt)
**`logs/eval_results.jsonl`**: Evaluation metrics only

## Configuration

Edit settings in the `.py` files:

```python
# Retrieval
TOP_K = 3                    # Chunks to retrieve
SIM_THRESHOLD = 0.35         # Minimum similarity
CHUNK_SIZE = 500             # Tokens per chunk

# LLM
LLM_MODEL = "qwen3:1.7b"     # Ollama model
LLM_TEMPERATURE = 0.1        # Determinism

# v2 Only: Confidence gating
MIN_OVERALL_CONFIDENCE = 0.50    # Threshold (0.0-1.0)
CONFIDENCE_GATE_ACTION = "flag"  # reject/flag/warn
```
## 📁 Project Structure

```
rag-pipeline/
├── setup_v1.ps1                    # Full setup for v1
├── setup_v2.ps1                    # Full setup for v2
├── evaluate_v1.ps1                 # Run v1 batch questions
├── evaluate_v2.ps1                 # Run v2 batch questions
├── rag_pipeline_eval_v1.py         # v1 Python script
├── rag_pipeline_eval_v2.py         # v2 Python script (with confidence)
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── raw/                        # 📥 PUT YOUR DOCUMENTS HERE
│   └── processed/
│       ├── manifest.jsonl          # Document metadata
│       └── chunks.jsonl            # Text chunks
│
├── index/
│   ├── faiss.index                 # FAISS vector index
│   └── faiss_meta.json             # Chunk metadata
│
├── logs/
│   ├── rag_logs.jsonl              # Detailed query logs
│   └── eval_results.jsonl          # Evaluation metrics
│
└── .venv/                          # Virtual environment (auto-created)
```

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| **First time setup (v1)** | `.\setup_v1.ps1` |
| **First time setup (v2)** | `.\setup_v2.ps1` |
| **Re-run batch (v1)** | `.\evaluate_v1.ps1` |
| **Re-run batch (v2)** | `.\evaluate_v2.ps1` |
| **Single query** | `python rag_pipeline_eval_vX.py query --question "..."` |
| **Rebuild index** | `python rag_pipeline_eval_vX.py run_all` |
| **Activate venv** | `.\.venv\Scripts\Activate.ps1` |
| **Check logs** | `Get-Content logs/eval_results.jsonl \| ConvertFrom-Json` |

## Troubleshooting

**"Python 3.12 required"**: Install Python 3.12 from python.org

**"Pipeline index not found"**: Run `setup_vX.ps1` first

**"LLM error"**: Verify Ollama is running (`ollama list`)

**Scripts won't run**: Both setup scripts handle execution policy automatically
