# rag_pipeline.py — FAISS-based RAG pipeline (Windows-friendly)
import argparse
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss


# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "index"
LOG_PATH = ROOT / "logs" / "rag_logs.jsonl"
EVAL_PATH = ROOT / "logs" / "eval_results.jsonl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3          # reduced from 5 — shorter prompt, faster inference
SIM_THRESHOLD = 0.35

LLM_TIMEOUT     = 180  # seconds — raised to survive slow cold-starts
LLM_NUM_PREDICT = -1  
LLM_TEMPERATURE = 0.1  # lower = faster + more deterministic

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen3:1.7b"
PROMPT_VERSION = "v2.3"  # Updated to v2.3 with confidence gating

# -----------------------------
# CONFIDENCE GATING THRESHOLDS
# -----------------------------
# These thresholds control answer quality before returning to user
CONFIDENCE_GATE_ENABLED = True  # Set to False to disable confidence filtering

# Minimum overall confidence threshold - only this is checked for gating
# Individual metrics (groundedness, precision, recall, evidence, coverage) are 
# still computed and logged, but not used for filtering
MIN_OVERALL_CONFIDENCE = 0.50      # Composite confidence score [0.0-1.0]

# Action to take when confidence is below threshold
CONFIDENCE_GATE_ACTION = "flag"    # Options: "reject", "flag", "warn"
# - "reject": Return a refusal message, don't show the answer
# - "flag": Show answer with a prominent warning banner
# - "warn": Show answer with a subtle confidence notice

# Note: Individual metric thresholds are no longer used for gating.
# The overall confidence score is a weighted combination of:
#   - Groundedness (30%): Fraction of sentences with citations
#   - Citation Precision (25%): Fraction of valid citations
#   - Evidence Strength (25%): Mean similarity of cited chunks
#   - Retrieval Coverage (15%): Max similarity in retrieved set
#   - Citation Recall (5%): Fraction of retrieved chunks cited


def ensure_dirs():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# CACHED RESOURCES
# Loaded once on first query, reused for every subsequent call.
# Saves 2-5s per question compared to reloading inside retrieve().
# -----------------------------
_embed_model = None
_faiss_index = None
_faiss_meta  = None


def _load_resources():
    global _embed_model, _faiss_index, _faiss_meta
    if _embed_model is None:
        logger.info("Loading embedding model...")
        _embed_model = SentenceTransformer(EMBED_MODEL)
    if _faiss_index is None:
        logger.info("Loading FAISS index...")
        _faiss_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "faiss_meta.json", "r", encoding="utf-8") as f:
            _faiss_meta = json.load(f)


# -----------------------------
# DATA MODELS
# -----------------------------
class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    page: int | None = None
    section: str | None = None
    text: str


# -----------------------------
# INGEST
# -----------------------------
def parse_pdf(path: Path):
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
    return pages


def parse_html(path: Path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return [{"page": None, "text": text}]


def parse_txt(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"page": None, "text": text}]


def ingest():
    ensure_dirs()
    manifest = DATA_PROCESSED / "manifest.jsonl"

    with manifest.open("w", encoding="utf-8") as mf:
        for path in tqdm(list(DATA_RAW.glob("**/*")), desc="Ingesting"):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            if suffix == ".pdf":
                pages = parse_pdf(path)
            elif suffix in {".html", ".htm"}:
                pages = parse_html(path)
            elif suffix in {".txt", ".md"}:
                pages = parse_txt(path)
            else:
                continue

            doc_id = str(uuid.uuid4())
            for page in pages:
                rec = {
                    "doc_id": doc_id,
                    "source_path": str(path),
                    "page": page["page"],
                    "text": page["text"],
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Ingestion complete.")


# -----------------------------
# CHUNKING
# -----------------------------
def tokenize(text):
    return re.findall(r"\w+|\S", text)


def detokenize(tokens):
    return " ".join(tokens)


def chunk_tokens(tokens, size, overlap):
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def detect_headers(lines):
    headers = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if re.match(r"^\d+(\.\d+)*\s+\w+", s) or s.isupper():
            headers.append(i)
    return headers


def chunk_by_section_or_tokens(text, doc_id, source_path, page):
    lines = text.splitlines()
    headers = detect_headers(lines)
    chunks = []

    if headers:
        boundaries = headers + [len(lines)]
        for i in range(len(headers)):
            start, end = boundaries[i], boundaries[i + 1]
            section_text = "\n".join(lines[start:end]).strip()
            if not section_text:
                continue
            section_name = lines[headers[i]].strip()
            chunk_id = f"{Path(source_path).stem}-{doc_id}-sec-{i}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_path=source_path,
                    page=page,
                    section=section_name,
                    text=section_text,
                )
            )
    else:
        tokens = tokenize(text)
        for i, tchunk in enumerate(chunk_tokens(tokens, CHUNK_SIZE, CHUNK_OVERLAP)):
            chunk_id = f"{Path(source_path).stem}-{doc_id}-tok-{i}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_path=source_path,
                    page=page,
                    section=None,
                    text=detokenize(tchunk),
                )
            )

    return chunks


def build_chunks():
    manifest = DATA_PROCESSED / "manifest.jsonl"
    out = DATA_PROCESSED / "chunks.jsonl"

    with manifest.open("r", encoding="utf-8") as mf, out.open("w", encoding="utf-8") as cf:
        for line in tqdm(mf, desc="Chunking"):
            rec = json.loads(line)
            for chunk in chunk_by_section_or_tokens(
                rec["text"], rec["doc_id"], rec["source_path"], rec["page"]
            ):
                cf.write(chunk.model_dump_json() + "\n")

    logger.info("Chunking complete.")


# -----------------------------
# EMBEDDING + INDEX (FAISS)
# -----------------------------
def embed_and_index():
    chunks_path = DATA_PROCESSED / "chunks.jsonl"

    model = SentenceTransformer(EMBED_MODEL)

    texts, ids, metas = [], [], []

    with chunks_path.open("r", encoding="utf-8") as cf:
        for line in cf:
            rec = ChunkRecord.model_validate_json(line)
            texts.append(rec.text)
            ids.append(rec.chunk_id)
            metas.append(
                {
                    "doc_id": rec.doc_id,
                    "source_path": rec.source_path,
                    "page": rec.page,
                    "section": rec.section,
                }
            )

    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "faiss_meta.json", "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "metas": metas, "texts": texts}, f)

    logger.info("FAISS indexing complete.")


# -----------------------------
# RETRIEVE + GENERATE
# -----------------------------
def call_llm(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": LLM_NUM_PREDICT,
            "temperature": LLM_TEMPERATURE,
        },
    }

    try:
        r = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        content = body.get("response", "").strip()

        if not content:
            logger.warning("Ollama returned an empty response — model may need a restart.")
            return "LLM error: empty response from model"

        # Ollama sets done_reason="length" when num_predict is hit mid-generation
        if body.get("done_reason") == "length":
            logger.warning(
                f"Output truncated at {LLM_NUM_PREDICT} tokens (done_reason=length). "
                "Raise LLM_NUM_PREDICT if answers are being cut off."
            )

        return content
    except Exception as e:
        return f"LLM error: {e}"


def retrieve(question):
    _load_resources()

    q_emb = _embed_model.encode([question])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, idxs = _faiss_index.search(q_emb, TOP_K)

    idxs   = idxs[0]
    scores = scores[0]

    ids    = _faiss_meta["ids"]
    metas  = _faiss_meta["metas"]
    texts  = _faiss_meta["texts"]

    retrieved_ids    = [ids[i]   for i in idxs]
    retrieved_texts  = [texts[i] for i in idxs]
    retrieved_metas = [metas[i] for i in idxs]
    retrieved_scores = scores.tolist()

    return retrieved_ids, retrieved_texts, retrieved_metas, retrieved_scores


# FIX 1: Rewritten prompt with explicit inline citation instruction and
#         a concrete few-shot example so small models (llama3.2:3b) follow it.
def build_prompt(question, ids, docs, metas, sims):
    blocks = []
    for cid, doc, meta, sim in zip(ids, docs, metas, sims):
        header = f"[CHUNK_ID={cid} | DOC_ID={meta['doc_id']} | SIM={sim:.3f}]"
        loc = f"Source: {meta['source_path']} | Page: {meta['page']} | Section: {meta['section']}"
        blocks.append(f"{header}\n{loc}\n{doc}\n")

    context = "\n\n".join(blocks)

    # Build a dynamic few-shot example using the first real chunk ID so
    # the model sees exactly the format it must reproduce.
    example_id = ids[0] if ids else "some-chunk-id"

    return f"""You are a cautious, citation-accurate assistant.

STRICT RULES — follow every one of them:
1. Base your answer ONLY on the chunks provided below.
2. Every factual claim in your answer MUST be followed immediately by the
   chunk ID it came from, written exactly as [CHUNK_ID=<id>].
3. If a sentence draws on multiple chunks, cite all of them:
   e.g. "... [CHUNK_ID=id1] [CHUNK_ID=id2]"
4. Do NOT place all citations at the end. Cite inline, right after the claim.
5. If the chunks do not contain enough information, respond with exactly:
   "I cannot answer this from the provided corpus."
6. Never fabricate or guess chunk IDs.

EXAMPLE of correct citation style (using a real chunk ID from this query):
  "FActScore measures factual precision of LLM outputs [CHUNK_ID={example_id}].
   It decomposes a generation into atomic claims and verifies each one against
   a reference corpus [CHUNK_ID={example_id}]."

Question:
{question}

Context chunks:
{context}

Answer (cite every claim inline with [CHUNK_ID=...]):""".strip()


def extract_citations(answer):
    return set(re.findall(r"\[CHUNK_ID=([^\]]+)\]", answer))


# -----------------------------
# EVALUATION METRICS
# -----------------------------
def compute_groundedness(answer: str, cited_ids: set) -> float:
    """
    Fraction of non-empty sentences in the answer that contain at least
    one inline [CHUNK_ID=...] citation.

    Returns a float in [0.0, 1.0], or 0.0 for empty / unanswerable replies.
    """
    # Strip the no-answer sentinel and WARNING footer before scoring
    clean = answer
    for marker in [
        "I cannot answer this from the provided corpus.",
        "[WARNING: No chunk citations were found in this answer.",
    ]:
        clean = clean.replace(marker, "")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if not sentences:
        return 0.0

    cited_sentence_count = sum(
        1 for s in sentences if re.search(r"\[CHUNK_ID=", s)
    )
    return round(cited_sentence_count / len(sentences), 4)


def compute_citation_precision(cited_ids: set, retrieved_ids: list) -> float:
    """
    Fraction of chunk IDs cited in the answer that actually belong to the
    set of retrieved chunks (i.e. were legitimately available to the model).

    Returns 1.0 when no citations are present (nothing was fabricated),
    and a value in [0.0, 1.0] otherwise.
    """
    if not cited_ids:
        return 1.0
    allowed = set(retrieved_ids)
    valid = cited_ids & allowed
    return round(len(valid) / len(cited_ids), 4)


def compute_citation_recall(cited_ids: set, retrieved_ids: list) -> float:
    """
    Fraction of retrieved chunks that were actually cited in the answer.
    
    Citation recall measures how comprehensively the model used the available
    retrieved evidence when generating its answer.

    Returns 0.0 when no chunks were retrieved (edge case),
    and a value in [0.0, 1.0] otherwise indicating the proportion of
    retrieved chunks that were cited.
    """
    if not retrieved_ids:
        return 0.0
    allowed = set(retrieved_ids)
    valid = cited_ids & allowed
    return round(len(valid) / len(allowed), 4)


def compute_evidence_strength(
    cited_ids: set,
    retrieved_ids: list,
    sims: list,
) -> float:
    """
    Mean cosine similarity of the chunks that were both cited in the answer
    AND legitimately retrieved — i.e. the average retrieval confidence of
    the evidence the model actually drew on.

    Interpretation:
      1.0  — perfect semantic match between question and cited chunks
      >=0.7 — strong evidence; answer is well-supported
      0.5-0.7 — moderate evidence; treat with some caution
      <0.5 — weak evidence; answer may be tenuously grounded
      0.0  — no valid citations (below-threshold, error, or hallucinated IDs)

    Args:
        cited_ids:     chunk IDs extracted from the answer text.
        retrieved_ids: ordered list of chunk IDs returned by retrieve().
        sims:          cosine similarity scores, parallel to retrieved_ids.

    Returns a float in [0.0, 1.0].
    """
    if not cited_ids:
        return 0.0

    # Build a map from chunk_id to similarity for the retrieved set
    sim_map = {cid: sim for cid, sim in zip(retrieved_ids, sims)}

    # Only score IDs that are both cited AND legitimately retrieved
    valid_sims = [sim_map[cid] for cid in cited_ids if cid in sim_map]

    if not valid_sims:
        return 0.0

    return round(float(np.mean(valid_sims)), 4)


def compute_evidence_strength_stats(
    cited_ids: set,
    retrieved_ids: list,
    sims: list,
) -> dict:
    """
    Detailed statistics about the evidence strength of cited chunks.
    
    Returns a dictionary with:
      - mean: average similarity (same as compute_evidence_strength)
      - min: weakest cited chunk similarity
      - max: strongest cited chunk similarity
      - std: standard deviation of similarities
      - count: number of valid citations
    
    Args:
        cited_ids:     chunk IDs extracted from the answer text.
        retrieved_ids: ordered list of chunk IDs returned by retrieve().
        sims:          cosine similarity scores, parallel to retrieved_ids.
    
    Returns a dict with statistical measures.
    """
    if not cited_ids:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "count": 0,
        }
    
    # Build a map from chunk_id to similarity for the retrieved set
    sim_map = {cid: sim for cid, sim in zip(retrieved_ids, sims)}
    
    # Only score IDs that are both cited AND legitimately retrieved
    valid_sims = [sim_map[cid] for cid in cited_ids if cid in sim_map]
    
    if not valid_sims:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "count": 0,
        }
    
    return {
        "mean": round(float(np.mean(valid_sims)), 4),
        "min": round(float(np.min(valid_sims)), 4),
        "max": round(float(np.max(valid_sims)), 4),
        "std": round(float(np.std(valid_sims)), 4),
        "count": len(valid_sims),
    }


def compute_retrieval_coverage(retrieved_sims: list) -> float:
    """
    Measures how well the top retrieved chunks cover the question.
    Uses the max similarity score from the retrieved set.
    
    Interpretation:
      >=0.7 — excellent coverage; strong matches found
      0.5-0.7 — good coverage; relevant chunks retrieved
      0.35-0.5 — marginal coverage; borderline relevance
      <0.35 — poor coverage; may not have relevant info
    
    Args:
        retrieved_sims: similarity scores of retrieved chunks
    
    Returns a float in [0.0, 1.0].
    """
    if not retrieved_sims:
        return 0.0
    return round(float(max(retrieved_sims)), 4)


def compute_overall_confidence(
    groundedness: float,
    citation_precision: float,
    citation_recall: float,
    evidence_strength: float,
    retrieval_coverage: float,
) -> dict:
    """
    Computes an overall confidence score and categorical rating for the answer.
    
    The confidence score is a weighted combination of multiple quality signals:
      - Groundedness (30%): Are claims cited?
      - Citation Precision (25%): Are citations valid?
      - Evidence Strength (25%): How relevant are cited chunks?
      - Retrieval Coverage (15%): Was good evidence available?
      - Citation Recall (5%): How comprehensive was evidence usage?
    
    Returns a dict with:
      - score: weighted confidence score [0.0, 1.0]
      - level: categorical rating (very_high, high, medium, low, very_low)
      - factors: breakdown of contributing factors
    
    Args:
        groundedness: fraction of sentences with citations
        citation_precision: fraction of valid citations
        citation_recall: fraction of retrieved chunks cited
        evidence_strength: mean similarity of cited chunks
        retrieval_coverage: max similarity of retrieved chunks
    
    Returns a dict with score, level, and factors.
    """
    # Weighted combination
    weights = {
        "groundedness": 0.30,
        "citation_precision": 0.25,
        "evidence_strength": 0.25,
        "retrieval_coverage": 0.15,
        "citation_recall": 0.05,
    }
    
    score = (
        weights["groundedness"] * groundedness +
        weights["citation_precision"] * citation_precision +
        weights["evidence_strength"] * evidence_strength +
        weights["retrieval_coverage"] * retrieval_coverage +
        weights["citation_recall"] * citation_recall
    )
    score = round(score, 4)
    
    # Categorical rating
    if score >= 0.85:
        level = "very_high"
    elif score >= 0.70:
        level = "high"
    elif score >= 0.50:
        level = "medium"
    elif score >= 0.30:
        level = "low"
    else:
        level = "very_low"
    
    return {
        "score": score,
        "level": level,
        "factors": {
            "groundedness": groundedness,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "evidence_strength": evidence_strength,
            "retrieval_coverage": retrieval_coverage,
        }
    }


def evaluate_confidence_gate(
    overall_confidence: dict,
    groundedness: float,
    citation_precision: float,
    evidence_strength: float,
    retrieval_coverage: float,
) -> dict:
    """
    Evaluates whether an answer meets minimum confidence threshold.
    
    Only checks the overall composite confidence score, not individual metrics.
    Individual metrics are still computed and logged for analysis.
    
    Returns a dict with:
      - passed: bool indicating if overall confidence threshold met
      - failures: list of failed criteria (only overall_confidence if failed)
      - action: recommended action (reject/flag/warn/pass)
      - message: user-facing message if confidence is low
    
    Args:
        overall_confidence: output from compute_overall_confidence
        groundedness: fraction of sentences with citations (logged but not gated)
        citation_precision: fraction of valid citations (logged but not gated)
        evidence_strength: mean similarity of cited chunks (logged but not gated)
        retrieval_coverage: max similarity of retrieved chunks (logged but not gated)
    
    Returns evaluation dict.
    """
    if not CONFIDENCE_GATE_ENABLED:
        return {
            "passed": True,
            "failures": [],
            "action": "pass",
            "message": None,
        }
    
    failures = []
    
    # Only check overall confidence score
    if overall_confidence["score"] < MIN_OVERALL_CONFIDENCE:
        failures.append({
            "criterion": "overall_confidence",
            "actual": overall_confidence["score"],
            "required": MIN_OVERALL_CONFIDENCE,
            "delta": round(MIN_OVERALL_CONFIDENCE - overall_confidence["score"], 4),
        })
    
    passed = len(failures) == 0
    
    # Determine action and message
    if passed:
        action = "pass"
        message = None
    else:
        action = CONFIDENCE_GATE_ACTION
        
        # Build user-facing message
        if action == "reject":
            message = (
                "I cannot provide a confident answer to this question based on the available information. "
                f"The answer quality score is {overall_confidence['score']:.2f} (confidence level: {overall_confidence['level']}), "
                f"which is below the minimum threshold of {MIN_OVERALL_CONFIDENCE:.2f} required for a reliable response.\n\n"
                "This may be because:\n"
                "- The retrieved documents don't contain sufficient relevant information\n"
                "- The question requires information not present in the corpus\n"
                "- The evidence found is not strong enough to support a definitive answer\n\n"
                "Please try rephrasing your question or consult additional sources."
            )
        elif action == "flag":
            message = (
                f"⚠️ LOW CONFIDENCE WARNING ⚠️\n\n"
                f"Answer Confidence: {overall_confidence['level'].upper()} ({overall_confidence['score']:.2f}/1.00)\n\n"
                f"This answer falls below the confidence threshold of {MIN_OVERALL_CONFIDENCE:.2f}. "
                f"Please verify this information independently before relying on it.\n\n"
                "---"
            )
        else:  # warn
            message = (
                f"Note: This answer has moderate confidence (score: {overall_confidence['score']:.2f}, "
                f"level: {overall_confidence['level']}). Please use with appropriate caution."
            )
    
    return {
        "passed": passed,
        "failures": failures,
        "action": action,
        "message": message,
    }


# -----------------------------
# EVAL RECORD WRITER
# -----------------------------
def write_eval_record(
    *,
    query_id: str,
    groundedness: float,
    citation_precision: float,
    citation_recall: float,
    evidence_strength: float,
    evidence_strength_stats: dict,
    retrieval_coverage: float,
    overall_confidence: dict,
    confidence_gate_eval: dict,
    status: str,
):
    """Append one evaluation record to EVAL_PATH with minimal fields."""
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Only include the essential fields requested
    record = {
        "query_id": query_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "overall_confidence": {
            "score": overall_confidence["score"],
            "level": overall_confidence["level"],
            "factors": {
                "groundedness": groundedness,
                "citation_precision": citation_precision,
                "citation_recall": citation_recall,
                "evidence_strength": evidence_strength,
                "retrieval_coverage": retrieval_coverage,
            }
        }
    }

    with EVAL_PATH.open("a", encoding="utf-8") as ef:
        ef.write(json.dumps(record, ensure_ascii=False) + "\n")

    gate_status = "✓ PASSED" if confidence_gate_eval["passed"] else f"✗ {confidence_gate_eval['action'].upper()}"
    logger.info(
        f"Eval recorded → {EVAL_PATH}  "
        f"[query_id={query_id}, confidence={overall_confidence['level']} ({overall_confidence['score']:.3f}), "
        f"gate={gate_status}, status={status}]"
    )


# -----------------------------
# STRUCTURED QUERY LOGGING
# -----------------------------
def write_query_log(
    *,
    query_id: str,
    question: str,
    prompt: str,
    retrieved_chunks: list,
    answer: str,
    cited_ids: set,
    status: str,
    latency_s: float,
    groundedness: float = 0.0,
    citation_precision: float = 1.0,
    citation_recall: float = 0.0,
    evidence_strength: float = 0.0,
    evidence_strength_stats: dict = None,
    retrieval_coverage: float = 0.0,
    overall_confidence: dict = None,
    confidence_gate_eval: dict = None,
):
    """Append one JSONL record to LOG_PATH capturing the full query trace."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if evidence_strength_stats is None:
        evidence_strength_stats = {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
    
    if overall_confidence is None:
        overall_confidence = {"score": 0.0, "level": "very_low", "factors": {}}
    
    if confidence_gate_eval is None:
        confidence_gate_eval = {"passed": False, "failures": [], "action": "reject", "message": None}

    record = {
        "query_id": query_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "status": status,                    # "ok" | "no_citations" | "hallucinated_ids" | "llm_error" | "below_threshold" | "low_confidence"
        "latency_s": round(latency_s, 3),
        "question": question,
        "prompt": prompt,
        "retrieved_chunks": retrieved_chunks, # list of {chunk_id, sim, source_path, page, section, text}
        "cited_chunk_ids": sorted(cited_ids),
        "answer": answer,
        "groundedness": groundedness,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "evidence_strength": evidence_strength,
        "evidence_strength_stats": evidence_strength_stats,
        "retrieval_coverage": retrieval_coverage,
        "overall_confidence": overall_confidence,
        "confidence_gate_eval": confidence_gate_eval,
    }

    with LOG_PATH.open("a", encoding="utf-8") as lf:
        lf.write(json.dumps(record, ensure_ascii=False) + "\n")

    gate_status = "passed" if confidence_gate_eval["passed"] else confidence_gate_eval["action"]
    logger.info(f"Query logged → {LOG_PATH}  [query_id={query_id}, status={status}, confidence={overall_confidence['level']}, gate={gate_status}]")


# FIX 2: Treat an answer with zero citations as an error, not a silent pass.
def answer_question(question):
    query_id = str(uuid.uuid4())
    t_start = time.monotonic()

    ids, docs, metas, sims = retrieve(question)

    # Helper: build the chunk list for the log regardless of outcome
    def chunk_log_entries():
        return [
            {
                "chunk_id": cid,
                "sim": round(sim, 4),
                "source_path": meta["source_path"],
                "page": meta["page"],
                "section": meta["section"],
                "text": doc,
            }
            for cid, doc, meta, sim in zip(ids, docs, metas, sims)
        ]

    # --- Below similarity threshold: log and bail early ---
    if not ids or max(sims) < SIM_THRESHOLD:
        answer = "I cannot answer this from the provided corpus."
        groundedness = 0.0
        citation_precision = 1.0
        citation_recall = 0.0
        evidence_strength = 0.0
        evidence_strength_stats = {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
        retrieval_coverage = compute_retrieval_coverage(sims) if sims else 0.0
        overall_confidence = compute_overall_confidence(
            groundedness, citation_precision, citation_recall, evidence_strength, retrieval_coverage
        )
        confidence_gate_eval = evaluate_confidence_gate(
            overall_confidence, groundedness, citation_precision, evidence_strength, retrieval_coverage
        )
        write_query_log(
            query_id=query_id,
            question=question,
            prompt="",
            retrieved_chunks=chunk_log_entries(),
            answer=answer,
            cited_ids=set(),
            status="below_threshold",
            latency_s=time.monotonic() - t_start,
            groundedness=groundedness,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            evidence_strength=evidence_strength,
            evidence_strength_stats=evidence_strength_stats,
            retrieval_coverage=retrieval_coverage,
            overall_confidence=overall_confidence,
            confidence_gate_eval=confidence_gate_eval,
        )
        write_eval_record(
            query_id=query_id,
            groundedness=groundedness,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            evidence_strength=evidence_strength,
            evidence_strength_stats=evidence_strength_stats,
            retrieval_coverage=retrieval_coverage,
            overall_confidence=overall_confidence,
            confidence_gate_eval=confidence_gate_eval,
            status="below_threshold",
        )
        return answer

    prompt = build_prompt(question, ids, docs, metas, sims)
    answer = call_llm(prompt)

    # --- LLM transport/timeout error ---
    if answer.startswith("LLM error:"):
        groundedness = 0.0
        citation_precision = 1.0
        citation_recall = 0.0
        evidence_strength = 0.0
        evidence_strength_stats = {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
        retrieval_coverage = compute_retrieval_coverage(sims)
        overall_confidence = compute_overall_confidence(
            groundedness, citation_precision, citation_recall, evidence_strength, retrieval_coverage
        )
        confidence_gate_eval = evaluate_confidence_gate(
            overall_confidence, groundedness, citation_precision, evidence_strength, retrieval_coverage
        )
        write_query_log(
            query_id=query_id,
            question=question,
            prompt=prompt,
            retrieved_chunks=chunk_log_entries(),
            answer=answer,
            cited_ids=set(),
            status="llm_error",
            latency_s=time.monotonic() - t_start,
            groundedness=groundedness,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            evidence_strength=evidence_strength,
            evidence_strength_stats=evidence_strength_stats,
            retrieval_coverage=retrieval_coverage,
            overall_confidence=overall_confidence,
            confidence_gate_eval=confidence_gate_eval,
        )
        write_eval_record(
            query_id=query_id,
            groundedness=groundedness,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            evidence_strength=evidence_strength,
            evidence_strength_stats=evidence_strength_stats,
            retrieval_coverage=retrieval_coverage,
            overall_confidence=overall_confidence,
            confidence_gate_eval=confidence_gate_eval,
            status="llm_error",
        )
        return answer

    cited = extract_citations(answer)
    allowed = set(ids)

    # --- Hallucinated chunk IDs ---
    if cited and not cited.issubset(allowed):
        logger.warning(f"Hallucinated chunk IDs detected: {cited - allowed}")
        safe_answer = (
            "Generated answer referenced invalid chunk IDs. "
            "Refusing to provide fabricated citations."
        )
        groundedness = compute_groundedness(answer, cited)
        citation_precision = compute_citation_precision(cited, ids)
        citation_recall = compute_citation_recall(cited, ids)
        evidence_strength = compute_evidence_strength(cited, ids, sims)
        evidence_strength_stats = compute_evidence_strength_stats(cited, ids, sims)
        retrieval_coverage = compute_retrieval_coverage(sims)
        overall_confidence = compute_overall_confidence(
            groundedness, citation_precision, citation_recall, evidence_strength, retrieval_coverage
        )
        confidence_gate_eval = evaluate_confidence_gate(
            overall_confidence, groundedness, citation_precision, evidence_strength, retrieval_coverage
        )
        write_query_log(
            query_id=query_id,
            question=question,
            prompt=prompt,
            retrieved_chunks=chunk_log_entries(),
            answer=safe_answer,
            cited_ids=cited,
            status="hallucinated_ids",
            latency_s=time.monotonic() - t_start,
            groundedness=groundedness,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            evidence_strength=evidence_strength,
            evidence_strength_stats=evidence_strength_stats,
            retrieval_coverage=retrieval_coverage,
            overall_confidence=overall_confidence,
            confidence_gate_eval=confidence_gate_eval,
        )
        write_eval_record(
            query_id=query_id,
            groundedness=groundedness,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            evidence_strength=evidence_strength,
            evidence_strength_stats=evidence_strength_stats,
            retrieval_coverage=retrieval_coverage,
            overall_confidence=overall_confidence,
            confidence_gate_eval=confidence_gate_eval,
            status="hallucinated_ids",
        )
        return safe_answer

    # --- Answer returned but no citations produced ---
    no_citation_signal = "I cannot answer this from the provided corpus."
    if not cited and no_citation_signal not in answer:
        logger.warning("LLM returned an answer with no citations — flagging.")
        flagged_answer = (
            f"{answer}\n\n"
            "[WARNING: No chunk citations were found in this answer. "
            "Treat it as unverified.]"
        )
        groundedness = 0.0
        citation_precision = 1.0
        citation_recall = 0.0
        evidence_strength = 0.0
        evidence_strength_stats = {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
        retrieval_coverage = compute_retrieval_coverage(sims)
        overall_confidence = compute_overall_confidence(
            groundedness, citation_precision, citation_recall, evidence_strength, retrieval_coverage
        )
        confidence_gate_eval = evaluate_confidence_gate(
            overall_confidence, groundedness, citation_precision, evidence_strength, retrieval_coverage
        )
        write_query_log(
            query_id=query_id,
            question=question,
            prompt=prompt,
            retrieved_chunks=chunk_log_entries(),
            answer=flagged_answer,
            cited_ids=set(),
            status="no_citations",
            latency_s=time.monotonic() - t_start,
            groundedness=groundedness,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            evidence_strength=evidence_strength,
            evidence_strength_stats=evidence_strength_stats,
            retrieval_coverage=retrieval_coverage,
            overall_confidence=overall_confidence,
            confidence_gate_eval=confidence_gate_eval,
        )
        write_eval_record(
            query_id=query_id,
            groundedness=groundedness,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            evidence_strength=evidence_strength,
            evidence_strength_stats=evidence_strength_stats,
            retrieval_coverage=retrieval_coverage,
            overall_confidence=overall_confidence,
            confidence_gate_eval=confidence_gate_eval,
            status="no_citations",
        )
        return flagged_answer

    # --- Happy path ---
    groundedness = compute_groundedness(answer, cited)
    citation_precision = compute_citation_precision(cited, ids)
    citation_recall = compute_citation_recall(cited, ids)
    evidence_strength = compute_evidence_strength(cited, ids, sims)
    evidence_strength_stats = compute_evidence_strength_stats(cited, ids, sims)
    retrieval_coverage = compute_retrieval_coverage(sims)
    overall_confidence = compute_overall_confidence(
        groundedness, citation_precision, citation_recall, evidence_strength, retrieval_coverage
    )
    
    # --- CONFIDENCE GATE: Evaluate answer quality before returning ---
    confidence_gate_eval = evaluate_confidence_gate(
        overall_confidence, groundedness, citation_precision, evidence_strength, retrieval_coverage
    )
    
    # Determine final status and answer based on confidence gate
    if confidence_gate_eval["passed"]:
        final_status = "ok"
        final_answer = answer
    else:
        final_status = "low_confidence"
        action = confidence_gate_eval["action"]
        
        if action == "reject":
            # Replace answer with refusal message
            final_answer = confidence_gate_eval["message"]
            logger.warning(
                f"Confidence gate REJECTED answer: "
                f"confidence={overall_confidence['score']:.3f} ({overall_confidence['level']}), "
                f"failures={len(confidence_gate_eval['failures'])}"
            )
        elif action == "flag":
            # Prepend warning banner to answer
            final_answer = f"{confidence_gate_eval['message']}\n\n{answer}"
            logger.warning(
                f"Confidence gate FLAGGED answer: "
                f"confidence={overall_confidence['score']:.3f} ({overall_confidence['level']}), "
                f"failures={len(confidence_gate_eval['failures'])}"
            )
        else:  # warn
            # Append subtle warning to answer
            final_answer = f"{answer}\n\n{confidence_gate_eval['message']}"
            logger.info(
                f"Confidence gate WARNED on answer: "
                f"confidence={overall_confidence['score']:.3f} ({overall_confidence['level']})"
            )
    
    write_query_log(
        query_id=query_id,
        question=question,
        prompt=prompt,
        retrieved_chunks=chunk_log_entries(),
        answer=final_answer,
        cited_ids=cited,
        status=final_status,
        latency_s=time.monotonic() - t_start,
        groundedness=groundedness,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        evidence_strength=evidence_strength,
        evidence_strength_stats=evidence_strength_stats,
        retrieval_coverage=retrieval_coverage,
        overall_confidence=overall_confidence,
        confidence_gate_eval=confidence_gate_eval,
    )
    write_eval_record(
        query_id=query_id,
        groundedness=groundedness,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        evidence_strength=evidence_strength,
        evidence_strength_stats=evidence_strength_stats,
        retrieval_coverage=retrieval_coverage,
        overall_confidence=overall_confidence,
        confidence_gate_eval=confidence_gate_eval,
        status=final_status,
    )
    return final_answer



# -----------------------------
# BATCH QUESTIONS
# -----------------------------
BATCH_QUESTIONS = {
    "direct": [
        "What does FActScore measure and what are its known failure modes?",
        "What is the AIS framework and how does it define a fully attributable statement?",
        "What is TruthfulQA's inverse scaling finding?",
        "What hallucination rate does FAVA report for ChatGPT and Llama 2?",
        "How does ALCE define citation recall/precision and what were best-model results?",
        "What is the original RAG paradigm and its factuality improvement claim?",
        "How do hallucination rates differ between data-driven and prompt-driven feedback systems for student assignments?",
        "What are the three RAG paradigms and what distinguishes Modular RAG?",
        "What does Self-RAG introduce and what gains does it report over standard RAG?",
        "What does the Wharton report conclude about CoT for reasoning models?",
    ],
    "synthesis": [
        "Compare the findings of the 2022 foundational Chain-of-Thought (CoT) study with the 2025 technical report regarding the effectiveness of CoT for different model types.",
        "How do FActScore, ALCE, and RAGAS operationalize 'faithfulness'?",
        "Contrast the performance improvements in citation quality achieved by standard RAG architectures versus the Agent-based citation approach in S25.",
        "Discuss the variance in reported ChatGPT hallucination rates across the HaluEval benchmark and the FavaBench empirical study.",
        "Based on the provided surveys, how does the definition of 'faithfulness' in model explanations differ from 'faithfulness' in general reasoning tasks?",
        "Do S05 and S10 together support that CoT reliably improves reasoning faithfulness and citation quality?",
    ],
    "edge": [
        "Does the provided corpus contain a specific list of the top five most reliable Large Language Models for performing robotic surgery?",
        "According to the HaluEval 2.0 study, does Chain-of-Thought reasoning consistently reduce hallucinations regardless of the model's scale?",
        "What specific GPU and hardware requirements are listed in the Self-RAG paper for reproducing their results?",
        "Is there any evidence in the provided manifest suggesting that retrieval augmentation can occasionally increase rather than decrease hallucination rates?",
        "Provide a direct quote from S18 explaining exactly why a model explanation is considered unfaithful if it uses concept-based influence.",
        "Does the corpus contain direct evidence that structured prompts reduce hallucination in open-domain QA?",
    ],
}


def run_batch(categories: list[str] | None = None, delay: float = 0.0):
    """
    Run all questions in BATCH_QUESTIONS and print results to stdout.

    Args:
        categories: list of category keys to run, e.g. ["direct", "edge"].
                    Pass None to run all categories.
        delay:      optional sleep in seconds between questions (useful to
                    avoid overwhelming a local LLM server).
    """
    import time as _time

    cats = categories or list(BATCH_QUESTIONS.keys())
    total = sum(len(BATCH_QUESTIONS[c]) for c in cats if c in BATCH_QUESTIONS)
    run = 0

    for cat in cats:
        questions = BATCH_QUESTIONS.get(cat)
        if questions is None:
            logger.warning(f"Unknown category '{cat}' — skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"  CATEGORY: {cat.upper()}  ({len(questions)} questions)")
        print(f"{'='*70}")

        for i, question in enumerate(questions, 1):
            run += 1
            print(f"\n[{run}/{total}] {cat.upper()} Q{i}")
            print(f"Q: {question}")
            print("-" * 60)
            answer = answer_question(question)
            print(f"A: {answer}")
            print()

            if delay > 0:
                _time.sleep(delay)

    print(f"\n{'='*70}")
    print(f"  Batch complete — {run} questions answered.")
    print(f"  Logs written to: {LOG_PATH}")
    print(f"{'='*70}\n")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest")
    sub.add_parser("chunk")
    sub.add_parser("index")
    sub.add_parser("run_all")

    q = sub.add_parser("query")
    q.add_argument("--question", required=True)

    b = sub.add_parser("batch", help="Run all predefined benchmark questions")
    b.add_argument(
        "--categories",
        nargs="+",
        choices=["direct", "synthesis", "edge"],
        default=None,
        help="Which question categories to run (default: all)",
    )
    b.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between questions (default: 0)",
    )

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest()
    elif args.cmd == "chunk":
        build_chunks()
    elif args.cmd == "index":
        embed_and_index()
    elif args.cmd == "run_all":
        ingest()
        build_chunks()
        embed_and_index()
    elif args.cmd == "query":
        print(answer_question(args.question))
    elif args.cmd == "batch":
        run_batch(categories=args.categories, delay=args.delay)


if __name__ == "__main__":
    main()