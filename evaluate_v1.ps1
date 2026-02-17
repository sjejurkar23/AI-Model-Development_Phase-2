Write-Host "=== RAG Pipeline v1 Batch Evaluation ===" -ForegroundColor Cyan

# Ensure we are in the script directory
Set-Location -Path $PSScriptRoot

# -----------------------------
# 1. Check if required files exist
# -----------------------------
if (-Not (Test-Path "rag_pipeline_eval_v1.py")) {
    Write-Host "ERROR: rag_pipeline_eval_v1.py not found." -ForegroundColor Red
    exit 1
}

if (-Not (Test-Path "index/faiss.index")) {
    Write-Host "ERROR: Pipeline index not found. Run setup_v1.ps1 first." -ForegroundColor Red
    exit 1
}

# -----------------------------
# 2. Activate virtual environment
# -----------------------------
if (-Not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found. Run setup_v1.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Activating virtual environment..."
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\.venv\Scripts\Activate.ps1

# -----------------------------
# 3. Run batch evaluation (v1)
# -----------------------------
Write-Host "Running v1 batch evaluation questions..." -ForegroundColor Cyan

python rag_pipeline_eval_v1.py batch

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Batch evaluation v1 encountered errors." -ForegroundColor Yellow
    exit 1
}

# -----------------------------
# 4. Display results summary
# -----------------------------
Write-Host "`n=== Evaluation v1 Complete ===" -ForegroundColor Green
Write-Host "Results saved to:" -ForegroundColor Cyan
Write-Host "  - logs/rag_logs.jsonl (detailed query logs)" -ForegroundColor White
Write-Host "  - logs/eval_results.jsonl (evaluation metrics)" -ForegroundColor White

# Count total queries
if (Test-Path "logs/eval_results.jsonl") {
    $totalQueries = (Get-Content "logs/eval_results.jsonl" | Measure-Object -Line).Lines
    Write-Host "`nTotal queries logged: $totalQueries" -ForegroundColor Cyan
}
