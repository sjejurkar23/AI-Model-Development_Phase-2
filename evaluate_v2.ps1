Write-Host "=== RAG Pipeline v2 Batch Evaluation ===" -ForegroundColor Cyan

# Ensure we are in the script directory
Set-Location -Path $PSScriptRoot

# -----------------------------
# 1. Check if required files exist
# -----------------------------
if (-Not (Test-Path "rag_pipeline_eval_v2.py")) {
    Write-Host "ERROR: rag_pipeline_eval_v2.py not found." -ForegroundColor Red
    exit 1
}

if (-Not (Test-Path "index/faiss.index")) {
    Write-Host "ERROR: Pipeline index not found. Run setup_v2.ps1 first." -ForegroundColor Red
    exit 1
}

# -----------------------------
# 2. Activate virtual environment
# -----------------------------
if (-Not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found. Run setup_v2.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Activating virtual environment..."
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\.venv\Scripts\Activate.ps1

# -----------------------------
# 3. Run batch evaluation (v2)
# -----------------------------
Write-Host "Running v2 batch evaluation questions..." -ForegroundColor Cyan
Write-Host "(v2 includes citation recall + confidence scoring)" -ForegroundColor Yellow

python rag_pipeline_eval_v2.py batch

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Batch evaluation v2 encountered errors." -ForegroundColor Yellow
    exit 1
}

# -----------------------------
# 4. Display results summary
# -----------------------------
Write-Host "`n=== Evaluation v2 Complete ===" -ForegroundColor Green
Write-Host "Results saved to:" -ForegroundColor Cyan
Write-Host "  - logs/rag_logs.jsonl (detailed query logs)" -ForegroundColor White
Write-Host "  - logs/eval_results.jsonl (evaluation metrics with confidence scores)" -ForegroundColor White

# Count total queries
if (Test-Path "logs/eval_results.jsonl") {
    $totalQueries = (Get-Content "logs/eval_results.jsonl" | Measure-Object -Line).Lines
    Write-Host "`nTotal queries logged: $totalQueries" -ForegroundColor Cyan
}

# Show confidence distribution if jq is available
$jqInstalled = Get-Command jq -ErrorAction SilentlyContinue
if ($jqInstalled) {
    Write-Host "`nConfidence level distribution:" -ForegroundColor Cyan
    Get-Content "logs/eval_results.jsonl" | jq -r '.overall_confidence.level' | Group-Object | Select-Object Count, Name | Format-Table -AutoSize
}
