Write-Host "=== RAG Pipeline Setup v1 (Windows) ===" -ForegroundColor Cyan

# Ensure we are in the script directory
Set-Location -Path $PSScriptRoot

# -----------------------------
# 0. Verify Python version
# -----------------------------
$pyVersion = python --version 2>&1
Write-Host "Detected $pyVersion"

if ($pyVersion -notmatch "3\.12") {
    Write-Host "ERROR: Python 3.12 is required for FAISS on Windows." -ForegroundColor Red
    exit 1
}

# -----------------------------
# 1. Verify files exist
# -----------------------------
if (-Not (Test-Path "requirements.txt")) {
    Write-Host "ERROR: requirements.txt not found in project directory." -ForegroundColor Red
    exit 1
}

if (-Not (Test-Path "rag_pipeline_eval_v1.py")) {
    Write-Host "ERROR: rag_pipeline_eval_v1.py not found in project directory." -ForegroundColor Red
    exit 1
}

# -----------------------------
# 2. Create project folders (non-destructive)
# -----------------------------
Write-Host "Creating project folders..."

$folders = @(
    "data",
    "data/raw",
    "data/processed",
    "index",
    "logs"
)

foreach ($folder in $folders) {
    if (-Not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
    }
}

# Create empty log files if missing
if (-Not (Test-Path "logs/rag_logs.jsonl")) {
    New-Item -ItemType File -Path "logs/rag_logs.jsonl" | Out-Null
}
if (-Not (Test-Path "logs/eval_results.jsonl")) {
    New-Item -ItemType File -Path "logs/eval_results.jsonl" | Out-Null
}

Write-Host "Folders verified." -ForegroundColor Green

# -----------------------------
# 3. Create virtual environment
# -----------------------------
Write-Host "Creating virtual environment..."

python -m venv .venv

if (-Not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment was not created." -ForegroundColor Red
    exit 1
}

Write-Host "Virtual environment created." -ForegroundColor Green

# -----------------------------
# 4. Activate venv
# -----------------------------
Write-Host "Activating virtual environment..."

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\.venv\Scripts\Activate.ps1

Write-Host "Virtual environment activated." -ForegroundColor Green

# -----------------------------
# 5. Install dependencies
# -----------------------------
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow

python.exe -m pip install --upgrade pip
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Dependency installation failed." -ForegroundColor Red
    exit 1
}

# -----------------------------
# 6. Verify FAISS installed correctly
# -----------------------------
pip show faiss-cpu > $null

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: FAISS did not install correctly. Ensure requirements.txt uses faiss-cpu==1.8.0." -ForegroundColor Red
    exit 1
}

Write-Host "Dependencies installed successfully." -ForegroundColor Green

# -----------------------------
# 7. Build RAG Pipeline Index (v1)
# -----------------------------
Write-Host "`n=== Building RAG Pipeline v1 (ingest + chunk + index) ===" -ForegroundColor Cyan

python rag_pipeline_eval_v1.py run_all

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Pipeline v1 build failed." -ForegroundColor Red
    exit 1
}

Write-Host "Pipeline v1 index built successfully." -ForegroundColor Green

# -----------------------------
# 8. Run Batch Evaluation (v1)
# -----------------------------
Write-Host "`n=== Running Batch Evaluation Questions (v1) ===" -ForegroundColor Cyan

python rag_pipeline_eval_v1.py batch

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Batch evaluation v1 encountered errors." -ForegroundColor Yellow
}

Write-Host "Batch evaluation v1 complete." -ForegroundColor Green

# -----------------------------
# 9. Done
# -----------------------------
Write-Host "`n=== Setup Complete (v1)! ===" -ForegroundColor Cyan
Write-Host "Pipeline v1 is ready. You can now:" -ForegroundColor Cyan
Write-Host "  - Query: python rag_pipeline_eval_v1.py query --question 'Your question here'" -ForegroundColor White
Write-Host "  - Re-run batch: python rag_pipeline_eval_v1.py batch" -ForegroundColor White
Write-Host "  - Check logs: logs/rag_logs.jsonl and logs/eval_results.jsonl" -ForegroundColor White
