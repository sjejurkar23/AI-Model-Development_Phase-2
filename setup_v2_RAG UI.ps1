Write-Host "=== RAG Pipeline Setup v2 (Windows) ===" -ForegroundColor Cyan

# -----------------------------
# PRE-FLIGHT: Execution Policy
# -----------------------------
try {
    $currentPolicy = Get-ExecutionPolicy -Scope CurrentUser
    if ($currentPolicy -eq "Restricted" -or $currentPolicy -eq "Undefined") {
        Write-Host "Setting execution policy to RemoteSigned..." -ForegroundColor Yellow
        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
        Write-Host "Execution policy updated." -ForegroundColor Green
    } else {
        Write-Host "Execution policy OK: $currentPolicy" -ForegroundColor Green
    }
} catch {
    Write-Host "WARNING: Could not set execution policy automatically." -ForegroundColor Yellow
    Write-Host "If issues persist, run manually: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force" -ForegroundColor Cyan
}

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

if (-Not (Test-Path "rag_pipeline_eval_v2.py")) {
    Write-Host "ERROR: rag_pipeline_eval_v2.py not found in project directory." -ForegroundColor Red
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
# 7. Build RAG Pipeline Index
# -----------------------------
Write-Host "`n=== Building RAG Pipeline (ingest + chunk + index) ===" -ForegroundColor Cyan

python rag_pipeline_eval_v2.py run_all

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Pipeline build failed." -ForegroundColor Red
    exit 1
}

Write-Host "Pipeline index built successfully." -ForegroundColor Green

# -----------------------------
# 8. Done — skip batch evaluation
# (Run batch manually via: python rag_pipeline_eval_v2.py batch)
# (Or use the Research Portal UI: streamlit run app.py)
# -----------------------------
Write-Host "`n=== Setup Complete! ===" -ForegroundColor Cyan
Write-Host "Index is ready. To launch the Research Portal, open the run_portal.bat file" -ForegroundColor Green
Write-Host ""
Write-Host "To run batch evaluation manually, open and new terminal and run:" -ForegroundColor Green
Write-Host "  python rag_pipeline_eval_v2.py batch" -ForegroundColor Yellow