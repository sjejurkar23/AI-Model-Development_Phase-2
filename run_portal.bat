@echo off
echo Starting Research Portal...
powershell -ExecutionPolicy Bypass -Command "& {Set-Location '%~dp0'; .\.venv\Scripts\Activate.ps1; streamlit run app.py}"
pause