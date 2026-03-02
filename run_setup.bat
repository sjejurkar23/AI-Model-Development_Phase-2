@echo off
echo Launching RAG setup with execution policy bypass...
powershell -ExecutionPolicy Bypass -File "%~dp0setup_v2_RAG UI.ps1"
pause