@echo off
title LLM Server — Full Stack (llama-server + Python UI)

cd /d "%~dp0"

echo =============================================
echo   LLM Server — Full System
echo   llama-server (CUDA) + Python (UI/STT/TTS/Search)
echo =============================================
echo.

:: Set CUDA paths for llama-server
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "PATH=%CD%\llama-cpp;%CD%\.venv\Lib\site-packages\torch\lib;%PATH%"

:: Kill any previous instances
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 2 /nobreak >nul

:: Start llama-server (LLM backend with CUDA)
echo [1/3] Starting llama-server (CUDA LLM backend)...
start /B "" "%CD%\llama-cpp\llama-server.exe" ^
    --model "%CD%\models\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf" ^
    --host 0.0.0.0 --port 8080 ^
    --n-gpu-layers -1 --main-gpu 0 ^
    --ctx-size 2048 --threads 6 --batch-size 512 ^
    --temp 0.7 --chat-template chatml ^
    --parallel 1 --no-mmap --log-disable

echo   llama-server starting on port 8080...
timeout /t 8 /nobreak >nul

:: Check llama-server health
echo [2/3] Verifying llama-server...
python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8080/health'); print('  llama-server:', 'OK' if r.status==200 else 'FAIL')" 2>&1

:: Start Python server
echo [3/3] Starting Python server (FastAPI + Web UI)...
echo.
echo   Web UI (HTTPS): https://localhost:8443
echo   API:            https://localhost:8443/v1/chat/completions
echo   Search API:     https://localhost:8443/api/search
echo   HTTP:           http://localhost:8000
echo.

python server.py

pause
