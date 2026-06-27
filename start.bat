@echo off
title LLM Server — Web UI only (llama-server must already be running)
cd /d "%~dp0"

echo =============================================
echo   LLM Server — Python Web UI
echo   (Requires llama-server on port 8080)
echo =============================================
echo.

:: Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] PyTorch not found — VRAM stats unavailable, whisper will use CPU
)

echo.
echo Starting Python server...
echo.
echo   Web UI (HTTPS): https://localhost:8443
echo   API:            https://localhost:8443/v1/chat/completions
echo   HTTP → HTTPS:   http://localhost:8000
echo.
echo   NOTE: Chrome warns on self-signed certs.
echo   Click "Advanced" > "Proceed to localhost" to continue.
echo.

python server.py

pause
