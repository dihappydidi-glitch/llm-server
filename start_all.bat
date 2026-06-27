@echo off
title LLM Server — Full Stack (llama-server + Python UI)
cd /d "%~dp0"

echo =============================================
echo   LLM Server — Full System
echo   llama-server (CUDA) + Python (UI/STT/TTS/Search)
echo =============================================
echo.

:: ── Model selection (via Python for reliability) ─────
for /f "delims=" %%i in ('python -x -c "
import sys, glob, os, re

models_dir = os.path.join(os.getcwd(), 'models')
gguf_files = sorted(glob.glob(os.path.join(models_dir, '**', '*.gguf'), recursive=True))

# Group sharded models (00001-of-NNNNN)
def is_shard(name):
    return bool(re.search(r'-\d{5}-of-\d{5}\.gguf$', name, re.I))

shard_groups = {}
singles = []
for f in gguf_files:
    name = os.path.basename(f)
    if is_shard(name):
        base = re.sub(r'-\d{5}-of-\d{5}\.gguf$', '', name)
        if base not in shard_groups:
            shard_groups[base] = f
    else:
        singles.append(f)

# Build model list: first shard for sharded, or single file
models = singles + list(shard_groups.values())
models.sort(key=lambda f: os.path.basename(f).lower())

if not models:
    sys.exit(1)

if len(models) == 1:
    print(models[0])
else:
    print('MULTIPLE')
    for i, m in enumerate(models, 1):
        name = os.path.basename(m)
        size = os.path.getsize(m) / 1024**3
        print(f'{i}. {name}  ({size:.1f} GiB)', file=sys.stderr)
    import sys
    print('', file=sys.stderr)
    choice = input(f'Select model [1-{len(models)}]: ').strip() or '1'
    print(models[int(choice)-1])
"') do set "MODEL_PATH=%%i"

if "%MODEL_PATH%"=="" (
    echo [ERROR] No .gguf models found.
    pause
    exit /b 1
)

for /f "delims=" %%i in ("%MODEL_PATH%") do echo Selected: %%i
echo.

:: ── Set CUDA paths ────────────────────────────────────
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "PATH=%CD%\llama-cpp;%CD%\.venv\Lib\site-packages\torch\lib;%PATH%"

:: ── Kill old processes ────────────────────────────────
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 2 /nobreak >nul

:: ── Start llama-server ────────────────────────────────
echo [1/3] Starting llama-server (CUDA LLM backend)...
start /B "" "%CD%\llama-cpp\llama-server.exe" ^
    --model "%MODEL_PATH%" ^
    --host 0.0.0.0 --port 8080 ^
    --n-gpu-layers -1 --main-gpu 0 ^
    --ctx-size 2048 --threads 6 --batch-size 512 ^
    --temp 0.7 --chat-template chatml ^
    --parallel 1 --no-mmap --log-disable

echo   llama-server starting on port 8080...
timeout /t 8 /nobreak >nul

:: ── Verify ──────────────────────────────────────────────
echo [2/3] Verifying llama-server...
python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8080/health'); print('  llama-server:', 'OK' if r.status==200 else 'FAIL')" 2>&1

:: ── Start Python server ────────────────────────────────
echo [3/3] Starting Python server (FastAPI + Web UI)...
echo.
echo   Web UI (HTTPS): https://localhost:8443
echo   API:            https://localhost:8443/v1/chat/completions
echo   HTTP:           http://localhost:8000
echo.

python server.py

pause
