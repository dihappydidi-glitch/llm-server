# LLM Server — Project State

**Date:** 2026-06-27
**Version:** 2.1.0 (llama-server proxy)

---

## Architecture

```
User
  │
  ▼
┌─────────────────────────┐      ┌──────────────────────────┐
│  Python FastAPI          │      │  llama-server (CUDA)     │
│  :8443 (HTTPS)          │─────▶│  :8080                   │
│  :8000 → redirects      │      │  Qwen 2.5 7B Q4_K_M     │
│                         │      │  ~15 tok/s              │
│  ├─ Web UI (chat)       │      │  VRAM: 4.3/6.0 GiB      │
│  ├─ STT (faster-whisper)│      │  GPU: 1905 MHz boost     │
│  ├─ TTS (edge-tts)      │      └──────────────────────────┘
│  ├─ Web Search (DDG)    │
│  └─ Admin API           │
└─────────────────────────┘

HTTP (:8000) → 301 redirect → HTTPS (:8443) — single uvicorn process.
```

## Components

### 1. llama-server (LLM inference)

| Param | Value |
|-------|-------|
| Binary | `llama-cpp/llama-server.exe` (b9821, CUDA 12.4) |
| Model | Qwen 2.5 7B Instruct, Q4_K_M (sharded GGUF, ~4.6 GiB) |
| Speed | 15.1 tok/s gen, 35.3 tok/s preproc |
| VRAM | 4.3 GiB of 6.0 GiB |

Key flags: `--n-gpu-layers -1`, `--no-mmap`, `--ctx-size 2048`, `--chat-template chatml`

### 2. Python FastAPI (Web UI + API)

One file: `server.py` (870 lines)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/favicon.ico` | SVG favicon |
| GET | `/api/health` | Status check |
| GET | `/api/stats` | Statistics + VRAM |
| GET | `/api/models` | List GGUF files |
| POST | `/api/generate` | Text generation (stream/non-stream) |
| POST | `/v1/chat/completions` | OpenAI-compatible chat |
| POST | `/api/stt` | Speech-to-Text (faster-whisper) |
| POST | `/api/tts` | Text-to-Speech (edge-tts) |
| GET | `/api/tts/voices` | List TTS voices |
| GET/POST | `/api/search` | Web search (DuckDuckGo) |
| GET | `/v1/models` | OpenAI model list |
| POST | `/api/admin/load` | Connect llama-server |
| POST | `/api/admin/unload` | Disconnect |
| POST | `/api/admin/reload` | Reconnect |

Admin endpoints accept `X-API-Key` header (set `ADMIN_API_KEY` env var).

### 3. Web Search

- **Library**: `ddgs` (DuckDuckGo Search)
- **Auto-trigger**: messages containing "поищи", "найди", "search", "internet", "новости", etc.
  inject search results into context automatically.

### 4. Voice (STT/TTS)

- **STT**: `faster-whisper` (tiny model, CUDA when available)
- **TTS**: `edge-tts` (Microsoft Edge online TTS)
- Lazy-loaded on first request; graceful fallback when not installed.

## File structure

```
C:\Users\611marco\llm-server\
├── server.py              # Main server (FastAPI + proxy)  ← 870 lines
├── voice.py               # STT/TTS module
├── config.json            # Configuration
├── requirements.txt       # Python dependencies
├── download_model.py      # HuggingFace model downloader
├── .gitignore
├── PROJECT_STATE.md       # This file
│
├── models/
│   └── Qwen2.5-7B-Instruct-GGUF/
│       ├── qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf  (3.7 GiB)
│       └── qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf  (658 MiB)
│
├── llama-cpp/             # llama.cpp CUDA 12.4 binaries
│   ├── llama-server.exe
│   ├── llama-server-impl.dll
│   ├── ggml-cuda.dll      # CUDA backend (565 MB)
│   └── ggml-cpu-*.dll     # CPU backends per arch
│
├── templates/
│   └── index.html         # Web UI (dark theme, chat, voice)
│
├── certs/
│   ├── cert.pem           # Self-signed SSL cert (with SAN)
│   └── key.pem            # Private key
│
└── .venv/                 # Python 3.x virtualenv
```

## Configuration

**`config.json`:**

```json
{
    "models_dir": "C:/Users/611marco/llm-server/models",
    "n_ctx": 4096,
    "n_gpu_layers": -1,
    "host": "0.0.0.0",
    "port": 8000,
    "https_enabled": true,
    "https_port": 8443,
    "ssl_certfile": "certs/cert.pem",
    "ssl_keyfile": "certs/key.pem"
}
```

All config keys can be overridden via environment variables:
`MODELS_DIR`, `N_CTX`, `N_GPU_LAYERS`, `HOST`, `PORT`, `HTTPS_ENABLED`, `HTTPS_PORT`, `SSL_CERTFILE`, `SSL_KEYFILE`, `ADMIN_API_KEY`.

## Running

### Quick start:

```bash
start_all.bat
```

### Manual:

```bash
# Terminal 1: llama-server with CUDA
llama-cpp\llama-server.exe ^
    --model "models\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf" ^
    --host 0.0.0.0 --port 8080 --n-gpu-layers -1 --main-gpu 0 ^
    --ctx-size 2048 --threads 6 --batch-size 512 ^
    --temp 0.7 --chat-template chatml --parallel 1 --no-mmap --log-disable

# Terminal 2: Python server
python server.py
```

### Endpoints:

- **Web UI (HTTPS):** https://localhost:8443
- **API:** https://localhost:8443/v1/chat/completions
- **HTTP** http://localhost:8000 — auto-redirects to HTTPS

## Performance

| Metric | Value | Note |
|--------|-------|------|
| LLM generation | **15.1 tok/s** | Qwen 2.5 7B Q4_K_M on GTX 1660 |
| LLM preproc | **35.3 tok/s** | Prompt processing |
| GPU util | ~40% | 1905 MHz boost clock |
| VRAM used | 4.3 GiB | Of 6.0 GiB |
| Model load | ~8 s | With `--no-mmap` |
| Web search | ~2-5 s | DuckDuckGo |

## Known issues & mitigations

| Issue | Cause | Resolution |
|-------|-------|------------|
| 2.6 tok/s (CPU) | llama-cpp-python without CUDA | External llama-server with CUDA 12.4 |
| 0xc000001d | Pre-built wheel for sm_80+, GTX 1660 is sm_75 | Official llama.cpp binaries with sm_75 support |
| 3.1 tok/s (GPU idle) | Zombie llama-server processes | Kill all before start (`taskkill /F /IM llama-server.exe`) |
| GPU 300 MHz, 0% util | Missing `--no-mmap` | Added `--no-mmap` flag |

## Security notes

- **CORS**: `allow_origins=["*"]` — intended for local network use only
- **Admin API**: protected by `X-API-Key` header when `ADMIN_API_KEY` env var is set
- **HTTPS**: self-signed cert — browser will show a warning; click "Advanced → Proceed"
- **Microphone**: requires HTTPS or localhost; click the 🎤 button to activate (no auto-start)

## TODO

- [ ] Conversation history / save/load
- [ ] Markdown rendering in chat (vs plain text)
- [ ] Streaming TTS for low-latency voice
- [ ] RAG / document search
- [ ] Voice activity detection (VAD) for hands-free
- [ ] Select TTS voice in Web UI
- [ ] Push 20+ tok/s via split-layer GPU/CPU
