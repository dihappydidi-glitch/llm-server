#!/usr/bin/env python3
"""
LLM Server — FastAPI + external llama-server backend
Web UI + OpenAI-compatible API + CUDA + STT/TTS + Web Search
Architecture: Python (UI/STT/TTS/Search) → llama-server (LLM on :8080)

Designed for GTX 1660 6GB + Qwen 2.5 7B Q4_K_M (~4.3 GiB VRAM).
"""

import io
import json
import os
import re
import sys
import time
import asyncio
import logging
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, List, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging — suppress noisy WinError 10054 from asyncio proactor
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("llm-server")

# Suppress the ConnectionResetError traceback that spams logs on Windows
# when clients disconnect abruptly.  The error is harmless.
_log_errors = logging.getLogger("asyncio")


class _SuppressConnectionReset(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "ConnectionResetError" not in msg and "WinError 10054" not in msg


_log_errors.addFilter(_SuppressConnectionReset())

# ---------------------------------------------------------------------------
# PATH setup — CUDA DLLs for faster-whisper, ffmpeg for audio decode
# ---------------------------------------------------------------------------
try:
    _torch_lib = os.path.join(
        os.path.dirname(__import__("torch").__file__), "lib"
    )
    if os.path.isdir(_torch_lib):
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
except Exception:
    log.info("PyTorch not available — CUDA DLL path not set (llama-server handles its own)")

_ffmpeg_path = r"C:\Users\611marco\ffmpeg\ffmpeg-8.1.1-essentials_build\bin"
if os.path.isdir(_ffmpeg_path) and os.path.isfile(os.path.join(_ffmpeg_path, "ffmpeg.exe")):
    os.environ["PATH"] = _ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
    log.info(f"ffmpeg path added: {_ffmpeg_path}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLAMA_SERVER_URL = "http://localhost:8080"

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")  # empty = no auth required


@dataclass
class ServerConfig:
    models_dir: str = os.environ.get("MODELS_DIR", "C:/Users/611marco/llm-server/models")
    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 40
    default_repetition_penalty: float = 1.05
    n_gpu_layers: int = int(os.environ.get("N_GPU_LAYERS", "-1"))
    n_ctx: int = int(os.environ.get("N_CTX", "4096"))
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))
    https_enabled: bool = os.environ.get("HTTPS_ENABLED", "1").lower() in ("1", "true", "yes")
    https_port: int = int(os.environ.get("HTTPS_PORT", "8443"))
    ssl_certfile: str = os.environ.get("SSL_CERTFILE", "certs/cert.pem")
    ssl_keyfile: str = os.environ.get("SSL_KEYFILE", "certs/key.pem")


@dataclass
class ModelState:
    config: ServerConfig = field(default_factory=ServerConfig)
    loaded: bool = False
    model_name: str = "none"
    model_path: str = ""
    start_time: float = 0.0
    n_requests: int = 0
    n_tokens_generated: int = 0


state = ModelState()
_model_load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Reusable HTTP client for llama-server proxy
# ---------------------------------------------------------------------------
_http_client: Optional[httpx.AsyncClient] = None
_http_client_lock = threading.Lock()


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=LLAMA_SERVER_URL,
        timeout=httpx.Timeout(300.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
    )


def get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        with _http_client_lock:
            if _http_client is None or _http_client.is_closed:
                _http_client = _make_client()
    return _http_client


async def close_client():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


async def check_llama_server() -> bool:
    try:
        r = await get_client().get("/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _ensure_loaded():
    """If state says not loaded but llama-server is reachable, mark loaded."""
    if state.loaded:
        return
    try:
        c = httpx.Client(base_url=LLAMA_SERVER_URL, timeout=3.0)
        ok = c.get("/health").status_code == 200
        c.close()
        if ok:
            state.loaded = True
            state.start_time = time.time()
            log.info("llama-server reconnected")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Voice module (lazy-loaded)
# ---------------------------------------------------------------------------
_VOICE_FAILED = object()  # sentinel
_voice = None


def get_voice():
    global _voice
    if _voice is not None:
        return _voice if _voice is not _VOICE_FAILED else None
    try:
        from voice import transcribe_audio, synthesize_speech, list_tts_voices
        _voice = {"transcribe": transcribe_audio, "synthesize": synthesize_speech, "voices": list_tts_voices}
        log.info("Voice module loaded")
        return _voice
    except Exception as e:
        log.warning(f"Voice module not available: {e}")
        _voice = _VOICE_FAILED
        return None


# ---------------------------------------------------------------------------
# Web Search
# ---------------------------------------------------------------------------
async def web_search(query: str, max_results: int = 5) -> list:
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")})
        return results
    except Exception as e:
        log.warning(f"Web search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=0, le=200)
    repetition_penalty: float = Field(1.05, ge=1.0, le=2.0)
    stream: bool = False


class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(5, ge=1, le=20)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------
def find_gguf_model(models_dir: str) -> Optional[str]:
    """Find the best GGUF model — handles sharded files and scores by quant quality."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return None
    gguf_files = list(models_path.rglob("*.gguf"))
    if not gguf_files:
        return None

    def is_shard(p: Path) -> bool:
        return bool(re.search(r"-\d{5}-of-\d{5}\.gguf$", p.name, re.IGNORECASE))

    def shard_index(p: Path) -> int:
        m = re.search(r"-(\d{5})-of-\d{5}\.gguf$", p.name, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    shards = [f for f in gguf_files if is_shard(f)]
    singles = [f for f in gguf_files if not is_shard(f)]

    candidates = {}
    for f in singles:
        candidates[f.stem] = f

    shard_groups = {}
    for f in shards:
        base = re.sub(r"-\d{5}-of-\d{5}$", "", f.stem)
        if base not in shard_groups or shard_index(f) < shard_index(shard_groups[base]):
            shard_groups[base] = f
    for base, f in shard_groups.items():
        candidates[base] = f

    def score(item) -> int:
        name = item[0].lower()
        s = 0
        if "q4_k_m" in name:
            s += 100
        if "q4" in name:
            s += 50
        if "qwen" in name:
            s += 10
        return s

    best = max(candidates.items(), key=lambda item: (score(item), item[0]))
    best_path = str(best[1])

    log.info(f"Models: {len(gguf_files)} .gguf, {len(singles)} single(s), {len(shard_groups)} shard group(s)")
    log.info(f"Selected: {best[0]} ({best_path})")
    return best_path


def connect_llama_server():
    """Connect to external llama-server backend.  Called once on startup."""
    try:
        c = httpx.Client(base_url=LLAMA_SERVER_URL, timeout=5.0)
        ok = c.get("/health").status_code == 200
        if ok:
            try:
                r2 = c.get("/v1/models")
                if r2.status_code == 200:
                    models = r2.json().get("data", [])
                    if models:
                        state.model_name = models[0].get("id", "llama-server")
            except Exception:
                pass
        c.close()
    except Exception:
        ok = False

    if not ok:
        log.warning("llama-server not reachable on %s — will retry on first request", LLAMA_SERVER_URL)
        return False

    model_path = find_gguf_model(state.config.models_dir)
    if model_path:
        state.model_name = Path(model_path).stem
        state.model_name = re.sub(r"-00001-of-\d{5}$", "", state.model_name)
        state.model_path = model_path

    state.loaded = True
    state.start_time = time.time()
    log.info("Connected to llama-server backend  →  %s/v1/chat/completions", LLAMA_SERVER_URL)
    log.info("Model: %s", state.model_name)
    return True


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    config = state.config
    log.info("=" * 50)
    log.info("LLM Server starting …")
    log.info("Models directory: %s", config.models_dir)
    log.info("=" * 50)

    if state.loaded:
        log.info("Model already loaded: %s", state.model_name)
    else:
        acquired = _model_load_lock.acquire(blocking=False)
        if not acquired:
            log.info("Model loading in another thread — waiting …")
            with _model_load_lock:
                pass
            log.info("Model loaded by other thread: %s", state.model_name)
        else:
            try:
                model_path = find_gguf_model(config.models_dir)
                if model_path:
                    try:
                        connect_llama_server()
                    except Exception as e:
                        log.error("Failed to connect: %s", e)
                        log.error("Server starting WITHOUT model — use /api/admin/load later")
                else:
                    log.warning("No GGUF model found — place a .gguf file in models/")
            finally:
                _model_load_lock.release()

    yield

    await close_client()
    log.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM Server", description="Local LLM inference with llama.cpp + CUDA", version="2.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# API Key guard
# ---------------------------------------------------------------------------
def _verify_admin(request: Request):
    if not ADMIN_API_KEY:
        return
    key = request.headers.get("X-API-Key", "")
    if key != ADMIN_API_KEY:
        raise HTTPException(403, "Forbidden: invalid or missing X-API-Key header")


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>LLM Server</h1>")


# Stops the favicon 404 spam in logs
_FAVICON_SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
<rect width="32" height="32" rx="6" fill="#4c9aff"/>
<text x="16" y="23" font-size="20" text-anchor="middle" fill="#fff" font-family="sans-serif" font-weight="bold">L</text>
</svg>"""


@app.get("/favicon.ico")
async def favicon():
    return Response(_FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------------------
# Health / Stats
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    llm_ok = await check_llama_server()
    return {
        "status": "connected" if llm_ok else "disconnected",
        "model": state.model_name,
        "loaded": state.loaded,
        "uptime_seconds": int(time.time() - state.start_time) if state.start_time else 0,
    }


@app.get("/api/stats")
async def stats():
    vram_allocated = vram_reserved = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
            vram_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
    except Exception:
        pass
    return {
        "model": state.model_name,
        "loaded": state.loaded,
        "uptime_seconds": int(time.time() - state.start_time) if state.start_time else 0,
        "requests_served": state.n_requests,
        "tokens_generated": state.n_tokens_generated,
        "vram_gb": {"allocated": round(vram_allocated, 2), "reserved": round(vram_reserved, 2)},
        "config": {"n_ctx": state.config.n_ctx, "n_gpu_layers": state.config.n_gpu_layers},
    }


@app.get("/api/models")
async def list_models():
    models_dir = Path(state.config.models_dir)
    models = []
    for f in models_dir.rglob("*.gguf"):
        models.append({"name": f.stem, "path": str(f), "size_gb": round(f.stat().st_size / 1024 ** 3, 2),
                       "loaded": f == Path(state.model_path) if state.model_path else False})
    return {"models": models}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate(req: GenerateRequest):
    _ensure_loaded()
    if not state.loaded:
        raise HTTPException(503, "llama-server not running on port 8080")
    state.n_requests += 1

    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    params = {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "repetition_penalty": req.repetition_penalty,
    }

    if req.stream:
        return StreamingResponse(
            _proxy_stream(messages, params, raw_text=True),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        async with httpx.AsyncClient(base_url=LLAMA_SERVER_URL, timeout=httpx.Timeout(300.0)) as client:
            r = await client.post("/v1/chat/completions", json={"messages": messages, **params, "stream": False})
            r.raise_for_status()
            result = r.json()
    except httpx.ConnectError:
        raise HTTPException(503, "Cannot connect to llama-server on port 8080")
    except Exception as e:
        raise HTTPException(502, f"llama-server error: {e}")

    output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"model": state.model_name, "response": output.strip(), "tokens_generated": len(output) // 4}


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completions
# ---------------------------------------------------------------------------
SEARCH_TRIGGERS = [
    "поищи", "найди", "ищи", "search", "find", "look up",
    "интернет", "internet", "новости", "news", "что нового",
    "проверь", "check", "weather", "погода", "узнай",
]


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    _ensure_loaded()
    if not state.loaded:
        raise HTTPException(503, "llama-server not running on port 8080")
    state.n_requests += 1

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    chat_messages = [m for m in messages if m["role"] != "system"]
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")

    # ---- Web search auto-trigger ----
    search_results = None
    last_text = chat_messages[-1]["content"].lower() if chat_messages else ""
    if any(t in last_text for t in SEARCH_TRIGGERS):
        log.info("Web search triggered by user request")
        try:
            search_results = await web_search(chat_messages[-1]["content"])
            if search_results:
                context = (
                    "Web search results for the user's query:\n\n"
                    + "\n\n".join(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}" for r in search_results)
                    + "\n\nUse these results to answer the user's question. Cite sources when relevant."
                )
                chat_messages.insert(0, {"role": "system", "content": context})
                log.info("Injected %d search result(s)", len(search_results))
        except Exception as e:
            log.warning("Web search error: %s", e)

    # ---- Build params ----
    params = {
        "max_tokens": req.max_tokens or state.config.default_max_tokens,
        "temperature": req.temperature if req.temperature is not None else state.config.default_temperature,
        "top_p": req.top_p if req.top_p is not None else state.config.default_top_p,
        "top_k": req.top_k or state.config.default_top_k,
        "repetition_penalty": req.repetition_penalty or state.config.default_repetition_penalty,
    }
    if req.stop:
        params["stop"] = [req.stop] if isinstance(req.stop, str) else req.stop

    # ---- Stream or single-shot ----
    if req.stream:
        return StreamingResponse(
            _proxy_stream(chat_messages, params),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        async with httpx.AsyncClient(base_url=LLAMA_SERVER_URL, timeout=httpx.Timeout(300.0)) as client:
            r = await client.post("/v1/chat/completions", json={"messages": chat_messages, **params, "stream": False})
            r.raise_for_status()
            result = r.json()
    except httpx.ConnectError:
        raise HTTPException(503, "Cannot connect to llama-server on port 8080")
    except Exception as e:
        raise HTTPException(502, f"llama-server error: {e}")

    output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = result.get("usage", {})

    response_data = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or state.model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": len(output) // 4, "total_tokens": 0},
    }
    if search_results:
        response_data["_search_results"] = search_results
    return response_data


async def _proxy_stream(messages: list, params: dict, raw_text: bool = False):
    """Stream from llama-server SSE, reformatting for OpenAI API."""
    chunk_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())
    model = state.model_name

    if not raw_text:
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    try:
        async with httpx.AsyncClient(base_url=LLAMA_SERVER_URL, timeout=httpx.Timeout(300.0)) as client:
            payload = {"messages": messages, "stream": True, **params}
            async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            state.n_tokens_generated += 1
                            if raw_text:
                                yield text
                            else:
                                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"
                    except json.JSONDecodeError:
                        continue
                    await asyncio.sleep(0)
    except Exception as e:
        log.error("Stream error: %s", e)
        if not raw_text:
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'error'}]})}\n\n"

    if not raw_text:
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
@app.post("/api/admin/load")
async def admin_load(request: Request):
    _verify_admin(request)
    ok = await check_llama_server()
    if not ok:
        raise HTTPException(502, "llama-server not running on port 8080")
    try:
        async with httpx.AsyncClient(base_url=LLAMA_SERVER_URL, timeout=5.0) as c:
            r = await c.get("/v1/models")
            if r.status_code == 200:
                models = r.json().get("data", [])
                if models:
                    state.model_name = models[0].get("id", "llama-server")
    except Exception:
        pass
    state.loaded = True
    state.start_time = time.time()
    log.info("Admin: connected to llama-server: %s", state.model_name)
    return {"status": "ok", "model": state.model_name, "backend": "llama-server"}


@app.post("/api/admin/unload")
async def admin_unload(request: Request):
    _verify_admin(request)
    if not state.loaded:
        return {"status": "ok", "message": "No model loaded"}
    name = state.model_name
    state.loaded = False
    state.model_name = "none"
    state.start_time = 0.0
    log.info("Admin: unloaded model: %s", name)
    return {"status": "ok", "model": name, "message": "Model unloaded"}


@app.post("/api/admin/reload")
async def admin_reload(request: Request):
    _verify_admin(request)
    state.loaded = False
    ok = await check_llama_server()
    if ok:
        state.loaded = True
        state.start_time = time.time()
        return {"status": "ok", "model": state.model_name}
    raise HTTPException(502, "llama-server not running")


# ---------------------------------------------------------------------------
# Voice API
# ---------------------------------------------------------------------------
@app.post("/api/stt")
async def speech_to_text(request: Request):
    v = get_voice()
    if not v:
        raise HTTPException(503, "Voice module not available (pip install faster-whisper edge-tts)")
    audio_data = await request.body()
    if not audio_data or len(audio_data) < 100:
        raise HTTPException(400, "No audio data or file too small")
    log.info("STT request: %d bytes", len(audio_data))
    try:
        text = await v["transcribe"](audio_data)
        return {"text": text, "language": "ru"}
    except Exception as e:
        log.error("STT error: %s", e)
        raise HTTPException(500, f"Transcription failed: {e}")


@app.post("/api/tts")
async def text_to_speech(request: Request):
    v = get_voice()
    if not v:
        raise HTTPException(503, "Voice module not available (pip install edge-tts)")
    try:
        data = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        text = data.get("text", "")
        voice = data.get("voice", None)
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON: {e}")
    if not text:
        raise HTTPException(400, "No text provided")
    try:
        log.info("TTS request: %d chars, voice=%s", len(text), voice)
        audio_bytes = await v["synthesize"](text, voice)
        if not audio_bytes or len(audio_bytes) < 100:
            raise HTTPException(502, "TTS service returned empty audio")
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="speech.mp3"', "Content-Length": str(len(audio_bytes))},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("TTS error: %s: %s", e.__class__.__name__, e)
        raise HTTPException(500, f"TTS failed: {e}")


@app.get("/api/tts/voices")
async def tts_voices():
    v = get_voice()
    if not v:
        raise HTTPException(503, "Voice module not available")
    try:
        return {"voices": await v["voices"]()}
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Web Search API
# ---------------------------------------------------------------------------
@app.post("/api/search")
async def api_search(req: SearchRequest):
    log.info("Web search: '%s' (max=%d)", req.query, req.max_results)
    results = await web_search(req.query, req.max_results)
    return {"query": req.query, "results": results, "count": len(results)}


@app.get("/api/search")
async def api_search_get(query: str = "", max_results: int = 5):
    if not query:
        return {"query": "", "results": [], "count": 0}
    results = await web_search(query, min(max_results, 20))
    return {"query": query, "results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# OpenAI-compatible model listing
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def openai_list_models():
    models = []
    models_dir = Path(state.config.models_dir)
    if models_dir.exists():
        for f in models_dir.rglob("*.gguf"):
            name = re.sub(r"-00001-of-\d{5}$", "", f.stem)
            models.append({"id": name, "object": "model", "created": int(f.stat().st_mtime), "owned_by": "local"})
    if not models:
        models.append({"id": state.model_name or "none", "object": "model", "created": int(time.time()), "owned_by": "local"})
    return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# Main — single uvicorn + stdlib redirect server
# ---------------------------------------------------------------------------
def _make_redirect_server(host: str, http_port: int, https_url: str):
    """Build a tiny HTTP server that 301-redirects everything to HTTPS."""

    class RedirectHandler(BaseHTTPRequestHandler):
        def _redirect(self):
            self.send_response(301)
            self.send_header("Location", https_url + self.path)
            self.end_headers()

        def do_GET(self):
            self._redirect()

        def do_HEAD(self):
            self._redirect()

        def do_POST(self):
            self._redirect()

        def do_PUT(self):
            self._redirect()

        def do_DELETE(self):
            self._redirect()

        def do_PATCH(self):
            self._redirect()

        def do_OPTIONS(self):
            self._redirect()

        # Silence per-request logs
        def log_message(self, fmt, *args):
            pass

    server = HTTPServer((host, http_port), RedirectHandler)
    return server


def main():
    # Load config file
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(state.config, key):
                    setattr(state.config, key, value)
                    log.info("Config: %s = %s", key, value)
        except Exception as e:
            log.warning("Config error: %s", e)

    log.info("Python: %s", sys.version)

    # Scan for GGUF models
    model_path = find_gguf_model(state.config.models_dir)
    if model_path:
        log.info("GGUF model: %s (%.1f GiB)", Path(model_path).name, os.path.getsize(model_path) / 1024 ** 3)
    else:
        log.info("No GGUF models found in %s — place a .gguf file and restart", state.config.models_dir)

    log.info("HTTP:  http://%s:%d", state.config.host, state.config.port)

    ssl_key = Path(__file__).parent / state.config.ssl_keyfile
    ssl_cert = Path(__file__).parent / state.config.ssl_certfile
    ssl_ok = ssl_key.exists() and ssl_cert.exists()

    if ssl_ok and state.config.https_enabled:
        log.info("HTTPS: https://%s:%d", state.config.host, state.config.https_port)
        log.info("HTTP → HTTPS redirect on http://%s:%d", state.config.host, state.config.port)

        # Start the stdlib redirect server in a daemon thread
        https_url = f"https://{state.config.host}:{state.config.https_port}"
        redirect_server = _make_redirect_server(state.config.host, state.config.port, https_url)
        t = threading.Thread(target=redirect_server.serve_forever, daemon=True)
        t.start()
        log.info("Redirect server running on port %d", state.config.port)

        # Main app on HTTPS
        uvicorn.run(
            app,
            host=state.config.host,
            port=state.config.https_port,
            ssl_keyfile=str(ssl_key),
            ssl_certfile=str(ssl_cert),
            log_level="info",
        )
        # On shutdown, stop the redirect server
        redirect_server.shutdown()
    else:
        if not ssl_ok:
            log.warning("SSL cert/key not found in certs/ — HTTPS disabled")
        else:
            log.info("HTTPS disabled by config")
        log.info("Server: http://%s:%d", state.config.host, state.config.port)

        uvicorn.run(app, host=state.config.host, port=state.config.port, log_level="info")


if __name__ == "__main__":
    main()
