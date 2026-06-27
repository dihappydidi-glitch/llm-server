"""
Voice I/O module for LLM Server
- STT: faster-whisper (tiny/base/small model, GPU via CTranslate2)
- TTS: edge-tts (Microsoft Edge free TTS API)
"""

import io
import os
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("llm-server.voice")

# ---------------------------------------------------------------------------
# STT — Speech-to-Text with faster-whisper
# ---------------------------------------------------------------------------

_whisper_model = None
_WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")


def get_whisper_model():
    """Lazy-load the Whisper model (loaded on first request)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    log.info(f"Loading Whisper model '{_WHISPER_MODEL_SIZE}'...")
    try:
        import torch
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Whisper device={device}, compute={compute_type}")
    except Exception:
        compute_type = "int8"
        device = "cpu"

    from faster_whisper import WhisperModel
    _whisper_model = WhisperModel(
        _WHISPER_MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        cpu_threads=4,
        num_workers=2,
        download_root=str(Path(__file__).parent / "models" / "whisper"),
    )
    log.info(f"Whisper model '{_WHISPER_MODEL_SIZE}' loaded")
    return _whisper_model


async def transcribe_audio(audio_data: bytes) -> str:
    """
    Transcribe audio bytes to text using faster-whisper.
    Accepts WebM (browser), WAV, OGG, MP3, etc.

    Uses PyAV for decoding → numpy array → faster-whisper.
    Falls back to temp-file path if PyAV fails.
    """
    model = get_whisper_model()

    # Try: decode with PyAV to numpy array (fastest path)
    import numpy as np
    try:
        samples = _decode_audio_to_float32(audio_data)
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None, lambda: model.transcribe(
                samples,
                language="ru",
                beam_size=3,
                vad_filter=True,
                without_timestamps=True,
            )
        )
        text_parts = [seg.text.strip() for seg in segments]
        result = " ".join(text_parts)
        duration = len(samples) / 16000
        log.info(f"STT ({info.language}/{duration:.1f}s): {result[:60]}...")
        return result
    except Exception as e:
        log.warning(f"PyAV decode failed ({e})")

    # Fallback: write temp file, let faster-whisper use ffmpeg
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None, lambda: model.transcribe(
                tmp_path,
                beam_size=3,
                vad_filter=True,
                language="ru",
                without_timestamps=True,
            )
        )
        text_parts = [seg.text.strip() for seg in segments]
        result = " ".join(text_parts)
        log.info(f"STT ({info.language}/{info.duration:.1f}s): {result[:60]}...")
        return result
    except Exception as e:
        log.error(f"STT error: {e}")
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _decode_audio_to_float32(audio_data: bytes) -> "np.ndarray":
    """Decode audio bytes → float32 numpy array (16 kHz mono) using PyAV."""
    import av
    import numpy as np

    input_container = av.open(io.BytesIO(audio_data))
    all_samples = []
    audio_stream = input_container.streams.audio[0]
    input_rate = audio_stream.sample_rate

    for frame in input_container.decode(audio=0):
        arr = frame.to_ndarray()
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.shape[0] > 1:
            arr = arr.mean(axis=0, keepdims=True)
        all_samples.append(arr)

    if not all_samples:
        raise ValueError("No audio frames decoded")

    samples = np.concatenate(all_samples, axis=1).flatten().astype(np.float32)

    # Resample to 16 kHz
    if input_rate != 16000:
        ratio = 16000 / input_rate
        new_len = int(len(samples) * ratio)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, new_len),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)

    return samples


# ---------------------------------------------------------------------------
# TTS — Text-to-Speech with edge-tts
# ---------------------------------------------------------------------------

_TTS_VOICE = os.environ.get("TTS_VOICE", "ru-RU-SvetlanaNeural")
_TTS_VOICE_EN = os.environ.get("TTS_VOICE_EN", "en-US-AriaNeural")


async def synthesize_speech(text: str, voice: Optional[str] = None) -> bytes:
    """Synthesize speech from text → MP3 bytes via edge-tts."""
    import edge_tts

    voice = voice or _TTS_VOICE
    has_cyrillic = any("Ѐ" <= c <= "ӿ" for c in text)
    if not has_cyrillic:
        voice = _TTS_VOICE_EN

    log.info(f"TTS: {len(text)} chars, voice={voice}")

    communicate = edge_tts.Communicate(text, voice)
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])

    result = bytes(audio_data)
    log.info(f"TTS generated: {len(result)} bytes")
    return result


async def list_tts_voices() -> list[dict]:
    """List available edge-tts voices (ru/en)."""
    import edge_tts

    voices = await edge_tts.list_voices()
    return [
        {"name": v["ShortName"], "locale": v.get("Locale", ""), "gender": v.get("Gender", "")}
        for v in voices if "ru" in v.get("Locale", "") or "en" in v.get("Locale", "")
    ]
