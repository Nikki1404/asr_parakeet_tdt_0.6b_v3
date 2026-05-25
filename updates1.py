#app/asr_engines/base.py-
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class EngineCaps:
    streaming: bool
    partials: bool
    ttft_meaningful: bool


class ASRSession(Protocol):
    def accept_pcm16(self, pcm16: bytes) -> None: ...
    def step_if_ready(self) -> Optional[str]: ...
    def finalize(self, pad_ms: int) -> str: ...


class ASREngine(ABC):
    caps: EngineCaps

    @abstractmethod
    def load(self) -> float:
        ...

    @abstractmethod
    def new_session(
        self,
        max_buffer_ms: int,
        language: Optional[str] = None,
    ) -> ASRSession:
        ...

#app/asr_engines/parakeet_asr.py
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
import wave
from typing import Optional

from app.asr_engines.base import ASREngine, EngineCaps

log = logging.getLogger("parakeet_server")


def safe_text(x) -> str:
    if x is None:
        return ""

    if isinstance(x, str):
        return x

    if isinstance(x, (list, tuple)) and x:
        return safe_text(x[0])

    if hasattr(x, "text"):
        try:
            return x.text or ""
        except Exception:
            return ""

    try:
        return str(x)
    except Exception:
        return ""


_LATIN_RE = re.compile(r"^[\x00-\xFF\u00C0-\u024F\s.,;:!?'\"\-()\d]+$")


def _is_gibberish(text: str, audio_ms: int) -> bool:
    t = text.strip()

    if not t:
        return True

    if not _LATIN_RE.match(t):
        log.debug("[GIBBERISH] non-latin rejected: %r", t)
        return True

    words = t.split()

    if len(words) >= 4 and len(set(w.lower() for w in words)) == 1:
        log.debug("[GIBBERISH] repeated-token loop rejected: %r", t)
        return True

    if audio_ms > 3000 and len(words) < 2:
        log.debug("[GIBBERISH] too few words rejected: %r", t)
        return True

    if len(t) == 1 and not t.isalpha():
        log.debug("[GIBBERISH] single non-alpha rejected: %r", t)
        return True

    return False


class ParakeetASR(ASREngine):
    caps = EngineCaps(
        streaming=False,
        partials=True,
        ttft_meaningful=True,
    )

    def __init__(
        self,
        model_name: str,
        device: str,
        sample_rate: int,
        language: Optional[str] = None,
        partial_tail_ms: int = 6000,
    ):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate
        self.language = language
        self.partial_tail_ms = partial_tail_ms
        self.model = None

        self.end_silence_ms = 700
        self.min_utt_ms = 300
        self.finalize_pad_ms = 0

        # Less aggressive partials reduce decoder instability
        self.partial_interval_sec = 2.5

    def load(self) -> float:
        import nemo.collections.asr as nemo_asr

        t0 = time.time()

        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        return time.time() - t0

    def new_session(
        self,
        max_buffer_ms: int,
        language: Optional[str] = None,
    ) -> "ParakeetSession":
        effective_language = language if language in ("en", "es") else self.language

        return ParakeetSession(
            engine=self,
            max_buffer_ms=max_buffer_ms,
            language=effective_language,
        )


_MIN_PARTIAL_MS = 600


class ParakeetSession:
    def __init__(
        self,
        engine: ParakeetASR,
        max_buffer_ms: int,
        language: Optional[str] = None,
    ):
        self.engine = engine
        self.language = language

        log.info("[ASR_SESSION] language=%s", language or "auto-detect")

        self.max_buffer_samples = int(engine.sr * max_buffer_ms / 1000)

        self.audio = bytearray()
        self.current_text = ""
        self.last_partial_time = 0.0
        self._audio_ms = 0

    def accept_pcm16(self, pcm16: bytes) -> None:
        self.audio.extend(pcm16)

        self._audio_ms = int(len(self.audio) / 2 / self.engine.sr * 1000)

        max_bytes = self.max_buffer_samples * 2

        if len(self.audio) > max_bytes:
            self.audio = self.audio[-max_bytes:]
            self._audio_ms = int(len(self.audio) / 2 / self.engine.sr * 1000)

    def step_if_ready(self) -> Optional[str]:
        now = time.time()

        if not self.audio:
            return None

        if self._audio_ms < _MIN_PARTIAL_MS:
            return None

        if (now - self.last_partial_time) < self.engine.partial_interval_sec:
            return None

        self.last_partial_time = now

        # IMPORTANT:
        # partials decode only recent tail audio to reduce hallucination/drift
        text = self._transcribe(tail_only=True).strip()

        if not text:
            return None

        if _is_gibberish(text, self._audio_ms):
            return None

        if text == self.current_text:
            return None

        if self.current_text and self.current_text.startswith(text):
            if len(text) < len(self.current_text) * 0.8:
                return None

        self.current_text = text
        return text

    def finalize(self, pad_ms: int) -> str:
        # Final decodes full utterance buffer for maximum completeness
        final = self._transcribe(tail_only=False).strip()

        if final and not _is_gibberish(final, self._audio_ms):
            self.current_text = final

        out = self.current_text.strip()

        self.audio.clear()
        self.current_text = ""
        self.last_partial_time = 0.0
        self._audio_ms = 0

        return out

    def _transcribe(self, tail_only: bool = False) -> str:
        tmp_path = None

        try:
            audio = bytes(self.audio)

            if tail_only:
                tail_bytes = int(self.engine.sr * self.engine.partial_tail_ms / 1000) * 2
                audio = audio[-tail_bytes:]

            if not audio:
                return ""

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                f.write(self._to_wav(audio, self.engine.sr))

            results = self._call_model(tmp_path, self.language)

            if not results:
                return ""

            return safe_text(results[0]).strip()

        except Exception as exc:
            log.warning("[ASR] transcription error: %s", exc)
            return ""

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _call_model(self, wav_path: str, language: Optional[str]):
        model = self.engine.model

        if not language:
            return model.transcribe([wav_path])

        try:
            return model.transcribe([wav_path], language_id=language)
        except TypeError:
            log.warning("[ASR] language_id unsupported, trying language")

        try:
            return model.transcribe([wav_path], language=language)
        except TypeError:
            log.warning("[ASR] language unsupported, using auto-detect")

        return model.transcribe([wav_path])

    @staticmethod
    def _to_wav(pcm: bytes, sr: int) -> bytes:
        buf = io.BytesIO()

        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)

        return buf.getvalue()
        
        
#app/main.py-
from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Optional

from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect

from app.config import load_config
from app.factory import build_engine
from app.streaming_session import StreamingSession

cfg = load_config()

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("parakeet_server")

app = FastAPI()
engine = build_engine(cfg)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    _PROM_ENABLED = True
except ImportError:
    _PROM_ENABLED = False


@app.on_event("startup")
async def startup():
    load_sec = engine.load()

    log.info("Model loaded in %.2fs", load_sec)
    log.info("Default language: %s", cfg.language or "auto-detect")
    log.info("Audio saving: %s", cfg.save_audio)
    log.info("Audio save dir: %s", cfg.audio_save_dir)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "backend": cfg.asr_backend,
        "model": cfg.model_name,
        "language": cfg.language or "auto-detect",
        "save_audio": cfg.save_audio,
        "audio_save_dir": cfg.audio_save_dir,
    }


@app.get("/metrics")
async def metrics():
    if not _PROM_ENABLED:
        return Response(
            content=json.dumps({"error": "prometheus_client not installed"}),
            media_type="application/json",
            status_code=503,
        )

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/asr/ml/ws")
async def ws_asr(
    ws: WebSocket,
    lang: Optional[str] = Query(default=None),
):
    await ws.accept()

    client = ws.client
    effective_lang = lang if lang in ("en", "es") else cfg.language

    log.info("[CONNECT] %s | lang=%s", client, effective_lang or "auto-detect")

    session = StreamingSession(
        engine=engine,
        cfg=cfg,
        language_override=effective_lang,
    )

    try:
        while True:
            data = await ws.receive_bytes()

            loop = asyncio.get_running_loop()

            events = await loop.run_in_executor(
                None,
                session.process_chunk,
                data,
            )

            for ev in events:
                if len(ev) == 3:
                    ev_type, text, ttfb_ms = ev

                    log.info(
                        "[%s] ttfb=%s | %s",
                        ev_type.upper(),
                        ttfb_ms,
                        text,
                    )

                    await ws.send_text(
                        json.dumps(
                            {
                                "type": ev_type,
                                "text": text,
                                "t_start": ttfb_ms,
                            }
                        )
                    )

                else:
                    ev_type, text = ev

                    log.info("[%s] %s", ev_type.upper(), text)

                    await ws.send_text(
                        json.dumps(
                            {
                                "type": ev_type,
                                "text": text,
                            }
                        )
                    )

    except WebSocketDisconnect:
        log.info("[DISCONNECT] %s", client)

        try:
            session.saver.save_full_session()
        except Exception as exc:
            log.warning("[SESSION_SAVE] failed: %s", exc)

    except Exception as exc:
        log.exception("[WS_ERROR] %s", exc)

        try:
            session.saver.save_full_session()
        except Exception as save_exc:
            log.warning("[SESSION_SAVE] failed after error: %s", save_exc)
        
#app/streaming_session.py-
from __future__ import annotations

import logging
import time
from typing import Optional

from app.audio_saver import AudioSaver
from app.filler_filter import strip_fillers
from app.vad import AdaptiveEnergyVAD

log = logging.getLogger("parakeet_server")


class StreamingSession:
    def __init__(
        self,
        engine,
        cfg,
        language_override: Optional[str] = None,
    ):
        self.engine = engine
        self.cfg = cfg

        self.language = (
            language_override
            if language_override in ("en", "es")
            else cfg.language
        )

        log.info("[SESSION] language=%s", self.language or "auto-detect")

        self.vad = AdaptiveEnergyVAD(
            cfg.sample_rate,
            cfg.vad_frame_ms,
            cfg.vad_start_margin,
            cfg.vad_min_noise_rms,
            cfg.pre_speech_ms,
        )

        self.session = engine.new_session(
            cfg.max_utt_ms,
            language=self.language,
        )

        self.frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
        self.raw_buf = bytearray()

        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0

        self.t_utt_start: Optional[float] = None
        self.t_first_partial: Optional[float] = None

        self.saver = AudioSaver(
            enabled=cfg.save_audio,
            output_dir=cfg.audio_save_dir,
            sample_rate=cfg.sample_rate,
            max_saved=cfg.audio_max_saved,
        )

    def process_chunk(self, pcm: bytes) -> list:
        events = []

        # IMPORTANT:
        # Save EXACT raw PCM bytes received from client for this websocket session.
        # This happens BEFORE VAD, ASR, trimming, filtering, or endpointing.
        if self.cfg.save_audio:
            self.saver.append_session_audio(pcm)

        self.raw_buf.extend(pcm)

        while len(self.raw_buf) >= self.frame_bytes:
            frame = bytes(self.raw_buf[: self.frame_bytes])
            del self.raw_buf[: self.frame_bytes]

            is_speech, pre = self.vad.push_frame(frame)

            self.silence_ms = (
                0 if is_speech else self.silence_ms + self.cfg.vad_frame_ms
            )

            if pre and not self.utt_started:
                self.utt_started = True
                self.utt_audio_ms = 0
                self.silence_ms = 0
                self.t_utt_start = time.time()
                self.t_first_partial = None

                log.info("[UTT_START] utterance started")

                self.saver.start_utterance()

                self.session.accept_pcm16(pre)
                self.saver.add_frame(pre)

            if not self.utt_started:
                continue

            self.session.accept_pcm16(frame)
            self.saver.add_frame(frame)

            self.utt_audio_ms += self.cfg.vad_frame_ms

            if self.engine.caps.partials and self.utt_audio_ms >= 800:
                raw_partial = self.session.step_if_ready()

                if raw_partial:
                    clean_partial = strip_fillers(raw_partial, self.language)

                    if clean_partial:
                        if self.t_first_partial is None:
                            self.t_first_partial = time.time()

                        ttfb_ms = int(
                            (self.t_first_partial - self.t_utt_start) * 1000
                        )

                        events.append(("partial", clean_partial, ttfb_ms))

            end_silence = getattr(self.engine, "end_silence_ms", 700)
            min_utt = getattr(self.engine, "min_utt_ms", 300)
            pad = getattr(self.engine, "finalize_pad_ms", 0)

            if (
                not is_speech
                and self.utt_audio_ms >= min_utt
                and self.silence_ms >= end_silence
            ):
                raw_final = self.session.finalize(pad)

                if raw_final:
                    final = strip_fillers(raw_final, self.language)

                    if final:
                        ttfb_ms = (
                            int((self.t_first_partial - self.t_utt_start) * 1000)
                            if self.t_first_partial is not None
                            else None
                        )

                        log.info(
                            "[UTT_END] dur=%dms ttfb=%s | %r",
                            self.utt_audio_ms,
                            ttfb_ms,
                            final,
                        )

                        events.append(("transcript", final, ttfb_ms))

                        self.saver.save(
                            transcript=final,
                            language=self.language,
                            ttfb_ms=ttfb_ms,
                        )

                    else:
                        log.info("[UTT_END] filler-only discarded")
                        self.saver.discard()

                else:
                    log.info("[UTT_END] empty transcript discarded")
                    self.saver.discard()

                self._reset()

        return events

    def _reset(self) -> None:
        log.info("[RESET] session reset after utterance")

        self.vad.reset()

        self.session = self.engine.new_session(
            self.cfg.max_utt_ms,
            language=self.language,
        )

        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None
        
#app/vad.py-
from collections import deque

import numpy as np


class AdaptiveEnergyVAD:

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int,
        start_margin: float,
        min_noise_rms: float,
        pre_speech_ms: int,
    ):

        self.sr = sample_rate

        self.frame_ms = frame_ms

        self.start_margin = start_margin

        self.min_noise_rms = min_noise_rms

        self.frame_samples = int(
            self.sr * self.frame_ms / 1000
        )

        self.frame_bytes = self.frame_samples * 2

        self.pre_frames = max(
            1,
            int(pre_speech_ms / frame_ms),
        )

        self.ring = deque(maxlen=self.pre_frames)

        self.in_speech = False

        self.noise_rms = min_noise_rms

    def reset(self):

        self.ring.clear()

        self.in_speech = False

        self.noise_rms = self.min_noise_rms

    def _rms(self, pcm16: bytes) -> float:

        x = (
            np.frombuffer(
                pcm16,
                dtype=np.int16,
            )
            .astype(np.float32)
            / 32768.0
        )

        return float(
            np.sqrt(np.mean(x * x) + 1e-12)
        )

    def push_frame(self, frame_pcm16: bytes):

        e = self._rms(frame_pcm16)

        if not self.in_speech:

            alpha = 0.95

            self.noise_rms = max(
                self.min_noise_rms,
                alpha * self.noise_rms
                + (1 - alpha) * e,
            )

        threshold = (
            self.noise_rms
            * self.start_margin
        )

        is_speech = e >= threshold

        self.ring.append(frame_pcm16)

        pre_roll = None

        if (
            not self.in_speech
            and is_speech
        ):

            self.in_speech = True

            pre_roll = b"".join(self.ring)

        return is_speech, pre_roll

#factory.py-
from app.config import Config
from app.asr_engines.parakeet_asr import ParakeetASR


def build_engine(cfg: Config) -> ParakeetASR:

    if cfg.asr_backend == "parakeet":

        return ParakeetASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
            language=cfg.language,
            partial_tail_ms=cfg.partial_tail_ms,
        )

    raise ValueError(
        f"Unsupported ASR_BACKEND={cfg.asr_backend}"
    )
    
#config.py-
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass
class Config:

    # ============================================================
    # MODEL
    # ============================================================

    asr_backend: str = "parakeet"

    device: str = "cuda"

    sample_rate: int = 16000

    # ============================================================
    # LANGUAGE
    # ============================================================

    # None | "en" | "es"
    language: Optional[str] = None

    # ============================================================
    # AUDIO SAVING
    # ============================================================

    save_audio: bool = True

    audio_save_dir: str = "./asr_audio_chunks"

    audio_max_saved: int = 500

    # ============================================================
    # VAD
    # ============================================================

    vad_frame_ms: int = 30

    vad_start_margin: float = 2.5

    vad_min_noise_rms: float = 0.003

    pre_speech_ms: int = 300

    # ============================================================
    # LIMITS
    # ============================================================

    max_utt_ms: int = 30000

    partial_tail_ms: int = 6000

    log_level: str = "INFO"

    model_name: str = ""


MODEL_MAP = {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
}


def load_config() -> Config:

    cfg = Config()

    if not cfg.model_name:
        cfg = replace(
            cfg,
            model_name=MODEL_MAP[cfg.asr_backend],
        )

    return cfg


#app/filler_filter.py-
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger("parakeet_server")


_FILLERS_EN = [
    "um",
    "uh",
    "hmm",
    "hm",
    "mm",
    "er",
    "ah",
]

_FILLERS_ES = [
    "eh",
    "em",
    "eeh",
    "mmm",
    "hijole",
    "hijoles",
    "andale",
]


def _build_pattern(words: list[str]):

    parts = sorted(
        [re.escape(w) for w in words],
        key=len,
        reverse=True,
    )

    return re.compile(
        r"(?<!\w)(?:"
        + "|".join(parts)
        + r")(?!\w)[,]?\s*",
        re.IGNORECASE,
    )


_PAT_EN = _build_pattern(_FILLERS_EN)

_PAT_ES = _build_pattern(_FILLERS_ES)

_PAT_BOTH = _build_pattern(
    _FILLERS_EN + _FILLERS_ES
)

_REPETITION_RE = re.compile(
    r"\b(\w+)(?:\s+\1){1,3}\b",
    re.IGNORECASE,
)


def strip_fillers(
    text: str,
    language: Optional[str] = None,
) -> str:

    if not text:
        return text

    original = text

    if language == "en":
        pat = _PAT_EN

    elif language == "es":
        pat = _PAT_ES

    else:
        pat = _PAT_BOTH

    cleaned = pat.sub(" ", text)

    cleaned = _REPETITION_RE.sub(
        r"\1",
        cleaned,
    )

    cleaned = re.sub(
        r" {2,}",
        " ",
        cleaned,
    ).strip()

    cleaned = re.sub(
        r"^[,;.\s]+",
        "",
        cleaned,
    ).strip()

    if (
        cleaned
        and original[0].isupper()
    ):
        cleaned = (
            cleaned[0].upper()
            + cleaned[1:]
        )

    if len(cleaned) < 2:
        return ""

    if cleaned != original:
        log.debug(
            "[FILLER] %r -> %r",
            original,
            cleaned,
        )

    return cleaned

#audio_saver.py-
from __future__ import annotations

import json
import logging
import re
import uuid
import wave

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("parakeet_server")


def _slug(
    text: str,
    max_len: int = 40,
):

    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        "_",
        text,
    )

    return (
        text[:max_len].strip("_")
        or "utterance"
    )


class AudioSaver:

    def __init__(
        self,
        enabled: bool,
        output_dir: str,
        sample_rate: int,
        max_saved: int = 500,
    ):

        self.enabled = enabled

        self.output_dir = Path(output_dir)

        self.sample_rate = sample_rate

        self.max_saved = max_saved

        self.utterance_dir = (
            self.output_dir / "utterances"
        )

        self.websocket_dir = (
            self.output_dir
            / "websocket_sessions"
        )

        self._buffer = bytearray()

        self._utt_start = None

        self._counter = 0

        # FULL WEBSOCKET SESSION AUDIO
        self._session_audio = bytearray()

        self._session_id = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S")

        self._session_uuid = uuid.uuid4().hex[:8]

        if self.enabled:

            self.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.utterance_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.websocket_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            log.info(
                "[AUDIO_SAVER] output=%s",
                self.output_dir.resolve(),
            )

    # ============================================================
    # FULL WEBSOCKET AUDIO
    # ============================================================

    def append_session_audio(
        self,
        pcm16: bytes,
    ):

        if not self.enabled:
            return

        self._session_audio.extend(pcm16)

    def save_full_session(self):

        if (
            not self.enabled
            or not self._session_audio
        ):
            return None

        stem = (
            f"session_"
            f"{self._session_id}_"
            f"{self._session_uuid}"
        )

        wav_path = (
            self.websocket_dir
            / f"{stem}.wav"
        )

        json_path = (
            self.websocket_dir
            / f"{stem}.json"
        )

        audio = bytes(self._session_audio)

        duration_ms = int(
            len(audio)
            / 2
            / self.sample_rate
            * 1000
        )

        try:

            with wave.open(
                str(wav_path),
                "wb",
            ) as wf:

                wf.setnchannels(1)

                wf.setsampwidth(2)

                wf.setframerate(
                    self.sample_rate
                )

                wf.writeframes(audio)

        except Exception as exc:

            log.error(
                "[AUDIO_SAVER] session WAV failed: %s",
                exc,
            )

            return None

        meta = {
            "session_id": self._session_id,
            "uuid": self._session_uuid,
            "duration_ms": duration_ms,
            "sample_rate": self.sample_rate,
            "wav_file": wav_path.name,
        }

        try:

            json_path.write_text(
                json.dumps(
                    meta,
                    indent=2,
                )
            )

        except Exception as exc:

            log.warning(
                "[AUDIO_SAVER] session JSON failed: %s",
                exc,
            )

        log.info(
            "[AUDIO_SAVER] websocket session saved: %s",
            wav_path.name,
        )

        return wav_path

    # ============================================================
    # UTTERANCE AUDIO
    # ============================================================

    def start_utterance(self):

        if not self.enabled:
            return

        self._buffer = bytearray()

        self._utt_start = datetime.now(
            timezone.utc
        )

    def add_frame(
        self,
        pcm16: bytes,
    ):

        if not self.enabled:
            return

        self._buffer.extend(pcm16)

    def save(
        self,
        transcript: str,
        language: Optional[str] = None,
        ttfb_ms: Optional[int] = None,
    ):

        if (
            not self.enabled
            or not self._buffer
        ):
            return None

        self._counter += 1

        ts = (
            self._utt_start
            or datetime.now(timezone.utc)
        )

        stem = (
            f"{ts.strftime('%Y%m%d_%H%M%S')}_"
            f"{self._counter:04d}_"
            f"{_slug(transcript)}"
        )

        wav_path = (
            self.utterance_dir
            / f"{stem}.wav"
        )

        json_path = (
            self.utterance_dir
            / f"{stem}.json"
        )

        audio = bytes(self._buffer)

        duration_ms = int(
            len(audio)
            / 2
            / self.sample_rate
            * 1000
        )

        try:

            with wave.open(
                str(wav_path),
                "wb",
            ) as wf:

                wf.setnchannels(1)

                wf.setsampwidth(2)

                wf.setframerate(
                    self.sample_rate
                )

                wf.writeframes(audio)

        except Exception as exc:

            log.error(
                "[AUDIO_SAVER] utterance WAV failed: %s",
                exc,
            )

            return None

        meta = {
            "index": self._counter,
            "timestamp": ts.isoformat(),
            "duration_ms": duration_ms,
            "transcript": transcript,
            "language": language or "auto",
            "ttfb_ms": ttfb_ms,
            "sample_rate": self.sample_rate,
            "wav_file": wav_path.name,
        }

        try:

            json_path.write_text(
                json.dumps(
                    meta,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        except Exception as exc:

            log.warning(
                "[AUDIO_SAVER] utterance JSON failed: %s",
                exc,
            )

        self._buffer = bytearray()

        self._utt_start = None

        log.info(
            "[AUDIO_SAVER] utterance saved: %s",
            wav_path.name,
        )

        return wav_path

    def discard(self):

        self._buffer = bytearray()

        self._utt_start = None

#Dockerfile-
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    build-essential \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p ./asr_audio_chunks/websocket_sessions \
    && mkdir -p ./asr_audio_chunks/utterances

RUN python -c "\
import nemo.collections.asr as nemo_asr; \
m = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3'); \
print('Model cached')"

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--ws-ping-interval", "20", "--ws-ping-timeout", "120"]


#client.py-
"""
Parakeet ASR – Terminal Microphone / File Streaming Client
"""

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional

import numpy as np
import pyaudio
import websockets

# ============================================================
# CONFIG
# ============================================================

# CHANGE ONLY THIS
WS_URL = "wss://cx-asr.exlservice.com/asr/ml/ws"

SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16

CHUNK_MS = 30

CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000

CHUNK_BYTES = CHUNK_SAMPLES * 2

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)

BANNER = rf"""
╔══════════════════════════════════════════════════════════════╗
║        Parakeet-TDT-0.6B-v3  |  Real-Time ASR Client        ║
║            Stable Partial Streaming + Session Save          ║
║            WebSocket: {WS_URL[:52]:<52}║
╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
Commands
  q         Quit
  d         List available audio input devices

Notes
  - Partial transcripts stream while speaking
  - Final transcript appears after endpointing
  - Server saves COMPLETE websocket session audio
"""

# ============================================================
# AUDIO CAPTURE
# ============================================================


class MicCapture:

    def __init__(
        self,
        device_index: Optional[int] = None,
    ):

        self.queue: Queue[bytes] = Queue(maxsize=512)

        self._stop = threading.Event()

        self._device = device_index

        self._thread = threading.Thread(
            target=self._capture,
            daemon=True,
        )

    def start(self):

        self._thread.start()

    def stop(self):

        self._stop.set()

    def _capture(self):

        pa = pyaudio.PyAudio()

        stream = None

        try:

            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self._device,
                frames_per_buffer=CHUNK_SAMPLES,
            )

            while not self._stop.is_set():

                data = stream.read(
                    CHUNK_SAMPLES,
                    exception_on_overflow=False,
                )

                self.queue.put(data)

        except OSError as exc:

            print(
                f"\n[ERROR] Audio capture failed: {exc}",
                file=sys.stderr,
            )

            self._stop.set()

        finally:

            try:

                if stream is not None:

                    stream.stop_stream()

                    stream.close()

            except Exception:
                pass

            pa.terminate()


def list_devices():

    pa = pyaudio.PyAudio()

    print("\nAvailable audio input devices:\n")

    for i in range(pa.get_device_count()):

        info = pa.get_device_info_by_index(i)

        if info["maxInputChannels"] > 0:

            print(
                f"  [{i}] {info['name']} "
                f"({int(info['defaultSampleRate'])} Hz)"
            )

    pa.terminate()

    print()

# ============================================================
# AUDIO LOADING
# ============================================================


def load_audio_as_16k_pcm(path: str) -> bytes:

    try:

        import soundfile as sf

        data, sr = sf.read(
            path,
            dtype="float32",
            always_2d=False,
        )

        if data.ndim == 2:
            data = data.mean(axis=1)

        if sr != SAMPLE_RATE:
            data = _resample(data, sr, SAMPLE_RATE)

        return _float32_to_pcm16(data)

    except Exception as sf_err:

        try:

            from pydub import AudioSegment

            seg = AudioSegment.from_file(path)

            seg = seg.set_channels(1)

            seg = seg.set_frame_rate(SAMPLE_RATE)

            seg = seg.set_sample_width(2)

            return seg.raw_data

        except Exception as pd_err:

            raise RuntimeError(
                f"Cannot load '{path}'.\n"
                f"soundfile : {sf_err}\n"
                f"pydub     : {pd_err}\n"
                f"Tip: install ffmpeg."
            )


def _resample(
    data: np.ndarray,
    orig_sr: int,
    target_sr: int,
):

    try:

        import librosa

        return librosa.resample(
            data,
            orig_sr=orig_sr,
            target_sr=target_sr,
        )

    except ImportError:
        pass

    try:

        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(orig_sr, target_sr)

        return resample_poly(
            data,
            target_sr // g,
            orig_sr // g,
        ).astype(np.float32)

    except ImportError:
        pass

    duration = len(data) / orig_sr

    old_times = np.linspace(
        0,
        duration,
        len(data),
        endpoint=False,
    )

    new_len = int(duration * target_sr)

    new_times = np.linspace(
        0,
        duration,
        new_len,
        endpoint=False,
    )

    return np.interp(
        new_times,
        old_times,
        data,
    ).astype(np.float32)


def _float32_to_pcm16(data: np.ndarray) -> bytes:

    return (
        np.clip(data, -1.0, 1.0)
        * 32767
    ).astype(np.int16).tobytes()

# ============================================================
# RECEIVER
# ============================================================


async def _receiver(ws):

    last_partial = ""

    try:

        async for raw in ws:

            try:
                msg = json.loads(raw)

            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            text = msg.get("text", "")

            t_start = msg.get("t_start")

            ts = time.strftime("%H:%M:%S")

            # ====================================================
            # PARTIAL
            # ====================================================

            if msg_type == "partial":

                # IMPORTANT:
                # partials are COMPLETE hypotheses,
                # not incremental append-only chunks.
                last_partial = text

                if t_start is not None:

                    print(
                        f"\r  ⏳ [{ts}] ({t_start} ms) {text:<120}",
                        end="",
                        flush=True,
                    )

                else:

                    print(
                        f"\r  ⏳ [{ts}] {text:<120}",
                        end="",
                        flush=True,
                    )

            # ====================================================
            # FINAL
            # ====================================================

            elif msg_type == "transcript":

                # clear partial line
                print(f"\r{' ' * 160}\r", end="")

                if t_start is not None:

                    print(
                        f"[{ts}] Final | first partial at {t_start} ms"
                    )

                else:

                    print(f"[{ts}] Final")

                print(f"  ✅  {text}")

                print("─" * 72)

                last_partial = ""

    except websockets.exceptions.ConnectionClosed:

        if last_partial:
            print()

        print("[Connection closed]")

# ============================================================
# STDIN READER
# ============================================================


def stdin_reader(cmd_queue: Queue):

    while True:

        try:

            line = sys.stdin.readline()

            if line:
                cmd_queue.put(line.strip())

        except Exception:
            break

# ============================================================
# MIC MODE
# ============================================================


async def run_mic(
    device_index: Optional[int],
    language: Optional[str],
):

    uri = WS_URL

    if language:
        uri += f"?lang={language}"

    print(BANNER)

    print(f"Connecting to {uri} …", end=" ", flush=True)

    try:

        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**23,
        ) as ws:

            print("connected ✓\n")

            print("🎙  Speak now – transcriptions appear below\n")

            print("─" * 72)

            mic = MicCapture(device_index)

            mic.start()

            cmd_queue: Queue[str] = Queue()

            threading.Thread(
                target=stdin_reader,
                args=(cmd_queue,),
                daemon=True,
            ).start()

            send_task = asyncio.create_task(
                _mic_sender(ws, mic, cmd_queue)
            )

            recv_task = asyncio.create_task(
                _receiver(ws)
            )

            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            mic.stop()

    except (
        OSError,
        websockets.exceptions.WebSocketException,
    ) as exc:

        print(
            f"failed ✗\n[ERROR] {exc}",
            file=sys.stderr,
        )

        sys.exit(1)


async def _mic_sender(
    ws,
    mic: MicCapture,
    cmd_queue: Queue,
):

    print(HELP)

    try:

        while True:

            try:

                cmd = cmd_queue.get_nowait()

                if cmd.lower() in (
                    "q",
                    "quit",
                    "exit",
                ):

                    print("\nBye!")

                    return

                elif cmd.lower() == "d":

                    print()

                    list_devices()

            except Empty:
                pass

            sent = 0

            while (
                not mic.queue.empty()
                and sent < 50
            ):

                frame = mic.queue.get_nowait()

                await ws.send(frame)

                sent += 1

            await asyncio.sleep(0.005)

    except websockets.exceptions.ConnectionClosed:

        print("\n[Connection closed by server]")

# ============================================================
# FILE MODE
# ============================================================


async def run_files(
    files: List[str],
    speed: float,
    language: Optional[str],
):

    uri = WS_URL

    if language:
        uri += f"?lang={language}"

    print(BANNER)

    print(
        f"  Mode   : FILE streaming at {speed:.1f}×"
    )

    print(f"  Server : {uri}")

    print(
        f"  Files  : "
        f"{', '.join(Path(f).name for f in files)}\n"
    )

    for filepath in files:

        p = Path(filepath)

        if not p.exists():

            print(
                f"  [SKIP] File not found: {filepath}\n"
            )

            continue

        print(
            f"  Loading {p.name} …",
            end=" ",
            flush=True,
        )

        try:

            pcm = load_audio_as_16k_pcm(
                str(p)
            )

        except RuntimeError as exc:

            print(f"FAILED ✗\n  {exc}\n")

            continue

        duration_sec = (
            len(pcm)
            / 2
            / SAMPLE_RATE
        )

        print(f"{duration_sec:.1f}s ✓")

        print(f"\n{'═' * 72}")

        print(
            f"  📄 {p.name} ({duration_sec:.1f}s)"
        )

        print(f"{'═' * 72}\n")

        try:

            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=20,
                max_size=2**23,
            ) as ws:

                recv_task = asyncio.create_task(
                    _receiver(ws)
                )

                await _file_sender(
                    ws,
                    pcm,
                    speed,
                )

                await asyncio.sleep(3.0)

                await ws.close()

                await asyncio.sleep(0.2)

                if not recv_task.done():
                    recv_task.cancel()

        except (
            OSError,
            websockets.exceptions.WebSocketException,
        ) as exc:

            print(
                f"\n[ERROR] WebSocket failed: {exc}"
            )

        print()

    print("All files processed.")

# ============================================================
# FILE SENDER
# ============================================================


async def _file_sender(
    ws,
    pcm: bytes,
    speed: float,
):

    if speed <= 0:
        raise ValueError(
            "--speed must be > 0"
        )

    chunk_delay = (
        (CHUNK_MS / 1000.0)
        / speed
    )

    offset = 0

    while (
        offset + CHUNK_BYTES
        <= len(pcm)
    ):

        chunk = pcm[
            offset: offset + CHUNK_BYTES
        ]

        offset += CHUNK_BYTES

        await ws.send(chunk)

        await asyncio.sleep(chunk_delay)

    leftover = pcm[offset:]

    if leftover:

        await ws.send(
            leftover
            + bytes(
                CHUNK_BYTES
                - len(leftover)
            )
        )

# ============================================================
# CLI
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="Parakeet ASR client"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Mic device index",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List audio devices",
    )

    parser.add_argument(
        "--file",
        nargs="+",
        metavar="FILE",
        help="Audio file(s)",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed",
    )

    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "es"],
        default=None,
        help="Force language",
    )

    return parser.parse_args()

# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":

    args = parse_args()

    if args.list:

        list_devices()

        sys.exit(0)

    if args.file:

        asyncio.run(
            run_files(
                args.file,
                args.speed,
                args.lang,
            )
        )

    else:

        asyncio.run(
            run_mic(
                args.device,
                args.lang,
            )
        )


#jenkinsfile
pipeline {
    agent {
        label 'cicd'
    }
    environment {
        ACCOUNT_ID = "058264113403"
        REGION = "us-east-1"
        ENVIRONMENT = "develop"
        AWS_ROLE_ARN = "arn:aws:iam::${ACCOUNT_ID}:role/EXLJenkinsCrossAccountRole-BU"
        EKS_CLUSTER           = "CX-APPS-AA-EKS-DEV"
    }
    stages {
        stage('Security Scanning') {
            steps {
                script {
                    try {
                        sh "ls -al"
                        // build propagate: false, job: "ISGADSO/autonomous_agent/bu-digital-cx-aa-app/${env.BRANCH_NAME}"

                        } catch (Exception e) {
                            echo "Failed to trigger downstream job: ${e.getMessage()}"
                            currentBuild.result = 'FAILURE'

                        }
                 }
            }
        }  
        stage('Set Environment Variables based on branch'){
            steps{
                script{
                    def commitHash = sh(script: 'git rev-parse --short=6 HEAD', returnStdout: true).trim()
                    sh "echo ${env.BRANCH_NAME}"
                    if (env.BRANCH_NAME == 'dev') {
                        env.ENVIRONMENT           = "develop"
                        env.IMAGE_REPO_NAME       = "asr-realtime-ml"
                        env.ECR_REPO_URI          = "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${env.IMAGE_REPO_NAME}"
                        env.IMAGE_TAG             = "develop_${commitHash}"
                        env.DEPLOYMENT_NAME       = "asr-realtime-ml"
                        env.CONTAINER_NAME        = "asr-realtime-ml-container"
                        env.K8_NAMESPACE          = "cx-speech"
                        env.FOLDER_NAME           = "."
                        // env.DOCKER_COMPOSE_FILE   = "docker-compose.yml"
                        // env.AZURE_REPO            = "asr-realtime"
			            env.DOCKER_FILE		  = "Dockerfile"
                        sh "echo ${env.FOLDER_NAME}"
                    }
                }
            }
        }
        stage('Build and Push Docker Image') {
            steps {
                script {
                    // Build and push Docker image to ECR    
                    sh "echo ${env.FOLDER_NAME}"            
                    withAWS(role: AWS_ROLE_ARN, region: REGION , useNode: true) {
			 sh "aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin '$ACCOUNT_ID'.dkr.ecr.'$REGION'.amazonaws.com"
                    	sh "export TAG=${env.IMAGE_TAG}"
			sh "docker build -t $IMAGE_REPO_NAME:$IMAGE_TAG -f ./$DOCKER_FILE ."
                    	sh "docker images"
                       
                        sh "echo docker tag  ${env.IMAGE_REPO_NAME}:${env.IMAGE_TAG} ${env.ECR_REPO_URI}:${env.IMAGE_TAG}"
                        sh "docker tag ${env.IMAGE_REPO_NAME}:${env.IMAGE_TAG} ${env.ECR_REPO_URI}:${env.IMAGE_TAG}"
                        sh "echo docker push"
                        sh "docker push ${env.ECR_REPO_URI}:${env.IMAGE_TAG}"
                   }
                }
            }
        }

	stage('Deploy EKS') {
            steps {
                script {
                    // Build and push Docker image to ECR                
                    withAWS(role: AWS_ROLE_ARN, region: REGION , useNode: true) {
			// def secretJson = sh(
	  //                       script: "aws secretsmanager get-secret-value --secret-id dev/transform-cx/ds --region $AWS_REGION --query SecretString --output text > config.ini",
	  //                       returnStdout: true
   //                  	).trim()
			// sh "head -n 10 config.ini"
                        sh "echo Setting kubeconfig"
                        sh "aws eks --region ${AWS_REGION} update-kubeconfig --name ${EKS_CLUSTER}"
                        sh "kubectl config get-contexts"
                        sh "aws eks describe-cluster --name ${EKS_CLUSTER} --region ${AWS_REGION}"
			// sh "kubectl create configmap ${env.DEPLOYMENT_NAME}-config --from-file=config.ini --namespace=${env.K8_NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -"
                        sh "echo Deployment EKS Started"
                        sh "kubectl set image deployment/${env.DEPLOYMENT_NAME} ${env.CONTAINER_NAME}=${env.ECR_REPO_URI}:${env.IMAGE_TAG} -n ${env.K8_NAMESPACE}"
                        sh "kubectl rollout restart deployment/${env.DEPLOYMENT_NAME} -n ${env.K8_NAMESPACE}"
                   }
                }
            }
        }
 
    }
}
