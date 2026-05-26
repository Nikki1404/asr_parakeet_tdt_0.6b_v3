#app/config.py-
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass
class Config:
    asr_backend: str = "parakeet"
    device: str = "cuda"
    sample_rate: int = 16000

    # Client must send --lang en or --lang es.
    language: Optional[str] = None
    require_client_language: bool = True

    save_audio: bool = True
    audio_save_dir: str = "./asr_audio_chunks"
    audio_max_saved: int = 500

    # Accuracy-first custom VAD
    vad_frame_ms: int = 30
    vad_start_margin: float = 2.0
    vad_min_noise_rms: float = 0.002
    pre_speech_ms: int = 600

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
        cfg = replace(cfg, model_name=MODEL_MAP[cfg.asr_backend])

    return cfg


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


#app/asr_engines/parakeet_asr.py-
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
import wave
from difflib import SequenceMatcher
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


_LATIN_RE = re.compile(
    r"^[\x00-\xFF\u00C0-\u024F\s.,;:!?'\"\-()\d]+$"
)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(
        None,
        a.lower().strip(),
        b.lower().strip(),
    ).ratio()


def _is_gibberish(text: str) -> bool:
    t = text.strip()

    if not t:
        return True

    if not _LATIN_RE.match(t):
        return True

    words = t.split()

    # repeated hallucinations
    if len(words) >= 4 and len(set(w.lower() for w in words)) == 1:
        return True

    # invalid single-char noise
    if len(t) == 1 and not t.isalpha():
        return True

    return False


def _score_candidate(text: str, audio_ms: int) -> int:
    if not text or _is_gibberish(text):
        return -100

    t = text.strip().lower()
    words = t.split()

    score = 0

    # ------------------------------------------------------------
    # VERY IMPORTANT:
    # Prefer SHORT outputs for SHORT speech.
    #
    # Prevent:
    # "three four"
    # -> "go on to four of it"
    # ------------------------------------------------------------

    if audio_ms <= 2000:

        if len(words) <= 4:
            score += 12

        if len(words) >= 7:
            score -= 20

    # ------------------------------------------------------------
    # Penalize multilingual hallucinations seen in testing
    # ------------------------------------------------------------

    bad_tokens = [
        "devět",
        "tři",
        "kět",
        "spalny",
        "spalnę",
        "ś",
    ]

    if any(tok in t for tok in bad_tokens):
        score -= 40

    # ------------------------------------------------------------
    # Prefer command / digit style utterances
    # ------------------------------------------------------------

    common_short_words = {
        "zero", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "okay", "ok", "yes", "no", "right",
        "hola", "gracias", "sí", "si",
    }

    if any(
        w.strip(".,!?").lower() in common_short_words
        for w in words
    ):
        score += 8

    # Natural language reward
    score += min(len(words), 6)

    return score


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

        # --------------------------------------------------------
        # ACCURACY-FIRST ENDPOINTING
        #
        # Short utterances need:
        # - more silence
        # - more context
        # - no clipping
        # --------------------------------------------------------

        self.end_silence_ms = 1100
        self.min_utt_ms = 700
        self.finalize_pad_ms = 300

        # --------------------------------------------------------
        # Slower partials = more stable decoder
        # --------------------------------------------------------

        self.partial_interval_sec = 2.0

    def load(self) -> float:
        import nemo.collections.asr as nemo_asr

        t0 = time.time()

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            self.model_name
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        return time.time() - t0

    def new_session(
        self,
        max_buffer_ms: int,
        language: Optional[str] = None,
    ) -> "ParakeetSession":

        effective_language = (
            language
            if language in ("en", "es")
            else self.language
        )

        return ParakeetSession(
            engine=self,
            max_buffer_ms=max_buffer_ms,
            language=effective_language,
        )


_MIN_PARTIAL_MS = 900


class ParakeetSession:

    def __init__(
        self,
        engine: ParakeetASR,
        max_buffer_ms: int,
        language: Optional[str] = None,
    ):
        self.engine = engine
        self.language = language

        log.info(
            "[ASR_SESSION] locked_language=%s",
            language,
        )

        self.max_buffer_samples = int(
            engine.sr * max_buffer_ms / 1000
        )

        self.audio = bytearray()

        self.current_text = ""
        self.last_good_partial = ""

        self.last_partial_time = 0.0
        self._audio_ms = 0

    def accept_pcm16(self, pcm16: bytes) -> None:
        self.audio.extend(pcm16)

        self._audio_ms = int(
            len(self.audio) / 2 / self.engine.sr * 1000
        )

        max_bytes = self.max_buffer_samples * 2

        if len(self.audio) > max_bytes:
            self.audio = self.audio[-max_bytes:]

            self._audio_ms = int(
                len(self.audio) / 2 / self.engine.sr * 1000
            )

    def step_if_ready(self) -> Optional[str]:

        now = time.time()

        if not self.audio:
            return None

        # --------------------------------------------------------
        # Don't decode tiny unstable buffers
        # --------------------------------------------------------

        if self._audio_ms < _MIN_PARTIAL_MS:
            return None

        if (
            now - self.last_partial_time
        ) < self.engine.partial_interval_sec:
            return None

        self.last_partial_time = now

        # --------------------------------------------------------
        # Tail-only partial decode
        # --------------------------------------------------------

        text = self._transcribe(
            tail_only=True,
            pad_ms=250,
        ).strip()

        if not text:
            return None

        if _is_gibberish(text):
            return None

        if text == self.current_text:
            return None

        self.current_text = text
        self.last_good_partial = text

        log.info(
            "[PARTIAL_SELECT] audio_ms=%s | locked_lang=%s | text=%r",
            self._audio_ms,
            self.language,
            text,
        )

        return text

    def finalize(self, pad_ms: int) -> str:

        candidates = []

        # --------------------------------------------------------
        # MAIN FINAL DECODE
        # --------------------------------------------------------

        full_final = self._transcribe(
            tail_only=False,
            pad_ms=pad_ms,
        ).strip()

        if full_final:
            candidates.append(
                ("full_final", full_final)
            )

        # --------------------------------------------------------
        # SHORT UTTERANCE RETRY
        #
        # More silence padding helps:
        # - "three four"
        # - "repo"
        # - "okay"
        # --------------------------------------------------------

        if self._audio_ms <= 2500:

            retry_final = self._transcribe(
                tail_only=False,
                pad_ms=600,
            ).strip()

            if retry_final:
                candidates.append(
                    ("short_retry", retry_final)
                )

        # --------------------------------------------------------
        # Stable partial candidate
        # --------------------------------------------------------

        if self.last_good_partial:
            candidates.append(
                (
                    "last_good_partial",
                    self.last_good_partial,
                )
            )

        best_source = ""
        best = ""
        best_score = -999

        log.info(
            "[FINAL_START] audio_ms=%s | locked_lang=%s | last_partial=%r | candidates=%d",
            self._audio_ms,
            self.language,
            self.last_good_partial,
            len(candidates),
        )

        for source, candidate in candidates:

            score = _score_candidate(
                candidate,
                self._audio_ms,
            )

            sim = None

            # ----------------------------------------------------
            # VERY IMPORTANT:
            #
            # Protect stable partial from
            # hallucinated final drift.
            #
            # Example:
            #
            # partial:
            # "three four"
            #
            # final:
            # "go on to four of it"
            # ----------------------------------------------------

            if (
                self.last_good_partial
                and candidate != self.last_good_partial
            ):
                sim = _similarity(
                    candidate,
                    self.last_good_partial,
                )

                if sim < 0.45:
                    score -= 25

                elif sim < 0.65:
                    score -= 12

            log.info(
                "[FINAL_CANDIDATE] source=%s | audio_ms=%s | score=%s | similarity=%s | text=%r",
                source,
                self._audio_ms,
                score,
                f"{sim:.2f}" if sim is not None else "NA",
                candidate,
            )

            if score > best_score:
                best = candidate
                best_source = source
                best_score = score

        if best and not _is_gibberish(best):
            self.current_text = best

        out = self.current_text.strip()

        log.info(
            "[FINAL_SELECT] source=%s | score=%s | audio_ms=%s | selected=%r",
            best_source,
            best_score,
            self._audio_ms,
            out,
        )

        # --------------------------------------------------------
        # RESET SESSION STATE
        # --------------------------------------------------------

        self.audio.clear()

        self.current_text = ""
        self.last_good_partial = ""

        self.last_partial_time = 0.0
        self._audio_ms = 0

        return out

    def _transcribe(
        self,
        tail_only: bool = False,
        pad_ms: int = 0,
    ) -> str:

        tmp_path = None

        try:

            audio = bytes(self.audio)

            # ----------------------------------------------------
            # Partial decode only uses recent tail
            # ----------------------------------------------------

            if tail_only:

                tail_bytes = int(
                    self.engine.sr
                    * self.engine.partial_tail_ms
                    / 1000
                ) * 2

                audio = audio[-tail_bytes:]

            # ----------------------------------------------------
            # Add trailing silence padding
            # ----------------------------------------------------

            if pad_ms > 0:

                pad_bytes = int(
                    self.engine.sr * pad_ms / 1000
                ) * 2

                audio = audio + (
                    b"\x00" * pad_bytes
                )

            if not audio:
                return ""

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as f:

                tmp_path = f.name

                f.write(
                    self._to_wav(
                        audio,
                        self.engine.sr,
                    )
                )

            results = self._call_model(
                tmp_path,
                self.language,
            )

            if not results:
                return ""

            text = safe_text(
                results[0]
            ).strip()

            return text

        except Exception as exc:

            log.warning(
                "[ASR_TRANSCRIBE_ERROR] %s",
                exc,
            )

            return ""

        finally:

            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _call_model(
        self,
        wav_path: str,
        language: Optional[str],
    ):

        model = self.engine.model

        # --------------------------------------------------------
        # Language locked from client side.
        #
        # NO AUTO-DETECT IN PRODUCTION.
        # --------------------------------------------------------

        if not language:
            return model.transcribe([wav_path])

        try:
            return model.transcribe(
                [wav_path],
                language_id=language,
            )

        except TypeError:

            log.warning(
                "[ASR] language_id unsupported, trying language"
            )

        try:
            return model.transcribe(
                [wav_path],
                language=language,
            )

        except TypeError:

            log.warning(
                "[ASR] language unsupported, using default"
            )

        return model.transcribe([wav_path])

    @staticmethod
    def _to_wav(
        pcm: bytes,
        sr: int,
    ) -> bytes:

        buf = io.BytesIO()

        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)

        return buf.getvalue()

#app/factory.py-
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

    raise ValueError(f"Unsupported ASR_BACKEND={cfg.asr_backend}")


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

        self.frame_samples = int(self.sr * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2

        self.pre_frames = max(1, int(pre_speech_ms / frame_ms))
        self.ring = deque(maxlen=self.pre_frames)

        self.in_speech = False
        self.noise_rms = min_noise_rms

    def reset(self):
        self.ring.clear()
        self.in_speech = False
        self.noise_rms = self.min_noise_rms

    def _rms(self, pcm16: bytes) -> float:
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def push_frame(self, frame_pcm16: bytes):
        e = self._rms(frame_pcm16)

        if not self.in_speech:
            alpha = 0.95
            self.noise_rms = max(
                self.min_noise_rms,
                alpha * self.noise_rms + (1 - alpha) * e,
            )

        threshold = self.noise_rms * self.start_margin
        is_speech = e >= threshold

        self.ring.append(frame_pcm16)

        pre_roll = None

        if not self.in_speech and is_speech:
            self.in_speech = True
            pre_roll = b"".join(self.ring)

        return is_speech, pre_roll

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

        self.language = language_override if language_override in ("en", "es") else cfg.language

        log.info("[SESSION] locked_language=%s", self.language)

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

            # Partials are display-only. Final accuracy is priority.
            if self.engine.caps.partials and self.utt_audio_ms >= 1200:
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

            end_silence = getattr(self.engine, "end_silence_ms", 1100)
            min_utt = getattr(self.engine, "min_utt_ms", 700)
            pad = getattr(self.engine, "finalize_pad_ms", 300)

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


#app/main.py-
from __future__ import annotations

import asyncio
import json
import logging
import sys

import numpy as np
import resampy
from fastapi import FastAPI, WebSocket
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


def upsample_if_needed(pcm: bytes, client_sample_rate: int) -> bytes:
    if not pcm:
        return pcm

    if client_sample_rate == cfg.sample_rate:
        return pcm

    log.debug(
        "[RESAMPLE] client_sr=%s -> server_sr=%s | bytes=%s",
        client_sample_rate,
        cfg.sample_rate,
        len(pcm),
    )

    try:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        y = resampy.resample(
            x,
            client_sample_rate,
            cfg.sample_rate,
        )

        y = np.clip(y, -1.0, 1.0)

        return (y * 32767.0).astype(np.int16).tobytes()

    except Exception as exc:
        log.warning("[RESAMPLE_FAILED] %s", exc)
        return pcm


@app.on_event("startup")
async def startup():
    load_sec = engine.load()

    log.info("[STARTUP] model_loaded_sec=%.2f", load_sec)
    log.info("[STARTUP] backend=%s", cfg.asr_backend)
    log.info("[STARTUP] model=%s", cfg.model_name)
    log.info("[STARTUP] server_sample_rate=%s", cfg.sample_rate)
    log.info("[STARTUP] audio_save=%s", cfg.save_audio)
    log.info("[STARTUP] audio_save_dir=%s", cfg.audio_save_dir)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "backend": cfg.asr_backend,
        "model": cfg.model_name,
        "sample_rate": cfg.sample_rate,
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
async def ws_asr(ws: WebSocket):
    await ws.accept()

    client = ws.client
    log.info("[CONNECT] client=%s", client)

    session = None

    try:
        # ----------------------------------------------------------
        # First message must be JSON config from client.
        # Endpoint remains only: ws://host/asr/ml/ws
        # ----------------------------------------------------------
        config_raw = await ws.receive_text()
        config_msg = json.loads(config_raw)

        if config_msg.get("type") != "config":
            await ws.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "text": "First message must be JSON: {'type':'config','lang':'en|es','sr':16000}",
                    }
                )
            )
            await ws.close()
            log.warning("[REJECT] client=%s | reason=first_message_not_config", client)
            return

        lang = config_msg.get("lang")
        client_sr = int(config_msg.get("sr", cfg.sample_rate))

        if lang not in ("en", "es"):
            await ws.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "text": "lang must be 'en' or 'es'",
                    }
                )
            )
            await ws.close()
            log.warning("[REJECT] client=%s | invalid_lang=%s", client, lang)
            return

        if client_sr <= 0:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "text": "sr must be a valid positive sample rate",
                    }
                )
            )
            await ws.close()
            log.warning("[REJECT] client=%s | invalid_sr=%s", client, client_sr)
            return

        log.info(
            "[SESSION_CONFIG] client=%s | locked_lang=%s | client_sr=%s | server_sr=%s",
            client,
            lang,
            client_sr,
            cfg.sample_rate,
        )

        session = StreamingSession(
            engine=engine,
            cfg=cfg,
            language_override=lang,
        )

        await ws.send_text(
            json.dumps(
                {
                    "type": "ready",
                    "lang": lang,
                    "client_sr": client_sr,
                    "server_sr": cfg.sample_rate,
                }
            )
        )

        chunk_count = 0
        total_input_bytes = 0
        total_resampled_bytes = 0

        while True:
            data = await ws.receive_bytes()

            chunk_count += 1
            total_input_bytes += len(data)

            data = upsample_if_needed(data, client_sr)
            total_resampled_bytes += len(data)

            if chunk_count % 100 == 0:
                log.info(
                    "[AUDIO_IN] client=%s | chunks=%d | input_bytes=%d | processed_bytes=%d",
                    client,
                    chunk_count,
                    total_input_bytes,
                    total_resampled_bytes,
                )

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
                        "[%s] client=%s | ttfb_ms=%s | text=%r",
                        ev_type.upper(),
                        client,
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

                    log.info(
                        "[%s] client=%s | text=%r",
                        ev_type.upper(),
                        client,
                        text,
                    )

                    await ws.send_text(
                        json.dumps(
                            {
                                "type": ev_type,
                                "text": text,
                            }
                        )
                    )

    except WebSocketDisconnect:
        log.info("[DISCONNECT] client=%s", client)

        if session is not None:
            try:
                session.saver.save_full_session()
            except Exception as exc:
                log.warning("[SESSION_SAVE_FAILED] %s", exc)

    except Exception as exc:
        log.exception("[WS_ERROR] client=%s | error=%s", client, exc)

        if session is not None:
            try:
                session.saver.save_full_session()
            except Exception as save_exc:
                log.warning("[SESSION_SAVE_FAILED_AFTER_ERROR] %s", save_exc)

        try:
            await ws.close()
        except Exception:
            pass

  #app/filler_filter.py-
  from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger("parakeet_server")


_FILLERS_EN = ["um", "uh", "hmm", "hm", "mm", "er", "ah"]
_FILLERS_ES = ["eh", "em", "eeh", "mmm", "hijole", "hijoles", "andale"]


def _build_pattern(words: list[str]):
    parts = sorted([re.escape(w) for w in words], key=len, reverse=True)

    return re.compile(
        r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)[,]?\s*",
        re.IGNORECASE,
    )


_PAT_EN = _build_pattern(_FILLERS_EN)
_PAT_ES = _build_pattern(_FILLERS_ES)
_PAT_BOTH = _build_pattern(_FILLERS_EN + _FILLERS_ES)

_REPETITION_RE = re.compile(r"\b(\w+)(?:\s+\1){1,3}\b", re.IGNORECASE)


def strip_fillers(text: str, language: Optional[str] = None) -> str:
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
    cleaned = _REPETITION_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    cleaned = re.sub(r"^[,;.\s]+", "", cleaned).strip()

    if cleaned and original[0].isupper():
        cleaned = cleaned[0].upper() + cleaned[1:]

    if len(cleaned) < 2:
        return ""

    if cleaned != original:
        log.debug("[FILLER] %r -> %r", original, cleaned)

    return cleaned

#client.py-
"""
Parakeet ASR – Terminal Microphone / File Streaming Client
Accuracy-first realtime ASR client for EN/ES locked sessions
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

# =============================================================================
# CONFIG
# =============================================================================

WS_URL = "wss://cx-asr.exlservice.com/asr/ml/ws"

SAMPLE_RATE = 16_000
CHANNELS = 1
FORMAT = pyaudio.paInt16

CHUNK_MS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2

LANGUAGE = "en"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BANNER = rf"""
╔══════════════════════════════════════════════════════════════╗
║        Parakeet-TDT-0.6B-v3  |  Real-Time ASR Client        ║
║        Locked-language realtime ASR (EN / ES only)          ║
║            WebSocket: {WS_URL[:52]:<52}║
╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
Commands
  q         Quit
  d         List available audio input devices

Notes
  - Language is locked from client side.
  - Partial transcripts are shown while speaking.
  - Final transcript arrives after endpointing on the server.
"""

# =============================================================================
# AUDIO CAPTURE
# =============================================================================


class MicCapture:

    def __init__(self, device_index: Optional[int] = None):
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

    print("\nAvailable audio input devices:")

    for i in range(pa.get_device_count()):

        info = pa.get_device_info_by_index(i)

        if info["maxInputChannels"] > 0:

            print(
                f"  [{i}] {info['name']}  "
                f"({int(info['defaultSampleRate'])} Hz)"
            )

    pa.terminate()

    print()

# =============================================================================
# AUDIO LOADING
# =============================================================================


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
            data = _resample(
                data,
                sr,
                SAMPLE_RATE,
            )

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
        np.clip(data, -1.0, 1.0) * 32767
    ).astype(np.int16).tobytes()

# =============================================================================
# CONFIG MESSAGE
# =============================================================================


async def send_config(ws):

    cfg = {
        "type": "config",
        "lang": LANGUAGE,
        "sr": SAMPLE_RATE,
    }

    await ws.send(json.dumps(cfg))

    print(
        f"[CONFIG] lang={LANGUAGE} | sr={SAMPLE_RATE}"
    )

# =============================================================================
# RECEIVER
# =============================================================================


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

            # ----------------------------------------------------
            # READY
            # ----------------------------------------------------

            if msg_type == "ready":

                print(
                    f"\n[SERVER_READY] "
                    f"lang={msg.get('lang')} | "
                    f"client_sr={msg.get('client_sr')} | "
                    f"server_sr={msg.get('server_sr')}\n"
                )

            # ----------------------------------------------------
            # ERROR
            # ----------------------------------------------------

            elif msg_type == "error":

                print(
                    f"\n[SERVER_ERROR] {text}\n"
                )

            # ----------------------------------------------------
            # PARTIAL
            # ----------------------------------------------------

            elif msg_type == "partial":

                last_partial = text

                if t_start is not None:

                    print(
                        f"\r  ⏳ [{ts}] ({t_start} ms) {text:<100}",
                        end="",
                        flush=True,
                    )

                else:

                    print(
                        f"\r  ⏳ [{ts}] {text:<100}",
                        end="",
                        flush=True,
                    )

            # ----------------------------------------------------
            # FINAL
            # ----------------------------------------------------

            elif msg_type == "transcript":

                print(f"\r{' ' * 140}\r", end="")

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

# =============================================================================
# STDIN READER
# =============================================================================


def stdin_reader(cmd_queue: Queue):

    while True:

        try:
            line = sys.stdin.readline()

            if line:
                cmd_queue.put(line.strip())

        except Exception:
            break

# =============================================================================
# MIC MODE
# =============================================================================


async def run_mic(device_index: Optional[int]):

    uri = WS_URL

    print(BANNER)

    print(
        f"Connecting to {uri} …",
        end=" ",
        flush=True,
    )

    try:

        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**23,
        ) as ws:

            await send_config(ws)

            print("connected ✓\n")

            print(
                "🎙  Speak now – transcriptions appear below\n"
            )

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
                _mic_sender(
                    ws,
                    mic,
                    cmd_queue,
                )
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

# =============================================================================
# FILE MODE
# =============================================================================


async def run_files(
    files: List[str],
    speed: float,
):

    uri = WS_URL

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
            pcm = load_audio_as_16k_pcm(str(p))

        except RuntimeError as exc:

            print(f"FAILED ✗\n  {exc}\n")

            continue

        duration_sec = (
            len(pcm) / 2 / SAMPLE_RATE
        )

        print(f"{duration_sec:.1f}s ✓")

        print(f"\n{'═' * 72}")

        print(
            f"  📄 {p.name} "
            f"({duration_sec:.1f}s)"
        )

        print(f"{'═' * 72}\n")

        try:

            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=20,
                max_size=2**23,
            ) as ws:

                await send_config(ws)

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

# =============================================================================
# FILE SENDER
# =============================================================================


async def _file_sender(
    ws,
    pcm: bytes,
    speed: float,
):

    if speed <= 0:
        raise ValueError("--speed must be > 0")

    chunk_delay = (
        CHUNK_MS / 1000.0
    ) / speed

    offset = 0

    while offset + CHUNK_BYTES <= len(pcm):

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
            + bytes(CHUNK_BYTES - len(leftover))
        )

# =============================================================================
# CLI
# =============================================================================


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
        choices=["en", "es"],
        default="en",
        help="Lock ASR language",
    )

    return parser.parse_args()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    args = parse_args()

    LANGUAGE = args.lang

    print(
        f"\n[LANGUAGE_LOCK] {LANGUAGE}\n"
    )

    if args.list:
        list_devices()
        sys.exit(0)

    if args.file:

        asyncio.run(
            run_files(
                args.file,
                args.speed,
            )
        )

    else:

        asyncio.run(
            run_mic(args.device)
        )


#sample websocket code
import asyncio
import json
import logging
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import websockets

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("parakeet_test")

# ============================================================================
# CONFIG
# ============================================================================

WS_URL = "wss://cx-asr.exlservice.com/asr/ml/ws"

LANGUAGE = "en"   # change to "es" for Spanish

TARGET_SR = 16000

CHUNK_MS = 30

CHUNK_SAMPLES = TARGET_SR * CHUNK_MS // 1000

CHUNK_BYTES = CHUNK_SAMPLES * 2

# ============================================================================
# AUDIO LOADING
# ============================================================================


def load_audio(filepath: str) -> bytes:

    audio, sr = sf.read(
        filepath,
        dtype="float32",
    )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:

        logger.info(
            "Resampling %s Hz -> %s Hz",
            sr,
            TARGET_SR,
        )

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR,
        )

    pcm = (
        np.clip(audio, -1.0, 1.0)
        * 32767
    ).astype(np.int16)

    return pcm.tobytes()

# ============================================================================
# SEND CONFIG
# ============================================================================


async def send_config(ws):

    cfg = {
        "type": "config",
        "lang": LANGUAGE,
        "sr": TARGET_SR,
    }

    await ws.send(json.dumps(cfg))

    logger.info(
        "Sent config | lang=%s | sr=%s",
        LANGUAGE,
        TARGET_SR,
    )

# ============================================================================
# RECEIVER
# ============================================================================


async def receiver(ws):

    async for msg in ws:

        if not isinstance(msg, str):
            continue

        try:
            obj = json.loads(msg)

        except Exception:
            continue

        typ = obj.get("type")

        txt = obj.get("text", "")

        ttfb = obj.get("t_start")

        ts = time.strftime("%H:%M:%S")

        # --------------------------------------------------------------------
        # READY
        # --------------------------------------------------------------------

        if typ == "ready":

            logger.info(
                "SERVER READY | lang=%s | client_sr=%s | server_sr=%s",
                obj.get("lang"),
                obj.get("client_sr"),
                obj.get("server_sr"),
            )

        # --------------------------------------------------------------------
        # ERROR
        # --------------------------------------------------------------------

        elif typ == "error":

            logger.error(
                "SERVER ERROR: %s",
                txt,
            )

        # --------------------------------------------------------------------
        # PARTIAL
        # --------------------------------------------------------------------

        elif typ == "partial":

            print(
                f"\r⏳ [{ts}] "
                f"({ttfb} ms) "
                f"{txt:<120}",
                end="",
                flush=True,
            )

        # --------------------------------------------------------------------
        # FINAL
        # --------------------------------------------------------------------

        elif typ == "transcript":

            print("\r" + " " * 180 + "\r", end="")

            print(
                f"[{ts}] FINAL "
                f"(first partial {ttfb} ms)"
            )

            print(f"✅ {txt}")

            print("─" * 80)

# ============================================================================
# SENDER
# ============================================================================


async def sender(
    ws,
    pcm_audio: bytes,
):

    offset = 0

    total_chunks = 0

    while offset < len(pcm_audio):

        chunk = pcm_audio[
            offset: offset + CHUNK_BYTES
        ]

        offset += CHUNK_BYTES

        if len(chunk) < CHUNK_BYTES:

            chunk += bytes(
                CHUNK_BYTES - len(chunk)
            )

        await ws.send(chunk)

        total_chunks += 1

        if total_chunks % 100 == 0:

            logger.info(
                "Sent chunks=%s",
                total_chunks,
            )

        await asyncio.sleep(
            CHUNK_MS / 1000
        )

    logger.info(
        "Finished audio streaming | chunks=%s",
        total_chunks,
    )

# ============================================================================
# MAIN STREAM
# ============================================================================


async def stream_audio(
    audio_file: str,
):

    pcm_audio = load_audio(audio_file)

    duration_sec = (
        len(pcm_audio) / 2 / TARGET_SR
    )

    logger.info(
        "Loaded audio | file=%s | duration=%.2fs",
        Path(audio_file).name,
        duration_sec,
    )

    async with websockets.connect(
        WS_URL,
        ping_interval=20,
        ping_timeout=20,
        max_size=None,
    ) as ws:

        logger.info(
            "Connected to websocket"
        )

        # ------------------------------------------------------------
        # IMPORTANT:
        # Send config FIRST
        # ------------------------------------------------------------

        await send_config(ws)

        recv_task = asyncio.create_task(
            receiver(ws)
        )

        await sender(
            ws,
            pcm_audio,
        )

        logger.info(
            "Waiting for final transcripts..."
        )

        await asyncio.sleep(8)

        await ws.close()

        recv_task.cancel()
