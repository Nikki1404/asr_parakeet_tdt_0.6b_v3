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
    def load(self) -> float: ...

    @abstractmethod
    def new_session(self) -> ASRSession: ...


#app/asr_engines/parakeet_asr.py-

"""
Streaming ASR engine — nvidia/parakeet-tdt-0.6b-v3
All config values come from app.config.Config; nothing is hardcoded here.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from app.asr_engines.base import ASREngine, EngineCaps
from app.config import Config

log = logging.getLogger("parakeet_engine")


def _safe_text(h: Any) -> str:
    if h is None:
        return ""
    if isinstance(h, str):
        return h
    if isinstance(h, (list, tuple)) and h:
        return _safe_text(h[0])
    if hasattr(h, "text"):
        try:
            return h.text or ""
        except Exception:
            return ""
    try:
        return str(h)
    except Exception:
        return ""


@dataclass
class _Timings:
    preproc: float = 0.0
    infer:   float = 0.0
    flush:   float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class ParakeetEngine(ASREngine):

    caps = EngineCaps(streaming=True, partials=True, ttft_meaningful=True)

    def __init__(self, cfg: Config):
        self.cfg   = cfg
        self.model = None

        self.shift_frames:     int   = 0
        self.pre_cache_frames: int   = 0
        self.hop_samples:      int   = 0
        self.drop_extra:       int   = 0
        self._stride_sec:      float = 0.01

        self.end_silence_ms  = cfg.end_silence_ms
        self.min_utt_ms      = cfg.min_utt_ms
        self.finalize_pad_ms = cfg.finalize_pad_ms

    # ── device helpers ────────────────────────────────────────

    def _to(self, t: torch.Tensor) -> torch.Tensor:
        return t.cuda(non_blocking=True) if self.cfg.device == "cuda" else t.cpu()

    def _cache_to(self, cache: tuple) -> tuple:
        return tuple(
            self._to(c) if isinstance(c, torch.Tensor) else c
            for c in cache
        )

    # ── load ──────────────────────────────────────────────────

    def load(self) -> float:
        import nemo.collections.asr as nemo_asr

        t0 = time.time()
        log.info("Loading model: %s  device=%s", self.cfg.model_id, self.cfg.device)

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            self.cfg.model_id, map_location="cpu"
        )
        log.info("Model downloaded / restored from cache")

        self.model = (
            self.model.cuda() if self.cfg.device == "cuda" else self.model.cpu()
        )
        log.info("Model moved to device: %s", self.cfg.device)

        self._set_att_context()
        self._configure_decoding()

        self.model.eval()
        try:
            self.model.preprocessor.featurizer.dither = 0.0
            log.debug("Dither disabled")
        except Exception:
            log.debug("Could not disable dither (non-fatal)")

        self._extract_streaming_cfg()
        self._warmup()

        elapsed = time.time() - t0
        log.info("Engine ready in %.2fs", elapsed)
        return elapsed

    def _set_att_context(self):
        size = [70, self.cfg.context_right]
        for arg in (size, tuple(size)):
            try:
                self.model.encoder.set_default_att_context_size(arg)
                log.info(
                    "Attention context set: left=70  right=%d",
                    self.cfg.context_right,
                )
                return
            except Exception:
                continue
        log.warning("Could not set attention context size — using model default")

    def _configure_decoding(self):
        ms = self.cfg.max_symbols
        strategies = [
            {"strategy": "greedy", "greedy": {"max_symbols": ms, "loop_labels": True,  "use_cuda_graph_decoder": False}},
            {"strategy": "greedy", "greedy": {"max_symbols": ms, "loop_labels": False, "use_cuda_graph_decoder": False}},
            {"strategy": "greedy", "greedy": {"max_symbols": ms}},
        ]
        for cfg_dict in strategies:
            try:
                self.model.change_decoding_strategy(
                    decoding_cfg=OmegaConf.create(cfg_dict)
                )
                log.info("Decoding strategy configured: %s", cfg_dict)
                return
            except Exception:
                continue
        log.warning("Could not set any custom decoding strategy — using model default")

    def _extract_streaming_cfg(self):
        try:
            scfg = self.model.encoder.streaming_cfg
            self.shift_frames = (
                scfg.shift_size[1]
                if isinstance(scfg.shift_size, (list, tuple))
                else scfg.shift_size
            )
            pre = scfg.pre_encode_cache_size
            self.pre_cache_frames = (
                pre[1] if isinstance(pre, (list, tuple)) else pre
            )
            self.drop_extra = int(getattr(scfg, "drop_extra_pre_encoded", 0))
            log.info(
                "Streaming cfg from model: shift=%d  pre_cache=%d  drop_extra=%d",
                self.shift_frames, self.pre_cache_frames, self.drop_extra,
            )
        except AttributeError:
            self.shift_frames     = self.cfg.default_shift_frames
            self.pre_cache_frames = self.cfg.default_pre_cache_frames
            self.drop_extra       = 0
            log.warning(
                "streaming_cfg not found on encoder — using config defaults: "
                "shift=%d  pre_cache=%d",
                self.shift_frames, self.pre_cache_frames,
            )

        try:
            self._stride_sec = float(
                self.model.cfg.preprocessor.get("window_stride", 0.01)
            )
        except Exception:
            self._stride_sec = 0.01
            log.warning("Could not read window_stride — defaulting to 0.01s")

        self.hop_samples = int(self._stride_sec * self.cfg.sample_rate)
        log.info(
            "Frame stride=%.3fs  hop_samples=%d",
            self._stride_sec, self.hop_samples,
        )

    @torch.inference_mode()
    def _warmup(self):
        log.info("Running warmup pass...")
        try:
            sess    = self.new_session()
            silence = np.zeros(self.cfg.sample_rate, dtype=np.float32)
            pcm16   = (silence * 32767).astype(np.int16).tobytes()
            sess.accept_pcm16(pcm16)
            sess.finalize(pad_ms=self.cfg.finalize_pad_ms)
            log.info("Warmup complete")
        except Exception as e:
            log.warning("Warmup failed (non-fatal): %s", e)

    # ── session factory ───────────────────────────────────────

    def new_session(self) -> "ParakeetSession":
        return ParakeetSession(self)

    # ── streaming step ────────────────────────────────────────

    @torch.inference_mode()
    def stream_transcribe(
        self,
        audio_f32:   np.ndarray,
        cache:       tuple,
        prev_hyp:    Any,
        prev_pred:   Any,
        emitted:     int,
        force_flush: bool = False,
    ):
        timings = _Timings()

        t0   = time.perf_counter()
        sig  = self._to(torch.from_numpy(audio_f32).unsqueeze(0))
        slen = torch.tensor([len(audio_f32)], device=sig.device)
        mel, _ = self.model.preprocessor(input_signal=sig, length=slen)
        timings.preproc = time.perf_counter() - t0

        available = int(mel.shape[-1]) - 1
        if available <= 0:
            return None, cache, prev_hyp, prev_pred, emitted, timings

        if (available - emitted) < self.shift_frames and not force_flush:
            return None, cache, prev_hyp, prev_pred, emitted, timings

        if emitted == 0:
            start, end, drop = 0, min(self.shift_frames, available), 0
        else:
            start = max(0, emitted - self.pre_cache_frames)
            end   = min(emitted + self.shift_frames, available)
            drop  = self.drop_extra

        chunk     = mel[:, :, start:end]
        chunk_len = torch.tensor([chunk.shape[-1]], device=chunk.device)
        cache     = self._cache_to(cache)

        t1 = time.perf_counter()
        try:
            prev_pred, texts, c0, c1, c2, prev_hyp = self.model.conformer_stream_step(
                processed_signal        = chunk,
                processed_signal_length = chunk_len,
                cache_last_channel      = cache[0],
                cache_last_time         = cache[1],
                cache_last_channel_len  = cache[2],
                keep_all_outputs        = False,
                previous_hypotheses     = prev_hyp,
                previous_pred_out       = prev_pred,
                drop_extra_pre_encoded  = drop,
                return_transcription    = True,
            )
            new_cache = (c0, c1, c2)
        except Exception as e:
            log.error("conformer_stream_step failed: %s", e)
            return None, cache, prev_hyp, prev_pred, emitted, timings

        timings.infer = time.perf_counter() - t1
        emitted = min(emitted + self.shift_frames, available)
        text    = _safe_text(texts).strip() if texts is not None else ""

        log.debug(
            "stream_step  flush=%s  preproc=%.3fs  infer=%.3fs  text=%r",
            force_flush, timings.preproc, timings.infer, text,
        )

        return text, new_cache, prev_hyp, prev_pred, emitted, timings


# ─────────────────────────────────────────────────────────────────────────────
# Per-connection session
# ─────────────────────────────────────────────────────────────────────────────

class ParakeetSession:

    def __init__(self, engine: ParakeetEngine):
        self.engine       = engine
        self._max_samples = int(
            engine.cfg.sample_rate * engine.cfg.max_utt_ms / 1000
        )
        self.audio        = np.array([], dtype=np.float32)
        self.cache        = None
        self.prev_hyp     = None
        self.prev_pred    = None
        self.emitted      = 0
        self.current_text = ""
        self.last_final   = ""
        self._trimmed     = False

        self._init_cache()

    def _init_cache(self):
        raw        = self.engine.model.encoder.get_initial_cache_state(batch_size=1)
        self.cache = self.engine._cache_to((raw[0], raw[1], raw[2]))

    def reset_stream_state(self):
        self._init_cache()
        self.prev_hyp     = self.prev_pred = None
        self.emitted      = 0
        self.current_text = ""
        self.audio        = np.array([], dtype=np.float32)
        self._trimmed     = False

    def accept_pcm16(self, pcm16: bytes) -> None:
        x          = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio = np.concatenate([self.audio, x])
        if len(self.audio) > self._max_samples:
            self.audio    = self.audio[-self._max_samples:]
            self._trimmed = True
            log.debug("Audio buffer trimmed to max %d samples", self._max_samples)

    def _is_new_text(self, t: Optional[str]) -> bool:
        if not t:
            return False
        n, o = t.strip(), self.current_text.strip()
        if not n or n == o:  return False
        if n.startswith(o):  return True
        if o.startswith(n):  return False
        return True

    def step_if_ready(self) -> Optional[str]:
        if self._trimmed and self.emitted > 0:
            log.debug("Buffer trimmed — resetting encoder cache")
            self._init_cache()
            self.prev_hyp = self.prev_pred = None
            self.emitted  = 0
            self._trimmed = False

        text, self.cache, self.prev_hyp, self.prev_pred, self.emitted, _ = (
            self.engine.stream_transcribe(
                self.audio, self.cache, self.prev_hyp,
                self.prev_pred, self.emitted, force_flush=False,
            )
        )
        if not self._is_new_text(text):
            return None
        self.current_text = text.strip()
        return self.current_text

    def finalize(self, pad_ms: int) -> str:
        log.debug("Finalizing utterance with %d ms pad", pad_ms)
        pad        = np.zeros(
            int(self.engine.cfg.sample_rate * pad_ms / 1000), dtype=np.float32
        )
        self.audio = np.concatenate([self.audio, pad])

        text, self.cache, self.prev_hyp, self.prev_pred, self.emitted, _ = (
            self.engine.stream_transcribe(
                self.audio, self.cache, self.prev_hyp,
                self.prev_pred, self.emitted, force_flush=True,
            )
        )
        if text:
            self.current_text = text.strip()

        final = self.current_text.strip()
        self.last_final = (
            (self.last_final + " " + final).strip() if final else self.last_final
        )
        self.reset_stream_state()
        return final


#app/config.py-

from dataclasses import dataclass
import os

# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for the entire application.
# Host / port / ping settings live ONLY in the Dockerfile CMD.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID            = "nvidia/parakeet-tdt-0.6b-v3"
SUPPORTED_LANGUAGES = ["en-US", "es-US"]
WS_PATH             = "/asr/ml/ws"


@dataclass(frozen=True)
class Config:

    # ── model ────────────────────────────────────────────────
    model_id: str = MODEL_ID

    # ── hardware ─────────────────────────────────────────────
    device:      str = os.getenv("DEVICE",      "cuda")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # ── streaming / encoder ──────────────────────────────────
    context_right:            int = int(os.getenv("CONTEXT_RIGHT",            "1"))
    default_shift_frames:     int = int(os.getenv("DEFAULT_SHIFT_FRAMES",     "8"))
    default_pre_cache_frames: int = int(os.getenv("DEFAULT_PRE_CACHE_FRAMES", "70"))

    # ── VAD ──────────────────────────────────────────────────
    vad_frame_ms:      int   = int(os.getenv("VAD_FRAME_MS",        "20"))
    vad_start_margin:  float = float(os.getenv("VAD_START_MARGIN",  "2.5"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.003"))
    pre_speech_ms:     int   = int(os.getenv("PRE_SPEECH_MS",       "300"))

    # ── endpointing ──────────────────────────────────────────
    end_silence_ms:  int = int(os.getenv("END_SILENCE_MS",  "700"))
    min_utt_ms:      int = int(os.getenv("MIN_UTT_MS",      "200"))
    finalize_pad_ms: int = int(os.getenv("FINALIZE_PAD_MS", "500"))

    # ── buffer ───────────────────────────────────────────────
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))

    # ── decoding ─────────────────────────────────────────────
    max_symbols: int = int(os.getenv("MAX_SYMBOLS", "10"))

    # ── audio saving ─────────────────────────────────────────
    # Set to "" to disable WAV saving
    audio_save_dir: str = os.getenv("AUDIO_SAVE_DIR", "/app/session_audio")

    # ── transcription log ────────────────────────────────────
    # Set to "" to disable file logging
    transcription_log_file: str = os.getenv(
        "TRANSCRIPTION_LOG", "/app/logs/transcriptions.log"
    )

    # ── app logging ──────────────────────────────────────────
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


#app/factory.py-

from app.config import Config
from app.asr_engines.parakeet_asr import ParakeetEngine


def build_engine(cfg: Config) -> ParakeetEngine:
    return ParakeetEngine(cfg)

#app/vad.py-

from collections import deque
import logging

import numpy as np

log = logging.getLogger("vad")


class AdaptiveEnergyVAD:
    """
    Adaptive noise-floor VAD with pre-speech ring buffer.
    Speech onset → flush buffered pre-roll so we never clip leading phonemes.
    """

    def __init__(
        self,
        sample_rate:   int,
        frame_ms:      int,
        start_margin:  float,
        min_noise_rms: float,
        pre_speech_ms: int,
    ):
        self.frame_ms      = frame_ms
        self.min_noise_rms = min_noise_rms
        self.start_margin  = start_margin

        samples_per_frame = int(sample_rate * frame_ms / 1000)
        self.frame_bytes  = samples_per_frame * 2

        pre_frames       = max(1, int(pre_speech_ms / frame_ms))
        self.ring: deque = deque(maxlen=pre_frames)

        self.in_speech = False
        self.noise_rms = min_noise_rms

        log.debug(
            "VAD init: frame_ms=%d  start_margin=%.2f  "
            "min_noise_rms=%.4f  pre_frames=%d",
            frame_ms, start_margin, min_noise_rms, pre_frames,
        )

    def reset(self):
        self.ring.clear()
        self.in_speech = False
        self.noise_rms = self.min_noise_rms
        log.debug("VAD reset")

    def _rms(self, pcm16: bytes) -> float:
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def push_frame(self, frame: bytes):
        """
        Returns (is_speech: bool, pre_roll: bytes | None).
        pre_roll is non-None only on the very first speech frame of an utterance.
        """
        energy = self._rms(frame)

        if not self.in_speech:
            self.noise_rms = max(
                self.min_noise_rms,
                0.95 * self.noise_rms + 0.05 * energy,
            )

        threshold = max(self.min_noise_rms, self.noise_rms) * self.start_margin
        is_speech = energy >= threshold
        self.ring.append(frame)

        pre_roll = None
        if not self.in_speech and is_speech:
            self.in_speech = True
            pre_roll       = b"".join(self.ring)
            log.debug(
                "VAD speech onset | energy=%.4f  threshold=%.4f  "
                "pre_roll=%d bytes",
                energy, threshold, len(pre_roll),
            )

        return is_speech, pre_roll


#app/streaming_session.py-

from __future__ import annotations

import logging
import time
from typing import Optional

from app.vad import AdaptiveEnergyVAD
from app.config import Config

log = logging.getLogger("streaming_session")


class StreamingSession:

    def __init__(self, engine, cfg: Config):
        self.engine = engine
        self.cfg    = cfg

        self.vad = AdaptiveEnergyVAD(
            sample_rate   = cfg.sample_rate,
            frame_ms      = cfg.vad_frame_ms,
            start_margin  = cfg.vad_start_margin,
            min_noise_rms = cfg.vad_min_noise_rms,
            pre_speech_ms = cfg.pre_speech_ms,
        )

        self.asr_session = engine.new_session()

        self.frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
        self._raw_buf    = bytearray()

        self.utt_started:     bool            = False
        self.utt_audio_ms:    int             = 0
        self.silence_ms:      int             = 0
        self.t_start:         Optional[float] = None
        self.t_first_partial: Optional[float] = None
        self._utt_index:      int             = 0

        log.debug(
            "StreamingSession created | frame_bytes=%d  "
            "end_silence_ms=%d  min_utt_ms=%d",
            self.frame_bytes,
            cfg.end_silence_ms,
            cfg.min_utt_ms,
        )

    def process_chunk(self, pcm: bytes) -> list[tuple]:
        events: list[tuple] = []
        self._raw_buf.extend(pcm)

        while len(self._raw_buf) >= self.frame_bytes:
            frame = bytes(self._raw_buf[: self.frame_bytes])
            del self._raw_buf[: self.frame_bytes]

            is_speech, pre_roll = self.vad.push_frame(frame)
            self.silence_ms = (
                0 if is_speech else self.silence_ms + self.cfg.vad_frame_ms
            )

            # ── speech onset ──────────────────────────────────
            if pre_roll and not self.utt_started:
                self._utt_index  += 1
                self.utt_started  = True
                self.utt_audio_ms = 0
                self.t_start      = time.time()
                self.t_first_partial = None
                self.asr_session.accept_pcm16(pre_roll)
                log.debug(
                    "Utterance #%d started | pre_roll=%d bytes",
                    self._utt_index, len(pre_roll),
                )

            if not self.utt_started:
                continue

            self.asr_session.accept_pcm16(frame)
            self.utt_audio_ms += self.cfg.vad_frame_ms

            # ── partials ──────────────────────────────────────
            if self.engine.caps.partials:
                partial = self.asr_session.step_if_ready()
                if partial:
                    if self.t_first_partial is None:
                        self.t_first_partial = time.time()
                    ttfb = int((self.t_first_partial - self.t_start) * 1000)
                    events.append(("partial", partial, ttfb))

            # ── endpointing ───────────────────────────────────
            if (
                not is_speech
                and self.utt_audio_ms >= self.cfg.min_utt_ms
                and self.silence_ms   >= self.cfg.end_silence_ms
            ):
                log.debug(
                    "Utterance #%d endpoint triggered | "
                    "utt_audio=%d ms  silence=%d ms",
                    self._utt_index,
                    self.utt_audio_ms,
                    self.silence_ms,
                )
                final = self.asr_session.finalize(self.cfg.finalize_pad_ms)
                if final:
                    ttfb = (
                        int((self.t_first_partial - self.t_start) * 1000)
                        if self.t_first_partial else None
                    )
                    events.append(("final", final, ttfb))
                self._reset_utt()

        return events

    def _reset_utt(self):
        self.vad.reset()
        self.utt_started     = False
        self.utt_audio_ms    = 0
        self.silence_ms      = 0
        self.t_start         = None
        self.t_first_partial = None
        log.debug("Utterance reset — VAD back to noise-floor tracking")


#app/main.py-

import asyncio
import json
import logging
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import resampy
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.config import Config, SUPPORTED_LANGUAGES, WS_PATH
from app.factory import build_engine
from app.streaming_session import StreamingSession

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

cfg = Config()

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log    = logging.getLogger("asr_server")
tx_log = logging.getLogger("transcription")

# transcription logger: stdout + optional file, no double-propagation to root
tx_log.setLevel(logging.INFO)
tx_log.propagate = False
tx_log.addHandler(logging.StreamHandler(sys.stdout))

if cfg.transcription_log_file:
    try:
        Path(cfg.transcription_log_file).parent.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(cfg.transcription_log_file, encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        tx_log.addHandler(_fh)
        log.info("Transcription log → %s", cfg.transcription_log_file)
    except Exception as e:
        log.warning("Could not open transcription log file: %s", e)

# ─────────────────────────────────────────────────────────────────────────────
# Audio save dir
# ─────────────────────────────────────────────────────────────────────────────

if cfg.audio_save_dir:
    try:
        Path(cfg.audio_save_dir).mkdir(parents=True, exist_ok=True)
        log.info("Session audio save dir → %s", cfg.audio_save_dir)
    except Exception as e:
        log.warning("Could not create audio save dir: %s", e)


def _save_wav(path: Path, pcm16_bytes: bytes, sample_rate: int) -> None:
    try:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16_bytes)
        log.info(
            "Session WAV saved → %s  (%.1f s)",
            path, len(pcm16_bytes) / 2 / sample_rate,
        )
    except Exception as e:
        log.warning("Could not save WAV %s: %s", path, e)


# ─────────────────────────────────────────────────────────────────────────────
# App + engine
# ─────────────────────────────────────────────────────────────────────────────

app    = FastAPI(title="Parakeet Real-Time ASR")
ENGINE = None

_server_start    = datetime.utcnow()
_total_sessions  = 0
_active_sessions = 0


@app.on_event("startup")
async def _startup():
    global ENGINE
    log.info("=" * 60)
    log.info("Parakeet Real-Time ASR server starting")
    log.info(
        "model=%s  device=%s  sample_rate=%d",
        cfg.model_id, cfg.device, cfg.sample_rate,
    )
    log.info("=" * 60)
    ENGINE   = build_engine(cfg)
    load_sec = await asyncio.get_running_loop().run_in_executor(None, ENGINE.load)
    log.info(
        "Engine ready in %.1fs | supported languages: %s",
        load_sec, SUPPORTED_LANGUAGES,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health  GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """200 once engine is loaded, 503 while still warming up."""
    engine_ready = ENGINE is not None and ENGINE.model is not None
    uptime       = (datetime.utcnow() - _server_start).total_seconds()
    body = {
        "status":              "ok" if engine_ready else "loading",
        "model":               cfg.model_id,
        "device":              cfg.device,
        "supported_languages": SUPPORTED_LANGUAGES,
        "sample_rate":         cfg.sample_rate,
        "uptime_seconds":      round(uptime, 1),
        "active_sessions":     _active_sessions,
        "total_sessions":      _total_sessions,
        "audio_save_dir":      cfg.audio_save_dir,
        "transcription_log":   cfg.transcription_log_file,
    }
    log.debug(
        "Health check → status=%s  active=%d",
        body["status"], _active_sessions,
    )
    return JSONResponse(content=body, status_code=200 if engine_ready else 503)


@app.websocket(WS_PATH)
async def ws_asr(ws: WebSocket):
    global _total_sessions, _active_sessions

    await ws.accept()

    # ── session identity ──────────────────────────────────────
    ts         = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    client_str = f"{ws.client.host}_{ws.client.port}" if ws.client else "unknown"
    session_id = f"{ts}_{client_str}"

    _total_sessions  += 1
    _active_sessions += 1

    log.info(
        "[%s] ━━━ SESSION OPEN ━━━  client=%s  active=%d  total=%d",
        session_id, ws.client, _active_sessions, _total_sessions,
    )

    # ── init handshake ────────────────────────────────────────
    log.info("[%s] Waiting for init message...", session_id)
    try:
        raw  = await asyncio.wait_for(ws.receive_text(), timeout=15.0)
        init = json.loads(raw)
    except asyncio.TimeoutError:
        log.warning("[%s] Init timeout (15s) — closing connection", session_id)
        _active_sessions -= 1
        await ws.close(code=4000)
        return
    except Exception as e:
        log.warning(
            "[%s] Bad init message: %s — closing connection", session_id, e
        )
        _active_sessions -= 1
        await ws.close(code=4001)
        return

    candidates = init.get("candidate_languages", SUPPORTED_LANGUAGES)
    client_sr  = int(init.get("sample_rate", cfg.sample_rate))

    log.info(
        "[%s] Init received | candidate_languages=%s  "
        "client_sr=%d  server_sr=%d",
        session_id, candidates, client_sr, cfg.sample_rate,
    )

    if client_sr != cfg.sample_rate:
        log.info(
            "[%s] Resampling enabled: %d Hz → %d Hz",
            session_id, client_sr, cfg.sample_rate,
        )

    # send session_ready ack
    await ws.send_text(json.dumps({
        "type":                "session_ready",
        "session_id":          session_id,
        "supported_languages": SUPPORTED_LANGUAGES,
        "model":               cfg.model_id,
    }))
    log.info("[%s] session_ready sent to client", session_id)

    # ── resampler ─────────────────────────────────────────────
    def maybe_resample(pcm: bytes) -> bytes:
        if not pcm or client_sr == cfg.sample_rate:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        y = resampy.resample(x, client_sr, cfg.sample_rate)
        return (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    # ── per-session state ─────────────────────────────────────
    session       = StreamingSession(ENGINE, cfg)
    loop          = asyncio.get_running_loop()
    audio_buf     = bytearray()
    partial_count = 0
    final_count   = 0
    chunk_count   = 0
    session_start = datetime.utcnow()

    log.info(
        "[%s] Streaming started — ready to receive audio chunks",
        session_id,
    )

    # ── main receive loop ─────────────────────────────────────
    try:
        while True:
            msg = await ws.receive()

            # ── disconnect ────────────────────────────────────
            if msg["type"] == "websocket.disconnect":
                log.info("[%s] WebSocket disconnect received", session_id)
                break

            # ── binary audio chunk ────────────────────────────
            data: Optional[bytes] = msg.get("bytes")
            if data is not None:
                original_len = len(data)
                data         = maybe_resample(data)
                audio_buf.extend(data)
                chunk_count += 1

                log.debug(
                    "[%s] Chunk #%04d | raw=%d B  resampled=%d B  "
                    "%.1f ms  total_buffered=%.1f s",
                    session_id, chunk_count,
                    original_len, len(data),
                    len(data) / 2 / cfg.sample_rate * 1000,
                    len(audio_buf) / 2 / cfg.sample_rate,
                )

                events = await loop.run_in_executor(
                    None, session.process_chunk, data
                )

                for ev_type, text, ttfb in events:
                    if ev_type == "partial":
                        partial_count += 1
                        tx_log.info(
                            "[%s] PARTIAL #%04d | %s",
                            session_id, partial_count, text,
                        )
                    elif ev_type == "final":
                        final_count += 1
                        ttfb_str = f"{ttfb}ms" if ttfb is not None else "n/a"
                        tx_log.info(
                            "[%s] FINAL   #%04d | ttfb=%-8s | %s",
                            session_id, final_count, ttfb_str, text,
                        )

                    await ws.send_text(json.dumps({
                        "type":    ev_type,
                        "text":    text,
                        "t_start": ttfb,
                    }))
                continue

            # ── text control message ──────────────────────────
            ctrl = msg.get("text")
            if ctrl:
                try:
                    ctrl_obj  = json.loads(ctrl)
                    ctrl_type = ctrl_obj.get("type", "unknown")
                    log.info(
                        "[%s] Control message: type=%s",
                        session_id, ctrl_type,
                    )
                    if ctrl_type == "end_session":
                        log.info(
                            "[%s] end_session received — closing cleanly",
                            session_id,
                        )
                        break
                except Exception as e:
                    log.warning(
                        "[%s] Could not parse control message: %s",
                        session_id, e,
                    )

    except WebSocketDisconnect:
        log.info("[%s] WebSocket disconnected (client closed)", session_id)
    except Exception:
        log.exception("[%s] Unexpected error in session loop", session_id)

    # ── session teardown ──────────────────────────────────────
    finally:
        _active_sessions -= 1
        duration = (datetime.utcnow() - session_start).total_seconds()
        audio_s  = len(audio_buf) / 2 / cfg.sample_rate

        log.info(
            "[%s] ━━━ SESSION CLOSED ━━━  "
            "duration=%.1fs  audio=%.1fs  chunks=%d  "
            "partials=%d  finals=%d  active=%d",
            session_id,
            duration, audio_s, chunk_count,
            partial_count, final_count,
            _active_sessions,
        )

        # save session audio
        if cfg.audio_save_dir and audio_buf:
            wav_path = Path(cfg.audio_save_dir) / f"{session_id}.wav"
            log.info("[%s] Saving session audio → %s", session_id, wav_path)
            _save_wav(wav_path, bytes(audio_buf), cfg.sample_rate)
        elif cfg.audio_save_dir and not audio_buf:
            log.info("[%s] No audio received — WAV not saved", session_id)
        else:
            log.debug(
                "[%s] Audio saving disabled (AUDIO_SAVE_DIR not set)",
                session_id,
            )


#client.py-
"""
Parakeet ASR — full-featured test client

    python client.py                           # mic, default local server
    python client.py --file audio.wav          # file mode
    python client.py --url wss://host/path     # custom server URL
    python client.py --health                  # health check and exit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Optional
from urllib.parse import urlparse, urlunparse
import urllib.request
import urllib.error

import numpy as np
import websockets

DEFAULT_WS_URL    = "ws://localhost:8001/asr/stream"
DEFAULT_LANGUAGES = ["en-US", "es-US"]
SERVER_SR         = 16_000
CHUNK_MS          = 100


# ── URL helpers ───────────────────────────────────────────────

def _health_url(ws_url: str) -> str:
    parsed = urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urlunparse((scheme, parsed.netloc, "/health", "", "", ""))


def _validate_ws_url(url: str) -> str:
    if urlparse(url).scheme not in ("ws", "wss"):
        print(f"[error] URL must start with ws:// or wss:// — got: {url}")
        sys.exit(1)
    return url


# ── audio helpers ─────────────────────────────────────────────

def _to_pcm16_16k(audio: np.ndarray, src_sr: int) -> bytes:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / 32768.0
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if src_sr != SERVER_SR:
        try:
            import resampy
            audio = resampy.resample(audio, src_sr, SERVER_SR)
        except ImportError:
            from math import gcd
            from scipy.signal import resample_poly
            g     = gcd(SERVER_SR, src_sr)
            audio = resample_poly(audio, SERVER_SR // g, src_sr // g).astype(np.float32)
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _load_audio_file(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        return audio, sr
    except Exception:
        pass
    try:
        from pydub import AudioSegment
        seg  = AudioSegment.from_file(path)
        raw  = np.array(seg.get_array_of_samples(), dtype=np.float32)
        raw /= 2 ** (seg.sample_width * 8 - 1)
        if seg.channels > 1:
            raw = raw.reshape(-1, seg.channels).mean(axis=1)
        return raw, seg.frame_rate
    except Exception as e:
        print(f"[error] cannot load {path}: {e}")
        sys.exit(1)


# ── display ───────────────────────────────────────────────────

class _Display:
    def __init__(self):
        self._partial_active = False

    def partial(self, text: str):
        sys.stdout.write(f"\r\033[K  [partial] {text}")
        sys.stdout.flush()
        self._partial_active = True

    def final(self, text: str, ttfb: Optional[int]):
        if self._partial_active:
            sys.stdout.write("\r\033[K")
        ttfb_str = f"  (ttfb {ttfb} ms)" if ttfb is not None else ""
        print(f"  [final]   {text}{ttfb_str}")
        self._partial_active = False

    def info(self, msg: str):
        if self._partial_active:
            sys.stdout.write("\r\033[K")
            self._partial_active = False
        print(msg)

    def separator(self):
        self.info("─" * 60)


# ── health check ──────────────────────────────────────────────

def check_health(ws_url: str) -> None:
    url = _health_url(ws_url)
    print(f"GET {url}")
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            body = json.loads(r.read())
            code = r.status
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        code = e.code
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)

    pad = max(len(k) for k in body) + 1
    print(f"\nHTTP {code}")
    print("─" * 50)
    for k, v in body.items():
        print(f"  {k:<{pad}}: {v}")
    print()
    sys.exit(0 if body.get("status") == "ok" else 1)


# ── WebSocket session ─────────────────────────────────────────

async def _run_session(
    ws_url:    str,
    languages: list[str],
    audio_gen: AsyncGenerator[bytes, None],
) -> None:
    display = _Display()
    try:
        async with websockets.connect(
            ws_url,
            ping_interval = 20,
            ping_timeout  = 120,
            max_size      = None,
        ) as ws:

            await ws.send(json.dumps({
                "candidate_languages": languages,
                "sample_rate":         SERVER_SR,
            }))

            ack = json.loads(await ws.recv())
            if ack.get("type") != "session_ready":
                display.info(f"[error] unexpected ack: {ack}")
                return

            display.separator()
            display.info(f"  model      : {ack.get('model', '?')}")
            display.info(f"  session_id : {ack.get('session_id', '?')}")
            display.info(f"  languages  : {ack.get('supported_languages', languages)}")
            display.separator()

            async def _recv():
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        msg = json.loads(raw)
                        t   = msg.get("type")
                        if t == "partial":
                            display.partial(msg.get("text", ""))
                        elif t == "final":
                            display.final(msg.get("text", ""), msg.get("t_start"))
                except websockets.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(_recv())

            try:
                async for chunk in audio_gen:
                    await ws.send(chunk)
            except (KeyboardInterrupt, asyncio.CancelledError):
                display.info("\n[client] interrupted — flushing final transcript...")

            try:
                await ws.send(json.dumps({"type": "end_session"}))
                await asyncio.sleep(1.5)
            except Exception:
                pass

            recv_task.cancel()
            display.separator()
            display.info("[client] session closed")

    except (websockets.InvalidURI, OSError, ConnectionRefusedError) as e:
        print(f"[error] cannot connect to {ws_url}: {e}")
        sys.exit(1)


# ── audio generators ──────────────────────────────────────────

async def _mic_generator(chunk_ms: int) -> AsyncGenerator[bytes, None]:
    import sounddevice as sd

    chunk_samples = int(SERVER_SR * chunk_ms / 1000)
    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _cb(indata, frames, t, status):
        if status:
            print(f"\n[mic] {status}", file=sys.stderr)
        mono  = indata[:, 0] if indata.ndim == 2 else indata
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(queue.put_nowait, pcm16)

    print(f"[mic] recording at {SERVER_SR} Hz, {chunk_ms} ms chunks")
    print("[mic] speak now — press Ctrl+C to stop\n")

    with sd.InputStream(
        samplerate = SERVER_SR,
        channels   = 1,
        dtype      = "float32",
        blocksize  = chunk_samples,
        callback   = _cb,
    ):
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n[mic] stopped")


async def _file_generator(path: str, chunk_ms: int) -> AsyncGenerator[bytes, None]:
    audio, src_sr = _load_audio_file(path)
    pcm16         = _to_pcm16_16k(audio, src_sr)

    chunk_bytes = int(SERVER_SR * chunk_ms / 1000) * 2
    duration_s  = len(pcm16) / 2 / SERVER_SR
    n_chunks    = (len(pcm16) + chunk_bytes - 1) // chunk_bytes

    print(f"[file] {path}")
    print(f"[file] duration={duration_s:.1f}s  chunks={n_chunks} × {chunk_ms} ms\n")

    t_start = time.perf_counter()
    for i, offset in enumerate(range(0, len(pcm16), chunk_bytes)):
        yield pcm16[offset: offset + chunk_bytes]
        target  = (i + 1) * chunk_ms / 1000
        elapsed = time.perf_counter() - t_start
        gap     = target - elapsed
        if gap > 0:
            await asyncio.sleep(gap)

    print("\n[file] stream complete — waiting for final transcript...")


# ── CLI ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parakeet ASR test client")
    p.add_argument("--url",       default=DEFAULT_WS_URL, metavar="URL",
                   help="Full WebSocket URL (default: %(default)s)")
    p.add_argument("--file",      default=None, metavar="PATH",
                   help="Audio file to transcribe (omit for microphone mode)")
    p.add_argument("--languages", default=DEFAULT_LANGUAGES, nargs="+", metavar="LANG",
                   help="Candidate BCP-47 language tags (default: en-US es-US)")
    p.add_argument("--chunk-ms",  default=CHUNK_MS, type=int, metavar="MS",
                   help="Audio chunk size in ms (default: 100)")
    p.add_argument("--health",    action="store_true",
                   help="Check server /health and exit")
    return p.parse_args()


async def _main() -> None:
    args   = _parse_args()
    ws_url = _validate_ws_url(args.url)

    if args.health:
        check_health(ws_url)
        return

    print("=" * 60)
    print("  Parakeet Real-Time ASR — test client")
    print("=" * 60)
    print(f"  url        : {ws_url}")
    print(f"  languages  : {args.languages}")
    print(f"  chunk      : {args.chunk_ms} ms")
    print(f"  mode       : {'file → ' + args.file if args.file else 'microphone'}")
    print("=" * 60 + "\n")

    if args.file:
        if not Path(args.file).exists():
            print(f"[error] file not found: {args.file}")
            sys.exit(1)
        gen = _file_generator(args.file, args.chunk_ms)
    else:
        gen = _mic_generator(args.chunk_ms)

    try:
        await _run_session(ws_url, args.languages, gen)
    except KeyboardInterrupt:
        print("\n[client] interrupted")


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


#parakeet_client.py-
"""
Parakeet ASR — integration sample (share this with API consumers)

Edit the CONFIG block only, then run:

    python parakeet_client.py --file path/to/audio.wav
    python parakeet_client.py --mic
"""

import asyncio
import json
import logging
import time

import librosa
import numpy as np
import soundfile as sf
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("parakeet_client")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — only edit this block
# ─────────────────────────────────────────────────────────────────────────────

WS_URL              = "wss://cx-asr-exl.service.com/asr/ml/ws"
CANDIDATE_LANGUAGES = ["en-US", "es-US"]   # parakeet auto-detects, informational only
TARGET_SR           = 16000                # must match server SAMPLE_RATE
CHUNK_MS            = 100                  # ms of audio per send iteration
WS_PING_INTERVAL    = 20                   # must match Dockerfile --ws-ping-interval
WS_PING_TIMEOUT     = 120                  # must match Dockerfile --ws-ping-timeout

# ─────────────────────────────────────────────────────────────────────────────
# derived — do not edit
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SAMPLES = TARGET_SR * CHUNK_MS // 1000
CHUNK_BYTES   = CHUNK_SAMPLES * 2


# ─────────────────────────────────────────────────────────────────────────────
# audio loader
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(filepath: str) -> bytes:
    audio, sr = sf.read(filepath, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# receiver
# ─────────────────────────────────────────────────────────────────────────────

async def receiver(ws):
    async for msg in ws:
        if not isinstance(msg, str):
            continue
        try:
            obj = json.loads(msg)
        except Exception:
            continue

        typ  = obj.get("type")
        txt  = obj.get("text", "")
        ttfb = obj.get("t_start")
        ts   = time.strftime("%H:%M:%S")

        if typ == "partial":
            print(
                f"\r⏳ [{ts}] ({ttfb} ms) {txt:<120}",
                end="", flush=True,
            )
        elif typ == "final":
            print("\r" + " " * 160 + "\r", end="")
            ttfb_str = f"{ttfb} ms" if ttfb is not None else "n/a"
            print(f"[{ts}] FINAL (ttfb {ttfb_str})")
            print(f"✅ {txt}")
            print("─" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# senders
# ─────────────────────────────────────────────────────────────────────────────

async def sender_file(ws, pcm_audio: bytes):
    offset  = 0
    t_start = time.perf_counter()
    i       = 0

    while offset < len(pcm_audio):
        chunk   = pcm_audio[offset: offset + CHUNK_BYTES]
        offset += CHUNK_BYTES
        if len(chunk) < CHUNK_BYTES:
            chunk += bytes(CHUNK_BYTES - len(chunk))
        await ws.send(chunk)

        i      += 1
        target  = i * CHUNK_MS / 1000
        elapsed = time.perf_counter() - t_start
        gap     = target - elapsed
        if gap > 0:
            await asyncio.sleep(gap)

    logger.info("File stream complete — waiting for final transcript...")
    await asyncio.sleep(4)


async def sender_mic(ws):
    import sounddevice as sd

    queue = asyncio.Queue()
    loop  = asyncio.get_running_loop()

    def _cb(indata, frames, t, status):
        if status:
            logger.warning("mic: %s", status)
        mono  = indata[:, 0] if indata.ndim == 2 else indata.flatten()
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(queue.put_nowait, pcm16)

    logger.info("Mic open at %d Hz — speak now, Ctrl+C to stop", TARGET_SR)

    with sd.InputStream(
        samplerate = TARGET_SR,
        channels   = 1,
        dtype      = "float32",
        blocksize  = CHUNK_SAMPLES,
        callback   = _cb,
    ):
        try:
            while True:
                chunk = await queue.get()
                await ws.send(chunk)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Mic stopped — waiting for final transcript...")
            await asyncio.sleep(4)


# ─────────────────────────────────────────────────────────────────────────────
# connect + run
# ─────────────────────────────────────────────────────────────────────────────

async def _connect_and_run(sender_coro):
    async with websockets.connect(
        WS_URL,
        ping_interval = WS_PING_INTERVAL,
        ping_timeout  = WS_PING_TIMEOUT,
        max_size      = None,
    ) as ws:
        logger.info("Connected → %s", WS_URL)

        await ws.send(json.dumps({
            "candidate_languages": CANDIDATE_LANGUAGES,
            "sample_rate":         TARGET_SR,
        }))

        ack = json.loads(await ws.recv())
        assert ack.get("type") == "session_ready", f"Unexpected ack: {ack}"
        logger.info(
            "Session ready | model=%s  id=%s",
            ack.get("model", "?"),
            ack.get("session_id", "?"),
        )

        recv_task = asyncio.create_task(receiver(ws))

        try:
            await sender_coro(ws)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                await ws.send(json.dumps({"type": "end_session"}))
            except Exception:
                pass
            recv_task.cancel()
            logger.info("Session closed")


async def stream_file(audio_file: str):
    pcm_audio = load_audio(audio_file)
    logger.info("Loaded audio | %.2f s", len(pcm_audio) / 2 / TARGET_SR)
    await _connect_and_run(lambda ws: sender_file(ws, pcm_audio))


async def stream_mic():
    await _connect_and_run(sender_mic)


# ─────────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Parakeet ASR integration sample")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", metavar="PATH", help="Audio file to transcribe")
    group.add_argument("--mic",  action="store_true", help="Stream from microphone")
    args = p.parse_args()

    try:
        if args.file:
            asyncio.run(stream_file(args.file))
        else:
            asyncio.run(stream_mic())
    except KeyboardInterrupt:
        pass


#Dockerfile-
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ── proxy (remove if not needed) ──────────────────────────────
ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

# ── system packages ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3.10-distutils \
        build-essential gcc g++ \
        ffmpeg libsndfile1 \
        libportaudio2 portaudio19-dev \
        git wget curl \
        libcudnn8 libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3    1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ── PyTorch CUDA 12.1 ──────────────────────────────────────────
RUN pip install --no-cache-dir \
        torch==2.4.0+cu121 \
        torchvision==0.19.0+cu121 \
        torchaudio==2.4.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

# ── app deps ───────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── source ─────────────────────────────────────────────────────
COPY app ./app

# ── runtime environment ────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV AUDIO_SAVE_DIR=/app/session_audio
ENV TRANSCRIPTION_LOG=/app/logs/transcriptions.log
ENV LOG_LEVEL=INFO

# ── bake model into image ──────────────────────────────────────
RUN python -c "
import nemo.collections.asr as nemo_asr
import logging
logging.basicConfig(level=logging.INFO)
logging.info('Downloading nvidia/parakeet-tdt-0.6b-v3 ...')
nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')
logging.info('Model cached OK')
"

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--ws-ping-interval", "20", "--ws-ping-timeout", "120"]



#client.py-

"""
Parakeet ASR — full-featured test client

    python client.py                           # mic, default local server
    python client.py --file audio.wav          # file mode
    python client.py --url wss://host/path     # custom server URL
    python client.py --health                  # health check and exit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Optional
from urllib.parse import urlparse, urlunparse
import urllib.request
import urllib.error

import numpy as np
import websockets

DEFAULT_WS_URL    = "ws://localhost:8001/asr/stream"
DEFAULT_LANGUAGES = ["en-US", "es-US"]
SERVER_SR         = 16_000
CHUNK_MS          = 100


# ── URL helpers ───────────────────────────────────────────────

def _health_url(ws_url: str) -> str:
    parsed = urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urlunparse((scheme, parsed.netloc, "/health", "", "", ""))


def _validate_ws_url(url: str) -> str:
    if urlparse(url).scheme not in ("ws", "wss"):
        print(f"[error] URL must start with ws:// or wss:// — got: {url}")
        sys.exit(1)
    return url


# ── audio helpers ─────────────────────────────────────────────

def _to_pcm16_16k(audio: np.ndarray, src_sr: int) -> bytes:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / 32768.0
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if src_sr != SERVER_SR:
        try:
            import resampy
            audio = resampy.resample(audio, src_sr, SERVER_SR)
        except ImportError:
            from math import gcd
            from scipy.signal import resample_poly
            g     = gcd(SERVER_SR, src_sr)
            audio = resample_poly(audio, SERVER_SR // g, src_sr // g).astype(np.float32)
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _load_audio_file(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        return audio, sr
    except Exception:
        pass
    try:
        from pydub import AudioSegment
        seg  = AudioSegment.from_file(path)
        raw  = np.array(seg.get_array_of_samples(), dtype=np.float32)
        raw /= 2 ** (seg.sample_width * 8 - 1)
        if seg.channels > 1:
            raw = raw.reshape(-1, seg.channels).mean(axis=1)
        return raw, seg.frame_rate
    except Exception as e:
        print(f"[error] cannot load {path}: {e}")
        sys.exit(1)


# ── display ───────────────────────────────────────────────────

class _Display:
    def __init__(self):
        self._partial_active = False

    def partial(self, text: str):
        sys.stdout.write(f"\r\033[K  [partial] {text}")
        sys.stdout.flush()
        self._partial_active = True

    def final(self, text: str, ttfb: Optional[int]):
        if self._partial_active:
            sys.stdout.write("\r\033[K")
        ttfb_str = f"  (ttfb {ttfb} ms)" if ttfb is not None else ""
        print(f"  [final]   {text}{ttfb_str}")
        self._partial_active = False

    def info(self, msg: str):
        if self._partial_active:
            sys.stdout.write("\r\033[K")
            self._partial_active = False
        print(msg)

    def separator(self):
        self.info("─" * 60)


# ── health check ──────────────────────────────────────────────

def check_health(ws_url: str) -> None:
    url = _health_url(ws_url)
    print(f"GET {url}")
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            body = json.loads(r.read())
            code = r.status
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        code = e.code
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)

    pad = max(len(k) for k in body) + 1
    print(f"\nHTTP {code}")
    print("─" * 50)
    for k, v in body.items():
        print(f"  {k:<{pad}}: {v}")
    print()
    sys.exit(0 if body.get("status") == "ok" else 1)


# ── WebSocket session ─────────────────────────────────────────

async def _run_session(
    ws_url:    str,
    languages: list[str],
    audio_gen: AsyncGenerator[bytes, None],
) -> None:
    display = _Display()
    try:
        async with websockets.connect(
            ws_url,
            ping_interval = 20,
            ping_timeout  = 120,
            max_size      = None,
        ) as ws:

            await ws.send(json.dumps({
                "candidate_languages": languages,
                "sample_rate":         SERVER_SR,
            }))

            ack = json.loads(await ws.recv())
            if ack.get("type") != "session_ready":
                display.info(f"[error] unexpected ack: {ack}")
                return

            display.separator()
            display.info(f"  model      : {ack.get('model', '?')}")
            display.info(f"  session_id : {ack.get('session_id', '?')}")
            display.info(f"  languages  : {ack.get('supported_languages', languages)}")
            display.separator()

            async def _recv():
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        msg = json.loads(raw)
                        t   = msg.get("type")
                        if t == "partial":
                            display.partial(msg.get("text", ""))
                        elif t == "final":
                            display.final(msg.get("text", ""), msg.get("t_start"))
                except websockets.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(_recv())

            try:
                async for chunk in audio_gen:
                    await ws.send(chunk)
            except (KeyboardInterrupt, asyncio.CancelledError):
                display.info("\n[client] interrupted — flushing final transcript...")

            try:
                await ws.send(json.dumps({"type": "end_session"}))
                await asyncio.sleep(1.5)
            except Exception:
                pass

            recv_task.cancel()
            display.separator()
            display.info("[client] session closed")

    except (websockets.InvalidURI, OSError, ConnectionRefusedError) as e:
        print(f"[error] cannot connect to {ws_url}: {e}")
        sys.exit(1)


# ── audio generators ──────────────────────────────────────────

async def _mic_generator(chunk_ms: int) -> AsyncGenerator[bytes, None]:
    import sounddevice as sd

    chunk_samples = int(SERVER_SR * chunk_ms / 1000)
    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _cb(indata, frames, t, status):
        if status:
            print(f"\n[mic] {status}", file=sys.stderr)
        mono  = indata[:, 0] if indata.ndim == 2 else indata
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(queue.put_nowait, pcm16)

    print(f"[mic] recording at {SERVER_SR} Hz, {chunk_ms} ms chunks")
    print("[mic] speak now — press Ctrl+C to stop\n")

    with sd.InputStream(
        samplerate = SERVER_SR,
        channels   = 1,
        dtype      = "float32",
        blocksize  = chunk_samples,
        callback   = _cb,
    ):
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n[mic] stopped")


async def _file_generator(path: str, chunk_ms: int) -> AsyncGenerator[bytes, None]:
    audio, src_sr = _load_audio_file(path)
    pcm16         = _to_pcm16_16k(audio, src_sr)

    chunk_bytes = int(SERVER_SR * chunk_ms / 1000) * 2
    duration_s  = len(pcm16) / 2 / SERVER_SR
    n_chunks    = (len(pcm16) + chunk_bytes - 1) // chunk_bytes

    print(f"[file] {path}")
    print(f"[file] duration={duration_s:.1f}s  chunks={n_chunks} × {chunk_ms} ms\n")

    t_start = time.perf_counter()
    for i, offset in enumerate(range(0, len(pcm16), chunk_bytes)):
        yield pcm16[offset: offset + chunk_bytes]
        target  = (i + 1) * chunk_ms / 1000
        elapsed = time.perf_counter() - t_start
        gap     = target - elapsed
        if gap > 0:
            await asyncio.sleep(gap)

    print("\n[file] stream complete — waiting for final transcript...")


# ── CLI ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parakeet ASR test client")
    p.add_argument("--url",       default=DEFAULT_WS_URL, metavar="URL",
                   help="Full WebSocket URL (default: %(default)s)")
    p.add_argument("--file",      default=None, metavar="PATH",
                   help="Audio file to transcribe (omit for microphone mode)")
    p.add_argument("--languages", default=DEFAULT_LANGUAGES, nargs="+", metavar="LANG",
                   help="Candidate BCP-47 language tags (default: en-US es-US)")
    p.add_argument("--chunk-ms",  default=CHUNK_MS, type=int, metavar="MS",
                   help="Audio chunk size in ms (default: 100)")
    p.add_argument("--health",    action="store_true",
                   help="Check server /health and exit")
    return p.parse_args()


async def _main() -> None:
    args   = _parse_args()
    ws_url = _validate_ws_url(args.url)

    if args.health:
        check_health(ws_url)
        return

    print("=" * 60)
    print("  Parakeet Real-Time ASR — test client")
    print("=" * 60)
    print(f"  url        : {ws_url}")
    print(f"  languages  : {args.languages}")
    print(f"  chunk      : {args.chunk_ms} ms")
    print(f"  mode       : {'file → ' + args.file if args.file else 'microphone'}")
    print("=" * 60 + "\n")

    if args.file:
        if not Path(args.file).exists():
            print(f"[error] file not found: {args.file}")
            sys.exit(1)
        gen = _file_generator(args.file, args.chunk_ms)
    else:
        gen = _mic_generator(args.chunk_ms)

    try:
        await _run_session(ws_url, args.languages, gen)
    except KeyboardInterrupt:
        print("\n[client] interrupted")


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass



INFO:     ('172.17.0.1', 34046) - "WebSocket /asr/ml/ws" [accepted]
2026-06-01 13:58:58,482 | INFO | asr_server | [20260601_135858_481989_172.17.0.1_34046] ━━━ SESSION OPEN ━━━  client=Address(host='172.17.0.1', port=34046)  active=1  total=2
2026-06-01 13:58:58,482 | INFO | asr_server | [20260601_135858_481989_172.17.0.1_34046] Waiting for init message...
INFO:     connection open
2026-06-01 13:58:58,771 | INFO | asr_server | [20260601_135858_481989_172.17.0.1_34046] Init received | candidate_languages=['en-US', 'es-US']  client_sr=16000  server_sr=16000
2026-06-01 13:58:58,771 | INFO | asr_server | [20260601_135858_481989_172.17.0.1_34046] session_ready sent to client
2026-06-01 13:58:58,771 | INFO | asr_server | [20260601_135858_481989_172.17.0.1_34046] Streaming started — ready to receive audio chunks
INFO:     10.90.126.122:31438 - "GET / HTTP/1.1" 404 Not Found
2026-06-01 13:59:01,594 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,598 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,601 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,604 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,608 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,611 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,614 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,617 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,620 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,709 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,712 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,715 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,718 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,721 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,808 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
INFO:     10.90.126.54:56614 - "GET / HTTP/1.1" 404 Not Found
2026-06-01 13:59:01,812 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,815 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,818 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,821 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,909 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,912 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,916 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,919 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:01,922 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,005 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,008 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,011 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,014 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,018 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,108 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,111 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,114 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,117 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,120 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,204 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,207 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,210 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,213 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,216 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,319 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,323 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,326 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,329 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,332 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,404 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,407 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,410 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,413 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,417 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,510 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,514 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,517 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,520 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,523 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,605 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,609 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,612 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,615 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,618 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,708 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,711 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,714 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,717 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,720 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,807 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,811 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,814 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,817 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,820 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,908 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,911 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,914 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,917 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:02,920 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,003 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,007 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,010 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,013 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,016 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,109 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,112 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,116 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,119 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,122 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,204 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,207 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,210 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,213 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,217 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,310 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,314 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,317 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,320 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,323 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,407 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,410 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,413 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,416 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,420 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,510 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,513 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,516 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,519 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,523 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,606 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,609 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,612 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,615 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,618 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,709 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,712 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,715 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,718 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,721 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,806 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,810 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,813 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,816 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,819 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,909 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,912 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,915 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,918 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:03,921 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,005 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,008 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,012 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,015 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,018 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,109 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,112 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,115 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,118 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,122 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,205 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,209 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,212 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,215 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,218 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,309 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,313 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,316 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,319 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,322 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,404 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,407 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,411 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,414 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,417 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,511 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,514 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,518 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,521 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,524 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,604 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,607 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,610 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,614 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,617 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,707 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,711 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,714 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,717 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,720 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,806 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,810 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,813 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,816 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,819 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,908 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,911 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,914 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,917 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:04,920 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,005 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,008 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,011 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,014 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,018 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,108 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,111 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,114 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,118 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,121 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,205 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,208 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,211 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,215 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,218 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,309 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,312 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,316 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,319 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,322 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,404 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,407 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,410 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,414 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,417 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,511 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,515 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,518 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,522 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,525 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,606 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,610 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,613 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,616 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,619 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,709 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,712 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,716 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,719 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,722 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,803 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,806 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,809 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,812 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,816 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,908 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,912 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,915 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,918 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:05,922 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,005 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,009 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,012 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,015 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,018 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,107 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,110 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,114 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,117 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,120 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,202 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,206 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,209 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,212 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,215 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,308 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,311 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,314 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,317 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,320 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,406 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,410 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,413 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,416 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,419 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,508 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,511 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,515 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,518 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,521 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,603 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,606 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,610 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,613 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,616 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,708 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,711 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:06,715 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,242 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,311 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,313 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,316 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,319 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,322 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,404 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,407 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,409 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,412 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,415 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,510 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,513 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,516 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,518 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,521 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,606 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,609 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,612 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,615 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,617 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,713 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,716 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,719 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,721 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,724 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,806 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,809 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,812 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,815 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,817 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,913 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,915 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,918 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,921 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:12,924 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,006 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,009 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,012 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,014 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,017 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,113 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,115 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,118 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,121 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,124 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,205 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,208 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,211 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,214 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,216 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,312 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,315 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,317 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,320 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,323 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,405 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,407 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,410 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,413 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,416 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,513 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,516 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,518 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,521 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,524 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,608 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,611 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,614 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,617 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,620 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,709 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,712 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,715 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,718 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,721 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,806 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,809 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,812 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,814 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,817 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,909 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,912 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,915 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,918 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:13,920 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,006 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,008 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,011 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,014 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,017 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,116 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,119 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,122 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,125 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,128 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,204 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,207 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,209 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,212 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,215 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,314 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,317 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,320 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,323 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,326 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,407 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,410 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,413 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,416 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,419 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,513 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,516 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,519 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,522 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,524 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,610 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,613 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,615 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,618 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,621 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,713 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,716 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,718 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,721 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,724 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,810 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,813 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,816 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,819 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,821 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,913 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,916 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,919 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,922 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:14,925 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,005 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,008 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,011 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,014 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,017 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,114 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,117 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,120 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,122 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,125 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,209 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,212 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,214 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,217 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,220 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,312 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,315 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,318 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:15,321 | ERROR | parakeet_engine | conformer_stream_step failed: cannot reshape tensor of 0 elements into shape [1, 8, -1, 0] because the unspecified dimension size -1 can be any value and is ambiguous
2026-06-01 13:59:21,707 | INFO | asr_server | [20260601_135858_481989_172.17.0.1_34046] Control message: type=end_session
2026-06-01 13:59:21,707 | INFO | asr_server | [20260601_135858_481989_172.17.0.1_34046] end_session received — closing cleanly
2026-06-01 13:59:21,707 | INFO | asr_server | [20260601_135858_481989_172.17.0.1_34046] ━━━ SESSION CLOSED ━━━  duration=22.9s  audio=21.2s  chunks=212  partials=0  finals=0  active=0
2026-06-01 13:59:21,707 | INFO | asr_server | [20260601_135858_481989_172.17.0.1_34046] Saving session audio → /app/session_audio/20260601_135858_481989_172.17.0.1_34046.wav
2026-06-01 13:59:21,709 | INFO | asr_server | Session WAV saved → /app/session_audio/20260601_135858_481989_172.17.0.1_34046.wav  (21.2 s)
INFO:     connection closed
