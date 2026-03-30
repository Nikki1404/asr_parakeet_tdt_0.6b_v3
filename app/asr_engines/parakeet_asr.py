import time
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from app.asr_engines.base import ASREngine, EngineCaps


def clean_text(text: Optional[str]) -> str:
    """
    Normalize text safely.
    """
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def detect_language_from_text(text: str) -> str:
    """
    Lightweight EN / ES detector.
    ASR already auto-detects language.
    This is only for downstream tagging.
    """
    text = text.lower()

    spanish_words = {
        "hola", "gracias", "cuenta", "necesito",
        "ayuda", "por favor", "buenos", "dias",
        "mañana", "usted", "señor", "señora",
        "numero", "correo", "telefono", "quiero",
        "como", "estoy", "estás", "estamos",
    }

    english_words = {
        "hello", "thank", "account", "help",
        "please", "good", "morning", "today",
        "email", "phone", "number", "want",
        "how", "i", "you", "we", "need",
    }

    es_score = sum(word in text for word in spanish_words)
    en_score = sum(word in text for word in english_words)

    return "es" if es_score > en_score else "en"


@dataclass
class StreamTimings:
    preproc_sec: float = 0.0
    infer_sec: float = 0.0
    flush_sec: float = 0.0


class ParakeetASR(ASREngine):
    """
    Realtime Parakeet ASR wrapper.
    """

    caps = EngineCaps(
        streaming=True,
        partials=True,
        ttft_meaningful=True,
    )

    def __init__(
        self,
        model_name: str,
        device: str,
        sample_rate: int,
    ):
        self.model_name = model_name

        self.device = (
            "cuda"
            if device == "cuda"
            and torch.cuda.is_available()
            else "cpu"
        )

        self.sr = sample_rate

        # endpointing
        self.end_silence_ms = 800
        self.min_utt_ms = 250
        self.finalize_pad_ms = 400

        # partial tuning
        self.partial_window_sec = 5.0
        self.partial_step_sec = 0.5

        self.model = None

    def load(self) -> float:
        """
        Load model using exact HF / NeMo syntax.
        """
        import nemo.collections.asr as nemo_asr

        t0 = time.time()

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name
        )

        if self.device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        self.model.eval()

        self._warmup()

        return time.time() - t0

    def _warmup(self):
        """
        Warm up model once.
        """
        try:
            sess = self.new_session(max_buffer_ms=5000)

            silence = np.zeros(
                int(self.sr * 1.0),
                dtype=np.float32
            )

            pcm16 = (
                silence * 32767
            ).astype(np.int16).tobytes()

            sess.accept_pcm16(pcm16)
            sess.finalize(400)

        except Exception:
            pass

    def new_session(self, max_buffer_ms: int):
        return ParakeetSession(self, max_buffer_ms)


class ParakeetSession:
    """
    Per websocket session state.
    """

    def __init__(
        self,
        engine: ParakeetASR,
        max_buffer_ms: int
    ):
        self.engine = engine

        self.max_buffer_samples = int(
            engine.sr * max_buffer_ms / 1000
        )

        self.audio = np.array([], dtype=np.float32)

        self.current_text = ""
        self.current_lang = "en"

        self.last_partial = ""
        self.last_final_text = ""

        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0

        self.chunks = 0
        self.last_decode_samples = 0

    def reset_stream_state(self):
        self.audio = np.array([], dtype=np.float32)

        self.current_text = ""
        self.current_lang = "en"

        self.last_partial = ""
        self.last_decode_samples = 0

        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

    def accept_pcm16(self, pcm16: bytes):
        if not pcm16:
            return

        x = (
            np.frombuffer(
                pcm16,
                dtype=np.int16
            ).astype(np.float32)
            / 32768.0
        )

        if x.size == 0:
            return

        self.audio = np.concatenate([self.audio, x])

        if len(self.audio) > self.max_buffer_samples:
            self.audio = self.audio[-self.max_buffer_samples:]

    def _transcribe(
        self,
        audio
    ) -> Tuple[str, str, StreamTimings]:

        timings = StreamTimings()

        if len(audio) == 0:
            return "", "en", timings

        t0 = time.perf_counter()

        audio = audio.astype(np.float32)

        timings.preproc_sec += (
            time.perf_counter() - t0
        )

        t1 = time.perf_counter()

        with torch.inference_mode():
            result = self.engine.model.transcribe(
                [audio],
                batch_size=1,
                verbose=False
            )

        timings.infer_sec += (
            time.perf_counter() - t1
        )

        text = ""

        try:
            if isinstance(result, list) and len(result) > 0:
                item = result[0]

                if isinstance(item, str):
                    text = item
                elif hasattr(item, "text"):
                    text = item.text
                else:
                    text = str(item)

        except Exception:
            text = ""

        text = clean_text(text)

        lang = detect_language_from_text(text)

        return text, lang, timings

    def _is_new_text(
        self,
        text: Optional[str]
    ) -> bool:

        if text is None:
            return False

        new = clean_text(text)
        old = clean_text(self.current_text)

        if not new:
            return False

        if new == old:
            return False

        if old and new.startswith(old):
            return True

        if old and old.startswith(new):
            return False

        return True

    def step_if_ready(self):
        min_samples = int(
            self.engine.partial_step_sec
            * self.engine.sr
        )

        if len(self.audio) < min_samples:
            return None

        if (
            len(self.audio)
            - self.last_decode_samples
        ) < min_samples:
            return None

        self.last_decode_samples = len(self.audio)

        rolling_samples = int(
            self.engine.partial_window_sec
            * self.engine.sr
        )

        chunk = self.audio[-rolling_samples:]

        text, lang, t = self._transcribe(chunk)

        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec

        if not self._is_new_text(text):
            return None

        self.current_text = text
        self.current_lang = lang
        self.last_partial = text

        self.chunks += 1

        return text, lang

    def finalize(self, pad_ms: int):
        if len(self.audio) == 0:
            return "", "en"

        pad = np.zeros(
            int(
                self.engine.sr
                * pad_ms / 1000
            ),
            dtype=np.float32
        )

        full_audio = np.concatenate(
            [self.audio, pad]
        )

        t0 = time.perf_counter()

        text, lang, t = self._transcribe(
            full_audio
        )

        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec

        self.utt_flush += (
            time.perf_counter() - t0
        )

        final = text or self.current_text
        final_lang = lang or self.current_lang

        self.last_final_text = (
            (
                self.last_final_text
                + " "
                + final
            ).strip()
            if final
            else self.last_final_text
        )

        self.reset_stream_state()

        return final, final_lang
