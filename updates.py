#app/batch_infer.py-
import time
import threading
import queue
import logging

log = logging.getLogger("batch_infer")


class BatchItem:
    def __init__(self, wav_path):
        self.wav_path = wav_path
        self.result = None
        self.event = threading.Event()


class DynamicBatchInferer:

    def __init__(self, model, max_batch_size=16, max_wait_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self.q = queue.Queue()
        self.running = True

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, wav_path):
        item = BatchItem(wav_path)
        self.q.put(item)
        return item

    def _worker(self):
        while self.running:
            batch = []
            start = time.time()

            # wait for at least 1 request
            item = self.q.get()
            batch.append(item)

            # dynamic collection window
            while True:
                elapsed = (time.time() - start) * 1000

                if len(batch) >= self.max_batch_size:
                    break

                if elapsed >= self.max_wait_ms:
                    break

                try:
                    item = self.q.get_nowait()
                    batch.append(item)
                except queue.Empty:
                    time.sleep(0.002)

            try:
                paths = [b.wav_path for b in batch]

                t0 = time.perf_counter()
                results = self.model.transcribe(paths)
                infer_time = time.perf_counter() - t0

                log.info(
                    f"BATCH | size={len(batch)} "
                    f"time={infer_time:.2f}s "
                    f"queue={self.q.qsize()}"
                )

                for item, res in zip(batch, results):
                    item.result = res
                    item.event.set()

            except Exception as e:
                log.exception(f"BATCH ERROR: {e}")
                for item in batch:
                    item.result = ""
                    item.event.set()


#app/asr_engines/parakeet_asr.py-
import io
import os
import time
import wave
import tempfile
import logging

from app.asr_engines.base import ASREngine, EngineCaps
from app.batch_infer import DynamicBatchInferer

log = logging.getLogger("parakeet_engine")


def safe_text(x):
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
            log.exception("Error extracting text")
            return ""
    return str(x)


class ParakeetASR(ASREngine):

    caps = EngineCaps(
        streaming=False,
        partials=True,
        ttft_meaningful=True,
    )

    def __init__(self, model_name, device, sample_rate):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate
        self.model = None

        self.end_silence_ms = 700
        self.min_utt_ms = 300
        self.finalize_pad_ms = 0

        # 🔥 reduced partial frequency (IMPORTANT)
        self.partial_interval_sec = 3.0

        self.batcher = None

    def load(self):
        import nemo.collections.asr as nemo_asr

        log.info(f"Loading model: {self.model_name}")
        t0 = time.time()

        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 🔥 dynamic batcher
        self.batcher = DynamicBatchInferer(
            self.model,
            max_batch_size=16,
            max_wait_ms=50,
        )

        log.info("Model ready on %s", self.device)
        return time.time() - t0

    def new_session(self, max_buffer_ms):
        return ParakeetSession(self, max_buffer_ms)


class ParakeetSession:

    def __init__(self, engine, max_buffer_ms):
        self.engine = engine
        self.max_buffer_samples = int(engine.sr * max_buffer_ms / 1000)

        self.audio = bytearray()
        self.current_text = ""
        self.last_partial_time = 0.0

    def accept_pcm16(self, pcm16):
        self.audio.extend(pcm16)

        max_bytes = self.max_buffer_samples * 2
        if len(self.audio) > max_bytes:
            self.audio = self.audio[-max_bytes:]

    def step_if_ready(self):
        now = time.time()

        if not self.audio:
            return None

        if (now - self.last_partial_time) < self.engine.partial_interval_sec:
            return None

        self.last_partial_time = now

        text = self._transcribe().strip()

        if not text:
            return None

        if text == self.current_text:
            return None

        if self.current_text and self.current_text.startswith(text):
            return None

        self.current_text = text
        return text

    def finalize(self, pad_ms):
        final = self._transcribe().strip()

        if final:
            self.current_text = final

        out = self.current_text.strip()

        self.audio.clear()
        self.current_text = ""
        self.last_partial_time = 0.0

        return out

    def _transcribe(self):
        if not self.audio:
            return ""

        tmp_path = None
        audio_sec = len(self.audio) / 2 / self.engine.sr

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(self._pcm_to_wav(bytes(self.audio)))

            # 🔥 submit to dynamic batcher
            item = self.engine.batcher.submit(tmp_path)

            wait_start = time.perf_counter()
            item.event.wait()
            total_wait = time.perf_counter() - wait_start

            result = safe_text(item.result).strip()

            log.info(
                f"TRANSCRIBE | audio={audio_sec:.2f}s "
                f"wait={total_wait:.2f}s"
            )

            return result

        except Exception as e:
            log.exception(f"TRANSCRIBE ERROR: {e}")
            return ""

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    log.exception("Temp cleanup failed")

    def _pcm_to_wav(self, pcm):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.engine.sr)
            wf.writeframes(pcm)
        return buf.getvalue()


#main.py-
import asyncio
import json
import logging
import sys
import time
import uuid

from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.config import load_config
from app.factory import build_engine
from app.streaming_session import StreamingSession

cfg = load_config()

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("parakeet_server")

app = FastAPI()
engine = build_engine(cfg)

# metrics
active_infer = 0


@app.on_event("startup")
async def startup():
    load_sec = engine.load()
    log.info(f"Model loaded in {load_sec:.2f}s")


@app.websocket("/ws")
async def ws_asr(ws: WebSocket):
    global active_infer

    conn_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    await ws.accept()
    log.info(f"[{conn_id}] CONNECTED")

    session = StreamingSession(engine, cfg)

    try:
        while True:
            data = await ws.receive_bytes()

            loop = asyncio.get_running_loop()
            start = time.perf_counter()

            try:
                active_infer += 1
                log.info(f"[{conn_id}] INFER START | active={active_infer}")

                events = await loop.run_in_executor(
                    None,
                    session.process_chunk,
                    data,
                )

                elapsed = time.perf_counter() - start

                log.info(
                    f"[{conn_id}] INFER END | active={active_infer} "
                    f"time={elapsed:.2f}s"
                )

                if active_infer > 2:
                    log.warning(f"SYSTEM OVERLOAD | active={active_infer}")

            except Exception as e:
                log.exception(f"[{conn_id}] INFER ERROR: {e}")
                continue

            finally:
                active_infer -= 1

            for ev in events:
                if len(ev) == 3:
                    ev_type, text, ttfb_ms = ev
                    await ws.send_text(json.dumps({
                        "type": ev_type,
                        "text": text,
                        "t_start": ttfb_ms,
                    }))
                else:
                    ev_type, text = ev
                    await ws.send_text(json.dumps({
                        "type": ev_type,
                        "text": text,
                    }))

    except WebSocketDisconnect:
        log.info(f"[{conn_id}] DISCONNECTED")

    finally:
        log.info(
            f"[{conn_id}] CLOSED | duration={time.time() - start_time:.2f}s"
        )



#streaming_sessioon.py-
import time
import logging
from app.vad import AdaptiveEnergyVAD

log = logging.getLogger("streaming_session")


class StreamingSession:

    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg

        self.vad = AdaptiveEnergyVAD(
            cfg.sample_rate,
            cfg.vad_frame_ms,
            cfg.vad_start_margin,
            cfg.vad_min_noise_rms,
            cfg.pre_speech_ms,
        )

        self.session = engine.new_session(cfg.max_utt_ms)

        self.frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
        self.raw_buf = bytearray()

        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0

        self.t_utt_start = None
        self.t_first_partial = None

        log.info(
            f"SESSION INIT | frame_bytes={self.frame_bytes} "
            f"vad_frame_ms={cfg.vad_frame_ms}"
        )

    def process_chunk(self, pcm):
        events = []

        if not pcm:
            log.warning("Empty PCM chunk received")
            return events

        self.raw_buf.extend(pcm)

        log.debug(f"Chunk received | size={len(pcm)}")

        while len(self.raw_buf) >= self.frame_bytes:
            frame = bytes(self.raw_buf[:self.frame_bytes])
            del self.raw_buf[:self.frame_bytes]

            try:
                is_speech, pre = self.vad.push_frame(frame)
            except Exception as e:
                log.exception(f"VAD ERROR: {e}")
                is_speech, pre = True, None

            self.silence_ms = 0 if is_speech else self.silence_ms + self.cfg.vad_frame_ms

            # 🔥 Speech start
            if pre and not self.utt_started:
                self.utt_started = True
                self.utt_audio_ms = 0
                self.silence_ms = 0
                self.t_utt_start = time.time()
                self.t_first_partial = None

                self.session.accept_pcm16(pre)

                log.info("UTTERANCE STARTED")

            if not self.utt_started:
                continue

            self.session.accept_pcm16(frame)
            self.utt_audio_ms += self.cfg.vad_frame_ms

            # 🔥 Partial
            if self.engine.caps.partials:
                try:
                    text = self.session.step_if_ready()
                except Exception as e:
                    log.exception(f"PARTIAL ERROR: {e}")
                    text = None

                if text:
                    if self.t_first_partial is None:
                        self.t_first_partial = time.time()

                    ttfb_ms = int((self.t_first_partial - self.t_utt_start) * 1000)

                    log.info(f"PARTIAL | {text}")

                    events.append(("partial", text, ttfb_ms))

            # 🔥 Endpoint detection
            if (
                not is_speech
                and self.utt_audio_ms >= self.engine.min_utt_ms
                and self.silence_ms >= self.engine.end_silence_ms
            ):
                log.info(
                    f"ENDPOINT | utt_ms={self.utt_audio_ms} "
                    f"silence_ms={self.silence_ms}"
                )

                try:
                    final = self.session.finalize(self.engine.finalize_pad_ms)
                except Exception as e:
                    log.exception(f"FINAL ERROR: {e}")
                    final = ""

                if final:
                    ttfb_ms = (
                        int((self.t_first_partial - self.t_utt_start) * 1000)
                        if self.t_first_partial is not None
                        else None
                    )

                    log.info(f"FINAL | {final}")

                    events.append(("transcript", final, ttfb_ms))
                else:
                    log.warning("FINAL EMPTY")

                self.reset()

        return events

    def reset(self):
        log.info("SESSION RESET")

        self.vad.reset()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--ws-ping-interval", "30", "--ws-ping-timeout", "300"]


Transcribing:   0%|          | 0/1 [00:00<?, ?it/s]
2026-05-04 15:24:09,559 | ERROR | parakeet_engine | TRANSCRIBE ERROR: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
Traceback (most recent call last):
  File "/app/app/asr_engines/parakeet_asr.py", line 134, in _transcribe
    results = self.engine.model.transcribe([tmp_path])
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/rnnt_models.py", line 308, in transcribe
    return super().transcribe(
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/mixins/transcription.py", line 270, in transcribe
    for processed_outputs in generator:
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/mixins/transcription.py", line 369, in transcribe_generator
    model_outputs = self._transcribe_forward(test_batch, transcribe_cfg)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/rnnt_models.py", line 936, in _transcribe_forward
    encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])
  File "/usr/local/lib/python3.10/dist-packages/nemo/core/classes/common.py", line 1141, in wrapped_call
    outputs = wrapped(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/rnnt_models.py", line 706, in forward
    encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/core/classes/common.py", line 1141, in wrapped_call
    outputs = wrapped(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/modules/conformer_encoder.py", line 586, in forward
    return self.forward_internal(
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/modules/conformer_encoder.py", line 635, in forward_internal
    audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/submodules/subsampling.py", line 425, in forward
    x = self.conv(x)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 219, in forward
    input = module(input)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 458, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 454, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR





Transcribing:   0%|          | 0/1 [00:00<?, ?it/s]
2026-05-04 15:25:37,223 | ERROR | parakeet_engine | TRANSCRIBE ERROR: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
Traceback (most recent call last):
  File "/app/app/asr_engines/parakeet_asr.py", line 134, in _transcribe
    results = self.engine.model.transcribe([tmp_path])
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/rnnt_models.py", line 308, in transcribe
    return super().transcribe(
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/mixins/transcription.py", line 270, in transcribe
    for processed_outputs in generator:
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/mixins/transcription.py", line 369, in transcribe_generator
    model_outputs = self._transcribe_forward(test_batch, transcribe_cfg)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/rnnt_models.py", line 936, in _transcribe_forward
    encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])
  File "/usr/local/lib/python3.10/dist-packages/nemo/core/classes/common.py", line 1141, in wrapped_call
    outputs = wrapped(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/rnnt_models.py", line 706, in forward
    encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/core/classes/common.py", line 1141, in wrapped_call
    outputs = wrapped(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/modules/conformer_encoder.py", line 586, in forward
    return self.forward_internal(
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/modules/conformer_encoder.py", line 635, in forward_internal
    audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/submodules/subsampling.py", line 425, in forward
    x = self.conv(x)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 219, in forward
    input = module(input)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 458, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 454, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
2026-05-04 15:25:37,231 | INFO | parakeet_server | [88b838fd] INFER START | active=8
2026-05-04 15:25:37,236 | INFO | parakeet_server | [a9b33817] INFER END | active=8 time=0.03s
2026-05-04 15:25:37,235 | ERROR | parakeet_engine | TRANSCRIBE ERROR: Cannot unfreeze partially without first freezing the module with `freeze()`
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/mixins/transcription.py", line 369, in transcribe_generator
    model_outputs = self._transcribe_forward(test_batch, transcribe_cfg)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/rnnt_models.py", line 936, in _transcribe_forward
    encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])
  File "/usr/local/lib/python3.10/dist-packages/nemo/core/classes/common.py", line 1141, in wrapped_call
    outputs = wrapped(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/rnnt_models.py", line 706, in forward
    encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/core/classes/common.py", line 1141, in wrapped_call
    outputs = wrapped(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/modules/conformer_encoder.py", line 586, in forward
    return self.forward_internal(
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/modules/conformer_encoder.py", line 635, in forward_internal
    audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/submodules/subsampling.py", line 425, in forward
    x = self.conv(x)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/container.py", line 219, in forward
    input = module(input)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 458, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 454, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/app/app/asr_engines/parakeet_asr.py", line 134, in _transcribe
    results = self.engine.model.transcribe([tmp_path])
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/rnnt_models.py", line 308, in transcribe
    return super().transcribe(
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/mixins/transcription.py", line 270, in transcribe
    for processed_outputs in generator:
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/mixins/transcription.py", line 382, in transcribe_generator
    self._transcribe_on_end(transcribe_cfg)
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/mixins/transcription.py", line 763, in _transcribe_on_end
    self.encoder.unfreeze(partial=True)
  File "/usr/local/lib/python3.10/dist-packages/nemo/core/classes/module.py", line 114, in unfreeze
    raise ValueError("Cannot unfreeze partially without first freezing the module with `freeze()`")
ValueError: Cannot unfreeze partially without first freezing the module with `freeze()`
