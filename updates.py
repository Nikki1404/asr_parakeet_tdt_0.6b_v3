for this code 
"""
Parakeet ASR – Terminal Microphone Client
Captures microphone audio, sends it to the WebSocket server in real time,
and prints transcriptions as they arrive. Works for English & Spanish
(parakeet-tdt-0.6b-v3 auto-detects the language).

Usage:
    # Microphone mode (default)
    python client.py (local testing)
    python client.py --host 34.118.200.125 --port 8001 (GCP)
    

    # File mode – streams audio file(s) chunk by chunk at real-time speed
    python client.py --host 34.118.200.125 --port 8001 --file english.wav
    python client.py --host 34.118.200.125 --port 8001 --file english.wav spanish.mp3
    python client.py --host 34.118.200.125 --port 8001 --file audio.wav --speed 1.5


Dependencies:
    pip install pyaudio websockets soundfile numpy
    pip install pydub   (optional, for MP3/OGG)
    apt install ffmpeg  (optional, for non-WAV resampling)
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

# Config
SAMPLE_RATE   = 16_000
CHANNELS      = 1
FORMAT        = pyaudio.paInt16
CHUNK_MS      = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000   # 480 samples
CHUNK_BYTES   = CHUNK_SAMPLES * 2                # int16 → 2 bytes per sample

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║        Parakeet-TDT-0.6B-v3  |  Real-Time ASR Client        ║
║          Auto language detection (EN / ES / more)           ║
║          WebSocket: ws://localhost:8001/ws                   ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
Commands
  [Enter]   Manual flush – force transcription of buffered audio
  q         Quit
  d         List available audio input devices

Test phrases
  EN: "The quick brown fox jumps over the lazy dog"
  ES: "El zorro marrón rápido salta sobre el perro perezoso"
"""


# Audio capture (mic)

class MicCapture:
    """Captures microphone audio into a thread-safe queue."""

    def __init__(self, device_index: Optional[int] = None):
        self.queue: Queue[bytes] = Queue(maxsize=512)
        self._stop   = threading.Event()
        self._device = device_index
        self._thread = threading.Thread(target=self._capture, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _capture(self):
        pa = pyaudio.PyAudio()
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
                data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                self.queue.put(data)
        except OSError as exc:
            print(f"\n[ERROR] Audio capture failed: {exc}", file=sys.stderr)
            self._stop.set()
        finally:
            try:
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
            print(f"  [{i}] {info['name']}  ({int(info['defaultSampleRate'])} Hz)")
    pa.terminate()
    print()


#  Audio loading (file) 
def load_audio_as_16k_pcm(path: str) -> bytes:
    """
    Load any audio file → raw int16 LE PCM at 16 kHz mono.
    Tries soundfile first (WAV/FLAC/OGG), then pydub (MP3 etc).
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            data = _resample(data, sr, SAMPLE_RATE)
        return _float32_to_pcm16(data)
    except Exception as sf_err:
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(path)
            seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
            return seg.raw_data
        except Exception as pd_err:
            raise RuntimeError(
                f"Cannot load '{path}'.\n"
                f"  soundfile : {sf_err}\n"
                f"  pydub     : {pd_err}\n"
                f"  Tip: install ffmpeg for MP3/OGG support."
            )


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import librosa
        return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_sr, target_sr)
        return resample_poly(data, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError:
        pass
    # Pure numpy linear interpolation fallback
    duration  = len(data) / orig_sr
    old_times = np.linspace(0, duration, len(data))
    new_times = np.linspace(0, duration, int(duration * target_sr))
    return np.interp(new_times, old_times, data).astype(np.float32)


def _float32_to_pcm16(data: np.ndarray) -> bytes:
    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


# ── Shared receiver (same for mic and file modes)

async def _receiver(ws):
    """Receives partial + final transcripts and prints them."""
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            text     = msg.get("text", "")
            dur_ms   = msg.get("duration_ms", 0)
            ts       = time.strftime("%H:%M:%S")

            if msg_type == "partial":
                print(f"\r  ⏳ [{ts}] {text:<80}", end="", flush=True)

            elif msg_type == "transcript":
                rtf = msg.get("rtf", 0)
                print(f"\r{' ' * 90}\r", end="")
                print(f"[{ts}] ({dur_ms/1000:.1f}s | RTF {rtf:.2f}x)")
                print(f"  ✅  {text}")
                print("─" * 64)

    except websockets.exceptions.ConnectionClosed:
        print("\n[Connection closed]")


# ── stdin reader 

def stdin_reader(cmd_queue: Queue):
    while True:
        try:
            line = sys.stdin.readline()
            if line:
                cmd_queue.put(line.strip())
        except Exception:
            break


#  MODE 1 : Microphone 

async def run_mic(host: str, port: int, device_index: Optional[int]):
    uri = f"ws://{host}:{port}/ws"
    print(BANNER)
    print(f"Connecting to {uri} …", end=" ", flush=True)

    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
            print("connected ✓\n")
            print("🎙  Speak now – transcriptions appear below\n")
            print("─" * 64)

            mic = MicCapture(device_index)
            mic.start()

            cmd_queue: Queue[str] = Queue()
            threading.Thread(target=stdin_reader, args=(cmd_queue,), daemon=True).start()

            send_task = asyncio.create_task(_mic_sender(ws, mic, cmd_queue))
            recv_task = asyncio.create_task(_receiver(ws))

            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            mic.stop()

    except (OSError, websockets.exceptions.WebSocketException) as exc:
        print(f"failed ✗\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


async def _mic_sender(ws, mic: MicCapture, cmd_queue: Queue):
    print(HELP)
    try:
        while True:
            try:
                cmd = cmd_queue.get_nowait()
                if cmd.lower() in ("q", "quit", "exit"):
                    print("\nBye!")
                    return
                elif cmd == "":
                    await ws.send(json.dumps({"cmd": "flush"}).encode())
                    print("[flushed]\n", flush=True)
                elif cmd.lower() == "d":
                    list_devices()
            except Empty:
                pass

            sent = 0
            while not mic.queue.empty() and sent < 50:
                frame = mic.queue.get_nowait()
                await ws.send(frame)
                sent += 1

            await asyncio.sleep(0.005)

    except websockets.exceptions.ConnectionClosed:
        print("\n[Connection closed by server]")


# File streaming
async def run_files(host: str, port: int, files: List[str], speed: float):
    uri = f"ws://{host}:{port}/ws"
    print(BANNER)
    print(f"  Mode   : FILE streaming at {speed:.1f}× real-time speed")
    print(f"  Server : {uri}")
    print(f"  Files  : {', '.join(Path(f).name for f in files)}\n")

    for filepath in files:
        p = Path(filepath)
        if not p.exists():
            print(f"  [SKIP] File not found: {filepath}\n")
            continue

        # Load and resample
        print(f"  Loading {p.name} …", end=" ", flush=True)
        try:
            pcm = load_audio_as_16k_pcm(str(p))
        except RuntimeError as e:
            print(f"FAILED ✗\n  {e}\n")
            continue

        duration_sec = len(pcm) / 2 / SAMPLE_RATE
        print(f"{duration_sec:.1f}s  ✓")
        print(f"\n{'═' * 64}")
        print(f"  📄 {p.name}  ({duration_sec:.1f}s)  — language auto-detected by model")
        print(f"{'═' * 64}\n")

        try:
            async with websockets.connect(
                uri, ping_interval=20, ping_timeout=20, max_size=2**23
            ) as ws:
                send_task = asyncio.create_task(
                    _file_sender(ws, pcm, speed)
                )
                recv_task = asyncio.create_task(_receiver(ws))

                # Receiver exits on ConnectionClosed; sender exits when done
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()

        except (OSError, websockets.exceptions.WebSocketException) as exc:
            print(f"\n  [ERROR] WebSocket failed: {exc}")

        print()   # blank line between files

    print("  All files processed.")


async def _file_sender(ws, pcm: bytes, speed: float):
    """Stream PCM bytes to server chunk by chunk at real-time speed."""
    chunk_delay = (CHUNK_MS / 1000) / speed   # e.g. 0.03s at 1×, 0.015s at 2×
    offset      = 0

    while offset + CHUNK_BYTES <= len(pcm):
        chunk   = pcm[offset: offset + CHUNK_BYTES]
        offset += CHUNK_BYTES
        await ws.send(chunk)
        await asyncio.sleep(chunk_delay)

    # Pad and send any leftover bytes
    leftover = pcm[offset:]
    if leftover:
        await ws.send(leftover + bytes(CHUNK_BYTES - len(leftover)))

    # Small pause then manual flush to force final transcript
    await asyncio.sleep(0.3)
    await ws.send(json.dumps({"cmd": "flush"}).encode())

    # Wait long enough for the server to respond with the final transcript
    await asyncio.sleep(10)


def parse_args():
    p = argparse.ArgumentParser(description="Parakeet ASR client – mic or file mode")
    p.add_argument("--host",   default="localhost", help="Server host (default: localhost)")
    p.add_argument("--port",   type=int, default=8001, help="Server port (default: 8001)")
    p.add_argument("--device", type=int, default=None, metavar="INDEX",
                   help="Mic device index for mic mode (see --list)")
    p.add_argument("--list",   action="store_true",
                   help="List audio input devices and exit")
    p.add_argument("--file",   nargs="+", metavar="FILE",
                   help="File mode: stream audio file(s) instead of mic "
                        "(WAV / MP3 / FLAC / OGG)")
    p.add_argument("--speed",  type=float, default=1.0,
                   help="File mode: playback speed multiplier "
                        "(default 1.0 = real-time, 2.0 = 2× faster)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        list_devices()
        sys.exit(0)

    if args.file:
        asyncio.run(run_files(args.host, args.port, args.file, args.speed))
    else:
        asyncio.run(run_mic(args.host, args.port, args.device))
        

why getting this now 
(venv) PS C:\Users\re_nikitav\Documents\parakeet-asr-multilingual> python client.py --host 34.118.200.125 --port 8001 --file  C:\Users\re_nikitav\Downloads\load_testing.wav

╔══════════════════════════════════════════════════════════════╗
║        Parakeet-TDT-0.6B-v3  |  Real-Time ASR Client        ║
║          Auto language detection (EN / ES / more)           ║
║          WebSocket: ws://localhost:8001/ws                   ║
╚══════════════════════════════════════════════════════════════╝

  Mode   : FILE streaming at 1.0× real-time speed
  Server : ws://34.118.200.125:8001/ws
  Files  : load_testing.wav

  Loading load_testing.wav … 572.8s  ✓

════════════════════════════════════════════════════════════════
  📄 load_testing.wav  (572.8s)  — language auto-detected by model
════════════════════════════════════════════════════════════════


  [ERROR] WebSocket failed: did not receive a valid HTTP response

  All files processed.
(venv) PS C:\Users\re_nikitav\Documents\parakeet-asr-multilingual>

earlier same code was working file for same  python client.py --host 34.118.200.125 --port 8001 --file  C:\Users\re_nikitav\Downloads\load_testing.wav

and i was getting transcription but now getting this error 

this is gcp vm hosted docker which i am accesing on this 
                                   --host 34.118.200.125 --port 8001

which was working earlier for exact same code , host and port nothing is changed so why all of sufdden started getting error 
    
in server logs 
i am getting this 

026-04-01 14:04:31,169 [INFO] [vad-silence] Final transcription 3.39 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00,  9.77it/s]
2026-04-01 14:04:31,291 [INFO]   → FINAL "So we had bot and the same thing and you can hear me."  [0.12s RTF 0.04x]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 10.68it/s]
2026-04-01 14:04:31,306 [ERROR] Transcription error: Cannot unfreeze partially without first freezing the module with `freeze()`
Traceback (most recent call last):
  File "/app/server.py", line 117, in transcribe_pcm
    results = asr_model.transcribe([tmp_path])
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
2026-04-01 14:04:31,678 [INFO] [vad-silence] Final transcription 1.20 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 15.41it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 14.81it/s]
2026-04-01 14:04:31,811 [ERROR] Transcription error: Cannot unfreeze partially without first freezing the module with `freeze()`
Traceback (most recent call last):
  File "/app/server.py", line 117, in transcribe_pcm
    results = asr_model.transcribe([tmp_path])
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
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 22.03it/s]
2026-04-01 14:04:32,289 [INFO] [vad-silence] Final transcription 1.14 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.77it/s]
2026-04-01 14:04:32,388 [INFO] [vad-silence] Final transcription 1.53 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 21.22it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.64it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.99it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.91it/s]
2026-04-01 14:04:33,954 [ERROR] Unexpected error: 'bytes'
Traceback (most recent call last):
  File "/app/server.py", line 207, in websocket_endpoint
    raw = await ws.receive_bytes()
  File "/usr/local/lib/python3.10/dist-packages/starlette/websockets.py", line 127, in receive_bytes
    return typing.cast(bytes, message["bytes"])
KeyError: 'bytes'
2026-04-01 14:04:34,353 [INFO] [vad-silence] Final transcription 3.36 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.32it/s]
INFO:     connection closed
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.17it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.16it/s]
2026-04-01 14:04:56,547 [INFO] [vad-silence] Final transcription 1.86 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.42it/s]
2026-04-01 14:04:56,618 [INFO]   → FINAL "Anyways,"  [0.07s RTF 0.04x]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.98it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.78it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.86it/s]
2026-04-01 14:05:37,325 [INFO] [vad-silence] Final transcription 2.34 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.42it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.62it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.62it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.41it/s]
2026-04-01 14:06:12,061 [INFO] [vad-silence] Final transcription 2.25 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.58it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.66it/s]
2026-04-01 14:06:48,134 [INFO] [vad-silence] Final transcription 1.23 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 18.95it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.95it/s]
2026-04-01 14:07:23,978 [INFO] [vad-silence] Final transcription 2.22 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.96it/s]
2026-04-01 14:07:24,047 [INFO]   → FINAL "De envenaste"  [0.07s RTF 0.03x]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 21.18it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 18.35it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.93it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.43it/s]
2026-04-01 14:08:16,991 [INFO] [vad-silence] Final transcription 2.73 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.10it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.01it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 18.90it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.10it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.74it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.18it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.99it/s]
2026-04-01 14:09:18,940 [INFO] [vad-silence] Final transcription 4.53 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.47it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 21.05it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.39it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.29it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 18.12it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 18.12it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 17.63it/s]
2026-04-01 14:10:10,669 [INFO] [vad-silence] Final transcription 4.20 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 18.56it/s]
2026-04-01 14:10:10,742 [INFO]   → FINAL "I was gonna give the girl it was a Plastic Pen, design,"  [0.07s RTF 0.02x]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.46it/s]
2026-04-01 14:10:11,953 [INFO] [vad-silence] Final transcription 2.13 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 21.12it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.06it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.64it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 18.76it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 18.19it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 16.56it/s]
2026-04-01 14:10:17,412 [INFO] [vad-silence] Final transcription 10.14 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 18.33it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.47it/s]
2026-04-01 14:10:18,125 [INFO] [vad-silence] Final transcription 1.23 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.84it/s]
2026-04-01 14:10:19,148 [INFO] [vad-silence] Final transcription 0.78 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.15it/s]
2026-04-01 14:10:19,217 [INFO]   → FINAL "A church!"  [0.07s RTF 0.09x]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.40it/s]
2026-04-01 14:10:20,164 [INFO] [vad-silence] Final transcription 0.99 s …
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.43it/s]
2026-04-01 14:10:20,231 [INFO]   → FINAL "Ame!"  [0.07s RTF 0.07x]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.89it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.20it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 20.75it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.50it/s]
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 19.85it/s]
2026-04-01 14:11:07,564 [INFO] Client disconnected: 223.185.131.203:22081
2026-04-01 14:11:07,564 [INFO] [disconnect-flush] Final transcription 3.09 s …
INFO:     connection closed
Transcribing: 100%|██████████| 1/1 [00:00<00:00, 18.16it/s]
2026-04-01 14:11:07,639 [INFO]   → FINAL "Surven the beach!"  [0.07s RTF 0.02x]
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/app/server.py", line 207, in websocket_endpoint
    raw = await ws.receive_bytes()
  File "/usr/local/lib/python3.10/dist-packages/starlette/websockets.py", line 126, in receive_bytes
    self._raise_on_disconnect(message)
  File "/usr/local/lib/python3.10/dist-packages/starlette/websockets.py", line 113, in _raise_on_disconnect
    raise WebSocketDisconnect(message["code"], message.get("reason"))
starlette.websockets.WebSocketDisconnect: (<CloseCode.ABNORMAL_CLOSURE: 1006>, '')

During handling of the above exception, another exception occurred:
    await send(message)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/protocols/websockets/websockets_impl.py", line 360, in asgi_send
    raise RuntimeError(msg % message_type)
RuntimeError: Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response already completed.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
WARNING:  Invalid HTTP request received.
