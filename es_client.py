"""
Parakeet ASR – Terminal Microphone / File Streaming Client

Features
- Microphone mode: streams live mic audio to a WebSocket server
- File mode: streams audio file(s) chunk by chunk at real-time speed
- Supports server messages:
    {"type": "partial", "text": "...", "t_start": 123}
    {"type": "transcript", "text": "...", "t_start": 123}

Examples
    # List input devices
    python client.py --list

    # Microphone mode (local server)
    python client.py
    python client.py --host 34.118.200.125 --port 8001
    python client.py --device 2

    # Microphone mode (Cloud Run WebSocket)
    python client.py --url wss://parakeet-custom-vad-150916788856.us-central1.run.app/ws

    # File mode
    python client.py --file english.wav
    python client.py --file english.wav spanish.mp3 --speed 1.5

    # File mode against Cloud Run WebSocket
    python client.py --url wss://parakeet-custom-vad-150916788856.us-central1.run.app/ws --file english.wav
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

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

SAMPLE_RATE = 16_000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_MS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║        Parakeet-TDT-0.6B-v3  |  Real-Time ASR Client        ║
║          Auto language detection (EN / ES / more)           ║
║      Local WS or Cloud Run WSS endpoint supported           ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
Commands
  q         Quit
  d         List available audio input devices

Notes
  - Partial transcripts are shown while speaking.
  - Final transcript arrives after endpointing on the server.
"""

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def build_ws_uri(url: Optional[str], host: str, port: int) -> str:
    if url:
        return url
    return f"ws://{host}:{port}/ws"


# ---------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------


class MicCapture:
    """Captures microphone audio into a thread-safe queue."""

    def __init__(self, device_index: Optional[int] = None):
        self.queue: Queue[bytes] = Queue(maxsize=512)
        self._stop = threading.Event()
        self._device = device_index
        self._thread = threading.Thread(target=self._capture, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

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
                data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                try:
                    self.queue.put(data, timeout=0.1)
                except Exception:
                    pass

        except OSError as exc:
            print(f"\n[ERROR] Audio capture failed: {exc}", file=sys.stderr)
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
            print(f"  [{i}] {info['name']}  ({int(info['defaultSampleRate'])} Hz)")
    pa.terminate()
    print()


# ---------------------------------------------------------------------
# Audio loading for file mode
# ---------------------------------------------------------------------


def load_audio_as_16k_pcm(path: str) -> bytes:
    """
    Load audio file -> raw PCM16 mono @ 16kHz.
    Tries soundfile first, then pydub.
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
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(orig_sr, target_sr)
        return resample_poly(data, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError:
        pass

    duration = len(data) / orig_sr
    old_times = np.linspace(0, duration, len(data), endpoint=False)
    new_len = int(duration * target_sr)
    new_times = np.linspace(0, duration, new_len, endpoint=False)
    return np.interp(new_times, old_times, data).astype(np.float32)


def _float32_to_pcm16(data: np.ndarray) -> bytes:
    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------


async def _receiver(ws):
    """
    Receives partial + final transcripts and prints them.
    Supported server message types:
      - partial
      - transcript
    """
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

            if msg_type == "partial":
                last_partial = text
                if t_start is not None:
                    print(f"\r  ⏳ [{ts}] ({t_start} ms) {text:<100}", end="", flush=True)
                else:
                    print(f"\r  ⏳ [{ts}] {text:<100}", end="", flush=True)

            elif msg_type == "transcript":
                print(f"\r{' ' * 140}\r", end="")
                if t_start is not None:
                    print(f"[{ts}] Final  | first partial at {t_start} ms")
                else:
                    print(f"[{ts}] Final")
                print(f"  ✅  {text}")
                print("─" * 72)
                last_partial = ""

    except websockets.exceptions.ConnectionClosed:
        if last_partial:
            print()
        print("[Connection closed]")


# ---------------------------------------------------------------------
# stdin command reader
# ---------------------------------------------------------------------


def stdin_reader(cmd_queue: Queue):
    while True:
        try:
            line = sys.stdin.readline()
            if line:
                cmd_queue.put(line.strip())
        except Exception:
            break


# ---------------------------------------------------------------------
# Microphone mode
# ---------------------------------------------------------------------


async def run_mic(host: str, port: int, device_index: Optional[int], url: Optional[str] = None):
    uri = build_ws_uri(url, host, port)
    print(BANNER)
    print(f"Connecting to {uri} …", end=" ", flush=True)

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**23,
            open_timeout=30,
        ) as ws:
            print("connected ✓\n")
            print("🎙  Speak now – transcriptions appear below\n")
            print("─" * 72)

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

            for task in pending:
                task.cancel()

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
                elif cmd.lower() == "d":
                    print()
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


# ---------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------


async def run_files(host: str, port: int, files: List[str], speed: float, url: Optional[str] = None):
    uri = build_ws_uri(url, host, port)

    print(BANNER)
    print(f"  Mode   : FILE streaming at {speed:.1f}× real-time speed")
    print(f"  Server : {uri}")
    print(f"  Files  : {', '.join(Path(f).name for f in files)}\n")

    for filepath in files:
        p = Path(filepath)
        if not p.exists():
            print(f"  [SKIP] File not found: {filepath}\n")
            continue

        print(f"  Loading {p.name} …", end=" ", flush=True)
        try:
            pcm = load_audio_as_16k_pcm(str(p))
        except RuntimeError as exc:
            print(f"FAILED ✗\n  {exc}\n")
            continue

        duration_sec = len(pcm) / 2 / SAMPLE_RATE
        print(f"{duration_sec:.1f}s ✓")

        print(f"\n{'═' * 72}")
        print(f"  📄 {p.name}  ({duration_sec:.1f}s)")
        print(f"{'═' * 72}\n")

        try:
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=20,
                max_size=2**23,
                open_timeout=30,
            ) as ws:
                recv_task = asyncio.create_task(_receiver(ws))
                await _file_sender(ws, pcm, speed)

                # give server time to endpoint trailing audio
                await asyncio.sleep(5.0)

                await ws.close()
                await asyncio.sleep(0.2)

                if not recv_task.done():
                    recv_task.cancel()

        except (OSError, websockets.exceptions.WebSocketException) as exc:
            print(f"\n  [ERROR] WebSocket failed: {exc}")

        print()

    print("  All files processed.")


async def _file_sender(ws, pcm: bytes, speed: float):
    """
    Stream PCM bytes chunk by chunk.
    speed=1.0 => real-time
    speed=2.0 => 2x faster than real-time
    """
    if speed <= 0:
        raise ValueError("--speed must be > 0")

    chunk_delay = (CHUNK_MS / 1000.0) / speed
    offset = 0

    while offset + CHUNK_BYTES <= len(pcm):
        chunk = pcm[offset : offset + CHUNK_BYTES]
        offset += CHUNK_BYTES
        await ws.send(chunk)
        await asyncio.sleep(chunk_delay)

    leftover = pcm[offset:]
    if leftover:
        padded = leftover + bytes(CHUNK_BYTES - len(leftover))
        await ws.send(padded)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parakeet ASR client – microphone or file mode"
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Full WebSocket URL, e.g. wss://parakeet-custom-vad-150916788856.us-central1.run.app/ws",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Server port (default: 8001)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Mic device index for mic mode (see --list)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List audio input devices and exit",
    )
    parser.add_argument(
        "--file",
        nargs="+",
        metavar="FILE",
        help="File mode: stream one or more audio files (wav/mp3/flac/ogg/etc)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="File mode playback speed multiplier (default: 1.0 = real-time)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        list_devices()
        sys.exit(0)

    if args.file:
        asyncio.run(run_files(args.host, args.port, args.file, args.speed, args.url))
    else:
        asyncio.run(run_mic(args.host, args.port, args.device, args.url))
