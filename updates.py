"""
Parakeet ASR – Batch Benchmark Client
Streams all MP3 files from downloads/audios/audios/ to the WebSocket server,
collects full transcriptions and per-response latencies.

Output (per file):
    results/<stem>/
        <stem>_latency.json
        <stem>_transcript.txt

Usage:
    python benchmark_client.py
    python benchmark_client.py --host 34.118.200.125 --port 8001
    python benchmark_client.py --host 34.118.200.125 --port 8001 --speed 2.0
    python benchmark_client.py --audio-dir path/to/audios --speed 2.0

Dependencies:
    pip install websockets soundfile numpy pydub
    apt install ffmpeg   (required for MP3 resampling)
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import websockets

# ── Audio config ───────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000
CHUNK_MS      = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000   # 480 samples
CHUNK_BYTES   = CHUNK_SAMPLES * 2                 # int16 → 2 bytes/sample

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_AUDIO_DIR = "downloads/audios/audios"
OUTPUT_ROOT       = "results"


# ── Audio loading ──────────────────────────────────────────────────────────────

def load_audio_as_16k_pcm(path: str) -> bytes:
    """Load any audio file → raw int16 LE PCM at 16 kHz mono."""
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
                f"  Tip: install ffmpeg for MP3 support."
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
    duration  = len(data) / orig_sr
    old_times = np.linspace(0, duration, len(data))
    new_times = np.linspace(0, duration, int(duration * target_sr))
    return np.interp(new_times, old_times, data).astype(np.float32)


def _float32_to_pcm16(data: np.ndarray) -> bytes:
    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


# ── Result container ───────────────────────────────────────────────────────────

class FileResult:
    def __init__(self, filepath: Path, audio_duration_sec: float):
        self.filepath                    = filepath
        self.audio_duration_sec          = audio_duration_sec
        self.timestamp                   = datetime.now().isoformat()
        self.detected_language           = "unknown"
        self.transcript_parts: List[str] = []
        self.latency_events: List[dict]  = []  # one entry per transcript response
        self.send_start: float           = 0.0
        self.total_processing_time_sec   = 0.0
        self.error: Optional[str]        = None


# ── WebSocket receiver ─────────────────────────────────────────────────────────

async def _receiver(ws, result: FileResult) -> None:
    """
    Collect every response from the server.
    Each 'transcript' message (is_final=True) is one latency event.
    Latency is measured from the moment the first chunk was sent.
    """
    response_num = 0

    try:
        async for raw in ws:
            now_ms = (time.perf_counter() - result.send_start) * 1000

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            msg_type = msg.get("type", "")
            text     = msg.get("text", "").strip()

            # Capture detected language if the server sends it
            if "language" in msg:
                result.detected_language = msg["language"]

            if msg_type == "transcript" and text:
                response_num += 1
                result.transcript_parts.append(text)
                result.latency_events.append({
                    "response_num": response_num,
                    "latency_ms":   now_ms,
                    "is_final":     True,
                    "words":        len(text.split()),
                })

            elif msg_type == "partial":
                # partials noted but not stored as latency events
                pass

    except websockets.exceptions.ConnectionClosed:
        pass


# ── WebSocket sender ───────────────────────────────────────────────────────────

async def _sender(ws, pcm: bytes, speed: float, result: FileResult) -> None:
    """Stream PCM to server; stamps result.send_start before the first chunk."""
    chunk_delay     = (CHUNK_MS / 1000) / speed
    offset          = 0
    result.send_start = time.perf_counter()

    while offset + CHUNK_BYTES <= len(pcm):
        chunk   = pcm[offset: offset + CHUNK_BYTES]
        offset += CHUNK_BYTES
        await ws.send(chunk)
        await asyncio.sleep(chunk_delay)

    # Send any leftover bytes (padded to full chunk)
    leftover = pcm[offset:]
    if leftover:
        await ws.send(leftover + bytes(CHUNK_BYTES - len(leftover)))

    # Force final transcript flush
    await asyncio.sleep(0.3)
    await ws.send(json.dumps({"cmd": "flush"}).encode())

    # Wait for server to finish all responses
    await asyncio.sleep(15)


# ── Per-file benchmark ─────────────────────────────────────────────────────────

async def benchmark_file(uri: str, filepath: Path, pcm: bytes, speed: float) -> FileResult:
    duration_sec = len(pcm) / 2 / SAMPLE_RATE
    result       = FileResult(filepath, duration_sec)
    wall_start   = time.time()

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=60,
            max_size=2 ** 24,   # 16 MB per frame — fine for 30ms PCM chunks
            open_timeout=30,
        ) as ws:
            send_task = asyncio.create_task(_sender(ws, pcm, speed, result))
            recv_task = asyncio.create_task(_receiver(ws, result))
            await asyncio.gather(send_task, recv_task, return_exceptions=True)

    except Exception as exc:
        result.error = str(exc)

    result.total_processing_time_sec = time.time() - wall_start
    return result


# ── Output writers ─────────────────────────────────────────────────────────────

def save_results(result: FileResult, output_root: str) -> Path:
    """
    Saves:
        results/<stem>/<stem>_latency.json
        results/<stem>/<stem>_transcript.txt
    Returns the folder path.
    """
    stem   = result.filepath.stem          # e.g. "spanish-1"  from  spanish-1.mp3
    folder = Path(output_root) / stem
    folder.mkdir(parents=True, exist_ok=True)

    # ── latency JSON ──────────────────────────────────────────────────────────
    latencies = result.latency_events
    if latencies:
        lat_values = [e["latency_ms"] for e in latencies]
        summary = {
            "total_responses": len(latencies),
            "final_responses": sum(1 for e in latencies if e["is_final"]),
            "avg_latency_ms":  sum(lat_values) / len(lat_values),
            "min_latency_ms":  min(lat_values),
            "max_latency_ms":  max(lat_values),
        }
    else:
        summary = {
            "total_responses": 0,
            "final_responses": 0,
            "avg_latency_ms":  None,
            "min_latency_ms":  None,
            "max_latency_ms":  None,
        }

    latency_payload = {
        "audio_file":                str(result.filepath),
        "audio_duration_sec":        result.audio_duration_sec,
        "total_processing_time_sec": result.total_processing_time_sec,
        "timestamp":                 result.timestamp,
        "model":                     "parakeet-tdt-0.6b-v3",
        "detected_language":         result.detected_language,
        "latencies":                 latencies,
        "summary":                   summary,
    }

    latency_path = folder / f"{stem}_latency.json"
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_payload, f, indent=2)

    # ── transcript TXT ────────────────────────────────────────────────────────
    transcript_path = folder / f"{stem}_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(" ".join(result.transcript_parts))

    return folder


# ── Batch runner ───────────────────────────────────────────────────────────────

async def run_benchmark(host: str, port: int, audio_dir: str, speed: float):
    uri        = f"ws://{host}:{port}/ws"
    audio_path = Path(audio_dir)

    if not audio_path.exists():
        print(f"[ERROR] Audio directory not found: {audio_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    mp3_files = sorted(audio_path.glob("*.mp3"))
    if not mp3_files:
        print(f"[ERROR] No MP3 files found in {audio_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    total = len(mp3_files)
    print(f"\n{'═' * 66}")
    print(f"  Parakeet ASR Benchmark Client")
    print(f"  Server     : {uri}")
    print(f"  Audio dir  : {audio_path.resolve()}")
    print(f"  Output dir : {Path(OUTPUT_ROOT).resolve()}/")
    print(f"  Files      : {total}")
    print(f"  Speed      : {speed:.1f}×")
    print(f"{'═' * 66}\n")

    for idx, filepath in enumerate(mp3_files, 1):
        size_mb = filepath.stat().st_size / (1024 ** 2)
        print(f"[{idx:>3}/{total}] {filepath.name}  ({size_mb:.1f} MB)", end="  ", flush=True)

        try:
            pcm = load_audio_as_16k_pcm(str(filepath))
        except RuntimeError as exc:
            print(f"LOAD FAILED ✗\n         {exc}\n")
            r = FileResult(filepath, 0)
            r.error = str(exc)
            save_results(r, OUTPUT_ROOT)
            continue

        duration_sec = len(pcm) / 2 / SAMPLE_RATE
        print(f"({duration_sec:.1f}s)", end="  ", flush=True)

        result = await benchmark_file(uri, filepath, pcm, speed)
        folder = save_results(result, OUTPUT_ROOT)

        if result.error:
            print(f"ERROR ✗  →  {result.error}")
        else:
            n       = len(result.latency_events)
            avg_lat = (
                sum(e["latency_ms"] for e in result.latency_events) / n if n else 0
            )
            preview = " ".join(result.transcript_parts)
            preview = (preview[:65] + "…") if len(preview) > 65 else preview
            print(
                f"✅  {n} responses | "
                f"avg={avg_lat:.0f}ms | "
                f"wall={result.total_processing_time_sec:.1f}s"
            )
            print(f"         📁 {folder}/")
            print(f"         ✏️  {preview}")

        print()

    print(f"{'═' * 66}")
    print(f"  All done! Results saved under: {Path(OUTPUT_ROOT).resolve()}/")
    print(f"{'═' * 66}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Parakeet ASR batch benchmark – per-file latency JSON + transcript TXT"
    )
    p.add_argument("--host",      default="localhost",       help="Server host (default: localhost)")
    p.add_argument("--port",      type=int, default=8001,    help="Server port (default: 8001)")
    p.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR, help=f"Directory with MP3 files (default: {DEFAULT_AUDIO_DIR})")
    p.add_argument("--speed",     type=float, default=2.0,   help="Streaming speed multiplier (default: 2.0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(args.host, args.port, args.audio_dir, args.speed))
