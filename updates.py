"""
Parakeet ASR – Batch Benchmark Client
Streams all MP3 files from downloads/audios/audios/ to the WebSocket server,
collects full transcriptions, and records TTFB / TTFT latency per file.

Output CSVs:
    transcriptions.csv  →  file_name | transcription
    latency.csv         →  file_name | ttfb_ms | ttft_ms

Usage:
    python benchmark_client.py
    python benchmark_client.py --host 34.118.200.125 --port 8001
    python benchmark_client.py --host 34.118.200.125 --port 8001 --speed 10.0
    python benchmark_client.py --audio-dir path/to/audios --speed 5.0

Dependencies:
    pip install websockets soundfile numpy pydub
    apt install ffmpeg   (required for MP3 resampling)
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import websockets

# ── Audio config
SAMPLE_RATE   = 16_000
CHUNK_MS      = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000   # 480 samples
CHUNK_BYTES   = CHUNK_SAMPLES * 2                 # int16 → 2 bytes/sample

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_AUDIO_DIR   = "downloads/audios/audios"
TRANSCRIPTIONS_CSV  = "transcriptions.csv"
LATENCY_CSV         = "latency.csv"


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
    # Numpy linear interpolation fallback
    duration  = len(data) / orig_sr
    old_times = np.linspace(0, duration, len(data))
    new_times = np.linspace(0, duration, int(duration * target_sr))
    return np.interp(new_times, old_times, data).astype(np.float32)


def _float32_to_pcm16(data: np.ndarray) -> bytes:
    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


# ── Per-file benchmark ─────────────────────────────────────────────────────────

class BenchmarkResult:
    def __init__(self, file_name: str):
        self.file_name      = file_name
        self.ttfb_ms: Optional[float] = None   # time to first byte (any server response)
        self.ttft_ms: Optional[float] = None   # time to first transcript (partial text)
        self.transcription  = ""
        self.error: Optional[str]     = None


async def _benchmark_receiver(
    ws,
    send_start_time: float,
    result: BenchmarkResult,
) -> None:
    """
    Receive messages from the server and record:
      TTFB – wall-clock ms from send_start until first message of any kind arrives.
      TTFT – wall-clock ms from send_start until first message with non-empty text.
    Collects the final transcript text (type == "transcript").
    """
    transcript_parts: List[str] = []

    try:
        async for raw in ws:
            now = time.perf_counter()
            elapsed_ms = (now - send_start_time) * 1000

            # TTFB: first byte / message received
            if result.ttfb_ms is None:
                result.ttfb_ms = elapsed_ms

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Raw binary ack or unknown — still counts as TTFB
                continue

            msg_type = msg.get("type", "")
            text     = msg.get("text", "").strip()

            # TTFT: first message that carries actual text
            if result.ttft_ms is None and text:
                result.ttft_ms = elapsed_ms

            if msg_type == "partial":
                pass  # already captured TTFT above

            elif msg_type == "transcript":
                if text:
                    transcript_parts.append(text)

    except websockets.exceptions.ConnectionClosed:
        pass

    result.transcription = " ".join(transcript_parts)


async def _benchmark_sender(ws, pcm: bytes, speed: float) -> float:
    """
    Stream PCM to the server at `speed`× real-time.
    Returns the perf_counter timestamp just before the first chunk is sent
    (used as the reference t=0 for latency measurements).
    """
    chunk_delay = (CHUNK_MS / 1000) / speed
    offset = 0

    # Record start time right before first send
    send_start = time.perf_counter()

    while offset + CHUNK_BYTES <= len(pcm):
        chunk   = pcm[offset: offset + CHUNK_BYTES]
        offset += CHUNK_BYTES
        await ws.send(chunk)
        await asyncio.sleep(chunk_delay)

    # Pad and send any leftover bytes
    leftover = pcm[offset:]
    if leftover:
        await ws.send(leftover + bytes(CHUNK_BYTES - len(leftover)))

    # Flush to force the final transcript
    await asyncio.sleep(0.3)
    await ws.send(json.dumps({"cmd": "flush"}).encode())

    return send_start


async def benchmark_file(
    uri: str,
    filepath: Path,
    pcm: bytes,
    speed: float,
) -> BenchmarkResult:
    result = BenchmarkResult(file_name=filepath.name)

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=60,
            max_size=2 ** 24,   # 16 MB – handles the 150 MB file's chunked frames
            open_timeout=30,
        ) as ws:
            # We need the sender to return its start-time to the receiver.
            # Use a shared list as a mutable container.
            start_time_holder: List[float] = []

            async def sender_wrapper():
                t = await _benchmark_sender(ws, pcm, speed)
                start_time_holder.append(t)
                # Give the receiver some extra time to collect the final transcript
                await asyncio.sleep(15)
                await ws.close()

            # Receiver needs the start time; we approximate it as "now" and refine
            # once the sender fires. Use a simple shared float via list.
            approx_start = time.perf_counter()

            async def receiver_wrapper():
                # Wait briefly so sender can record its precise start
                await asyncio.sleep(0.01)
                actual_start = start_time_holder[0] if start_time_holder else approx_start
                await _benchmark_receiver(ws, actual_start, result)

            recv_task = asyncio.create_task(receiver_wrapper())
            send_task = asyncio.create_task(sender_wrapper())

            await asyncio.gather(send_task, recv_task, return_exceptions=True)

    except Exception as exc:
        result.error = str(exc)

    return result


# ── Batch runner ───────────────────────────────────────────────────────────────

async def run_benchmark(host: str, port: int, audio_dir: str, speed: float):
    uri       = f"ws://{host}:{port}/ws"
    audio_path = Path(audio_dir)

    if not audio_path.exists():
        print(f"[ERROR] Audio directory not found: {audio_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    mp3_files = sorted(audio_path.glob("*.mp3"))
    if not mp3_files:
        print(f"[ERROR] No MP3 files found in {audio_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    total = len(mp3_files)
    print(f"\n{'═' * 64}")
    print(f"  Parakeet ASR Benchmark Client")
    print(f"  Server  : {uri}")
    print(f"  Dir     : {audio_path.resolve()}")
    print(f"  Files   : {total}")
    print(f"  Speed   : {speed:.1f}× real-time")
    print(f"{'═' * 64}\n")

    results: List[BenchmarkResult] = []

    for idx, filepath in enumerate(mp3_files, 1):
        size_mb = filepath.stat().st_size / (1024 ** 2)
        print(f"[{idx:>3}/{total}] {filepath.name}  ({size_mb:.1f} MB)", end="  ", flush=True)

        # Load audio
        try:
            pcm = load_audio_as_16k_pcm(str(filepath))
        except RuntimeError as exc:
            print(f"LOAD FAILED ✗\n         {exc}\n")
            r = BenchmarkResult(filepath.name)
            r.error = str(exc)
            results.append(r)
            continue

        duration_sec = len(pcm) / 2 / SAMPLE_RATE
        print(f"({duration_sec:.1f}s audio)", end="  ", flush=True)

        # Benchmark
        wall_start = time.time()
        result     = await benchmark_file(uri, filepath, pcm, speed)
        wall_elapsed = time.time() - wall_start

        results.append(result)

        if result.error:
            print(f"ERROR ✗  →  {result.error}")
        else:
            ttfb = f"{result.ttfb_ms:.0f}ms" if result.ttfb_ms is not None else "N/A"
            ttft = f"{result.ttft_ms:.0f}ms" if result.ttft_ms is not None else "N/A"
            preview = (result.transcription[:60] + "…") if len(result.transcription) > 60 else result.transcription
            print(f"TTFB={ttfb}  TTFT={ttft}  wall={wall_elapsed:.1f}s")
            print(f"         ✅  {preview}")

        print()

    # ── Write CSVs ────────────────────────────────────────────────────────────
    _write_transcriptions_csv(results)
    _write_latency_csv(results)

    print(f"\n{'═' * 64}")
    print(f"  Done! Results saved to:")
    print(f"    {Path(TRANSCRIPTIONS_CSV).resolve()}")
    print(f"    {Path(LATENCY_CSV).resolve()}")
    print(f"{'═' * 64}\n")

    # ── Summary stats ─────────────────────────────────────────────────────────
    successful = [r for r in results if r.error is None]
    ttfb_vals  = [r.ttfb_ms for r in successful if r.ttfb_ms is not None]
    ttft_vals  = [r.ttft_ms for r in successful if r.ttft_ms is not None]

    if ttfb_vals:
        print(f"  TTFB  avg={_avg(ttfb_vals):.0f}ms  "
              f"min={min(ttfb_vals):.0f}ms  max={max(ttfb_vals):.0f}ms")
    if ttft_vals:
        print(f"  TTFT  avg={_avg(ttft_vals):.0f}ms  "
              f"min={min(ttft_vals):.0f}ms  max={max(ttft_vals):.0f}ms")
    print(f"  Success: {len(successful)}/{total}\n")


def _avg(vals):
    return sum(vals) / len(vals)


# ── CSV writers ────────────────────────────────────────────────────────────────

def _write_transcriptions_csv(results: List[BenchmarkResult]):
    with open(TRANSCRIPTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "transcription"])
        for r in results:
            writer.writerow([r.file_name, r.transcription if not r.error else f"ERROR: {r.error}"])


def _write_latency_csv(results: List[BenchmarkResult]):
    with open(LATENCY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "ttfb_ms", "ttft_ms"])
        for r in results:
            writer.writerow([
                r.file_name,
                f"{r.ttfb_ms:.2f}" if r.ttfb_ms is not None else "N/A",
                f"{r.ttft_ms:.2f}" if r.ttft_ms is not None else "N/A",
            ])


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Parakeet ASR batch benchmark – streams MP3 files and records latency"
    )
    p.add_argument("--host",      default="localhost",         help="Server host (default: localhost)")
    p.add_argument("--port",      type=int, default=8001,      help="Server port (default: 8001)")
    p.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR,   help=f"Directory with MP3 files (default: {DEFAULT_AUDIO_DIR})")
    p.add_argument("--speed",     type=float, default=10.0,
                   help="Streaming speed multiplier (default: 10.0 = 10× faster than real-time, good for benchmarking)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(args.host, args.port, args.audio_dir, args.speed))



we need in this way 
filename_latency.json
{
  "audio_file": "audio/spanish-1.wav",
  "audio_duration_sec": 57.817687074829934,
  "total_processing_time_sec": 47.64312219619751,
  "timestamp": "2026-04-01T14:35:27.851017",
  "model": "nova-3",
  "language": "es",
  "latencies": [
    {
      "response_num": 1,
      "latency_ms": 3980.722665786743,
      "is_final": true,
      "words": 12
    },
    {
      "response_num": 2,
      "latency_ms": 6740.095853805542,
      "is_final": true,
      "words": 13
    },
    {
      "response_num": 3,
      "latency_ms": 9951.934337615967,
      "is_final": true,
      "words": 16
    },
    {
      "response_num": 4,
      "latency_ms": 17667.06657409668,
      "is_final": true,
      "words": 27
    },
    {
      "response_num": 5,
      "latency_ms": 20783.82396697998,
      "is_final": true,
      "words": 11
    },
    {
      "response_num": 6,
      "latency_ms": 25372.496843338013,
      "is_final": true,
      "words": 29
    },
    {
      "response_num": 7,
      "latency_ms": 29421.988248825073,
      "is_final": true,
      "words": 17
    },
    {
      "response_num": 8,
      "latency_ms": 32451.578378677368,
      "is_final": true,
      "words": 12
    },
    {
      "response_num": 9,
      "latency_ms": 42714.39003944397,
      "is_final": true,
      "words": 19
    },
    {
      "response_num": 10,
      "latency_ms": 42714.81513977051,
      "is_final": true,
      "words": 18
    },
    {
      "response_num": 11,
      "latency_ms": 43747.33901023865,
      "is_final": true,
      "words": 3
    },
    {
      "response_num": 12,
      "latency_ms": 46270.442485809326,
      "is_final": true,
      "words": 13
    }
  ],
  "summary": {
    "total_responses": 12,
    "final_responses": 12,
    "avg_latency_ms": 26818.05779536565,
    "min_latency_ms": 3980.722665786743,
    "max_latency_ms": 46270.442485809326
  }
}


and transcript as filename_transcript.txt
