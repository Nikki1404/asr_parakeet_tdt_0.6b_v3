#!/usr/bin/env python3
"""
Batch realtime transcription client for NVIDIA NIM WebSocket endpoint.

What it does:
- Reads audio files from an input folder
- Converts each file to 16 kHz mono PCM16 WAV using ffmpeg
- Streams 30-second chunks to a NIM realtime WebSocket endpoint
- Saves:
    <file_stem>_transcript.txt
    <file_stem>_latency.json

Latency JSON includes one entry per sent chunk.

Example:
python transcribe_realtime_batch.py \
  --ws-url "ws://localhost:9000/v1/realtime?intent=transcription" \
  --input-folder "/home/re_nikitav/audio_maria" \
  --output-folder "/home/re_nikitav/transcription_results" \
  --model "parakeet-ctc-0.6b-es" \
  --language "es"

Install:
pip install websockets
sudo apt-get update && sudo apt-get install -y ffmpeg
"""

import argparse
import asyncio
import base64
import datetime as dt
import json
import subprocess
import tempfile
import time
import traceback
import wave
from pathlib import Path
from typing import Any


# =========================================================
# AUDIO / STREAM CONFIG
# =========================================================
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = 2  # PCM16
WAV_HEADER_SIZE = 44

SEND_CHUNK_SEC = 30
SEND_CHUNK_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * SEND_CHUNK_SEC

SUPPORTED_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


# =========================================================
# HELPERS
# =========================================================
def now_iso() -> str:
    return dt.datetime.now().isoformat()


def ensure_ffmpeg() -> None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg not found")
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required but was not found in PATH. "
            "Install it first."
        ) from exc


def convert_to_wav(src_path: Path, wav_path: Path) -> float:
    """
    Convert any supported input file to 16k mono PCM16 WAV.
    Returns audio duration in seconds.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        str(NUM_CHANNELS),
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed for {src_path.name}\n{result.stderr}"
        )

    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration_sec = frames / float(rate)

    return duration_sec


def save_results(
    output_folder: Path,
    audio_path: Path,
    transcript_text: str,
    result_json: dict[str, Any],
) -> None:
    stem = audio_path.stem
    transcript_path = output_folder / f"{stem}_transcript.txt"
    latency_path = output_folder / f"{stem}_latency.json"

    transcript_path.write_text(transcript_text, encoding="utf-8")
    latency_path.write_text(
        json.dumps(result_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"SAVED -> {transcript_path}")
    print(f"SAVED -> {latency_path}")


# =========================================================
# CORE TRANSCRIPTION
# =========================================================
async def transcribe_file(
    audio_path: Path,
    ws_url: str,
    output_folder: Path,
    model: str,
    language: str,
    timeout_sec: int,
    ping_interval: int,
    ping_timeout: int,
) -> None:
    import websockets

    print(f"\nSTARTING -> {audio_path.name}")

    output_folder.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nim_asr_") as tmpdir:
        wav_path = Path(tmpdir) / f"{audio_path.stem}.wav"

        print(f"Converting -> {audio_path.name}")
        duration_sec = convert_to_wav(audio_path, wav_path)
        print(f"Converted -> {audio_path.name} | Duration: {duration_sec:.1f} sec")

        final_transcript_parts: list[str] = []
        all_events: list[dict[str, Any]] = []
        chunk_metrics: list[dict[str, Any]] = []

        start_time = time.time()
        first_response_time_ms = None
        first_final_time_ms = None
        response_num = 0

        # We track "first response after chunk send" for each chunk.
        # This gives a per-chunk latency metric.
        pending_chunk_indexes: list[int] = []

        try:
            async with websockets.connect(
                ws_url,
                ping_interval=ping_interval,
                ping_timeout=ping_timeout,
                max_size=64 * 1024 * 1024,
                close_timeout=30,
            ) as ws:

                # -----------------------------------------
                # 1) START SESSION
                # -----------------------------------------
                session_event = {
                    "event_id": f"session_{int(time.time() * 1000)}",
                    "type": "transcription_session.update",
                    "session": {
                        "modalities": ["text"],
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "language": language,
                            "model": model,
                            "prompt": "",
                        },
                        "input_audio_params": {
                            "sample_rate_hz": SAMPLE_RATE,
                            "num_channels": NUM_CHANNELS,
                        },
                    },
                }

                await ws.send(json.dumps(session_event))
                print(f"Session initialized -> {audio_path.name}")

                # -----------------------------------------
                # 2) SENDER
                # -----------------------------------------
                async def sender() -> None:
                    with open(wav_path, "rb") as fh:
                        fh.read(WAV_HEADER_SIZE)  # skip header

                        chunk_num = 0
                        while True:
                            chunk = fh.read(SEND_CHUNK_BYTES)
                            if not chunk:
                                break

                            chunk_num += 1
                            sent_at = time.time()
                            chunk_start_sec = (chunk_num - 1) * SEND_CHUNK_SEC
                            chunk_end_sec = min(chunk_num * SEND_CHUNK_SEC, duration_sec)

                            event = {
                                "event_id": f"chunk_{chunk_num:06d}",
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("utf-8"),
                            }

                            await ws.send(json.dumps(event))

                            chunk_metrics.append(
                                {
                                    "chunk_num": chunk_num,
                                    "audio_start_sec": round(chunk_start_sec, 3),
                                    "audio_end_sec": round(chunk_end_sec, 3),
                                    "bytes_sent": len(chunk),
                                    "sent_at_iso": dt.datetime.fromtimestamp(sent_at).isoformat(),
                                    "sent_at_epoch": sent_at,
                                    "first_response_latency_ms": None,
                                    "first_response_elapsed_ms": None,
                                    "first_response_event_type": None,
                                }
                            )
                            pending_chunk_indexes.append(len(chunk_metrics) - 1)

                            print(
                                f"{audio_path.name} | SENT CHUNK {chunk_num} | "
                                f"{chunk_start_sec:.0f}-{chunk_end_sec:.0f}s | "
                                f"{len(chunk)} bytes"
                            )

                    # finalize stream
                    await ws.send(
                        json.dumps(
                            {
                                "event_id": f"commit_{int(time.time() * 1000)}",
                                "type": "input_audio_buffer.commit",
                            }
                        )
                    )
                    await ws.send(
                        json.dumps(
                            {
                                "event_id": f"done_{int(time.time() * 1000)}",
                                "type": "input_audio_buffer.done",
                            }
                        )
                    )
                    print(f"{audio_path.name} | STREAM DONE")

                # -----------------------------------------
                # 3) RECEIVER
                # -----------------------------------------
                async def receiver() -> None:
                    nonlocal first_response_time_ms
                    nonlocal first_final_time_ms
                    nonlocal response_num

                    while True:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
                            now = time.time()
                            elapsed_ms = (now - start_time) * 1000.0

                            try:
                                msg = json.loads(raw)
                            except Exception:
                                continue

                            msg_type = msg.get("type", "")
                            response_num += 1

                            if first_response_time_ms is None:
                                first_response_time_ms = round(elapsed_ms, 2)

                            # Assign first response latency to oldest pending chunk
                            # when we see any meaningful response event.
                            meaningful_response = bool(msg_type)
                            if meaningful_response and pending_chunk_indexes:
                                oldest_idx = pending_chunk_indexes[0]
                                if chunk_metrics[oldest_idx]["first_response_latency_ms"] is None:
                                    sent_epoch = chunk_metrics[oldest_idx]["sent_at_epoch"]
                                    chunk_metrics[oldest_idx]["first_response_latency_ms"] = round(
                                        (now - sent_epoch) * 1000.0, 2
                                    )
                                    chunk_metrics[oldest_idx]["first_response_elapsed_ms"] = round(
                                        elapsed_ms, 2
                                    )
                                    chunk_metrics[oldest_idx]["first_response_event_type"] = msg_type
                                    pending_chunk_indexes.pop(0)

                            transcript = msg.get("transcript", "").strip()
                            is_final = msg_type.endswith(".completed")
                            is_delta = msg_type.endswith(".delta")

                            all_events.append(
                                {
                                    "response_num": response_num,
                                    "event_type": msg_type,
                                    "elapsed_ms": round(elapsed_ms, 2),
                                    "is_final": is_final,
                                    "is_delta": is_delta,
                                    "transcript_preview": transcript[:120] if transcript else "",
                                }
                            )

                            # Final transcript only
                            if transcript and is_final:
                                final_transcript_parts.append(transcript)

                                if first_final_time_ms is None:
                                    first_final_time_ms = round(elapsed_ms, 2)

                                print(
                                    f"{audio_path.name} | FINAL {response_num} | "
                                    f"ELAPSED {elapsed_ms:.0f} ms"
                                )

                                # stop when server marks last result
                                if msg.get("is_last_result", False):
                                    print(f"{audio_path.name} | LAST FINAL RECEIVED")
                                    break

                            if msg_type == "error":
                                print(f"{audio_path.name} | SERVER ERROR -> {msg}")
                                break

                        except asyncio.TimeoutError:
                            print(f"{audio_path.name} | timeout waiting for server response")
                            break

                        except websockets.exceptions.ConnectionClosed:
                            print(f"{audio_path.name} | websocket closed")
                            break

                await asyncio.gather(sender(), receiver())

        except Exception as exc:
            print(f"FAILED -> {audio_path.name}")
            print(str(exc))
            traceback.print_exc()
            return

        total_time_sec = time.time() - start_time
        transcript_text = "\n".join(final_transcript_parts)

        result_json = {
            "audio_file": str(audio_path),
            "audio_duration_sec": round(duration_sec, 3),
            "total_processing_time_sec": round(total_time_sec, 3),
            "timestamp": now_iso(),
            "endpoint": ws_url,
            "model": model,
            "language": language,
            "sample_rate": SAMPLE_RATE,
            "chunk_sec": SEND_CHUNK_SEC,
            "chunk_bytes": SEND_CHUNK_BYTES,
            "ttfb_ms": first_response_time_ms,
            "ttft_ms": first_final_time_ms,
            "summary": {
                "total_chunks_sent": len(chunk_metrics),
                "total_responses": len(all_events),
                "total_final_segments": sum(1 for e in all_events if e["is_final"]),
            },
            "chunk_latencies": [
                {
                    k: v
                    for k, v in metric.items()
                    if k != "sent_at_epoch"
                }
                for metric in chunk_metrics
            ],
            "events": all_events,
        }

        save_results(output_folder, audio_path, transcript_text, result_json)

        print(
            f"COMPLETED -> {audio_path.name} | "
            f"TOTAL {total_time_sec:.1f} sec | "
            f"CHUNKS {len(chunk_metrics)}"
        )


# =========================================================
# BATCH
# =========================================================
async def run_batch(
    ws_url: str,
    input_folder: Path,
    output_folder: Path,
    model: str,
    language: str,
    timeout_sec: int,
    ping_interval: int,
    ping_timeout: int,
) -> None:
    files = sorted(
        [
            f
            for f in input_folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ]
    )

    total_files = len(files)

    print("=" * 90)
    print(f"TOTAL FILES = {total_files}")
    print("MODE = SEQUENTIAL")
    print(f"WS URL = {ws_url}")
    print(f"INPUT = {input_folder}")
    print(f"OUTPUT = {output_folder}")
    print(f"MODEL = {model}")
    print(f"LANGUAGE = {language}")
    print(f"SEND = {SEND_CHUNK_SEC} sec chunks")
    print("=" * 90)

    if total_files == 0:
        print("No supported audio files found.")
        return

    for idx, file_path in enumerate(files, start=1):
        print(f"\n[{idx}/{total_files}] {file_path.name}")
        await transcribe_file(
            audio_path=file_path,
            ws_url=ws_url,
            output_folder=output_folder,
            model=model,
            language=language,
            timeout_sec=timeout_sec,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
        )

    print("\nALL FILES COMPLETED")


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ws-url",
        required=True,
        help='Example: "ws://localhost:9000/v1/realtime?intent=transcription"',
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Folder containing audio files",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder where transcript/json files will be saved",
    )
    parser.add_argument(
        "--model",
        default="parakeet-ctc-0.6b-es",
        help="Model name for session config",
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Language code, e.g. es or multi",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Receiver timeout per file",
    )
    parser.add_argument(
        "--ping-interval",
        type=int,
        default=20,
        help="WebSocket ping interval",
    )
    parser.add_argument(
        "--ping-timeout",
        type=int,
        default=60,
        help="WebSocket ping timeout",
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    args = parse_args()

    ensure_ffmpeg()

    asyncio.run(
        run_batch(
            ws_url=args.ws_url,
            input_folder=Path(args.input_folder),
            output_folder=Path(args.output_folder),
            model=args.model,
            language=args.language,
            timeout_sec=args.timeout_sec,
            ping_interval=args.ping_interval,
            ping_timeout=args.ping_timeout,
        )
    )


python transcribe_nim_session_batch.py \
  --base-url http://localhost:9000 \
  --input-folder /home/re_nikitav/audio_maria \
  --output-folder /home/re_nikitav/parakeet_es_results
