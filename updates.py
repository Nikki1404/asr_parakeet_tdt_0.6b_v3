import argparse
import asyncio
import json
import time
import statistics
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import websockets


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2

INPUT_FOLDER = Path("downloads/audios/audios")
OUTPUT_FOLDER = Path("transcription_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)


# =========================
# AUDIO LOADER (NO FFMPEG)
# =========================
def load_mp3_as_pcm16(filepath):
    """
    Load MP3 file as 16k mono PCM16 bytes
    No ffmpeg required
    """
    print(f"Loading audio -> {filepath.name}")

    audio, sr = librosa.load(
        str(filepath),
        sr=SAMPLE_RATE,
        mono=True
    )

    duration_sec = len(audio) / SAMPLE_RATE

    pcm16 = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16).tobytes()

    print(f"Loaded {filepath.name} | Duration: {duration_sec/60:.1f} min")

    return pcm16, duration_sec


# =========================
# SAVE RESULTS
# =========================
def save_results(filepath, transcript_text, result_json):
    output_dir = OUTPUT_FOLDER / filepath.stem
    output_dir.mkdir(exist_ok=True)

    transcript_path = output_dir / f"{filepath.stem}_transcript.txt"
    latency_path = output_dir / f"{filepath.stem}_latency.json"

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    print(f"SAVED -> {filepath.name}")


# =========================
# TRANSCRIBE ONE FILE
# =========================
async def transcribe_file(filepath, host, port, speed):
    uri = f"ws://{host}:{port}/ws"

    print(f"\nSTARTING -> {filepath.name}")

    pcm, duration_sec = load_mp3_as_pcm16(filepath)

    transcript_parts = []
    latencies = []

    start_time = time.time()

    first_response_time = None
    first_final_time = None
    response_num = 0

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**24
        ) as ws:

            async def sender():
                offset = 0
                chunk_delay = (CHUNK_MS / 1000) / speed

                while offset < len(pcm):
                    chunk = pcm[offset: offset + CHUNK_BYTES]
                    offset += CHUNK_BYTES

                    if len(chunk) < CHUNK_BYTES:
                        chunk += bytes(CHUNK_BYTES - len(chunk))

                    await ws.send(chunk)
                    await asyncio.sleep(chunk_delay)

                await asyncio.sleep(0.5)

                await ws.send(json.dumps({"cmd": "flush"}))

            async def receiver():
                nonlocal first_response_time
                nonlocal first_final_time
                nonlocal response_num

                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=60
                        )

                        now = time.time()

                        msg = json.loads(raw)

                        text = msg.get("text", "")
                        msg_type = msg.get("type", "")

                        latency_ms = (now - start_time) * 1000

                        response_num += 1

                        if first_response_time is None:
                            first_response_time = latency_ms

                        is_final = msg_type == "transcript"

                        if is_final and first_final_time is None:
                            first_final_time = latency_ms

                        if text:
                            transcript_parts.append(text)

                        latencies.append({
                            "response_num": response_num,
                            "latency_ms": latency_ms,
                            "is_final": is_final,
                            "words": len(text.split())
                        })

                        if is_final:
                            print(
                                f"{filepath.name} | "
                                f"FINAL {response_num} | "
                                f"TTFT {latency_ms:.0f} ms"
                            )

                    except asyncio.TimeoutError:
                        print(f"{filepath.name} | receiver timeout")
                        break

                    except websockets.exceptions.ConnectionClosed:
                        print(f"{filepath.name} | connection closed")
                        break

            await asyncio.gather(sender(), receiver())

    except Exception as e:
        print(f"FAILED -> {filepath.name}")
        print(str(e))
        traceback.print_exc()
        return

    total_time = time.time() - start_time

    transcript_text = "\n".join(transcript_parts)

    latency_values = [x["latency_ms"] for x in latencies]

    result_json = {
        "audio_file": str(filepath),
        "audio_duration_sec": duration_sec,
        "total_processing_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
        "model": "parakeet-tdt-0.6b-v3",
        "ttfb_ms": first_response_time,
        "ttft_ms": first_final_time,
        "latencies": latencies,
        "summary": {
            "total_responses": len(latencies),
            "final_responses": sum(x["is_final"] for x in latencies),
            "avg_latency_ms": (
                statistics.mean(latency_values)
                if latency_values else 0
            ),
            "min_latency_ms": (
                min(latency_values)
                if latency_values else 0
            ),
            "max_latency_ms": (
                max(latency_values)
                if latency_values else 0
            )
        }
    }

    save_results(filepath, transcript_text, result_json)

    print(
        f"COMPLETED -> {filepath.name} | "
        f"TTFB: {first_response_time:.0f} ms | "
        f"TTFT: {first_final_time:.0f} ms"
    )


# =========================
# BATCH RUNNER
# =========================
async def run_batch(host, port, workers, speed):
    files = sorted(INPUT_FOLDER.glob("*.mp3"))

    total_files = len(files)

    print("=" * 80)
    print(f"TOTAL FILES = {total_files}")
    print(f"WORKERS = {workers}")
    print(f"SPEED = {speed}x")
    print("=" * 80)

    semaphore = asyncio.Semaphore(workers)

    async def process_file(idx, file):
        async with semaphore:
            print(f"\n[{idx}/{total_files}] PROCESSING {file.name}")
            await transcribe_file(file, host, port, speed)

    tasks = [
        asyncio.create_task(process_file(idx, file))
        for idx, file in enumerate(files, start=1)
    ]

    await asyncio.gather(*tasks)

    print("\nALL FILES COMPLETED")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8001)

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Recommended: 2"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="Recommended: 4x"
    )

    return parser.parse_args()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        run_batch(
            args.host,
            args.port,
            args.workers,
            args.speed
        )
    )


async def receiver():
    nonlocal first_response_time
    nonlocal first_final_time
    nonlocal response_num

    while True:
        try:
            raw = await asyncio.wait_for(
                ws.recv(),
                timeout=60
            )

            now = time.time()

            msg = json.loads(raw)

            text = msg.get("text", "")
            msg_type = msg.get("type", "")

            latency_ms = (now - start_time) * 1000

            response_num += 1

            if first_response_time is None:
                first_response_time = latency_ms

            is_final = msg_type == "transcript"

            if is_final and first_final_time is None:
                first_final_time = latency_ms

            # Save ONLY final transcript
            if text and is_final:
                transcript_parts.append(text)

            latencies.append({
                "response_num": response_num,
                "latency_ms": latency_ms,
                "is_final": is_final,
                "words": len(text.split())
            })

            if is_final:
                print(
                    f"{filepath.name} | "
                    f"FINAL {response_num} | "
                    f"TTFT {latency_ms:.0f} ms"
                )

        except asyncio.TimeoutError:
            print(f"{filepath.name} | receiver timeout")
            break

        except websockets.exceptions.ConnectionClosed:
            print(f"{filepath.name} | connection closed")
            break
