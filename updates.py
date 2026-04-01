import argparse
import asyncio
import json
import statistics
import time
import traceback
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import websockets


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000

# Faster than 30 ms:
# server will split this into 30 ms VAD frames internally
CHUNK_MS = 1000
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2

INPUT_FOLDER = Path(r"C:/Users/re_nikitav/Downloads/audios/audios")
OUTPUT_FOLDER = Path("transcription_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Small yield to avoid overflooding socket buffers on very long files
SEND_YIELD_EVERY_N_CHUNKS = 8


# =========================
# AUDIO LOADER
# =========================
def load_mp3_as_pcm16(filepath: Path):
    print(f"Loading audio -> {filepath.name}")

    audio, _ = librosa.load(
        str(filepath),
        sr=SAMPLE_RATE,
        mono=True
    )

    duration_sec = len(audio) / SAMPLE_RATE
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    print(f"Loaded {filepath.name} | Duration: {duration_sec/60:.1f} min")
    return pcm16, duration_sec


# =========================
# FILE SELECTION
# =========================
def get_target_files(mode: str):
    all_files = sorted(INPUT_FOLDER.glob("*.mp3"))

    if mode == "maria":
        files = sorted(INPUT_FOLDER.glob("maria*.mp3"))
    elif mode == "rest":
        files = [f for f in all_files if not f.name.lower().startswith("maria")]
    elif mode == "all":
        files = all_files
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return files


# =========================
# SAVE RESULTS
# =========================
def save_results(filepath: Path, transcript_text: str, result_json: dict):
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
async def transcribe_file(filepath: Path, host: str, port: int):
    uri = f"ws://{host}:{port}/ws"

    print(f"\nSTARTING -> {filepath.name}")

    pcm, duration_sec = load_mp3_as_pcm16(filepath)

    transcript_parts = []
    latencies = []

    start_time = time.time()
    first_response_time = None
    first_final_time = None
    detected_language = None
    response_num = 0

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**24,
            close_timeout=30
        ) as ws:

            async def sender():
                offset = 0
                sent_chunks = 0

                while offset < len(pcm):
                    chunk = pcm[offset: offset + CHUNK_BYTES]
                    offset += CHUNK_BYTES

                    if len(chunk) < CHUNK_BYTES:
                        chunk += bytes(CHUNK_BYTES - len(chunk))

                    await ws.send(chunk)
                    sent_chunks += 1

                    # Tiny cooperative yield, but no realtime throttling
                    if sent_chunks % SEND_YIELD_EVERY_N_CHUNKS == 0:
                        await asyncio.sleep(0)

                # Force final flush
                await asyncio.sleep(0.2)
                await ws.send(json.dumps({"cmd": "flush"}).encode("utf-8"))

            async def receiver():
                nonlocal first_response_time
                nonlocal first_final_time
                nonlocal detected_language
                nonlocal response_num

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=90)
                        now = time.time()

                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")

                        msg = json.loads(raw)

                        text = msg.get("text", "")
                        msg_type = msg.get("type", "")
                        lang = msg.get("language") or msg.get("detected_language")

                        latency_ms = (now - start_time) * 1000
                        response_num += 1

                        if first_response_time is None:
                            first_response_time = latency_ms

                        is_final = msg_type == "transcript"

                        if is_final and first_final_time is None:
                            first_final_time = latency_ms

                        if lang and detected_language is None:
                            detected_language = lang

                        # Save only final transcript text
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
                                f"{filepath.name} | FINAL {response_num} | "
                                f"{latency_ms:.0f} ms | {len(text.split())} words"
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
        "detected_language": detected_language,
        "ttfb_ms": first_response_time,
        "ttft_ms": first_final_time,
        "latencies": latencies,
        "summary": {
            "total_responses": len(latencies),
            "final_responses": sum(x["is_final"] for x in latencies),
            "avg_latency_ms": statistics.mean(latency_values) if latency_values else 0,
            "min_latency_ms": min(latency_values) if latency_values else 0,
            "max_latency_ms": max(latency_values) if latency_values else 0,
        }
    }

    save_results(filepath, transcript_text, result_json)

    ttfb_str = f"{first_response_time:.0f} ms" if first_response_time is not None else "NA"
    ttft_str = f"{first_final_time:.0f} ms" if first_final_time is not None else "NA"

    print(
        f"COMPLETED -> {filepath.name} | "
        f"TTFB: {ttfb_str} | TTFT: {ttft_str}"
    )


# =========================
# BATCH RUNNER
# =========================
async def run_batch(host: str, port: int, workers: int, mode: str):
    files = get_target_files(mode)
    total_files = len(files)

    print("=" * 80)
    print(f"MODE = {mode}")
    print(f"TOTAL FILES = {total_files}")
    print(f"WORKERS = {workers}")
    print(f"CHUNK_MS = {CHUNK_MS}")
    print("=" * 80)

    if not files:
        print("No matching files found.")
        return

    semaphore = asyncio.Semaphore(workers)
    completed = 0

    async def process_file(idx: int, file: Path):
        nonlocal completed
        async with semaphore:
            print(f"\n[{idx}/{total_files}] PROCESSING {file.name}")
            await transcribe_file(file, host, port)
            completed += 1
            print(f"PROGRESS -> {completed}/{total_files} completed")

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
        "--mode",
        choices=["maria", "rest", "all"],
        default="maria",
        help="maria = only maria*.mp3, rest = everything except maria*.mp3, all = all mp3s"
    )

    return parser.parse_args()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        run_batch(
            host=args.host,
            port=args.port,
            workers=args.workers,
            mode=args.mode
        )
    )
