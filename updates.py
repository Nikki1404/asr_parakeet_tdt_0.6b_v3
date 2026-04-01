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

# SEND 30 SEC AUDIO BLOCKS
SEND_CHUNK_SEC = 30
SEND_CHUNK_SAMPLES = SAMPLE_RATE * SEND_CHUNK_SEC
SEND_CHUNK_BYTES = SEND_CHUNK_SAMPLES * 2

# STREAMING DELAY BETWEEN SENDS
SEND_DELAY_MS = 250

INPUT_FOLDER = Path("/home/re_nikitav/parakeet-asr-multilingual/audio_maria")
OUTPUT_FOLDER = Path("transcription_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)


# =========================
# AUDIO LOADER
# =========================
def load_mp3_as_pcm16(filepath):
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

    print(
        f"Loaded {filepath.name} | "
        f"Duration: {duration_sec:.1f} sec"
    )

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
async def transcribe_file(filepath, host, port):
    uri = f"ws://{host}:{port}/ws"

    print(f"\nSTARTING -> {filepath.name}")

    pcm, duration_sec = load_mp3_as_pcm16(filepath)

    final_transcript_parts = []
    latencies = []

    start_time = time.time()

    first_response_time = None
    first_final_time = None
    response_num = 0

    last_chunk_sent_time = None

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**24
        ) as ws:

            async def sender():
                nonlocal last_chunk_sent_time

                offset = 0

                while offset < len(pcm):
                    chunk = pcm[
                        offset: offset + SEND_CHUNK_BYTES
                    ]
                    offset += SEND_CHUNK_BYTES

                    last_chunk_sent_time = time.time()

                    await ws.send(chunk)

                    # fixed 250 ms pacing
                    await asyncio.sleep(
                        SEND_DELAY_MS / 1000
                    )

                # allow decoder to finalize
                await asyncio.sleep(1.0)

                await ws.send(
                    json.dumps({"cmd": "flush"})
                )

            async def receiver():
                nonlocal first_response_time
                nonlocal first_final_time
                nonlocal response_num
                nonlocal last_chunk_sent_time

                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=120
                        )

                        now = time.time()

                        msg = json.loads(raw)

                        text = msg.get("text", "")
                        msg_type = msg.get("type", "")

                        elapsed_ms = (
                            now - start_time
                        ) * 1000

                        if last_chunk_sent_time is not None:
                            response_latency_ms = (
                                now - last_chunk_sent_time
                            ) * 1000
                        else:
                            response_latency_ms = 0

                        response_num += 1

                        if first_response_time is None:
                            first_response_time = elapsed_ms

                        is_final = (
                            msg_type == "transcript"
                        )

                        if is_final and first_final_time is None:
                            first_final_time = elapsed_ms

                        # SAVE ONLY FINAL
                        if text and is_final:
                            final_transcript_parts.append(
                                text
                            )

                        latencies.append({
                            "response_num": response_num,
                            "latency_ms": round(
                                response_latency_ms, 2
                            ),
                            "elapsed_ms": round(
                                elapsed_ms, 2
                            ),
                            "is_final": is_final
                        })

                        if is_final:
                            print(
                                f"{filepath.name} | "
                                f"FINAL {response_num} | "
                                f"LATENCY {response_latency_ms:.0f} ms"
                            )

                    except asyncio.TimeoutError:
                        print(
                            f"{filepath.name} | timeout"
                        )
                        break

                    except websockets.exceptions.ConnectionClosed:
                        print(
                            f"{filepath.name} | closed"
                        )
                        break

            await asyncio.gather(
                sender(),
                receiver()
            )

    except Exception as e:
        print(f"FAILED -> {filepath.name}")
        print(str(e))
        traceback.print_exc()
        return

    total_time = time.time() - start_time

    # FINAL COMPLETE TRANSCRIPT ONLY
    transcript_text = "\n".join(
        final_transcript_parts
    )

    latency_values = [
        x["latency_ms"] for x in latencies
    ]

    result_json = {
        "audio_file": str(filepath),
        "audio_duration_sec": duration_sec,
        "total_processing_time_sec": round(
            total_time, 2
        ),
        "timestamp": datetime.now().isoformat(),
        "model": "parakeet-tdt-0.6b-v3",
        "ttfb_ms": round(
            first_response_time, 2
        ) if first_response_time else None,
        "ttft_ms": round(
            first_final_time, 2
        ) if first_final_time else None,
        "summary": {
            "avg_latency_ms": round(
                statistics.mean(latency_values), 2
            ) if latency_values else 0
        }
    }

    save_results(
        filepath,
        transcript_text,
        result_json
    )

    print(
        f"COMPLETED -> {filepath.name} | "
        f"TOTAL {total_time:.1f} sec"
    )


# =========================
# BATCH RUNNER
# =========================
async def run_batch(host, port):
    files = sorted(INPUT_FOLDER.glob("*.mp3"))

    total_files = len(files)

    print("=" * 80)
    print(f"TOTAL FILES = {total_files}")
    print("MODE = SEQUENTIAL")
    print("SEND = 30 sec")
    print("DELAY = 250 ms")
    print("=" * 80)

    for idx, file in enumerate(files, start=1):
        print(f"\n[{idx}/{total_files}] {file.name}")

        await transcribe_file(
            filepath=file,
            host=host,
            port=port
        )

    print("\nALL FILES COMPLETED")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8001)

    return parser.parse_args()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        run_batch(
            args.host,
            args.port
        )
    )


these are all the endpoints 

INFO 2026-03-31 08:51:16.390 http_api.py:201] Readying serving endpoints:
  0.0.0.0:9000/v1/audio/list_voices (GET)
  0.0.0.0:9000/v1/audio/synthesize (POST)
  0.0.0.0:9000/v1/audio/synthesize_online (POST)
  0.0.0.0:9000/v1/audio/transcriptions (POST)
  0.0.0.0:9000/v1/audio/translations (POST)
  0.0.0.0:9000/v1/health/live (GET)
  0.0.0.0:9000/v1/health/ready (GET)
  0.0.0.0:9000/v1/metrics (GET)
  0.0.0.0:9000/v1/license (GET)
  0.0.0.0:9000/v1/metadata (GET)
  0.0.0.0:9000/v1/version (GET)
  0.0.0.0:9000/v1/manifest (GET)
  0.0.0.0:9000/v1/realtime/health (GET)
  0.0.0.0:9000/v1/realtime/transcription_sessions (POST)
  0.0.0.0:9000/v1/realtime/synthesis_sessions (POST)

root@cx-asr-test:/home/re_nikitav# gcloud compute scp /home/re_nikitav/audio_maria.zip re_nikitav@cx-asr-v2:/home/re_nikitav/ --zone=us-central1-a 
WARNING: The private SSH key file for gcloud does not exist.
WARNING: The public SSH key file for gcloud does not exist.
WARNING: You do not have an SSH key for gcloud.
WARNING: SSH keygen will be executed to generate a key.
Generating public/private rsa key pair.
Enter passphrase (empty for no passphrase): 


                  /v1/realtime?intent=transcription"
