import argparse
import asyncio
import json
import time
import statistics
import traceback
from pathlib import Path

import numpy as np
import librosa
import websockets
import pandas as pd
from jiwer import wer


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000

CHUNK_MS = 250
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_MS / 1000)
CHUNK_BYTES = CHUNK_FRAMES * 2

SEND_DELAY_MS = 50

INPUT_FOLDER = Path(
    "/home/re_nikitav/parakeet-asr-multilingual/audio_samples"
)

OUTPUT_FILE = "nemotron_final_report.xlsx"

WEBSOCKET_ADDRESS = "ws://127.0.0.1:8002/asr/realtime-custom-vad"

BACKEND = "nemotron"


# =========================
# AUDIO LOADER
# =========================
def load_wav_as_pcm16(filepath):
    audio, sr = librosa.load(
        str(filepath),
        sr=SAMPLE_RATE,
        mono=True
    )

    duration_sec = len(audio) / SAMPLE_RATE

    pcm16 = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16).tobytes()

    return pcm16, duration_sec


# =========================
# REFERENCE
# =========================
def load_reference_text(filepath):
    txt_path = filepath.with_suffix(".txt")

    if txt_path.exists():
        return txt_path.read_text(
            encoding="utf-8"
        ).strip()

    return ""


# =========================
# WER
# =========================
def calculate_wer(ref, hyp):
    if not ref or not hyp:
        return None

    return round(
        wer(
            ref.lower(),
            hyp.lower()
        ) * 100,
        2
    )


# =========================
# TRANSCRIBE ONE FILE
# =========================
async def transcribe_file(filepath):
    pcm, duration_sec = load_wav_as_pcm16(filepath)

    final_transcript_parts = []
    latencies = []

    start_time = time.time()

    first_response_time = None
    first_final_time = None
    response_num = 0

    async with websockets.connect(
        WEBSOCKET_ADDRESS,
        max_size=None
    ) as ws:

        await ws.send(json.dumps({
            "backend": BACKEND,
            "sample_rate": SAMPLE_RATE
        }))

        offset = 0

        send_start_time = time.time()

        async def sender():
            nonlocal offset

            chunk_num = 0

            while offset < len(pcm):
                chunk = pcm[
                    offset: offset + CHUNK_BYTES
                ]

                offset += CHUNK_BYTES
                chunk_num += 1

                await ws.send(chunk)

                await asyncio.sleep(
                    SEND_DELAY_MS / 1000
                )

            await asyncio.sleep(0.5)

            # EOS
            await ws.send(b"")

        async def receiver():
            nonlocal first_response_time
            nonlocal first_final_time
            nonlocal response_num

            while True:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(),
                        timeout=120
                    )

                    now = time.time()

                    msg = json.loads(raw)

                    typ = msg.get("type")
                    text = msg.get("text", "")

                    response_num += 1

                    elapsed_ms = (
                        now - start_time
                    ) * 1000

                    if first_response_time is None:
                        first_response_time = elapsed_ms

                    if typ == "final":

                        if first_final_time is None:
                            first_final_time = elapsed_ms

                        final_transcript_parts.append(text)

                        latencies.append(elapsed_ms)

                except:
                    break

        await asyncio.gather(
            sender(),
            receiver()
        )

    total_time = time.time() - start_time

    transcript_text = "\n".join(
        final_transcript_parts
    )

    avg_latency = (
        statistics.mean(latencies)
        if latencies else None
    )

    reference_text = load_reference_text(filepath)

    calculated_wer = calculate_wer(
        reference_text,
        transcript_text
    )

    return {
        "file_name": filepath.name,
        "reference_txt": reference_text,
        "ttft_ms": round(first_final_time, 2) if first_final_time else None,
        "ttfb_ms": round(first_response_time, 2) if first_response_time else None,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else None,
        "wer": calculated_wer,
        "total_time_sec": round(total_time, 2),
        "transcription": transcript_text
    }


# =========================
# BATCH
# =========================
async def run_batch():
    files = sorted(INPUT_FOLDER.glob("*.wav"))

    rows = []

    for idx, file in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {file.name}")

        row = await transcribe_file(file)

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSAVED -> {OUTPUT_FILE}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    asyncio.run(run_batch())
