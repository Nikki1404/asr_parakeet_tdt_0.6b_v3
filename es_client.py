import asyncio
import json
import time
import statistics
from pathlib import Path
from datetime import datetime

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

SEND_DELAY_MS = 80

INPUT_FOLDER = Path(
    "/home/re_nikitav/parakeet-asr-multilingual/audio_samples"
)

OUTPUT_FILE = (
    f"nemotron_final_report_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
)

WEBSOCKET_ADDRESS = "ws://127.0.0.1:8002/asr/realtime-custom-vad"

BACKEND = "nemotron"


# =========================
# AUDIO LOADER
# =========================
def load_wav_as_pcm16(filepath):
    print(f"Loading -> {filepath.name}", flush=True)

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
        f"{filepath.name} loaded | "
        f"{duration_sec:.2f} sec",
        flush=True
    )

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
        max_size=None,
        ping_interval=30,
        ping_timeout=120,
        close_timeout=30
    ) as ws:

        # backend config
        await ws.send(json.dumps({
            "backend": BACKEND,
            "sample_rate": SAMPLE_RATE
        }))

        offset = 0

        async def sender():
            nonlocal offset

            chunk_num = 0
            total_chunks = len(pcm) // CHUNK_BYTES + 1

            while offset < len(pcm):
                chunk = pcm[
                    offset: offset + CHUNK_BYTES
                ]

                offset += CHUNK_BYTES
                chunk_num += 1

                await ws.send(chunk)

                if chunk_num % 20 == 0:
                    print(
                        f"{filepath.name} | "
                        f"sent chunk {chunk_num}/{total_chunks}",
                        flush=True
                    )

                await asyncio.sleep(
                    SEND_DELAY_MS / 1000
                )

            # IMPORTANT FIX
            # trailing silence before EOS
            silence = b"\x00\x00" * int(
                SAMPLE_RATE * 0.6
            )

            await ws.send(silence)

            await asyncio.sleep(1.0)

            print(
                f"{filepath.name} | sending EOS",
                flush=True
            )

            await ws.send(b"")

        async def receiver():
            nonlocal first_response_time
            nonlocal first_final_time
            nonlocal response_num

            last_message_time = time.time()
            final_received = False

            while True:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(),
                        timeout=300
                    )

                    last_message_time = time.time()

                    now = time.time()

                    msg = json.loads(raw)

                    typ = msg.get("type")
                    text = msg.get("text", "")

                    response_num += 1

                    elapsed_ms = (
                        now - start_time
                    ) * 1000

                    # LIVE TRANSCRIPTION
                    if text:
                        print(
                            f"{filepath.name} | "
                            f"{typ.upper()} | "
                            f"{text[:120]}",
                            flush=True
                        )

                    if first_response_time is None:
                        first_response_time = elapsed_ms

                    if typ == "final" and text:
                        final_received = True

                        if first_final_time is None:
                            first_final_time = elapsed_ms

                        final_transcript_parts.append(text)

                        latencies.append(elapsed_ms)

                except asyncio.TimeoutError:
                    idle_time = (
                        time.time() - last_message_time
                    )

                    # IMPORTANT FIX
                    if final_received and idle_time > 5:
                        print(
                            f"{filepath.name} | "
                            f"final transcription completed",
                            flush=True
                        )
                        break

                    if idle_time > 20:
                        print(
                            f"{filepath.name} | "
                            f"timeout after no messages",
                            flush=True
                        )
                        break

                    continue

                except websockets.exceptions.ConnectionClosed:
                    print(
                        f"{filepath.name} | "
                        f"websocket closed",
                        flush=True
                    )
                    break

                except Exception as e:
                    print(
                        f"{filepath.name} | "
                        f"receiver error: {e}",
                        flush=True
                    )
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

    if not files:
        raise FileNotFoundError(
            f"No wav files found in {INPUT_FOLDER}"
        )

    rows = []

    for idx, file in enumerate(files, start=1):
        print(
            f"\n[{idx}/{len(files)}] {file.name}",
            flush=True
        )

        row = await transcribe_file(file)

        rows.append(row)

    df = pd.DataFrame(rows)

    try:
        df.to_excel(
            OUTPUT_FILE,
            index=False
        )
        final_output = OUTPUT_FILE

    except PermissionError:
        final_output = (
            f"nemotron_final_report_backup_"
            f"{int(time.time())}.xlsx"
        )

        df.to_excel(
            final_output,
            index=False
        )

    print(
        f"\nSAVED -> {final_output}",
        flush=True
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    asyncio.run(run_batch())


see this 
poor-audio.wav | PARTIAL | Uh, and we're talking with Jenny. So what is your current occupation? I'm Assistant Director of Navy Davidson Private Ho
poor-audio.wav | sent chunk 260/421
poor-audio.wav | PARTIAL | Uh, and we're talking with Jenny. So what is your current occupation? I'm Assistant Director of Navy Davidson Private Ho
poor-audio.wav | PARTIAL | Uh, and we're talking with Jenny. So what is your current occupation? I'm Assistant Director of Navy Davidson Private Ho
poor-audio.wav | PARTIAL | Uh, and we're talking with Jenny. So what is your current occupation? I'm Assistant Director of Navy Davidson Private Ho
poor-audio.wav | PARTIAL | Uh, and we're talking with Jenny. So what is your current occupation? I'm Assistant Director of Navy Davidson Private Ho
poor-audio.wav | PARTIAL | Uh, and we're talking with Jenny. So what is your current occupation? I'm Assistant Director of Navy Davidson Private Ho
poor-audio.wav | PARTIAL | Uh, and we're talking with Jenny. So what is your current occupation? I'm Assistant Director of Navy Davidson Private Ho
poor-audio.wav | PARTIAL | Uh, and we're talking with Jenny. So what is your current occupation? I'm Assistant Director of Navy Davidson Private Ho
poor-audio.wav | PARTIAL | Uh, and we're talking with Jenny. So what is your current occupation? I'm Assistant Director of Navy Davidson Private Ho
poor-audio.wav | sent chunk 280/421
poor-audio.wav | sent chunk 300/421
poor-audio.wav | sent chunk 320/421
poor-audio.wav | sent chunk 340/421
poor-audio.wav | sent chunk 360/421
poor-audio.wav | sent chunk 380/421
poor-audio.wav | sent chunk 400/421
poor-audio.wav | sent chunk 420/421
poor-audio.wav | sending EOS
poor-audio.wav | timeout after no messages

SAVED -> nemotron_final_report_20260409_144257.xlsx
after chunk 260 i didn't get transcriptions

