import asyncio
import json
import time
import statistics
import wave
from pathlib import Path

import websockets
import pandas as pd
from jiwer import wer


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000

# EXACT SAME AS WORKING MIC CLIENT
CHUNK_MS = 80
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000

SEND_DELAY_MS = 80

INPUT_FOLDER = Path(r"C:\Users\re_nikitav\Downloads\audio_samples\english_samples")

OUTPUT_FILE = "nemotron_final_report.xlsx"

WEBSOCKET_ADDRESS = "ws://127.0.0.1:8002/asr/realtime-custom-vad"

BACKEND = "nemotron"


# =========================
# WAV LOADER
# =========================
def load_wav_as_pcm16(filepath):
    with wave.open(str(filepath), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

        if sr != SAMPLE_RATE:
            raise ValueError(
                f"{filepath.name}: expected {SAMPLE_RATE}Hz, got {sr}"
            )

        if channels != 1:
            raise ValueError(
                f"{filepath.name}: expected mono, got {channels}"
            )

        if sample_width != 2:
            raise ValueError(
                f"{filepath.name}: expected 16-bit PCM"
            )

        pcm = wf.readframes(wf.getnframes())
        duration_sec = wf.getnframes() / SAMPLE_RATE

    return pcm, duration_sec


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

        async def sender():
            offset = 0

            while offset < len(pcm):
                chunk = pcm[
                    offset: offset + CHUNK_BYTES
                ]

                offset += CHUNK_BYTES

                await ws.send(chunk)

                await asyncio.sleep(
                    SEND_DELAY_MS / 1000
                )

            # IMPORTANT:
            # same flush logic as mic client
            silence = b"\x00\x00" * int(
                SAMPLE_RATE * 0.6
            )

            await ws.send(silence)

            await asyncio.sleep(0.5)

            # EOS
            await ws.send(b"")

        async def receiver():
            nonlocal first_response_time
            nonlocal first_final_time

            while True:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(),
                        timeout=60
                    )

                    now = time.time()

                    msg = json.loads(raw)

                    typ = msg.get("type")
                    text = msg.get("text", "")

                    elapsed_ms = (
                        now - start_time
                    ) * 1000

                    if first_response_time is None:
                        first_response_time = elapsed_ms

                    if typ == "final" and text:
                        if first_final_time is None:
                            first_final_time = elapsed_ms

                        final_transcript_parts.append(text)
                        latencies.append(elapsed_ms)

                    elif typ == "partial":
                        # ignore partial text for final report
                        pass

                except asyncio.TimeoutError:
                    break

                except websockets.exceptions.ConnectionClosed:
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
