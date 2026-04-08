import asyncio
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import websockets
from jiwer import wer

WS_URL = "ws://localhost:8001/ws"
AUDIO_FOLDER = Path("./audio_samples")
OUTPUT_FILE = "parakeet_tdt_benchmark.xlsx"

TARGET_SR = 16000
CHUNK_MS = 80
AUDIO_EXTENSIONS = {".ogg", ".wav", ".flac", ".mp3", ".m4a"}


def convert_to_wav(filepath):
    audio, _ = librosa.load(
        str(filepath),
        sr=TARGET_SR,
        mono=True
    )

    pcm16 = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16)

    wav_path = filepath.with_suffix(".wav")

    sf.write(
        str(wav_path),
        pcm16,
        TARGET_SR,
        subtype="PCM_16"
    )

    return wav_path


def load_reference_text(audio_file):
    txt_file = AUDIO_FOLDER / f"{audio_file.stem}.txt"
    return txt_file.read_text(encoding="utf-8").strip()


def normalize_text(text):
    text = text.lower().replace("\n", " ")
    return " ".join(text.split())


def calculate_wer(ref, pred):
    return round(
        wer(normalize_text(ref), normalize_text(pred)),
        4
    )


async def benchmark_tdt(wav_file):
    audio, _ = sf.read(str(wav_file))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    pcm_bytes = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16).tobytes()

    chunk_bytes = TARGET_SR * 2 * CHUNK_MS // 1000

    transcript_parts = []
    ttft = None
    ttfb = None

    overall_start = time.time()

    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"sample_rate": TARGET_SR}))
        send_start = time.time()

        for i in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[i:i + chunk_bytes]
            await ws.send(chunk)

            try:
                response = await asyncio.wait_for(
                    ws.recv(),
                    timeout=2
                )

                current_time = time.time()
                data = json.loads(response)

                text = data.get("text", "").strip()
                is_final = data.get("is_final", False)

                if text:
                    transcript_parts.append(text)

                    if ttft is None:
                        ttft = round(
                            current_time - send_start,
                            3
                        )

                    if is_final and ttfb is None:
                        ttfb = round(
                            current_time - send_start,
                            3
                        )

            except:
                pass

        await ws.send(json.dumps({"eof": True}))

    total_time = round(
        time.time() - overall_start,
        3
    )

    transcript = " ".join(transcript_parts)

    return {
        "ttft": ttft or total_time,
        "ttfb": ttfb or total_time,
        "total_time": total_time,
        "transcript": transcript
    }


async def main():
    rows = []

    for file in AUDIO_FOLDER.iterdir():
        if file.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        print(f"Processing -> {file.name}")

        ref_text = load_reference_text(file)
        wav_file = convert_to_wav(file)

        result = await benchmark_tdt(wav_file)

        rows.append({
            "file_name": file.name,
            "ref_txt": ref_text,
            "ttft": result["ttft"],
            "ttfb": result["ttfb"],
            "total_time": result["total_time"],
            "wer": calculate_wer(
                ref_text,
                result["transcript"]
            ),
            "transcript": result["transcript"]
        })

    pd.DataFrame(rows).to_excel(
        OUTPUT_FILE,
        index=False
    )

    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
