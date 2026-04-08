import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from jiwer import wer
import riva.client
from riva.client import (
    RecognitionConfig,
    StreamingRecognitionConfig,
    AudioEncoding
)

RIVA_URI = "34.118.200.125:50051"

AUDIO_FOLDER = Path("./audio_samples")
OUTPUT_FILE = "parakeet_ctc_benchmark.xlsx"

TARGET_SR = 16000
AUDIO_EXTENSIONS = {".ogg", ".wav", ".flac", ".mp3", ".m4a"}

LANGUAGE_CODE = "es-ES"
MODEL_NAME = "parakeet-ctc-0.6b-es"


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
    return " ".join(
        text.lower().replace("\n", " ").split()
    )


def calculate_wer(ref, pred):
    return round(
        wer(normalize_text(ref), normalize_text(pred)),
        4
    )


def benchmark_ctc(wav_file):
    audio, sr = sf.read(
        str(wav_file),
        dtype="int16"
    )

    audio_bytes = audio.tobytes()

    def audio_chunks():
        chunk_size = 3200
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i + chunk_size]

    auth = riva.client.Auth(uri=RIVA_URI)

    asr_service = riva.client.ASRService(auth)

    streaming_config = StreamingRecognitionConfig(
        config=RecognitionConfig(
            encoding=AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=TARGET_SR,
            audio_channel_count=1,
            language_code=LANGUAGE_CODE,
            model=MODEL_NAME
        ),
        interim_results=True
    )

    ttft = None
    ttfb = None
    transcript_parts = []

    start = time.time()

    responses = asr_service.streaming_response_generator(
        audio_chunks=audio_chunks(),
        streaming_config=streaming_config
    )

    for response in responses:
        now = time.time()

        for result in response.results:
            if not result.alternatives:
                continue

            text = result.alternatives[0].transcript.strip()

            if text:
                transcript_parts.append(text)

                if ttft is None:
                    ttft = round(now - start, 3)

                if result.is_final and ttfb is None:
                    ttfb = round(now - start, 3)

    total_time = round(time.time() - start, 3)

    return {
        "ttft": ttft or total_time,
        "ttfb": ttfb or total_time,
        "total_time": total_time,
        "transcript": " ".join(transcript_parts)
    }


def main():
    rows = []

    for file in AUDIO_FOLDER.iterdir():
        if file.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        print(f"Processing -> {file.name}")

        ref_text = load_reference_text(file)
        wav_file = convert_to_wav(file)

        result = benchmark_ctc(wav_file)

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
    main()
