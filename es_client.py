import json
import time
import statistics
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import riva.client


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000

SEND_CHUNK_SEC = 30
SEND_CHUNK_SAMPLES = SAMPLE_RATE * SEND_CHUNK_SEC
SEND_CHUNK_BYTES = SEND_CHUNK_SAMPLES * 2

INPUT_FOLDER = Path("/home/re_nikitav/audio_maria")
OUTPUT_FOLDER = Path("transcription_results_es")
OUTPUT_FOLDER.mkdir(exist_ok=True)

SERVER = "192.168.4.62:50051"
LANGUAGE = "es-US"

TARGET_FILES = {
    "maria1.mp3",
    "maria2.mp3",
    "maria4.mp3",
    "maria7.mp3",
    "maria10.mp3",
    "maria20.mp3",
    "maria21.mp3",
    "maria24.mp3"
}


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
# CHUNK GENERATOR
# =========================
def chunk_generator(pcm):
    offset = 0
    chunk_num = 0

    while offset < len(pcm):
        chunk = pcm[
            offset: offset + SEND_CHUNK_BYTES
        ]
        offset += SEND_CHUNK_BYTES
        chunk_num += 1

        yield chunk_num, chunk


# =========================
# TRANSCRIBE ONE FILE
# =========================
def transcribe_file(filepath, asr_service):
    print(f"\nSTARTING -> {filepath.name}")

    pcm, duration_sec = load_mp3_as_pcm16(filepath)

    final_transcript_parts = []
    latencies = []

    start_time = time.time()

    first_response_time = None
    first_final_time = None
    first_chunk_time = None
    send_start_time = None
    send_end_time = None

    response_num = 0
    prev_response_time = None

    try:
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=LANGUAGE,
                max_alternatives=1,
                enable_automatic_punctuation=True
            ),
            interim_results=False
        )

        send_start_time = time.time()

        def audio_stream():
            nonlocal first_chunk_time

            for chunk_num, chunk in chunk_generator(pcm):
                now = time.time()

                if first_chunk_time is None:
                    first_chunk_time = now

                print(
                    f"{filepath.name} | "
                    f"SENT CHUNK {chunk_num}"
                )

                yield chunk

        responses = asr_service.streaming_response_generator(
            audio_chunks=audio_stream(),
            streaming_config=config
        )

        send_end_time = time.time()

        prev_response_time = send_end_time

        for response in responses:
            now = time.time()

            if not response.results:
                continue

            response_latency_ms = (
                now - prev_response_time
            ) * 1000

            prev_response_time = now

            for result in response.results:
                response_num += 1

                transcript = (
                    result.alternatives[0].transcript
                    if result.alternatives else ""
                )

                is_final = result.is_final

                if first_response_time is None:
                    first_response_time = response_latency_ms

                if is_final and first_final_time is None:
                    first_final_time = response_latency_ms

                words = len(transcript.split()) if transcript else 0
                chars = len(transcript)

                latencies.append({
                    "response_num": response_num,
                    "response_latency_ms": round(
                        response_latency_ms, 4
                    ),
                    "is_final": is_final,
                    "words": words,
                    "char_count": chars
                })

                if transcript and is_final:
                    final_transcript_parts.append(transcript)

                    print(
                        f"{filepath.name} | "
                        f"FINAL {response_num}"
                    )

    except Exception as e:
        print(f"FAILED -> {filepath.name}")
        print(str(e))
        traceback.print_exc()
        return

    total_time = time.time() - start_time

    transcript_text = "\n".join(
        final_transcript_parts
    )

    latency_values = [
        x["response_latency_ms"]
        for x in latencies
    ]

    result_json = {
        "audio_file": str(filepath),
        "audio_duration_sec": duration_sec,
        "total_processing_time_sec": round(total_time, 4),
        "timestamp": datetime.now().isoformat(),
        "model": "parakeet-rnnt-1.1b",
        "language": LANGUAGE,
        "timing_metrics": {
            "send_duration_sec": round(
                send_end_time - send_start_time, 4
            ) if send_end_time else None,
            "ttft_sec": round(
                first_response_time / 1000, 4
            ) if first_response_time else None,
            "ttfb_sec": round(
                first_final_time / 1000, 4
            ) if first_final_time else None
        },
        "latencies": latencies,
        "summary": {
            "total_responses": len(latencies),
            "final_responses": sum(
                1 for x in latencies if x["is_final"]
            ),
            "avg_latency_ms": round(
                statistics.mean(latency_values), 4
            ) if latency_values else None,
            "min_latency_ms": min(latency_values) if latency_values else None,
            "max_latency_ms": max(latency_values) if latency_values else None
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
# MAIN
# =========================
def main():
    auth = riva.client.Auth(uri=SERVER)
    asr_service = riva.client.ASRService(auth)

    files = sorted([
        f for f in INPUT_FOLDER.glob("*.mp3")
        if f.name in TARGET_FILES
    ])

    print(f"TOTAL FILES = {len(files)}")

    for file in files:
        transcribe_file(file, asr_service)

    print("\nALL FILES COMPLETED")


if __name__ == "__main__":
    main()
