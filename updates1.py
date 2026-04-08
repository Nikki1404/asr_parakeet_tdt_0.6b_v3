import riva.client
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
import wave



INPUT_FOLDER = Path("/home/re_nikitav/audio_maria")
OUTPUT_FOLDER = Path("/home/re_nikitav/parakeet_1.1b_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

SERVER = "192.168.4.38:50051"
LANGUAGE = "es-US"

SAMPLE_RATE = 16000

# FAST BATCH MODE
CHUNK_SEC = 5
CHUNK_SIZE = SAMPLE_RATE * 2 * CHUNK_SEC

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac"
}

# CONVERT TO WAV
def convert_to_wav(input_file: Path, output_file: Path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(output_file)
        ],
        check=True
    )

# GET AUDIO DURATION
def get_audio_duration_sec(wav_path: Path):
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# TRANSCRIBE ONE FILE
def transcribe_file(asr_service, file_path: Path):
    print(f"\nSTARTING -> {file_path.name}")

    wav_path = OUTPUT_FOLDER / f"{file_path.stem}.wav"

    convert_to_wav(file_path, wav_path)

    audio_duration_sec = get_audio_duration_sec(
        wav_path
    )

    with open(wav_path, "rb") as f:
        audio_data = f.read()

    send_start_time = time.time()

    chunks = [
        audio_data[i:i + CHUNK_SIZE]
        for i in range(
            0,
            len(audio_data),
            CHUNK_SIZE
        )
    ]

    send_end_time = time.time()

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

    start_time = time.time()

    first_response_time = None
    first_final_time = None

    response_num = 0
    latencies = []
    final_parts = []

    responses = asr_service.streaming_response_generator(
        audio_chunks=chunks,
        streaming_config=config
    )

    for response in responses:
        now = time.time()

        for result in response.results:
            response_num += 1

            transcript = (
                result.alternatives[0].transcript
            )

            words = len(
                transcript.split()
            )

            chars = len(transcript)

            latency_from_start_ms = (
                now - start_time
            ) * 1000

            latency_from_send_start_ms = (
                now - send_start_time
            ) * 1000

            latency_from_send_end_ms = (
                now - send_end_time
            ) * 1000

            if first_response_time is None:
                first_response_time = (
                    latency_from_start_ms
                )

            if (
                result.is_final and
                first_final_time is None
            ):
                first_final_time = (
                    latency_from_start_ms
                )

            latencies.append({
                "response_num": response_num,
                "latency_from_start_ms": round(
                    latency_from_start_ms, 2
                ),
                "latency_from_send_start_ms": round(
                    latency_from_send_start_ms, 2
                ),
                "latency_from_send_end_ms": round(
                    latency_from_send_end_ms, 2
                ),
                "is_final": result.is_final,
                "words": words,
                "char_count": chars
            })

            if transcript:
                print(
                    f"{file_path.name} | "
                    f"RESP {response_num} | "
                    f"{transcript[:80]}"
                )

            if (
                result.is_final and
                transcript
            ):
                final_parts.append(
                    transcript
                )

    total_time = (
        time.time() - start_time
    )

    total_words = sum(
        x["words"] for x in latencies
    )

    total_chars = sum(
        x["char_count"] for x in latencies
    )

    final_count = sum(
        1 for x in latencies
        if x["is_final"]
    )

    latency_values = [
        x["latency_from_send_start_ms"]
        for x in latencies
    ]

    latency_json = {
        "audio_file": str(file_path),
        "audio_duration_sec": round(
            audio_duration_sec, 4
        ),
        "total_processing_time_sec": round(
            total_time, 4
        ),
        "timestamp": datetime.now().isoformat(),
        "model": "parakeet-rnnt-1.1b",
        "language": LANGUAGE,
        "timing_metrics": {
            "send_duration_sec": round(
                send_end_time -
                send_start_time, 4
            ),
            "first_response_latency_sec": round(
                first_response_time / 1000, 4
            ) if first_response_time else None,
            "first_final_latency_sec": round(
                first_final_time / 1000, 4
            ) if first_final_time else None
        },
        "latencies": latencies,
        "summary": {
            "total_responses": len(
                latencies
            ),
            "final_responses":
                final_count,
            "interim_responses":
                len(latencies) - final_count,
            "total_words":
                total_words,
            "total_characters":
                total_chars,
            "avg_latency_from_send_start_ms":
                round(
                    sum(latency_values) /
                    len(latency_values), 2
                ),
            "min_latency_from_send_start_ms":
                round(
                    min(latency_values), 2
                ),
            "max_latency_from_send_start_ms":
                round(
                    max(latency_values), 2
                )
        }
    }

    latency_file = (
        OUTPUT_FOLDER /
        f"{file_path.stem}_latency.json"
    )

    latency_file.write_text(
        json.dumps(
            latency_json,
            indent=2
        ),
        encoding="utf-8"
    )

    transcript_file = (
        OUTPUT_FOLDER /
        f"{file_path.stem}_transcription.txt"
    )

    transcript_file.write_text(
        "\n".join(final_parts),
        encoding="utf-8"
    )

    print(
        f"COMPLETED -> {file_path.name}"
    )


# MAIN
def main():
    auth = riva.client.Auth(uri=SERVER)

    asr_service = riva.client.ASRService(
        auth
    )

    files = sorted([
        f for f in INPUT_FOLDER.iterdir()
        if f.suffix.lower()
        in SUPPORTED_EXTENSIONS
    ])

    print(f"TOTAL FILES = {len(files)}")

    for file in files:
        transcribe_file(
            asr_service,
            file
        )

    print("\nALL FILES COMPLETED")


if __name__ == "__main__":
    main()
