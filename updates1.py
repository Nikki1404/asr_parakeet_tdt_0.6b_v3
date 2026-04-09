import argparse
import json
import time
import statistics
import traceback
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from jiwer import wer

import riva.client


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000

# 80 ms chunks — same as working code
CHUNK_SIZE = SAMPLE_RATE * 2 * 80 // 1000  # = 2560 bytes

INPUT_FOLDER = Path(
    "/home/re_nikitav/parakeet-asr-multilingual/audio_samples"
)

OUTPUT_FOLDER = Path("transcription_results_grpc")
OUTPUT_FOLDER.mkdir(exist_ok=True)

LOG_FILE = OUTPUT_FOLDER / "run.log"


# =========================
# LOGGER
# =========================
class Logger:
    def __init__(self, log_path):
        self.file = open(log_path, "a", encoding="utf-8", buffering=1)

    def log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.file.write(line + "\n")

    def close(self):
        self.file.close()


logger = Logger(LOG_FILE)


# =========================
# LOAD RAW WAV BYTES
# Read file as-is — riva handles the WAV header server-side
# =========================
def load_wav_bytes(filepath):
    logger.log(f"Loading -> {filepath.name}")

    with open(filepath, "rb") as f:
        audio_data = f.read()

    try:
        pcm_bytes = len(audio_data) - 44
        duration_sec = pcm_bytes / (SAMPLE_RATE * 2)
    except Exception:
        duration_sec = 0.0

    logger.log(
        f"Loaded {filepath.name} | "
        f"~{duration_sec:.1f} sec | "
        f"{len(audio_data)} bytes"
    )

    return audio_data, duration_sec


# =========================
# CHUNK GENERATOR
# =========================
def chunk_generator(audio_data, chunk_size, timing):
    timing["send_start_time"] = time.time()

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i: i + chunk_size]

        now = time.time()
        if timing.get("first_chunk_time") is None:
            timing["first_chunk_time"] = now

        yield chunk

    timing["send_end_time"] = time.time()


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

    logger.log(f"SAVED -> {filepath.name}")


# =========================
# REFERENCE TEXT
# =========================
def load_reference_text(filepath):
    txt_path = filepath.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return ""


# =========================
# EXCEL REPORT
# =========================
def generate_excel_report(rows):
    excel_path = OUTPUT_FOLDER / "final_report.xlsx"
    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False, engine="openpyxl")
    logger.log(f"EXCEL SAVED -> {excel_path}")


# =========================
# TRANSCRIBE ONE FILE
# =========================
def transcribe_file(filepath, server, language):
    logger.log(f"STARTING -> {filepath.name}")
    logger.log(f"Using server={server} language={language}")

    audio_data, duration_sec = load_wav_bytes(filepath)

    auth = riva.client.Auth(uri=server)
    asr_service = riva.client.ASRService(auth)

    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=language,
            max_alternatives=1,
            enable_automatic_punctuation=True,
        ),
        interim_results=True,
    )

    final_transcript_parts = []
    latencies = []

    start_time = time.time()

    first_response_time = None
    first_final_time = None
    response_num = 0

    timing = {
        "send_start_time": None,
        "first_chunk_time": None,
        "send_end_time": None,
    }

    try:
        chunks = chunk_generator(audio_data, CHUNK_SIZE, timing)

        responses = asr_service.streaming_response_generator(
            audio_chunks=chunks,
            streaming_config=config,
        )

        for response in responses:
            now = time.time()
            elapsed_ms = (now - start_time) * 1000

            for result in response.results:
                if not result.alternatives:
                    continue

                text = result.alternatives[0].transcript
                is_final = result.is_final

                if not text:
                    continue

                response_num += 1

                latency_from_send_start = (
                    (now - timing["send_start_time"]) * 1000
                    if timing["send_start_time"] else None
                )
                latency_from_first_chunk = (
                    (now - timing["first_chunk_time"]) * 1000
                    if timing["first_chunk_time"] else None
                )
                latency_from_send_end = (
                    (now - timing["send_end_time"]) * 1000
                    if timing["send_end_time"] else None
                )

                if first_response_time is None:
                    first_response_time = elapsed_ms

                if is_final and first_final_time is None:
                    first_final_time = elapsed_ms

                words = len(text.split()) if text else 0
                chars = len(text)

                latencies.append({
                    "response_num": response_num,
                    "latency_from_start_ms": round(elapsed_ms, 4),
                    "latency_from_send_start_ms": round(latency_from_send_start, 4) if latency_from_send_start is not None else None,
                    "latency_from_first_chunk_ms": round(latency_from_first_chunk, 4) if latency_from_first_chunk is not None else None,
                    "latency_from_send_end_ms": round(latency_from_send_end, 4) if latency_from_send_end is not None else None,
                    "is_final": is_final,
                    "words": words,
                    "char_count": chars,
                    "text_preview": text[:80],
                })

                if is_final:
                    final_transcript_parts.append(text)
                    lat_str = (
                        f"{latency_from_send_start:.0f} ms"
                        if latency_from_send_start is not None else "N/A"
                    )
                    logger.log(
                        f"{filepath.name} | FINAL #{response_num} | "
                        f"{words} words | latency {lat_str} | "
                        f"{text[:60]!r}"
                    )
                else:
                    logger.log(
                        f"{filepath.name} | interim #{response_num} | "
                        f"{text[:60]!r}"
                    )

    except Exception as e:
        logger.log(f"FAILED -> {filepath.name} | {str(e)}")
        traceback.print_exc()
        return None

    total_time = time.time() - start_time
    transcript_text = "\n".join(final_transcript_parts)

    send_start_time = timing["send_start_time"]
    send_end_time = timing["send_end_time"]
    first_chunk_time = timing["first_chunk_time"]

    latency_values_start = [
        x["latency_from_send_start_ms"]
        for x in latencies
        if x["latency_from_send_start_ms"] is not None
    ]
    latency_values_end = [
        x["latency_from_send_end_ms"]
        for x in latencies
        if x["latency_from_send_end_ms"] is not None
    ]

    result_json = {
        "audio_file": str(filepath),
        "audio_duration_sec": round(duration_sec, 2),
        "total_processing_time_sec": round(total_time, 4),
        "timestamp": datetime.now().isoformat(),
        "server": server,
        "language": language,
        "timing_metrics": {
            "send_duration_sec": round(
                send_end_time - send_start_time, 4
            ) if send_end_time and send_start_time else None,
            "first_response_latency_sec": round(
                first_response_time / 1000, 4
            ) if first_response_time else None,
            "first_final_latency_sec": round(
                first_final_time / 1000, 4
            ) if first_final_time else None,
            "time_to_first_chunk_sec": round(
                first_chunk_time - start_time, 4
            ) if first_chunk_time and start_time else None,
        },
        "latencies": latencies,
        "summary": {
            "total_responses": len(latencies),
            "final_responses": sum(1 for x in latencies if x["is_final"]),
            "interim_responses": sum(1 for x in latencies if not x["is_final"]),
            "total_words": sum(x["words"] for x in latencies if x["is_final"]),
            "total_characters": sum(x["char_count"] for x in latencies if x["is_final"]),
            "avg_latency_from_send_start_ms": round(
                statistics.mean(latency_values_start), 4
            ) if latency_values_start else None,
            "avg_latency_from_send_end_ms": round(
                statistics.mean(latency_values_end), 4
            ) if latency_values_end else None,
        },
    }

    save_results(filepath, transcript_text, result_json)

    reference_text = load_reference_text(filepath)
    calculated_wer = None

    if reference_text and transcript_text:
        try:
            calculated_wer = round(
                wer(reference_text.lower(), transcript_text.lower()) * 100, 2
            )
        except Exception:
            pass

    logger.log(
        f"DONE -> {filepath.name} | "
        f"total {total_time:.2f}s | "
        f"words {result_json['summary']['total_words']} | "
        f"WER {calculated_wer}%"
    )

    return {
        "file_name": filepath.name,
        "audio_duration_sec": round(duration_sec, 2),
        "reference_txt": reference_text,
        "transcription": transcript_text,
        "ttfb_ms": (
            result_json["timing_metrics"]["first_response_latency_sec"] * 1000
            if result_json["timing_metrics"]["first_response_latency_sec"] else None
        ),
        "ttft_ms": (
            result_json["timing_metrics"]["first_final_latency_sec"] * 1000
            if result_json["timing_metrics"]["first_final_latency_sec"] else None
        ),
        "avg_latency_from_send_start_ms": result_json["summary"]["avg_latency_from_send_start_ms"],
        "avg_latency_from_send_end_ms": result_json["summary"]["avg_latency_from_send_end_ms"],
        "total_words": result_json["summary"]["total_words"],
        "wer": calculated_wer,
        "total_time_sec": round(total_time, 4),
    }


# =========================
# BATCH RUNNER
# =========================
def run_batch(server, language):
    files = sorted(INPUT_FOLDER.glob("*.wav"))

    if not files:
        logger.log(f"No .wav files found in {INPUT_FOLDER}")
        return

    logger.log(
        f"Found {len(files)} files | "
        f"server={server} | "
        f"lang={language}"
    )

    report_rows = []

    for idx, file in enumerate(files, start=1):
        logger.log(f"--- [{idx}/{len(files)}] {file.name} ---")

        row = transcribe_file(
            filepath=file,
            server=server,
            language=language,
        )

        if row:
            report_rows.append(row)
        else:
            logger.log(f"SKIPPED (failed) -> {file.name}")

    if report_rows:
        generate_excel_report(report_rows)
    else:
        logger.log("No successful results — Excel not generated.")

    logger.log("ALL FILES COMPLETED")
    logger.close()


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch ASR transcription via Riva gRPC"
    )
    parser.add_argument(
        "--server",
        default="192.168.4.62:50051",
        help="gRPC server address (default: 192.168.4.62:50051)",
    )
    parser.add_argument(
        "--language",
        default="es-US",           # <-- FIXED: was en-US, now es-US
        help="Language code (default: es-US)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.log(
        f"=== RUN START | pid={os.getpid()} | "
        f"server={args.server} | "
        f"language={args.language} ==="
    )

    run_batch(server=args.server, language=args.language)



nohup python -u transcribe_grpc.py --server 192.168.4.62:50051 --language es-US \
  > transcription_results_grpc/nohup.out 2>&1 &


     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 184.6/184.6 kB 38.9 MB/s eta 0:00:00
Requirement already satisfied: cffi>=1.0 in ./env/lib/python3.11/site-packages (from soundfile==0.13.1) (2.0.0)
Building wheels for collected packages: PyAudio
  Building wheel for PyAudio (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Building wheel for PyAudio (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [28 lines of output]
      /tmp/pip-build-env-vnu5knp4/overlay/lib/python3.11/site-packages/setuptools/dist.py:765: SetuptoolsDeprecationWarning: License classifiers are deprecated.
      !!
      
              ********************************************************************************
              Please consider removing the following classifiers in favor of a SPDX license expression:
      
              License :: OSI Approved :: MIT License
      
              See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
              ********************************************************************************
      
      !!
        self._finalize_license_expression()
      running bdist_wheel
      running build
      running build_py
      creating build/lib.linux-x86_64-cpython-311/pyaudio
      copying src/pyaudio/__init__.py -> build/lib.linux-x86_64-cpython-311/pyaudio
      running build_ext
      building 'pyaudio._portaudio' extension
      creating build/temp.linux-x86_64-cpython-311/src/pyaudio
      x86_64-linux-gnu-gcc -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -fPIC -I/usr/local/include -I/usr/include -I/home/nikita_verma2/parakeet-asr-multilingual/env/include -I/usr/include/python3.11 -c src/pyaudio/device_api.c -o build/temp.linux-x86_64-cpython-311/src/pyaudio/device_api.o
      In file included from src/pyaudio/device_api.c:1:
      src/pyaudio/device_api.h:7:10: fatal error: Python.h: No such file or directory
          7 | #include "Python.h"
            |          ^~~~~~~~~~
      compilation terminated.
      error: command '/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for PyAudio
Failed to build PyAudio
ERROR: Could not build wheels for PyAudio, which is required to install pyproject.toml-based projects


cffi==2.0.0
numpy==2.4.4
PyAudio==0.2.14
pycparser==3.0
pydub==0.25.1
soundfile==0.13.1
websockets==16.0
