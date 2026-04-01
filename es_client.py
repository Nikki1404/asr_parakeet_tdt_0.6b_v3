import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import requests


# =========================================================
# CONFIG
# =========================================================
SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac"
}

MAX_WORKERS = 4


# =========================================================
# SAVE OUTPUTS
# =========================================================
def save_results(
    output_folder: Path,
    filepath: Path,
    transcript_text: str,
    latency_json: dict
):
    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    transcript_path = (
        output_folder /
        f"{filepath.stem}_transcript.txt"
    )

    latency_path = (
        output_folder /
        f"{filepath.stem}_latency.json"
    )

    transcript_path.write_text(
        transcript_text,
        encoding="utf-8"
    )

    latency_path.write_text(
        json.dumps(latency_json, indent=2),
        encoding="utf-8"
    )

    print(f"SAVED -> {transcript_path}")
    print(f"SAVED -> {latency_path}")


# =========================================================
# TRANSCRIBE SINGLE FILE
# =========================================================
def transcribe_single_file(
    filepath: Path,
    base_url: str,
    output_folder: Path
):
    print(f"STARTING -> {filepath.name}")

    base_url = base_url.strip().rstrip("/")

    url = f"{base_url}/v1/audio/transcriptions"

    start_time = time.time()

    # detect mime type
    if filepath.suffix.lower() == ".wav":
        mime_type = "audio/wav"
    else:
        mime_type = "audio/mpeg"

    with open(filepath, "rb") as f:
        response = requests.post(
            url,
            files={
                "file": (
                    filepath.name,
                    f,
                    mime_type
                )
            },
            timeout=7200
        )

    total_latency_ms = (
        time.time() - start_time
    ) * 1000

    # print debug if failure
    if response.status_code != 200:
        print(
            f"FAILED -> {filepath.name} | "
            f"STATUS {response.status_code}"
        )
        print(response.text)

    response.raise_for_status()

    result = response.json()

    transcript_text = (
        result.get("text")
        or result.get("transcript")
        or ""
    )

    latency_json = {
        "audio_file": str(filepath),
        "timestamp": datetime.now().isoformat(),
        "total_latency_ms": round(
            total_latency_ms, 2
        ),
        "total_latency_sec": round(
            total_latency_ms / 1000, 2
        ),
        "response": result
    }

    save_results(
        output_folder,
        filepath,
        transcript_text,
        latency_json
    )

    print(
        f"COMPLETED -> {filepath.name} | "
        f"{total_latency_ms/1000:.1f}s"
    )


# =========================================================
# RUN BATCH
# =========================================================
def run_batch(
    base_url: str,
    input_folder: Path,
    output_folder: Path
):
    files = sorted([
        f for f in input_folder.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    total_files = len(files)

    print(f"TOTAL FILES = {total_files}")

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:
        futures = []

        for file in files:
            future = executor.submit(
                transcribe_single_file,
                file,
                base_url,
                output_folder
            )
            futures.append(future)

        for future in futures:
            future.result()

    print("\nALL FILES COMPLETED")


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        required=True
    )

    parser.add_argument(
        "--input-folder",
        required=True
    )

    parser.add_argument(
        "--output-folder",
        required=True
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    args = parse_args()

    run_batch(
        args.base_url,
        Path(args.input_folder),
        Path(args.output_folder)
    )

python3 -c "import requests; print(requests.post('http://192.168.4.62:9000/v1/audio/transcriptions').text)"
