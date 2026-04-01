import argparse
import asyncio
import base64
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import wave

import websockets


# =========================================================
# CONFIG
# =========================================================
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

SEND_CHUNK_SEC = 30
SEND_CHUNK_BYTES = (
    SAMPLE_RATE *
    BYTES_PER_SAMPLE *
    SEND_CHUNK_SEC
)

SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac"
}


# =========================================================
# AUDIO CONVERSION
# =========================================================
def convert_to_wav(src_path: Path, wav_path: Path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src_path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(wav_path)
        ],
        check=True,
        capture_output=True
    )

    with wave.open(str(wav_path), "rb") as wf:
        duration_sec = (
            wf.getnframes() /
            wf.getframerate()
        )

    return duration_sec


# =========================================================
# TRANSCRIBE ONE FILE
# =========================================================
async def transcribe_file(
    filepath: Path,
    base_url: str,
    output_folder: Path
):
    print(f"\nSTARTING -> {filepath.name}")

    ws_url = (
        base_url
        .replace("http://", "ws://")
        .replace("https://", "wss://")
        .rstrip("/")
        + "/v1/realtime?intent=transcription"
    )

    print(f"WS URL -> {ws_url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = (
            Path(tmpdir) /
            f"{filepath.stem}.wav"
        )

        duration_sec = convert_to_wav(
            filepath,
            wav_path
        )

        transcript_parts = []
        chunk_latencies = []
        chunk_send_times = []

        start_time = time.time()

        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=60,
            max_size=2**26
        ) as ws:

            # minimal config only
            await ws.send(json.dumps({
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_params": {
                        "sample_rate_hz": SAMPLE_RATE,
                        "num_channels": 1
                    }
                }
            }))

            async def sender():
                with open(wav_path, "rb") as fh:
                    # skip wav header
                    fh.read(44)

                    chunk_num = 0

                    while True:
                        chunk = fh.read(
                            SEND_CHUNK_BYTES
                        )

                        if not chunk:
                            break

                        chunk_num += 1

                        chunk_send_times.append(
                            time.time()
                        )

                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                chunk
                            ).decode("utf-8")
                        }))

                        print(
                            f"{filepath.name} | "
                            f"SENT CHUNK {chunk_num}"
                        )

                await ws.send(json.dumps({
                    "type": "input_audio_buffer.commit"
                }))

                await ws.send(json.dumps({
                    "type": "input_audio_buffer.done"
                }))

            async def receiver():
                response_num = 0

                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=1800
                        )

                        msg = json.loads(raw)

                        response_num += 1

                        msg_type = msg.get(
                            "type",
                            ""
                        )

                        transcript = msg.get(
                            "transcript",
                            ""
                        ).strip()

                        is_final = msg_type.endswith(
                            ".completed"
                        )

                        latency_ms = (
                            time.time()
                            - chunk_send_times[
                                min(
                                    response_num - 1,
                                    len(chunk_send_times)-1
                                )
                            ]
                        ) * 1000

                        chunk_latencies.append({
                            "response_num": response_num,
                            "latency_ms": round(
                                latency_ms, 2
                            ),
                            "is_final": is_final
                        })

                        if transcript and is_final:
                            transcript_parts.append(
                                transcript
                            )

                            print(
                                f"{filepath.name} | "
                                f"FINAL {response_num}"
                            )

                            if msg.get(
                                "is_last_result",
                                False
                            ):
                                break

                    except asyncio.TimeoutError:
                        print(
                            f"{filepath.name} | timeout"
                        )
                        break

            await asyncio.gather(
                sender(),
                receiver()
            )

        total_time_sec = (
            time.time() - start_time
        )

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
            "\n".join(transcript_parts),
            encoding="utf-8"
        )

        latency_path.write_text(
            json.dumps({
                "audio_file": str(filepath),
                "timestamp": datetime.now().isoformat(),
                "audio_duration_sec": round(
                    duration_sec, 2
                ),
                "total_processing_sec": round(
                    total_time_sec, 2
                ),
                "chunk_latencies": chunk_latencies
            }, indent=2),
            encoding="utf-8"
        )

        print(f"SAVED -> {filepath.name}")


# =========================================================
# BATCH
# =========================================================
async def run_batch(
    base_url: str,
    input_folder: Path,
    output_folder: Path
):
    files = sorted([
        f for f in input_folder.iterdir()
        if f.suffix.lower()
        in SUPPORTED_EXTENSIONS
    ])

    print(f"TOTAL FILES = {len(files)}")

    for file in files:
        await transcribe_file(
            file,
            base_url,
            output_folder
        )

    print("\nALL FILES COMPLETED")


# =========================================================
# MAIN
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-url", required=True)
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output-folder", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        run_batch(
            args.base_url,
            Path(args.input_folder),
            Path(args.output_folder)
        )
    )

python3 transcribe_ws_minimal_batch.py \
  --base-url http://192.168.4.62:9000 \
  --input-folder /home/re_nikitav/audio_maria \
  --output-folder /home/re_nikitav/parakeet_es_results
