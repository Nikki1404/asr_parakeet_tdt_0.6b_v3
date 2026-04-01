```python
#!/usr/bin/env python3
"""
NVIDIA NIM realtime transcription batch client
(Session-based API)

Flow:
1. POST /v1/realtime/transcription_sessions
2. get websocket url
3. stream 30 sec chunks
4. save transcript + latency json
"""

import argparse
import asyncio
import base64
import json
import time
import traceback
import statistics
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import wave

import websockets
import requests


# =========================================================
# CONFIG
# =========================================================
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

SEND_CHUNK_SEC = 30
SEND_CHUNK_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * SEND_CHUNK_SEC

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac"
}


# =========================================================
# CONVERT TO WAV
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
            wf.getnframes() / wf.getframerate()
        )

    return duration_sec


# =========================================================
# CREATE SESSION
# =========================================================
def create_transcription_session(base_url: str):
    url = f"{base_url}/v1/realtime/transcription_sessions"

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={}
    )

    response.raise_for_status()

    result = response.json()

    # flexible field handling
    ws_url = (
        result.get("ws_url")
        or result.get("url")
        or result.get("websocket_url")
    )

    if not ws_url:
        raise ValueError(
            f"Could not find websocket url in response: {result}"
        )

    return ws_url


# =========================================================
# SAVE OUTPUTS
# =========================================================
def save_results(
    output_folder: Path,
    filepath: Path,
    transcript_text: str,
    result_json: dict
):
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
        json.dumps(result_json, indent=2),
        encoding="utf-8"
    )

    print(f"SAVED -> {transcript_path}")
    print(f"SAVED -> {latency_path}")


# =========================================================
# TRANSCRIBE FILE
# =========================================================
async def transcribe_file(
    filepath: Path,
    base_url: str,
    output_folder: Path
):
    print(f"\nSTARTING -> {filepath.name}")

    ws_url = create_transcription_session(base_url)

    print(f"SESSION WS URL -> {ws_url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / f"{filepath.stem}.wav"

        duration_sec = convert_to_wav(
            filepath,
            wav_path
        )

        final_transcript_parts = []
        chunk_latencies = []

        start_time = time.time()

        first_response_time = None
        first_final_time = None

        chunk_send_times = []

        try:
            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=60,
                max_size=2**26
            ) as ws:

                async def sender():
                    with open(wav_path, "rb") as fh:
                        fh.read(44)

                        chunk_num = 0

                        while True:
                            chunk = fh.read(
                                SEND_CHUNK_BYTES
                            )

                            if not chunk:
                                break

                            chunk_num += 1

                            sent_time = time.time()

                            chunk_send_times.append(
                                sent_time
                            )

                            await ws.send(
                                json.dumps({
                                    "event_id": f"chunk_{chunk_num}",
                                    "type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(
                                        chunk
                                    ).decode("utf-8")
                                })
                            )

                            print(
                                f"{filepath.name} | "
                                f"SENT CHUNK {chunk_num}"
                            )

                    await ws.send(
                        json.dumps({
                            "type": "input_audio_buffer.commit"
                        })
                    )

                    await ws.send(
                        json.dumps({
                            "type": "input_audio_buffer.done"
                        })
                    )

                async def receiver():
                    nonlocal first_response_time
                    nonlocal first_final_time

                    response_num = 0

                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(),
                                timeout=1800
                            )

                            now = time.time()

                            msg = json.loads(raw)

                            response_num += 1

                            elapsed_ms = (
                                now - start_time
                            ) * 1000

                            if first_response_time is None:
                                first_response_time = elapsed_ms

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

                            if chunk_send_times:
                                latency_ms = (
                                    now -
                                    chunk_send_times[
                                        min(
                                            response_num - 1,
                                            len(chunk_send_times)-1
                                        )
                                    ]
                                ) * 1000
                            else:
                                latency_ms = 0

                            chunk_latencies.append({
                                "response_num": response_num,
                                "latency_ms": round(
                                    latency_ms, 2
                                ),
                                "elapsed_ms": round(
                                    elapsed_ms, 2
                                ),
                                "is_final": is_final
                            })

                            if transcript and is_final:
                                final_transcript_parts.append(
                                    transcript
                                )

                                if first_final_time is None:
                                    first_final_time = elapsed_ms

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

        except Exception as e:
            print(str(e))
            traceback.print_exc()
            return

        total_time = (
            time.time() - start_time
        )

        transcript_text = "\n".join(
            final_transcript_parts
        )

        result_json = {
            "audio_file": str(filepath),
            "audio_duration_sec": duration_sec,
            "total_processing_time_sec": round(
                total_time, 2
            ),
            "timestamp": datetime.now().isoformat(),
            "chunk_sec": SEND_CHUNK_SEC,
            "chunk_latencies": chunk_latencies,
            "ttfb_ms": round(
                first_response_time, 2
            ) if first_response_time else None,
            "ttft_ms": round(
                first_final_time, 2
            ) if first_final_time else None,
            "summary": {
                "avg_latency_ms": round(
                    statistics.mean(
                        [x["latency_ms"]
                         for x in chunk_latencies]
                    ), 2
                ) if chunk_latencies else 0
            }
        }

        save_results(
            output_folder,
            filepath,
            transcript_text,
            result_json
        )

        print(
            f"COMPLETED -> {filepath.name}"
        )


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
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    for idx, file in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}]")
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


if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        run_batch(
            args.base_url,
            Path(args.input_folder),
            Path(args.output_folder)
        )
    )
```
