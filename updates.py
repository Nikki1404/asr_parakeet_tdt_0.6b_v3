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

        start_time = time.time()
        connection_start_time = time.time()

        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=60,
            max_size=2**26
        ) as ws:

            connection_time_sec = (
                time.time() -
                connection_start_time
            )

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

            send_start_time = None
            first_chunk_sent_time = None
            send_end_time = None

            first_byte_latency_sec = None
            first_response_latency_sec = None
            first_final_latency_sec = None

            async def sender():
                nonlocal send_start_time
                nonlocal first_chunk_sent_time
                nonlocal send_end_time

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

                        now = time.time()

                        if send_start_time is None:
                            send_start_time = now

                        if first_chunk_sent_time is None:
                            first_chunk_sent_time = now

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

                send_end_time = time.time()

            async def receiver():
                nonlocal first_byte_latency_sec
                nonlocal first_response_latency_sec
                nonlocal first_final_latency_sec

                response_num = 0

                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=1800
                        )

                        now = time.time()

                        if first_byte_latency_sec is None:
                            first_byte_latency_sec = (
                                now - start_time
                            )

                        msg = json.loads(raw)

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

                        if transcript:
                            response_num += 1

                            if first_response_latency_sec is None:
                                first_response_latency_sec = (
                                    now - start_time
                                )

                            if is_final and first_final_latency_sec is None:
                                first_final_latency_sec = (
                                    now - start_time
                                )

                            chunk_latencies.append({
                                "response_num": response_num,
                                "latency_from_start_ms":
                                    round(
                                        (now - start_time) * 1000,
                                        4
                                    ),
                                "latency_from_send_start_ms":
                                    round(
                                        (now - send_start_time) * 1000,
                                        4
                                    ),
                                "latency_from_first_chunk_ms":
                                    round(
                                        (now - first_chunk_sent_time) * 1000,
                                        4
                                    ),
                                "latency_from_send_end_ms":
                                    round(
                                        (now - send_end_time) * 1000,
                                        4
                                    ) if send_end_time else None,
                                "is_final": is_final,
                                "words": len(transcript.split()),
                                "char_count": len(transcript)
                            })

                            if is_final:
                                transcript_parts.append(transcript)

                                print(
                                    f"{filepath.name} | "
                                    f"FINAL {response_num}"
                                )

                        if msg.get("is_last_result", False):
                            break

                    except asyncio.TimeoutError:
                        break

            await asyncio.gather(
                sender(),
                receiver()
            )

        total_time_sec = time.time() - start_time

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

        latency_values_start = [
            x["latency_from_send_start_ms"]
            for x in chunk_latencies
        ]

        latency_values_end = [
            x["latency_from_send_end_ms"]
            for x in chunk_latencies
            if x["latency_from_send_end_ms"] is not None
        ]

        latency_json = {
            "audio_file": str(filepath),
            "audio_duration_sec": round(duration_sec, 4),
            "total_processing_time_sec": round(total_time_sec, 4),
            "timestamp": datetime.now().isoformat(),
            "model": "nova-3",
            "language": "multi",
            "timing_metrics": {
                "connection_time_sec": round(connection_time_sec, 4),
                "send_duration_sec": round(send_end_time - send_start_time, 4),
                "first_byte_latency_sec": round(first_byte_latency_sec, 4) if first_byte_latency_sec else None,
                "first_response_latency_sec": round(first_response_latency_sec, 4) if first_response_latency_sec else None,
                "first_final_latency_sec": round(first_final_latency_sec, 4) if first_final_latency_sec else None,
                "time_to_first_chunk_sec": round(first_chunk_sent_time - start_time, 4)
            },
            "latencies": chunk_latencies,
            "summary": {
                "total_responses": len(chunk_latencies),
                "final_responses": sum(1 for x in chunk_latencies if x["is_final"]),
                "interim_responses": sum(1 for x in chunk_latencies if not x["is_final"]),
                "total_words": sum(x["words"] for x in chunk_latencies),
                "total_characters": sum(x["char_count"] for x in chunk_latencies),
                "avg_latency_from_send_start_ms": round(sum(latency_values_start) / len(latency_values_start), 4),
                "min_latency_from_send_start_ms": round(min(latency_values_start), 4),
                "max_latency_from_send_start_ms": round(max(latency_values_start), 4),
                "avg_latency_from_send_end_ms": round(sum(latency_values_end) / len(latency_values_end), 4) if latency_values_end else None,
                "min_latency_from_send_end_ms": round(min(latency_values_end), 4) if latency_values_end else None,
                "max_latency_from_send_end_ms": round(max(latency_values_end), 4) if latency_values_end else None
            }
        }

        latency_path.write_text(
            json.dumps(latency_json, indent=2),
            encoding="utf-8"
        )

        print(f"SAVED -> {filepath.name}")


# =========================================================
# BATCH
# =========================================================
async def run_batch(base_url: str, input_folder: Path, output_folder: Path):
    files = sorted([
        f for f in input_folder.iterdir()
        if (
            f.suffix.lower() in SUPPORTED_EXTENSIONS
            and f.name in TARGET_FILES
        )
    ])

    print(f"TOTAL FILES = {len(files)}")
    print("FILES TO PROCESS:")
    for f in files:
        print(f" - {f.name}")

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

