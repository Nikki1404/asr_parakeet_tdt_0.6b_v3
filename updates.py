import asyncio
import websockets
import json
import base64
import subprocess
import tempfile
from pathlib import Path

WS_URL = "ws://192.168.4.62:9000/v1/realtime?intent=transcription"
INPUT_FILE = "/home/re_nikitav/audio_maria/maria1.mp3"

SAMPLE_RATE = 16000

# IMPORTANT: 80 ms realtime chunks
CHUNK_MS = 80
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000


def convert_to_wav(src_path: str, out_path: str):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            out_path
        ],
        check=True
    )


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_file = str(Path(tmpdir) / "test.wav")

        convert_to_wav(INPUT_FILE, wav_file)

        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=60,
            max_size=2**26
        ) as ws:

            print("CONNECTED")

            # minimal config
            await ws.send(json.dumps({
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_params": {
                        "sample_rate_hz": 16000,
                        "num_channels": 1
                    }
                }
            }))

            print("SESSION CONFIG SENT")

            async def sender():
                with open(wav_file, "rb") as f:
                    # skip wav header
                    f.read(44)

                    chunk_num = 0

                    while True:
                        chunk = f.read(CHUNK_BYTES)

                        if not chunk:
                            break

                        chunk_num += 1

                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                chunk
                            ).decode("utf-8")
                        }))

                        print(
                            f"SENT CHUNK {chunk_num} "
                            f"({CHUNK_MS} ms)"
                        )

                        # CRITICAL: realtime pacing
                        await asyncio.sleep(
                            CHUNK_MS / 1000
                        )

                await ws.send(json.dumps({
                    "type": "input_audio_buffer.commit"
                }))

                await ws.send(json.dumps({
                    "type": "input_audio_buffer.done"
                }))

                print("STREAM COMPLETED")

            async def receiver():
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=300
                        )

                        msg = json.loads(raw)

                        print("\nRAW RESPONSE:")
                        print(
                            json.dumps(
                                msg,
                                indent=2
                            )
                        )

                        # transcript print
                        if "transcript" in msg:
                            print(
                                "\nTRANSCRIPT:",
                                msg["transcript"]
                            )

                        if msg.get(
                            "is_last_result",
                            False
                        ):
                            print(
                                "\nFINAL RESULT RECEIVED"
                            )
                            break

                    except asyncio.TimeoutError:
                        print("NO MORE RESPONSES")
                        break

            await asyncio.gather(
                sender(),
                receiver()
            )


asyncio.run(main())



import asyncio
import websockets
import json
import base64
import subprocess
import tempfile
from pathlib import Path

WS_URL = "ws://192.168.4.38:9000/v1/realtime?intent=transcription"
INPUT_FILE = "/home/re_nikitav/audio_maria/maria1.mp3"

SAMPLE_RATE = 16000

# 80 ms chunk
CHUNK_MS = 80
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000

# safe 2x speed
SEND_DELAY_MS = 40

MODEL_NAME = (
    "parakeet-rnnt-1.1b-unified-ml-cs-"
    "universal-multi-asr-streaming-"
    "silero-vad-sortformer"
)


def convert_to_wav(src_path: str, out_path: str):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            out_path
        ],
        check=True
    )


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_file = str(Path(tmpdir) / "test.wav")

        convert_to_wav(INPUT_FILE, wav_file)

        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=60,
            max_size=2**26
        ) as ws:

            print("CONNECTED")

            # FINAL FIX
            await ws.send(json.dumps({
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "language": "es-US",
                        "model": MODEL_NAME
                    },
                    "input_audio_params": {
                        "sample_rate_hz": 16000,
                        "num_channels": 1
                    }
                }
            }))

            print("SESSION CONFIG SENT")

            async def sender():
                with open(wav_file, "rb") as f:
                    # skip wav header
                    f.read(44)

                    chunk_num = 0

                    while True:
                        chunk = f.read(CHUNK_BYTES)

                        if not chunk:
                            break

                        chunk_num += 1

                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                chunk
                            ).decode("utf-8")
                        }))

                        print(
                            f"SENT CHUNK {chunk_num}"
                        )

                        # 2x realtime
                        await asyncio.sleep(
                            SEND_DELAY_MS / 1000
                        )

                await ws.send(json.dumps({
                    "type": "input_audio_buffer.commit"
                }))

                await ws.send(json.dumps({
                    "type": "input_audio_buffer.done"
                }))

                print("STREAM COMPLETED")

            async def receiver():
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=600
                        )

                        msg = json.loads(raw)

                        print("\nRAW RESPONSE:")
                        print(
                            json.dumps(
                                msg,
                                indent=2
                            )
                        )

                        transcript = msg.get(
                            "transcript",
                            ""
                        )

                        if transcript:
                            print(
                                "\nTRANSCRIPT:",
                                transcript
                            )

                        if msg.get(
                            "is_last_result",
                            False
                        ):
                            print(
                                "\nFINAL RESULT RECEIVED"
                            )
                            break

                    except asyncio.TimeoutError:
                        print("TIMEOUT")
                        break

            await asyncio.gather(
                sender(),
                receiver()
            )


asyncio.run(main())





import riva.client
import time
import json
import subprocess
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_FOLDER = Path("/home/re_nikitav/audio_maria")
OUTPUT_FOLDER = Path("/home/re_nikitav/parakeet_1.1b_multilingual_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

SERVER = "192.168.4.38:50051"
LANGUAGE = "es-US"

SAMPLE_RATE = 16000
CHUNK_MS = 80
CHUNK_SIZE = SAMPLE_RATE * 2 * CHUNK_MS // 1000

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}


# =========================
# CONVERT TO WAV
# =========================
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


# =========================
# TRANSCRIBE ONE FILE
# =========================
def transcribe_file(asr_service, file_path: Path):
    print(f"\nSTARTING -> {file_path.name}")

    wav_path = OUTPUT_FOLDER / f"{file_path.stem}.wav"

    convert_to_wav(file_path, wav_path)

    with open(wav_path, "rb") as f:
        audio_data = f.read()

    chunks = [
        audio_data[i:i + CHUNK_SIZE]
        for i in range(0, len(audio_data), CHUNK_SIZE)
    ]

    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=LANGUAGE,
            max_alternatives=1,
            enable_automatic_punctuation=True
        ),
        interim_results=True
    )

    chunk_latency = []
    final_parts = []

    start_time = time.time()
    first_transcript_time = None

    responses = asr_service.streaming_response_generator(
        audio_chunks=chunks,
        streaming_config=config
    )

    chunk_num = 0

    for response in responses:
        recv_time = time.time()

        for result in response.results:
            transcript = result.alternatives[0].transcript

            chunk_num += 1

            latency_ms = (
                recv_time - start_time
            ) * 1000

            chunk_latency.append({
                "chunk_num": chunk_num,
                "latency_ms": round(latency_ms, 2),
                "is_final": result.is_final,
                "text": transcript
            })

            if transcript:
                print(
                    f"{file_path.name} | "
                    f"CHUNK {chunk_num} | "
                    f"{transcript}"
                )

            if transcript and first_transcript_time is None:
                first_transcript_time = latency_ms

            if result.is_final and transcript:
                final_parts.append(transcript)

    total_time = time.time() - start_time

    # =========================
    # SAVE TRANSCRIPT
    # =========================
    transcript_file = (
        OUTPUT_FOLDER /
        f"{file_path.stem}_transcription.txt"
    )

    transcript_text = "\n".join(final_parts)

    transcript_file.write_text(
        transcript_text,
        encoding="utf-8"
    )

    # =========================
    # SAVE LATENCY
    # =========================
    latency_file = (
        OUTPUT_FOLDER /
        f"{file_path.stem}_latency.json"
    )

    latency_json = {
        "audio_file": str(file_path),
        "total_time_sec": round(total_time, 2),
        "ttft_ms": round(first_transcript_time, 2)
        if first_transcript_time
        else None,
        "total_chunks": len(chunk_latency),
        "chunk_latency": chunk_latency
    }

    latency_file.write_text(
        json.dumps(latency_json, indent=2),
        encoding="utf-8"
    )

    print(f"COMPLETED -> {file_path.name}")


# =========================
# MAIN
# =========================
def main():
    auth = riva.client.Auth(uri=SERVER)
    asr_service = riva.client.ASRService(auth)

    files = sorted([
        f for f in INPUT_FOLDER.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    print(f"TOTAL FILES = {len(files)}")

    for file in files:
        transcribe_file(asr_service, file)

    print("\nALL FILES COMPLETED")


if __name__ == "__main__":
    main()



python3 -c "import socket; s=socket.socket(); print(s.connect_ex(('127.0.0.1',50051)))"
