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

MODEL_NAME = (
    "parakeet-0.6b-unified-ml-cs-es-US-"
    "asr-streaming-silero-vad-sortformer"
)

LANGUAGE = "es-US"


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
# TRANSCRIBE SINGLE FILE
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
        + "/v1/realtime"
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

            # SESSION CONFIG
            await ws.send(json.dumps({
                "event_id": "session_config",
                "type": "transcription_session.update",
                "session": {
                    "modalities": ["text"],
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "language": LANGUAGE,
                        "model": MODEL_NAME
                    },
                    "input_audio_params": {
                        "sample_rate_hz": SAMPLE_RATE,
                        "num_channels": 1
                    }
                }
            }))

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

                        chunk_send_times.append(
                            time.time()
                        )

                        await ws.send(json.dumps({
                            "event_id": f"chunk_{chunk_num}",
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                chunk
                            ).decode("utf-8")
                        }))

                        print(
                            f"{filepath.name} | "
                            f"SENT CHUNK {chunk_num}"
                        )

                # finalize stream
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
# BATCH RUNNER
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


(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# python3 transcribe_parakeet_es_batch.py --base-url http://192.168.4.62:9000 --input-folder
 /home/re_nikitav/audio_maria --output-folder /home/re_nikitav/parakeet_es_results
TOTAL FILES = 15

STARTING -> maria1.mp3
WS URL -> ws://192.168.4.62:9000/v1/realtime
Traceback (most recent call last):
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_parakeet_es_batch.py", line 347, in <module>
    asyncio.run(
  File "/usr/lib/python3.11/asyncio/runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/asyncio/base_events.py", line 653, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_parakeet_es_batch.py", line 311, in run_batch
    await transcribe_file(
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_parakeet_es_batch.py", line 112, in transcribe_file
    async with websockets.connect(
  File "/home/re_nikitav/parakeet-asr-multilingual/env/lib/python3.11/site-packages/websockets/asyncio/client.py", line 590, in __aenter__
    return await self
           ^^^^^^^^^^
  File "/home/re_nikitav/parakeet-asr-multilingual/env/lib/python3.11/site-packages/websockets/asyncio/client.py", line 546, in __await_impl__
    await self.connection.handshake(
  File "/home/re_nikitav/parakeet-asr-multilingual/env/lib/python3.11/site-packages/websockets/asyncio/client.py", line 115, in handshake
    raise self.protocol.handshake_exc
  File "/home/re_nikitav/parakeet-asr-multilingual/env/lib/python3.11/site-packages/websockets/client.py", line 327, in parse
    self.process_response(response)
  File "/home/re_nikitav/parakeet-asr-multilingual/env/lib/python3.11/site-packages/websockets/client.py", line 144, in process_response
    raise InvalidStatus(response)
websockets.exceptions.InvalidStatus: server rejected WebSocket connection: HTTP 403
(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# 


INFO:uvicorn.error:('192.168.4.47', 53954) - "WebSocket /v1/realtime?intent=transcription" [accepted]
INFO:realtime.core.connection_manager:Client 35e54899-6b56-4e91-a381-d430a6ef49da connected at 2026-03-31 12:31:17
I0331 12:31:17.387586 16895 grpc_riva_asr.cc:341] ASR->RivaSpeechRecognitionConfig::
INFO:realtime.asr.inference:Updating Riva config with session config: modalities=[<Modality.TEXT: 'text'>] input_audio_format=<AudioFormat.PCM16: 'pcm16'> input_audio_transcription=InputAudioTranscriptionConfig(language='es-US', model='parakeet-0.6b-unified-ml-cs-es-US-asr-streaming-silero-vad-sortformer', prompt=None) input_audio_params=InputAudioParams(sample_rate_hz=16000, num_channels=1) recognition_config=RecognitionConfig(max_alternatives=1, enable_automatic_punctuation=False, enable_word_time_offsets=False, enable_profanity_filter=False, enable_verbatim_transcripts=False, custom_configuration='') speaker_diarization=SpeakerDiarizationConfig(enable_speaker_diarization=False, max_speaker_count=8) word_boosting=WordBoostingConfig(enable_word_boosting=False, word_boosting_list=[]) endpointing_config=EndpointingConfig(start_history=0, start_threshold=0, stop_history=0, stop_threshold=0, stop_history_eou=0, stop_threshold_eou=0) input_min_chunk_seconds=0.08 id='sess_8b3754e6-ddc7-4b58-ab1f-76c6833fe75e' object='realtime.transcription_session' client_secret=None
INFO:uvicorn.error:connection open
