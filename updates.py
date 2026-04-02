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


for another model ths docker logs  how to do it for this 
I0331 08:51:15.842867  6829 riva_server.cc:307] Riva Conversational AI Server listening on 0.0.0.0:50051
W0331 08:51:15.842885  6829 stats_reporter.cc:41] No API key provided. Stats reporting disabled.
I0331 08:51:15.842888  6829 riva_server.cc:317] Metrics server disabled
INFO:inference:Riva gRPC Server is READY
INFO 2026-03-31 08:51:16.200 telemetry_handler.py:77] Telemetry mode TelemetryMode.OFF
WARNING:riva_http_server:No Offline ASR models found
INFO 2026-03-31 08:51:16.298 http_api.py:341] Starting HTTP Inference server, port: 9000, workers_count: 1, host: 0.0.0.0, log_level: info, SSL: disabled
INFO:uvicorn.error:Started server process [53]
INFO:uvicorn.error:Waiting for application startup.
INFO 2026-03-31 08:51:16.390 http_api.py:201] Readying serving endpoints:
  0.0.0.0:9000/v1/audio/list_voices (GET)
  0.0.0.0:9000/v1/audio/synthesize (POST)
  0.0.0.0:9000/v1/audio/synthesize_online (POST)
  0.0.0.0:9000/v1/audio/transcriptions (POST)
  0.0.0.0:9000/v1/audio/translations (POST)
  0.0.0.0:9000/v1/health/live (GET)
  0.0.0.0:9000/v1/health/ready (GET)
  0.0.0.0:9000/v1/metrics (GET)
  0.0.0.0:9000/v1/license (GET)
  0.0.0.0:9000/v1/metadata (GET)
  0.0.0.0:9000/v1/version (GET)
  0.0.0.0:9000/v1/manifest (GET)
  0.0.0.0:9000/v1/realtime/health (GET)
  0.0.0.0:9000/v1/realtime/transcription_sessions (POST)
  0.0.0.0:9000/v1/realtime/synthesis_sessions (POST)
INFO 2026-03-31 08:51:16.390 http_api.py:211] Welcome! Application is ready to receive API requests.
INFO:uvicorn.error:Application startup complete.
INFO:uvicorn.error:Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
WARNING:uvicorn.error:Invalid HTTP request received.
WARNING:uvicorn.error:Invalid HTTP request received.
WARNING:uvicorn.error:Invalid HTTP request received.
I0331 10:35:26.075955  6837 grpc_riva_asr.cc:413] ASRService.StreamingRecognize called.
I0331 10:35:26.082137  6837 grpc_riva_asr.cc:456] Using model parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming-silero-vad-sortformer from Triton localhost:8001 for inference
I0331 10:35:42.834745  6837 stats_builder.h:100] {"specversion":"1.0","type":"riva.asr.streamingrecognize.v1","source":"","subject":"","id":"41605a13-55b7-4e76-b00e-9243e08ef47e","datacontenttype":"application/json","time":"2026-03-31T10:35:26.075920594+00:00","data":{"release_version":"2.25.0","customer_uuid":"","ngc_org":"","ngc_team":"","ngc_org_team":"","container_uuid":"","language_code":"es-US","request_count":1,"audio_duration":57.81764221191406,"speech_duration":0.0,"status":0,"err_msg":""}}
I0331 10:36:47.776564  6848 grpc_riva_asr.cc:413] ASRService.StreamingRecognize called.
I0331 10:36:47.776744  6848 grpc_riva_asr.cc:456] Using model parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming-silero-vad-sortformer from Triton localhost:8001 for inference
I0331 10:36:50.743755  6848 stats_builder.h:100] {"specversion":"1.0","type":"riva.asr.streamingrecognize.v1","source":"","subject":"","id":"1b5bc6ca-ba67-4e51-aa7f-e14199e2ada0","datacontenttype":"application/json","time":"2026-03-31T10:36:47.77653716+00:00","data":{"release_version":"2.25.0","customer_uuid":"","ngc_org":"","ngc_team":"","ngc_org_team":"","container_uuid":"","language_code":"es-US","request_count":1,"audio_duration":19.040367126464845,"speech_duration":0.0,"status":0,"err_msg":""}}
I0331 10:57:54.863586  6848 grpc_riva_asr.cc:413] ASRService.StreamingRecognize called.
I0331 10:57:54.863758  6848 grpc_riva_asr.cc:456] Using model parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming-silero-vad-sortformer from Triton localhost:8001 for inference
I0331 10:57:57.887012  6848 stats_builder.h:100] {"specversion":"1.0","type":"riva.asr.streamingrecognize.v1","source":"","subject":"","id":"73311358-9455-49e8-bc3c-463d80501e4e","datacontenttype":"application/json","time":"2026-03-31T10:57:54.863527391+00:00","data":{"release_version":"2.25.0","customer_uuid":"","ngc_org":"","ngc_team":"","ngc_org_team":"","container_uuid":"","language_code":"es-US","request_count":1,"audio_duration":19.040367126464845,"speech_duration":0.0,"status":0,"err_msg":""}}
I0331 10:58:46.863085  6848 grpc_riva_asr.cc:413] ASRService.StreamingRecognize called.
I0331 10:58:46.863253  6848 grpc_riva_asr.cc:456] Using model parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming-silero-vad-sortformer from Triton localhost:8001 for inference
I0331 10:58:49.867614  6848 stats_builder.h:100] {"specversion":"1.0","type":"riva.asr.streamingrecognize.v1","source":"","subject":"","id":"b8c3a265-2a6e-435b-8376-e8b43ede1106","datacontenttype":"application/json","time":"2026-03-31T10:58:46.863058741+00:00","data":{"release_version":"2.25.0","customer_uuid":"","ngc_org":"","ngc_team":"","ngc_org_team":"","container_uuid":"","language_code":"es-US","request_count":1,"audio_duration":19.040367126464845,"speech_duration":0.0,"status":0,"err_msg":""}}
I0331 11:01:54.929942  6848 grpc_riva_asr.cc:413] ASRService.StreamingRecognize called.
I0331 11:01:54.930101  6848 grpc_riva_asr.cc:456] Using model parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming-silero-vad-sortformer from Triton localhost:8001 for inference
I0331 11:02:04.165468  6848 stats_builder.h:100] {"specversion":"1.0","type":"riva.asr.streamingrecognize.v1","source":"","subject":"","id":"bfc52bb8-262b-46ed-92d2-6c0174683d2d","datacontenttype":"application/json","time":"2026-03-31T11:01:54.929912963+00:00","data":{"release_version":"2.25.0","customer_uuid":"","ngc_org":"","ngc_team":"","ngc_org_team":"","container_uuid":"","language_code":"es-US","request_count":1,"audio_duration":57.81764221191406,"speech_duration":0.0,"status":0,"err_msg":""}}
WARNING:uvicorn.error:Invalid HTTP request received.
I0331 11:06:20.515439  6848 grpc_riva_asr.cc:413] ASRService.StreamingRecognize called.
I0331 11:06:20.515952  6848 grpc_riva_asr.cc:456] Using model parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming-silero-vad-sortformer from Triton localhost:8001 for inference
I0331 11:06:23.475203  6848 stats_builder.h:100] {"specversion":"1.0","type":"riva.asr.streamingrecognize.v1","source":"","subject":"","id":"dd1648d0-f139-4ac0-8b9d-7c0549d59f74","datacontenttype":"application/json","time":"2026-03-31T11:06:20.515415347+00:00","data":{"release_version":"2.25.0","customer_uuid":"","ngc_org":"","ngc_team":"","ngc_org_team":"","container_uuid":"","language_code":"es-US","request_count":1,"audio_duration":19.040367126464845,"speech_duration":0.0,"status":0,"err_msg":""}}
