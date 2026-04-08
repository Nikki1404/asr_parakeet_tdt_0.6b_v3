import subprocess
import time
import riva.client

SERVER = "192.168.4.62:50051"
INPUT_MP3 = "/home/re_nikitav/audio_maria/maria1.mp3"
TEMP_WAV = "/tmp/maria1_stream.wav"

SAMPLE_RATE = 16000

# 160 ms chunk
CHUNK_MS = 160
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000

# FAST SEND SPEED
SEND_DELAY_MS = 20

LANGUAGE = "es-US"

print("Converting mp3 -> wav...")

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        INPUT_MP3,
        "-t", "30",
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        TEMP_WAV
    ],
    check=True
)

print("Reading wav...")

with open(TEMP_WAV, "rb") as f:
    wav_bytes = f.read()

pcm = wav_bytes[44:]

print(f"PCM bytes = {len(pcm)}")


def audio_stream():
    offset = 0
    chunk_num = 0

    while offset < len(pcm):
        chunk = pcm[offset: offset + CHUNK_BYTES]
        offset += CHUNK_BYTES
        chunk_num += 1

        print(
            f"SENDING CHUNK {chunk_num} | "
            f"bytes={len(chunk)}"
        )

        yield chunk

        # FASTER THAN REAL TIME
        time.sleep(SEND_DELAY_MS / 1000.0)


auth = riva.client.Auth(uri=SERVER)
asr_service = riva.client.ASRService(auth)

config = riva.client.StreamingRecognitionConfig(
    config=riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE,
        max_alternatives=1,
        enable_automatic_punctuation=False
    ),
    interim_results=True
)

print("Starting streaming gRPC...")

start_time = time.time()
first_response_time = None
response_num = 0

responses = asr_service.streaming_response_generator(
    audio_chunks=audio_stream(),
    streaming_config=config
)

for response in responses:
    now = time.time()

    print("RAW RESPONSE RECEIVED")

    if not response.results:
        print("NO RESULTS")
        continue

    response_num += 1

    latency_ms = (now - start_time) * 1000

    if first_response_time is None:
        first_response_time = latency_ms
        print(
            f"TTFT = {first_response_time:.2f} ms"
        )

    for result in response.results:
        text = ""

        if result.alternatives:
            text = result.alternatives[0].transcript

        print(
            f"RESP {response_num} | "
            f"LAT={latency_ms:.2f} ms"
        )
        print(f"TEXT: {text}")
        print(f"FINAL: {result.is_final}")
        print("-" * 80)

print("COMPLETED")


(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# tail -f benchmark_run.log
nohup: ignoring input
Processing -> 0a12a9ea-af37-41ec-905f-3babb9580e97.flac
Trying Riva offline with language_code=es-US
Offline failed for language_code=es-US: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.INVALID_ARGUMENT
        details = "Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; "
        debug_error_string = "UNKNOWN:Error received from peer ipv4:192.168.4.62:50051 {grpc_status:3, grpc_message:"Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; "}"
>
Trying Riva offline with language_code=es-ES
Offline failed for language_code=es-ES: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.INVALID_ARGUMENT
        details = "Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; "
        debug_error_string = "UNKNOWN:Error received from peer ipv4:192.168.4.62:50051 {grpc_message:"Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; ", grpc_status:3}"
>
Trying Riva offline with language_code=es
Offline failed for language_code=es: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.INVALID_ARGUMENT
        details = "Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; "
        debug_error_string = "UNKNOWN:Error received from peer ipv4:192.168.4.62:50051 {grpc_message:"Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; ", grpc_status:3}"
>
Trying Riva streaming with language_code=es-US
Streaming failed for language_code=es-US: <_MultiThreadedRendezvous of RPC that terminated with:
        status = StatusCode.UNKNOWN
        details = "Exception iterating requests!"
        debug_error_string = "None"
>
Trying Riva streaming with language_code=es-ES
Streaming failed for language_code=es-ES: <_MultiThreadedRendezvous of RPC that terminated with:
        status = StatusCode.UNKNOWN
        details = "Exception iterating requests!"
        debug_error_string = "None"
>
Trying Riva streaming with language_code=es
Streaming failed for language_code=es: <_MultiThreadedRendezvous of RPC that terminated with:
        status = StatusCode.UNKNOWN
        details = "Exception iterating requests!"
        debug_error_string = "None"
>
Traceback (most recent call last):
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_benchmark_client.py", line 419, in <module>
    asyncio.run(main())
  File "/usr/lib/python3.11/asyncio/runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/asyncio/base_events.py", line 653, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_benchmark_client.py", line 393, in main
    ctc = benchmark_ctc_riva(wav_file)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_benchmark_client.py", line 353, in benchmark_ctc_riva
    raise RuntimeError(f"Riva ASR failed for all modes/language codes. Last error: {last_error}")
RuntimeError: Riva ASR failed for all modes/language codes. Last error: <_MultiThreadedRendezvous of RPC that terminated with:
        status = StatusCode.UNKNOWN
        details = "Exception iterating requests!"
        debug_error_string = "None"
>


these are DOcker logs from server for parakeet-ctc-0.6b-es model 
ne; 
I0408 14:25:23.556370 996197 stats_builder.h:100] {"specversion":"1.0","type":"riva.asr.recognize.v1","source":"","subject":"","id":"f8222c3e-73f1-493e-8648-b51f94f97f43","datacontenttype":"application/json","time":"2026-04-08T14:25:23.555960021+00:00","data":{"release_version":"2.24.0","customer_uuid":"","ngc_org":"","ngc_team":"","ngc_org_team":"","container_uuid":"","language_code":"es-ES","request_count":1,"audio_duration":0.0,"speech_duration":0.0,"status":3,"err_msg":"Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; "}}
I0408 14:25:23.639611 996197 grpc_riva_asr.cc:244] ASRService.Recognize called.
E0408 14:25:23.639964 996197 grpc_riva_asr.cc:264] Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; 

 
