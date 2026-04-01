```python id="kpjlwm"
import requests
from pathlib import Path
import time
import json

FILE_PATH = Path("/home/re_nikitav/audio_maria/maria1.mp3")
URL = "http://127.0.0.1:9000/v1/audio/transcriptions"

start = time.time()

with open(FILE_PATH, "rb") as f:
    response = requests.post(
        URL,
        files={
            "file": (
                FILE_PATH.name,
                f,
                "audio/mpeg"
            )
        },
        data={
            "language": "es"
        },
        timeout=7200
    )

print("STATUS:", response.status_code)
print("TEXT:", response.text[:1000])

response.raise_for_status()

result = response.json()

latency_sec = time.time() - start

print(json.dumps({
    "latency_sec": round(latency_sec, 2),
    "result": result
}, indent=2))
```
curl -X POST http://127.0.0.1:9000/v1/audio/transcriptions \
  -F "file=@/home/re_nikitav/audio_maria/maria1.mp3" \
  -F "language=es-US"



this i got from logs of deployed model from docker 
INFO:uvicorn.error:connection open
INFO:realtime.asr.inference:Updating Riva config with session config: modalities=[<Modality.TEXT: 'text'>] input_audio_format=<AudioFormat.NONE: 'none'> input_audio_transcription=InputAudioTranscriptionConfig(language='es-US', model='parakeet-0.6b-unified-ml-cs-es-US-asr-streaming-silero-vad-sortformer', prompt=None) input_audio_params=InputAudioParams(sample_rate_hz=44100, num_channels=1) recognition_config=RecognitionConfig(max_alternatives=1, enable_automatic_punctuation=False, enable_word_time_offsets=False, enable_profanity_filter=False, enable_verbatim_transcripts=False, custom_configuration='') speaker_diarization=SpeakerDiarizationConfig(enable_speaker_diarization=False, max_speaker_count=8) word_boosting=WordBoostingConfig(enable_word_boosting=False, word_boosting_list=[]) endpointing_config=EndpointingConfig(start_history=0, start_threshold=0.0, stop_history=0, stop_threshold=0.0, stop_history_eou=0, stop_threshold_eou=0.0) input_min_chunk_seconds=0.08
INFO:realtime.asr.inference:Starting inference task for client 35e54899-6b56-4e91-a381-d430a6ef49da
I0331 12:31:17.397459  1212 grpc_riva_asr.cc:413] ASRService.StreamingRecognize called.
I0331 12:31:17.400738  1212 grpc_riva_asr.cc:456] Using model parakeet-0.6b-unified-ml-cs-es-US-asr-streaming-silero-vad-sortformer from Triton localhost:8001 for inference
INFO:realtime.asr.inference:Received end-of-stream marker for client 35e54899-6b56-4e91-a381-d430a6ef49da. Total chunks sent to server: 525
but i am getting this 

(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# python3 transcribe_test.py 
STATUS: 400
TEXT: {"detail":"Bad Request, bad model: parakeet-0.6b-unified-ml-cs-es-US-asr-streaming-silero-vad-sortformer"}
Traceback (most recent call last):
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_test.py", line 30, in <module>
    response.raise_for_status()
  File "/home/re_nikitav/parakeet-asr-multilingual/env/lib/python3.11/site-packages/requests/models.py", line 1028, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: http://127.0.0.1:9000/v1/audio/transcriptions
(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# 
