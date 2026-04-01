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
        timeout=7200
    )

print("STATUS:", response.status_code)
print("TEXT:", response.text[:500])

response.raise_for_status()

result = response.json()

latency_sec = time.time() - start

print(json.dumps({
    "latency_sec": round(latency_sec, 2),
    "result": result
}, indent=2))

getting this 
(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# python3 transcribe_test.py 
STATUS: 400
TEXT: {"detail":"Bad Request, need model or language"}
Traceback (most recent call last):
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_test.py", line 27, in <module>
    response.raise_for_status()
  File "/home/re_nikitav/parakeet-asr-multilingual/env/lib/python3.11/site-packages/requests/models.py", line 1028, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: http://127.0.0.1:9000/v1/audio/transcriptions
