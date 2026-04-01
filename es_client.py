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
            "language": "es-US"
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
