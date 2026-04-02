```python
import asyncio
import websockets
import json
import base64

WS_URL = "ws://192.168.4.62:9000/v1/realtime?intent=transcription"
AUDIO_FILE = "/home/re_nikitav/audio_maria/maria1.mp3"

async def main():
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

        print("CONFIG SENT")

        # send one small fake chunk for quick test
        with open(AUDIO_FILE, "rb") as f:
            chunk = f.read(200000)

        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("utf-8")
        }))

        await ws.send(json.dumps({
            "type": "input_audio_buffer.commit"
        }))

        await ws.send(json.dumps({
            "type": "input_audio_buffer.done"
        }))

        print("AUDIO SENT")

        for _ in range(10):
            msg = await ws.recv()
            print("RAW RESPONSE:")
            print(msg)

asyncio.run(main())
```
