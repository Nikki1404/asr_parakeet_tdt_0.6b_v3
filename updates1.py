import asyncio
import json
import logging
import websockets
import soundfile as sf
import numpy as np
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# =====================================
# CONFIG
# =====================================
WEBSOCKET_ADDRESS = "ws://192.168.4.38:8001/ws"
TARGET_SR = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = TARGET_SR * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2


# =====================================
# AUDIO LOADER
# =====================================
def load_audio(filepath: str):
    audio, sr = sf.read(filepath, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR
        )

    pcm = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16)

    return pcm.tobytes()


# =====================================
# STREAM FUNCTION
# =====================================
async def stream_parakeet(audio_file: str):
    pcm_audio = load_audio(audio_file)

    async with websockets.connect(
        WEBSOCKET_ADDRESS,
        max_size=None
    ) as ws:

        print(f"\nConnected to {WEBSOCKET_ADDRESS}")
        print("🎙 Streaming in real time...\n")

        final_received = asyncio.Event()

        async def receive_task():
            try:
                async for msg in ws:
                    print(f"\nRAW -> {msg}")   # debug

                    if isinstance(msg, bytes):
                        msg = msg.decode("utf-8")

                    obj = json.loads(msg)

                    typ = obj.get("type")
                    txt = obj.get("text", "")

                    ts = time.strftime("%H:%M:%S")

                    if typ == "partial":
                        print(
                            f"\r⏳ [{ts}] {txt:<100}",
                            end="",
                            flush=True
                        )

                    elif typ in ["transcript", "final"]:
                        print("\r" + " " * 120, end="\r")
                        print(f"✅ [{ts}] FINAL: {txt}")

                        final_received.set()
                        break

            except websockets.exceptions.ConnectionClosed:
                print("\nConnection closed")

        async def send_task():
            offset = 0

            while offset < len(pcm_audio):
                chunk = pcm_audio[
                    offset: offset + CHUNK_BYTES
                ]
                offset += CHUNK_BYTES

                if len(chunk) < CHUNK_BYTES:
                    chunk += bytes(
                        CHUNK_BYTES - len(chunk)
                    )

                await ws.send(chunk)

                await asyncio.sleep(
                    CHUNK_MS / 1000
                )

            # send flush
            await asyncio.sleep(0.5)
            await ws.send(
                json.dumps({"cmd": "flush"})
            )

            print("\nFlush sent. Waiting for final transcript...")

            # wait max 10 sec for final
            try:
                await asyncio.wait_for(
                    final_received.wait(),
                    timeout=10
                )
            except asyncio.TimeoutError:
                print("\nTimeout waiting for final transcript")

        recv = asyncio.create_task(receive_task())
        send = asyncio.create_task(send_task())

        await send
        await recv



# =====================================
# MAIN
# =====================================
async def main():
    audio_file = "/home/nikita_verma2/0a12a9ea-af37-41ec-905f-3babb9580e97.wav"
    await stream_parakeet(audio_file)

✅ [18:20:33] FINAL: My sister in law quiere que haga, you know, she she's a linguist and so she wants to sort of track, you know, Spanglish. So she decided me if I would wear it at work.
TimeoutError: timed out while closing connection

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/nikita_verma2/parakeet-asr-multilingual/websocket_test.py", line 128, in <module>
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
  File "/home/nikita_verma2/parakeet-asr-multilingual/websocket_test.py", line 124, in main
    await stream_parakeet(audio_file)
  File "/home/nikita_verma2/parakeet-asr-multilingual/websocket_test.py", line 113, in stream_parakeet
    await asyncio.gather(
  File "/home/nikita_verma2/parakeet-asr-multilingual/websocket_test.py", line 100, in send_task
    await ws.send(chunk)
  File "/home/nikita_verma2/parakeet-asr-multilingual/env/lib/python3.11/site-packages/websockets/asyncio/connection.py", line 485, in send
    async with self.send_context():
  File "/usr/lib/python3.11/contextlib.py", line 204, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/nikita_verma2/parakeet-asr-multilingual/env/lib/python3.11/site-packages/websockets/asyncio/connection.py", line 965, in send_context
    raise self.protocol.close_exc from original_exc
websockets.exceptions.ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout; no close frame received
if __name__ == "__main__":
    asyncio.run(main())



