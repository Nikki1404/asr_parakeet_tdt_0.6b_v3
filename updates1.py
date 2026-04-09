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

        async def receive_task():
            try:
                async for msg in ws:
                    if isinstance(msg, str):
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

                # real-time speed
                await asyncio.sleep(
                    CHUNK_MS / 1000
                )

            # flush final transcript
            await asyncio.sleep(0.3)
            await ws.send(
                json.dumps({"cmd": "flush"})
            )

        await asyncio.gather(
            send_task(),
            receive_task()
        )


# =====================================
# MAIN
# =====================================
async def main():
    audio_file = "/home/nikita_verma2/0a12a9ea-af37-41ec-905f-3babb9580e97.wav"   # change file name
    await stream_parakeet(audio_file)


if __name__ == "__main__":
    asyncio.run(main())


