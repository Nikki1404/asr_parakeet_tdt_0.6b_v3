import asyncio
import json
import logging
import websockets
import soundfile as sf
import numpy as np

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
# MAIN STREAM FUNCTION
# =====================================
async def stream_parakeet(audio_file: str):
    event_queue = asyncio.Queue()

    pcm_audio = load_audio(audio_file)

    async with websockets.connect(
        WEBSOCKET_ADDRESS,
        max_size=None
    ) as ws:

        logger.info("Connected to websocket")

        async def receive_task():
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        obj = json.loads(msg)

                        typ = obj.get("type")
                        txt = obj.get("text", "")

                        logger.info(
                            f"Websocket received msg: {txt}, type: {typ}"
                        )

                        if typ == "partial":
                            await event_queue.put({
                                "type": "INTERIM_TRANSCRIPT",
                                "text": txt
                            })

                        elif typ in ["transcript", "final"]:
                            await event_queue.put({
                                "type": "FINAL_TRANSCRIPT",
                                "text": txt
                            })
                            break

            except websockets.exceptions.ConnectionClosed:
                pass

            # signal completion
            await event_queue.put(None)

        async def send_task():
            try:
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

                # final flush
                await asyncio.sleep(0.3)

                await ws.send(
                    json.dumps({"cmd": "flush"})
                )

            except Exception as e:
                logger.info(f"Error occurred: {e}")

        stask = asyncio.create_task(send_task())
        rtask = asyncio.create_task(receive_task())

        try:
            while True:
                event = await event_queue.get()

                if event is None:
                    break

                yield event

        finally:
            stask.cancel()
            rtask.cancel()


# =====================================
# TEST RUNNER
# =====================================
async def main():
    audio_file = "/home/nikita_verma2/maria1.wav"  # change this

    async for event in stream_parakeet(audio_file):
        print(f"\nEVENT RECEIVED -> {event}")


if __name__ == "__main__":
    asyncio.run(main())
