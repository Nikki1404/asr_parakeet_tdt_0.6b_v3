import asyncio
import json
import sys

import sounddevice as sd
import websockets


WS_URL = "ws://127.0.0.1:8002/asr/realtime-custom-vad"
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCK_MS = 80
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)


async def send_audio(ws):
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[mic-status] {status}", file=sys.stderr)
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    ):
        print("Recording started. Press Ctrl+C to stop.")
        while True:
            chunk = await q.get()
            await ws.send(chunk)


async def recv_msgs(ws):
    while True:
        msg = await ws.recv()
        obj = json.loads(msg)

        kind = obj.get("type")
        text = obj.get("text", "")
        lang = obj.get("language", "en")
        t_start = obj.get("t_start")

        if kind == "partial":
            print(f"[PARTIAL][{lang}] {text} (t_start={t_start} ms)")
        elif kind == "final":
            print(f"[FINAL][{lang}] {text} (t_start={t_start} ms)")
        else:
            print(obj)


async def main():
    async with websockets.connect(
        WS_URL,
        max_size=10_000_000,
        ping_interval=20,
        ping_timeout=120,
    ) as ws:
        await ws.send(json.dumps({
            "backend": "parakeet",
            "sample_rate": SAMPLE_RATE,
        }))

        sender = asyncio.create_task(send_audio(ws))
        receiver = asyncio.create_task(recv_msgs(ws))

        done, pending = await asyncio.wait(
            [sender, receiver],
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for task in pending:
            task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")