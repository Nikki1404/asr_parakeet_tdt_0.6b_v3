import asyncio
import json
import logging
import sys

import numpy as np
import resampy
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.config import load_config, Config, MODEL_MAP
from app.factory import build_engine
from app.streaming_session import StreamingSession
from app.asr_engines.base import ASREngine


cfg = load_config()

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("asr_server")
app = FastAPI()
ENGINE_CACHE: dict[str, ASREngine] = {}


async def preload_engines():
    log.info("Preloading ASR engines...")

    for backend, model_name in MODEL_MAP.items():
        try:
            log.info("Initializing engine: %s", backend)

            tmp_cfg = Config()
            object.__setattr__(tmp_cfg, "asr_backend", backend)
            object.__setattr__(tmp_cfg, "model_name", model_name)
            object.__setattr__(tmp_cfg, "device", cfg.device)
            object.__setattr__(tmp_cfg, "sample_rate", cfg.sample_rate)

            engine = build_engine(tmp_cfg)
            load_sec = engine.load()

            ENGINE_CACHE[backend] = engine
            log.info("Preloaded %s in %.2fs", backend, load_sec)

        except Exception:
            log.exception("Failed to preload %s", backend)

    log.info("All engines preloaded.")


@app.on_event("startup")
async def startup_event():
    log.info("Server startup initiated")
    await preload_engines()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_backends": list(ENGINE_CACHE.keys()),
        "sample_rate": cfg.sample_rate,
        "device": cfg.device,
    }


def get_engine(backend: str) -> ASREngine:
    if backend not in ENGINE_CACHE:
        raise ValueError(f"Engine {backend} not loaded")
    return ENGINE_CACHE[backend]


@app.websocket("/asr/realtime-custom-vad")
async def ws_asr(ws: WebSocket):
    log.info("WS connection request from %s", ws.client)
    await ws.accept()

    try:
        init = json.loads(await ws.receive_text())
    except Exception:
        await ws.close(code=4001)
        return

    backend = init.get("backend", cfg.asr_backend)
    client_sample_rate = int(init.get("sample_rate", cfg.sample_rate))

    if backend not in MODEL_MAP:
        await ws.close(code=4000)
        return

    engine = get_engine(backend)

    def upsample_if_needed(pcm: bytes):
        if not pcm or client_sample_rate == cfg.sample_rate:
            return pcm

        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        y = resampy.resample(x, client_sample_rate, cfg.sample_rate)
        y = np.clip(y, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16).tobytes()

    session = StreamingSession(engine, cfg)

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            data = msg.get("bytes")
            text_msg = msg.get("text")

            if text_msg:
                try:
                    obj = json.loads(text_msg)
                    if obj.get("eos") is True:
                        loop = asyncio.get_running_loop()
                        events = await loop.run_in_executor(None, session.flush)

                        for ev_type, text, lang, ttfb in events:
                            await ws.send_text(json.dumps({
                                "type": ev_type,
                                "text": text,
                                "language": lang,
                                "t_start": ttfb
                            }))
                        continue
                except Exception:
                    pass

            if data is None:
                continue

            data = upsample_if_needed(data)

            loop = asyncio.get_running_loop()
            events = await loop.run_in_executor(None, session.process_chunk, data)

            for ev_type, text, lang, ttfb in events:
                await ws.send_text(json.dumps({
                    "type": ev_type,
                    "text": text,
                    "language": lang,
                    "t_start": ttfb
                }))

    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s", ws.client)
    except Exception:
        log.exception("Error during websocket session")
    finally:
        try:
            loop = asyncio.get_running_loop()
            events = await loop.run_in_executor(None, session.flush)

            for ev_type, text, lang, ttfb in events:
                await ws.send_text(json.dumps({
                    "type": ev_type,
                    "text": text,
                    "language": lang,
                    "t_start": ttfb
                }))
        except Exception:
            pass

        log.info("WS session closed for %s", ws.client)