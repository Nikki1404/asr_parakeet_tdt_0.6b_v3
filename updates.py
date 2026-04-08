import asyncio
import json
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import websockets
from jiwer import wer

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================
WS_URL = "ws://192.168.4.62:8001/ws"
RIVA_URI = "192.168.4.62:50051"

AUDIO_FOLDER = Path("./audio_samples")
OUTPUT_FILE = "benchmark_grouped_report.xlsx"

TARGET_SR = 16000
CHUNK_MS = 80
WS_RECV_TIMEOUT_SEC = 2

AUDIO_EXTENSIONS = {".ogg", ".wav", ".flac", ".mp3", ".m4a"}

# Riva / CTC model settings
RIVA_LANGUAGE_CODE = "es-ES"
RIVA_MODEL_NAME = "parakeet-ctc-0.6b-es"

# 100 ms of 16k mono PCM16 = 1600 samples = 3200 bytes
RIVA_STREAM_CHUNK_BYTES = 3200


# =====================================================
# AUDIO CONVERSION
# =====================================================
def convert_to_wav(filepath: Path) -> Path:
    audio, _ = librosa.load(
        str(filepath),
        sr=TARGET_SR,
        mono=True
    )

    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    wav_path = filepath.with_suffix(".wav")

    sf.write(
        str(wav_path),
        pcm16,
        TARGET_SR,
        subtype="PCM_16"
    )

    return wav_path


# =====================================================
# REFERENCE TEXT
# =====================================================
def load_reference_text(audio_file: Path) -> str:
    txt_file = AUDIO_FOLDER / f"{audio_file.stem}.txt"

    if not txt_file.exists():
        raise FileNotFoundError(f"Reference text missing -> {txt_file}")

    return txt_file.read_text(encoding="utf-8").strip()


# =====================================================
# TEXT NORMALIZATION / WER
# =====================================================
def normalize_text(text: str) -> str:
    text = text.lower()
    for ch in [".", ",", "!", "?", ";", ":", '"', "'", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, "")
    text = text.replace("\n", " ")
    return " ".join(text.split())


def calculate_wer(reference: str, prediction: str) -> float:
    ref = normalize_text(reference)
    pred = normalize_text(prediction)
    return round(wer(ref, pred), 4)


# =====================================================
# WEBSOCKET BENCHMARK (parakeet_tdt_0.6b_v3)
# =====================================================
async def benchmark_tdt(wav_file: Path) -> dict:
    audio, _ = sf.read(str(wav_file))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    pcm_bytes = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16).tobytes()

    chunk_bytes = TARGET_SR * 2 * CHUNK_MS // 1000

    transcript_parts = []
    ttft = None
    ttfb = None

    overall_start = time.time()

    async with websockets.connect(
        WS_URL,
        ping_interval=20,
        ping_timeout=120,
        max_size=None
    ) as ws:
        await ws.send(json.dumps({"sample_rate": TARGET_SR}))
        send_start = time.time()

        for i in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[i:i + chunk_bytes]
            await ws.send(chunk)

            try:
                response = await asyncio.wait_for(
                    ws.recv(),
                    timeout=WS_RECV_TIMEOUT_SEC
                )
                current_time = time.time()
                data = json.loads(response)

                text = str(data.get("text", "")).strip()
                is_final = bool(data.get("is_final", False))

                if text:
                    transcript_parts.append(text)

                    if ttft is None:
                        ttft = round(current_time - send_start, 3)

                    if is_final and ttfb is None:
                        ttfb = round(current_time - send_start, 3)

            except asyncio.TimeoutError:
                pass

        await ws.send(json.dumps({"eof": True}))

        while True:
            try:
                response = await asyncio.wait_for(
                    ws.recv(),
                    timeout=WS_RECV_TIMEOUT_SEC
                )
                current_time = time.time()
                data = json.loads(response)

                text = str(data.get("text", "")).strip()
                is_final = bool(data.get("is_final", False))

                if text:
                    transcript_parts.append(text)

                    if ttft is None:
                        ttft = round(current_time - send_start, 3)

                    if is_final and ttfb is None:
                        ttfb = round(current_time - send_start, 3)

            except asyncio.TimeoutError:
                break
            except Exception:
                break

    total_time = round(time.time() - overall_start, 3)
    transcript = " ".join([t for t in transcript_parts if t]).strip()

    return {
        "ttft": ttft if ttft is not None else total_time,
        "ttfb": ttfb if ttfb is not None else total_time,
        "total_time": total_time,
        "transcript": transcript
    }


# =====================================================
# RIVA STREAMING BENCHMARK (parakeet-ctc-0.6b-es)
# =====================================================
def benchmark_ctc_riva(wav_file: Path) -> dict:
    import riva.client
    from riva.client import RecognitionConfig, StreamingRecognitionConfig, AudioEncoding

    audio, sr = sf.read(str(wav_file), dtype="int16")

    if sr != TARGET_SR:
        raise ValueError(f"Expected {TARGET_SR} Hz WAV, got {sr} Hz in {wav_file}")

    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.int16)

    audio_bytes = audio.tobytes()

    def audio_chunks():
        for i in range(0, len(audio_bytes), RIVA_STREAM_CHUNK_BYTES):
            yield audio_bytes[i:i + RIVA_STREAM_CHUNK_BYTES]

    auth = riva.client.Auth(uri=RIVA_URI)
    asr_service = riva.client.ASRService(auth)

    streaming_config = StreamingRecognitionConfig(
        config=RecognitionConfig(
            encoding=AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=TARGET_SR,
            audio_channel_count=1,
            language_code=RIVA_LANGUAGE_CODE,
            model=RIVA_MODEL_NAME,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            verbatim_transcripts=False,
        ),
        interim_results=True,
    )

    ttft = None
    ttfb = None
    transcript_parts = []

    start_time = time.time()

    responses = asr_service.streaming_response_generator(
        audio_chunks=audio_chunks(),
        streaming_config=streaming_config
    )

    for response in responses:
        now = time.time()

        if not getattr(response, "results", None):
            continue

        for result in response.results:
            if not result.alternatives:
                continue

            text = result.alternatives[0].transcript.strip()
            if not text:
                continue

            transcript_parts.append(text)

            if ttft is None:
                ttft = round(now - start_time, 3)

            if getattr(result, "is_final", False) and ttfb is None:
                ttfb = round(now - start_time, 3)

    total_time = round(time.time() - start_time, 3)
    transcript = " ".join([x for x in transcript_parts if x]).strip()

    return {
        "ttft": ttft if ttft is not None else total_time,
        "ttfb": ttfb if ttfb is not None else total_time,
        "total_time": total_time,
        "transcript": transcript
    }


# =====================================================
# EXCEL EXPORT
# =====================================================
def save_grouped_excel(rows):
    df = pd.DataFrame(rows)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="benchmark")


# =====================================================
# MAIN
# =====================================================
async def main():
    if not AUDIO_FOLDER.exists():
        raise FileNotFoundError(f"Audio folder not found: {AUDIO_FOLDER.resolve()}")

    rows = []

    audio_files = sorted(
        [
            f for f in AUDIO_FOLDER.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        ]
    )

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {AUDIO_FOLDER.resolve()}")

    for file in audio_files:
        print(f"Processing -> {file.name}", flush=True)

        ref_text = load_reference_text(file)
        wav_file = convert_to_wav(file)

        tdt = await benchmark_tdt(wav_file)
        ctc = benchmark_ctc_riva(wav_file)

        row = {
            ("", "file_name"): file.name,
            ("", "ref_txt"): ref_text,

            ("parakeet_tdt_0.6b_v3", "ttft"): tdt["ttft"],
            ("parakeet_tdt_0.6b_v3", "ttfb"): tdt["ttfb"],
            ("parakeet_tdt_0.6b_v3", "total_time"): tdt["total_time"],
            ("parakeet_tdt_0.6b_v3", "wer"): calculate_wer(ref_text, tdt["transcript"]),
            ("parakeet_tdt_0.6b_v3", "transcript"): tdt["transcript"],

            ("parakeet_ctc_0.6b_es", "ttft"): ctc["ttft"],
            ("parakeet_ctc_0.6b_es", "ttfb"): ctc["ttfb"],
            ("parakeet_ctc_0.6b_es", "total_time"): ctc["total_time"],
            ("parakeet_ctc_0.6b_es", "wer"): calculate_wer(ref_text, ctc["transcript"]),
            ("parakeet_ctc_0.6b_es", "transcript"): ctc["transcript"],
        }

        rows.append(row)

    save_grouped_excel(rows)
    print(f"Saved -> {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())




import asyncio, websockets

async def test():
    async with websockets.connect("ws://34.118.200.125:8001/ws") as ws:
        print("Connected")

asyncio.run(test())



import riva.client
auth = riva.client.Auth(uri="34.118.200.125:50051")
print("Riva connected")

(venv) PS C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\parakeet_client_testing> Test-NetConnection 34.118.200.125 -Port 8001


ComputerName     : 34.118.200.125
RemoteAddress    : 34.118.200.125
RemotePort       : 8001
InterfaceAlias   : Ethernet 2
SourceAddress    : 10.16.11.56
TcpTestSucceeded : True


