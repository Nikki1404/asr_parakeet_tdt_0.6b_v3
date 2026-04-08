import asyncio
import json
import time
import warnings
import wave
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


# =====================================================
# AUDIO CONVERSION
# =====================================================
def convert_to_wav(filepath: Path) -> Path:
    """
    Convert input audio to 16kHz mono 16-bit PCM WAV.
    If the file is already .wav, it will still be normalized and rewritten.
    """
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
    """
    Measures:
      - TTFT: first response containing text
      - TTFB: first final response containing text
      - Total time: end-to-end time for the file
    """
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
        # Initial config
        await ws.send(json.dumps({"sample_rate": TARGET_SR}))
        send_start = time.time()

        # Stream audio
        for i in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[i:i + chunk_bytes]
            await ws.send(chunk)

            # Try receiving any immediate partial/final response
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

        # End of stream
        await ws.send(json.dumps({"eof": True}))

        # Drain remaining responses
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

    # Join all text responses. Depending on server behavior this may contain
    # repeated partials; left as-is to preserve raw model output behavior.
    transcript = " ".join([t for t in transcript_parts if t]).strip()

    return {
        "ttft": ttft if ttft is not None else total_time,
        "ttfb": ttfb if ttfb is not None else total_time,
        "total_time": total_time,
        "transcript": transcript
    }


# =====================================================
# RIVA BENCHMARK (parakeet_ctc_0.6b_es)
# =====================================================
def benchmark_ctc_riva(wav_file: Path) -> dict:
    """
    Offline Riva ASR:
      TTFT = TTFB = total_time
    """
    import riva.client
    from riva.client import RecognitionConfig, AudioEncoding

    with wave.open(str(wav_file), "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        audio_bytes = wf.readframes(wf.getnframes())

    if sample_rate != 16000:
        raise ValueError(f"Expected 16k WAV, got {sample_rate} Hz in {wav_file}")
    if num_channels != 1:
        raise ValueError(f"Expected mono WAV, got {num_channels} channels in {wav_file}")
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sampwidth} in {wav_file}")

    auth = riva.client.Auth(uri=RIVA_URI)
    asr_service = riva.client.ASRService(auth)

    config = RecognitionConfig(
        encoding=AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,
        audio_channel_count=1,
        language_code="es-US",
        enable_automatic_punctuation=True,
        max_alternatives=1,
        verbatim_transcripts=False,
    )

    start = time.time()

    response = asr_service.offline_recognize(
        audio_bytes=audio_bytes,
        config=config
    )

    total_time = round(time.time() - start, 3)

    transcript_parts = []
    for result in response.results:
        if result.alternatives:
            transcript_parts.append(result.alternatives[0].transcript.strip())

    transcript = " ".join([x for x in transcript_parts if x]).strip()

    return {
        "ttft": total_time,
        "ttfb": total_time,
        "total_time": total_time,
        "transcript": transcript
    }


# =====================================================
# EXCEL EXPORT
# =====================================================
def save_grouped_excel(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="benchmark")


# =====================================================
# MAIN
# =====================================================
async def main() -> None:
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

        # TDT benchmark
        tdt = await benchmark_tdt(wav_file)

        # CTC benchmark
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
