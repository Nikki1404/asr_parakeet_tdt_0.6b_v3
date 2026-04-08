import asyncio
import websockets
import time
import json
import pandas as pd
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from jiwer import wer
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================
WS_URL = "ws://192.168.4.62:8001/ws"
RIVA_URI = "192.168.4.62:50051"

AUDIO_FOLDER = Path("./audio_samples")

TARGET_SR = 16000
CHUNK_MS = 80

OUTPUT_FILE = "benchmark_grouped_report.xlsx"

AUDIO_EXTENSIONS = [".ogg", ".wav", ".flac", ".mp3"]


# =====================================================
# CONVERT AUDIO TO WAV
# =====================================================
def convert_to_wav(filepath):
    audio, _ = librosa.load(
        str(filepath),
        sr=TARGET_SR,
        mono=True
    )

    wav_path = filepath.with_suffix(".wav")

    sf.write(
        str(wav_path),
        audio,
        TARGET_SR
    )

    return wav_path


# =====================================================
# LOAD REFERENCE TEXT
# =====================================================
def load_reference_text(audio_file):
    txt_file = AUDIO_FOLDER / f"{audio_file.stem}.txt"

    if not txt_file.exists():
        raise FileNotFoundError(
            f"Reference text missing -> {txt_file.name}"
        )

    with open(txt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


# =====================================================
# TEXT NORMALIZATION
# =====================================================
def normalize_text(text):
    return " ".join(
        text.lower()
        .replace("\n", " ")
        .replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .replace("?", "")
        .replace(";", "")
        .replace(":", "")
        .split()
    )


# =====================================================
# WER
# =====================================================
def calculate_wer(reference, prediction):
    return round(
        wer(
            normalize_text(reference),
            normalize_text(prediction)
        ),
        4
    )


# =====================================================
# WEBSOCKET BENCHMARK (TDT)
# =====================================================
async def benchmark_tdt(wav_file):
    audio, _ = sf.read(str(wav_file))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    pcm = (
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
        ping_timeout=120
    ) as ws:

        await ws.send(json.dumps({
            "sample_rate": TARGET_SR
        }))

        send_start = time.time()

        for i in range(0, len(pcm), chunk_bytes):
            chunk = pcm[i:i + chunk_bytes]

            await ws.send(chunk)

            try:
                response = await asyncio.wait_for(
                    ws.recv(),
                    timeout=2
                )

                current_time = time.time()

                data = json.loads(response)

                text = data.get("text", "").strip()

                if text:
                    transcript_parts.append(text)

                    if ttft is None:
                        ttft = round(
                            current_time - send_start,
                            3
                        )

                    if (
                        data.get("is_final", False)
                        and ttfb is None
                    ):
                        ttfb = round(
                            current_time - send_start,
                            3
                        )

            except asyncio.TimeoutError:
                pass

        await ws.send(json.dumps({"eof": True}))

        while True:
            try:
                response = await asyncio.wait_for(
                    ws.recv(),
                    timeout=2
                )

                current_time = time.time()

                data = json.loads(response)

                text = data.get("text", "").strip()

                if text:
                    transcript_parts.append(text)

                    if (
                        data.get("is_final", False)
                        and ttfb is None
                    ):
                        ttfb = round(
                            current_time - send_start,
                            3
                        )

            except:
                break

    total_time = round(
        time.time() - overall_start,
        3
    )

    return {
        "ttft": ttft if ttft else total_time,
        "ttfb": ttfb if ttfb else total_time,
        "total_time": total_time,
        "transcript": " ".join(transcript_parts)
    }


# =====================================================
# RIVA BENCHMARK (CTC)
# =====================================================
def benchmark_ctc_riva(wav_file):
    import riva.client

    auth = riva.client.Auth(uri=RIVA_URI)

    asr_service = riva.client.ASRService(auth)

    start = time.time()

    response = asr_service.offline_recognize(
        audio_file=str(wav_file),
        language_code="es-US"
    )

    total_time = round(
        time.time() - start,
        3
    )

    transcript = (
        response.results[0]
        .alternatives[0]
        .transcript
    )

    return {
        "ttft": total_time,
        "ttfb": total_time,
        "total_time": total_time,
        "transcript": transcript
    }


# =====================================================
# MAIN
# =====================================================
async def main():
    rows = []

    for file in AUDIO_FOLDER.iterdir():

        if file.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        print(f"Processing -> {file.name}")

        wav_file = convert_to_wav(file)

        ref_text = load_reference_text(file)

        # ---------------------------------------------
        # TDT
        # ---------------------------------------------
        tdt = await benchmark_tdt(wav_file)

        # ---------------------------------------------
        # CTC
        # ---------------------------------------------
        ctc = benchmark_ctc_riva(wav_file)

        row = {
            ("", "file_name"): file.name,
            ("", "ref_txt"): ref_text,

            ("parakeet_tdt_0.6b_v3", "ttft"): tdt["ttft"],
            ("parakeet_tdt_0.6b_v3", "ttfb"): tdt["ttfb"],
            ("parakeet_tdt_0.6b_v3", "total_time"): tdt["total_time"],
            ("parakeet_tdt_0.6b_v3", "wer"): calculate_wer(
                ref_text,
                tdt["transcript"]
            ),
            ("parakeet_tdt_0.6b_v3", "transcript"): tdt["transcript"],

            ("parakeet_ctc_0.6b_es", "ttft"): ctc["ttft"],
            ("parakeet_ctc_0.6b_es", "ttfb"): ctc["ttfb"],
            ("parakeet_ctc_0.6b_es", "total_time"): ctc["total_time"],
            ("parakeet_ctc_0.6b_es", "wer"): calculate_wer(
                ref_text,
                ctc["transcript"]
            ),
            ("parakeet_ctc_0.6b_es", "transcript"): ctc["transcript"]
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    df.columns = pd.MultiIndex.from_tuples(df.columns)

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:
        df.to_excel(writer, index=False)

    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())


gettign this 
(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# tail -f benchmark_run.log
nohup: ignoring input
Processing -> Monologue.ogg
Traceback (most recent call last):
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_benchmark_client.py", line 314, in <module>
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
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_benchmark_client.py", line 273, in main
    ctc = benchmark_ctc_riva(wav_file)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_benchmark_client.py", line 224, in benchmark_ctc_riva
    response = asr_service.offline_recognize(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: ASRService.offline_recognize() got an unexpected keyword argument 'audio_file'
