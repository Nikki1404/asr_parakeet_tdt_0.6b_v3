import time
import wave
from pathlib import Path

import pandas as pd
from jiwer import wer
import riva.client

# =====================================================
# CONFIG
# =====================================================
RIVA_URI = "34.118.200.125:50051"

AUDIO_FOLDER = Path(
    r"C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\audio_samples"
)

OUTPUT_FILE = "parakeet_ctc_benchmark.xlsx"

TARGET_SR = 16000
LANGUAGE = "es-US"


# =====================================================
# TEXT NORMALIZATION
# =====================================================
def normalize_text(text):
    return " ".join(
        text.lower().replace("\n", " ").split()
    )


def calculate_wer(ref, pred):
    return round(
        wer(normalize_text(ref), normalize_text(pred)),
        4
    )


# =====================================================
# LOAD REFERENCE TEXT
# =====================================================
def load_reference_text(audio_file):
    txt_file = AUDIO_FOLDER / f"{audio_file.stem}.txt"

    if not txt_file.exists():
        raise FileNotFoundError(
            f"Missing GT file -> {txt_file}"
        )

    return txt_file.read_text(
        encoding="utf-8"
    ).strip()


# =====================================================
# RIVA CTC BENCHMARK
# =====================================================
def benchmark_ctc(wav_file):
    auth = riva.client.Auth(uri=RIVA_URI)

    asr_service = riva.client.ASRService(auth)

    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=TARGET_SR,
            language_code=LANGUAGE,
            max_alternatives=1,
            enable_automatic_punctuation=True
        ),
        interim_results=True
    )

    start_time = time.time()

    # IMPORTANT FIX:
    # Read ONLY PCM frames, no WAV header
    with wave.open(wav_file, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

        if sample_rate != TARGET_SR:
            raise ValueError(
                f"Expected {TARGET_SR} Hz, got {sample_rate}"
            )

        if channels != 1:
            raise ValueError(
                f"Expected mono audio, got {channels} channels"
            )

        if sample_width != 2:
            raise ValueError(
                f"Expected 16-bit PCM, got {sample_width * 8}-bit"
            )

        audio_data = wf.readframes(wf.getnframes())

    # 80 ms chunking
    chunk_size = TARGET_SR * 2 * 80 // 1000

    chunks = [
        audio_data[i:i + chunk_size]
        for i in range(0, len(audio_data), chunk_size)
    ]

    ttft = None
    ttfb = None
    final_transcript = []

    responses = asr_service.streaming_response_generator(
        audio_chunks=chunks,
        streaming_config=config
    )

    for response in responses:
        current_time = time.time()

        for result in response.results:
            if not result.alternatives:
                continue

            transcript = (
                result.alternatives[0]
                .transcript
                .strip()
            )

            if transcript:

                if ttft is None:
                    ttft = round(
                        current_time - start_time,
                        3
                    )

                if result.is_final:
                    final_transcript.append(transcript)

                    if ttfb is None:
                        ttfb = round(
                            current_time - start_time,
                            3
                        )

    total_time = round(
        time.time() - start_time,
        3
    )

    transcript_text = " ".join(final_transcript)

    return {
        "ttft": ttft or total_time,
        "ttfb": ttfb or total_time,
        "total_time": total_time,
        "transcript": transcript_text
    }


# =====================================================
# MAIN
# =====================================================
def main():
    rows = []

    wav_files = sorted(
        AUDIO_FOLDER.glob("*.wav")
    )

    if not wav_files:
        raise FileNotFoundError(
            f"No wav files found in {AUDIO_FOLDER}"
        )

    for file in wav_files:
        print(f"Processing -> {file.name}")

        ref_text = load_reference_text(file)

        result = benchmark_ctc(str(file))

        rows.append({
            "file_name": file.name,
            "ref_txt": ref_text,
            "ttft": result["ttft"],
            "ttfb": result["ttfb"],
            "total_time": result["total_time"],
            "wer": calculate_wer(
                ref_text,
                result["transcript"]
            ),
            "transcript": result["transcript"]
        })

    df = pd.DataFrame(rows)

    df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
