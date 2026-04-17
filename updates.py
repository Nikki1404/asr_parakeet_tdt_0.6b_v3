import time
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk

# ==========================
# CONFIG
# ==========================
SPEECH_KEY = "xxxxxxxxxxxxxxxxxxx78dcc9363"
SPEECH_REGION = "eastus"

CANDIDATE_LANGUAGES = ["en-US", "es-US"]
AUDIO_FILE = "audio/maria1.mp3"


def transcribe_audio_auto_detect(file_path):
    print("=" * 60)
    print("Testing Azure STT with Auto Language Detection")
    print(f"File      : {file_path}")
    print(f"Candidates: {CANDIDATE_LANGUAGES}")
    print("=" * 60)

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    speech_config.output_format = speechsdk.OutputFormat.Detailed

    auto_detect_source_language_config = (
        speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=CANDIDATE_LANGUAGES
        )
    )

    audio_config = speechsdk.audio.AudioConfig(filename=file_path)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config
    )

    partial_results = []
    final_results = []

    start_time = time.time()
    first_partial_time = None
    first_final_time = None
    detected_language = None
    done = False

    def recognizing(evt):
        nonlocal first_partial_time, detected_language

        if evt.result.text:
            now = time.time()

            if first_partial_time is None:
                first_partial_time = now

            auto_lang = speechsdk.AutoDetectSourceLanguageResult(evt.result)
            detected_language = auto_lang.language

            latency = (now - start_time) * 1000
            partial_results.append({
                "text": evt.result.text,
                "latency_ms": latency,
                "detected_language": detected_language
            })

            print(f"[PARTIAL {latency:.0f} ms] ({detected_language}) {evt.result.text}")

    def recognized(evt):
        nonlocal first_final_time, detected_language

        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            now = time.time()

            if first_final_time is None:
                first_final_time = now

            auto_lang = speechsdk.AutoDetectSourceLanguageResult(evt.result)
            detected_language = auto_lang.language

            latency = (now - start_time) * 1000
            final_results.append({
                "text": evt.result.text,
                "latency_ms": latency,
                "detected_language": detected_language
            })

            print(f"[FINAL {latency:.0f} ms] ({detected_language}) {evt.result.text}")

    def canceled(evt):
        nonlocal done
        print("Recognition canceled.")
        if evt.result and evt.result.cancellation_details:
            print(f"Reason: {evt.result.cancellation_details.reason}")
            print(f"Details: {evt.result.cancellation_details.error_details}")
        done = True

    def stopped(evt):
        nonlocal done
        done = True

    recognizer.recognizing.connect(recognizing)
    recognizer.recognized.connect(recogned := recognized)
    recognizer.session_stopped.connect(stopped)
    recognizer.canceled.connect(canceled)

    recognizer.start_continuous_recognition()

    while not done:
        time.sleep(0.2)

    recognizer.stop_continuous_recognition()

    total_time = time.time() - start_time
    final_transcript = " ".join([x["text"] for x in final_results])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Detected Lang: {detected_language}")
    print(f"TTFT Partial : {(first_partial_time - start_time)*1000:.0f} ms" if first_partial_time else "No partial")
    print(f"TTFT Final   : {(first_final_time - start_time)*1000:.0f} ms" if first_final_time else "No final")
    print(f"Total Time   : {total_time:.2f} sec")

    print("\nFINAL TRANSCRIPT:")
    print(final_transcript)

    return {
        "detected_language": detected_language,
        "partial_results": partial_results,
        "final_results": final_results,
        "final_transcript": final_transcript,
        "total_time_sec": total_time
    }


if __name__ == "__main__":
    transcribe_audio_auto_detect(AUDIO_FILE)



pip install azure-cognitiveservices-speech
pip install python-dotenv mutagen


wss://parakeet-custom-vad-150916788856.us-central1.run.app/ws
