import time
import json
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk

# ==========================
# CONFIG
# ==========================
SPEECH_KEY = "YOUR_KEY"
SPEECH_REGION = "eastus"

LANGUAGES = ["en-US", "es-US"]

AUDIO_FILE = "audio/maria1.mp3"


def transcribe_audio(file_path, language="en-US"):
    print("=" * 60)
    print(f"Testing Azure STT")
    print(f"File      : {file_path}")
    print(f"Language  : {language}")
    print("=" * 60)

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    speech_config.speech_recognition_language = language
    speech_config.output_format = speechsdk.OutputFormat.Detailed

    audio_config = speechsdk.audio.AudioConfig(filename=file_path)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    partial_results = []
    final_results = []

    start_time = time.time()
    first_partial_time = None
    first_final_time = None

    done = False

    def recognizing(evt):
        nonlocal first_partial_time

        if evt.result.text:
            now = time.time()

            if first_partial_time is None:
                first_partial_time = now

            latency = (now - start_time) * 1000

            partial_results.append({
                "text": evt.result.text,
                "latency_ms": latency
            })

            print(f"[PARTIAL {latency:.0f} ms] {evt.result.text}")

    def recognized(evt):
        nonlocal first_final_time

        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            now = time.time()

            if first_final_time is None:
                first_final_time = now

            latency = (now - start_time) * 1000

            final_results.append({
                "text": evt.result.text,
                "latency_ms": latency
            })

            print(f"[FINAL {latency:.0f} ms] {evt.result.text}")

    def stop(evt):
        nonlocal done
        done = True

    recognizer.recognizing.connect(recognizing)
    recognizer.recognized.connect(recognized)
    recognizer.session_stopped.connect(stop)
    recognizer.canceled.connect(stop)

    recognizer.start_continuous_recognition()

    while not done:
        time.sleep(0.2)

    recognizer.stop_continuous_recognition()

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"TTFT Partial : {(first_partial_time - start_time)*1000:.0f} ms" if first_partial_time else "No partial")
    print(f"TTFT Final   : {(first_final_time - start_time)*1000:.0f} ms" if first_final_time else "No final")
    print(f"Total Time   : {total_time:.2f} sec")

    final_transcript = " ".join([x["text"] for x in final_results])

    print("\nFINAL TRANSCRIPT:")
    print(final_transcript)

    return {
        "language": language,
        "partial_results": partial_results,
        "final_results": final_results,
        "final_transcript": final_transcript,
        "total_time_sec": total_time
    }


if __name__ == "__main__":
    for lang in LANGUAGES:
        transcribe_audio(AUDIO_FILE, lang)




pip install azure-cognitiveservices-speech
pip install python-dotenv mutagen
