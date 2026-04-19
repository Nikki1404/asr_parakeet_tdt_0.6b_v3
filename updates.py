import os
import time
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment

# ==========================
# CONFIG
# ==========================
SPEECH_KEY = "YOUR_KEY"
SPEECH_REGION = "eastus"

CANDIDATE_LANGUAGES = ["en-US", "es-ES"]

INPUT_AUDIO_FILE = "audio/maria1.mp3"


def convert_to_wav(input_file):
    """
    Convert audio to:
    WAV / PCM / 16kHz / Mono / 16-bit
    """

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    output_file = str(input_path.with_suffix(".wav"))

    print(f"\nConverting audio:")
    print(f"FROM: {input_file}")
    print(f"TO  : {output_file}")

    audio = AudioSegment.from_file(input_file)

    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)  # 16-bit

    audio.export(output_file, format="wav")

    print("Conversion completed.\n")

    return output_file


def transcribe_audio_auto_detect(file_path):
    print("=" * 60)
    print("Azure STT Auto Language Detection")
    print("=" * 60)

    # Convert first
    wav_file = convert_to_wav(file_path)

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

    audio_config = speechsdk.audio.AudioConfig(filename=wav_file)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config
    )

    final_transcript = []
    detected_language = None
    first_partial_time = None
    first_final_time = None
    start_time = time.time()
    done = False

    def recognizing(evt):
        nonlocal first_partial_time, detected_language

        if evt.result.text:
            if first_partial_time is None:
                first_partial_time = time.time()

            lang_result = speechsdk.AutoDetectSourceLanguageResult(evt.result)
            detected_language = lang_result.language

            latency = (time.time() - start_time) * 1000

            print(f"[PARTIAL {latency:.0f} ms] ({detected_language}) {evt.result.text}")

    def recognized(evt):
        nonlocal first_final_time, detected_language

        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            if first_final_time is None:
                first_final_time = time.time()

            lang_result = speechsdk.AutoDetectSourceLanguageResult(evt.result)
            detected_language = lang_result.language

            latency = (time.time() - start_time) * 1000

            print(f"[FINAL {latency:.0f} ms] ({detected_language}) {evt.result.text}")

            final_transcript.append(evt.result.text)

    def stop(evt):
        nonlocal done
        done = True

    recognizer.recognizing.connect(recognizing)
    recognizer.recognized.connect(recognized)
    recognizer.session_stopped.connect(stop)
    recognizer.canceled.connect(stop)

    print("Starting recognition...\n")

    recognizer.start_continuous_recognition()

    while not done:
        time.sleep(0.2)

    recognizer.stop_continuous_recognition()

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Detected Language : {detected_language}")

    if first_partial_time:
        print(f"TTFT Partial      : {(first_partial_time - start_time)*1000:.0f} ms")

    if first_final_time:
        print(f"TTFT Final        : {(first_final_time - start_time)*1000:.0f} ms")

    print(f"Total Time        : {total_time:.2f} sec")

    print("\nFINAL TRANSCRIPT:")
    print(" ".join(final_transcript))


if __name__ == "__main__":
    transcribe_audio_auto_detect(INPUT_AUDIO_FILE)

(azure_test_env) PS C:\Users\re_nikitav\Documents\azure_asr_test> winget install --id Gyan.FFmpeg --source winget
Found FFmpeg [Gyan.FFmpeg] Version 8.1
This application is licensed to you by its owner.
Microsoft is not responsible for, nor does it grant any licenses to, third-party packages.
Downloading https://github.com/GyanD/codexffmpeg/releases/download/8.1/ffmpeg-8.1-full_build.zip
  ██████████████████████████████   236 MB /  236 MB
Successfully verified installer hash
Extracting archive...
Successfully extracted archive
Starting package install...
Path environment variable modified; restart your shell to use the new value.
Command line alias added: "ffmpeg"
Command line alias added: "ffplay"
Command line alias added: "ffprobe"
Successfully installed
(azure_test_env) PS C:\Users\re_nikitav\Documents\azure_asr_test> ffmpeg --version
ffmpeg : The term 'ffmpeg' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the
spelling of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ ffmpeg --version
+ ~~~~~~
    + CategoryInfo          : ObjectNotFound: (ffmpeg:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

(azure_test_env) PS C:\Users\re_nikitav\Documents\azure_asr_test> ffmpeg -version
ffmpeg : The term 'ffmpeg' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the
spelling of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ ffmpeg -version
+ ~~~~~~
    + CategoryInfo          : ObjectNotFound: (ffmpeg:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

(azure_test_env) PS C:\Users\re_nikitav\Documents\azure_asr_test>
