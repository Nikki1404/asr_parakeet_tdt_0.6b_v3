import time
import subprocess
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk

# =========================================================
# CONFIG
# =========================================================

# Azure Speech Service Credentials
SPEECH_KEY = "YOUR_AZURE_SPEECH_KEY"
SPEECH_REGION = "eastus"

# Candidate languages for Auto Detect
CANDIDATE_LANGUAGES = [
    "en-US",
    "es-ES"
]

# Input file (mp3 / wav / flac etc.)
INPUT_AUDIO_FILE = "audio/maria1.mp3"


# =========================================================
# AUDIO CONVERSION
# =========================================================

def convert_to_wav(input_file):
    """
    Convert input audio to:
    WAV / PCM / 16kHz / Mono / 16-bit

    This is the most stable format for Azure STT.
    Uses FFmpeg directly via subprocess.
    """

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    output_file = str(input_path.with_suffix(".wav"))

    print("=" * 60)
    print("STEP 1: AUDIO CONVERSION")
    print("=" * 60)
    print(f"FROM : {input_file}")
    print(f"TO   : {output_file}")
    print()

    command = [
        "ffmpeg",
        "-y",                       # overwrite output file
        "-i", input_file,           # input file
        "-ar", "16000",             # sample rate
        "-ac", "1",                 # mono
        "-sample_fmt", "s16",       # 16-bit PCM
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print("Audio conversion completed successfully.\n")
        return output_file

    except subprocess.CalledProcessError as e:
        print("FFmpeg conversion failed.")
        print("Please make sure FFmpeg is installed.")
        print("Install using: winget install ffmpeg")
        raise e


# =========================================================
# AZURE AUTO-DETECT TRANSCRIPTION
# =========================================================

def transcribe_audio_auto_detect(file_path):
    """
    Azure STT with:
    - Auto language detection
    - Partial transcripts
    - Final transcripts
    - TTFT measurement
    - Final summary
    """

    print("=" * 60)
    print("STEP 2: AZURE STT AUTO LANGUAGE DETECTION")
    print("=" * 60)

    # -----------------------------------------
    # Convert to WAV first
    # -----------------------------------------

    wav_file = convert_to_wav(file_path)

    print(f"Using WAV File: {wav_file}")
    print(f"Candidate Languages: {CANDIDATE_LANGUAGES}")
    print()

    # -----------------------------------------
    # Azure Speech Config
    # -----------------------------------------

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    speech_config.output_format = speechsdk.OutputFormat.Detailed

    # Optional:
    # Helps endpointing / silence handling
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
        "800"
    )

    # -----------------------------------------
    # Auto Detect Config
    # -----------------------------------------

    auto_detect_source_language_config = (
        speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=CANDIDATE_LANGUAGES
        )
    )

    # -----------------------------------------
    # Audio Config
    # -----------------------------------------

    audio_config = speechsdk.audio.AudioConfig(
        filename=wav_file
    )

    # -----------------------------------------
    # Recognizer
    # -----------------------------------------

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config
    )

    # =====================================================
    # Variables
    # =====================================================

    partial_results = []
    final_results = []

    final_transcript = []
    detected_language = None

    first_partial_time = None
    first_final_time = None

    start_time = time.time()
    done = False

    # =====================================================
    # CALLBACKS
    # =====================================================

    def recognizing(evt):
        """
        Partial transcript callback
        """

        nonlocal first_partial_time
        nonlocal detected_language

        if evt.result.text:
            current_time = time.time()

            if first_partial_time is None:
                first_partial_time = current_time

            try:
                lang_result = speechsdk.AutoDetectSourceLanguageResult(
                    evt.result
                )
                detected_language = lang_result.language
            except:
                detected_language = "Unknown"

            latency = (current_time - start_time) * 1000

            partial_results.append({
                "text": evt.result.text,
                "latency_ms": round(latency, 2),
                "language": detected_language
            })

            print(
                f"[PARTIAL {latency:.0f} ms] "
                f"({detected_language}) "
                f"{evt.result.text}"
            )

    def recognized(evt):
        """
        Final transcript callback
        """

        nonlocal first_final_time
        nonlocal detected_language

        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            current_time = time.time()

            if first_final_time is None:
                first_final_time = current_time

            try:
                lang_result = speechsdk.AutoDetectSourceLanguageResult(
                    evt.result
                )
                detected_language = lang_result.language
            except:
                detected_language = "Unknown"

            latency = (current_time - start_time) * 1000

            final_results.append({
                "text": evt.result.text,
                "latency_ms": round(latency, 2),
                "language": detected_language
            })

            final_transcript.append(evt.result.text)

            print(
                f"[FINAL   {latency:.0f} ms] "
                f"({detected_language}) "
                f"{evt.result.text}"
            )

    def canceled(evt):
        """
        Error / cancellation callback
        """

        nonlocal done

        print("\nRecognition canceled.")

        try:
            details = evt.result.cancellation_details

            print(f"Reason      : {details.reason}")
            print(f"Error Code  : {details.error_code}")
            print(f"Error Detail: {details.error_details}")

        except Exception:
            print("Could not fetch cancellation details.")

        done = True

    def session_stopped(evt):
        """
        Recognition complete callback
        """

        nonlocal done

        print("\nSession stopped.")
        done = True

    # =====================================================
    # Connect Events
    # =====================================================

    recognizer.recognizing.connect(recognizing)
    recognizer.recognized.connect(recognized)
    recognizer.canceled.connect(canceled)
    recognizer.session_stopped.connect(session_stopped)

    # =====================================================
    # Start Recognition
    # =====================================================

    print("Starting recognition...\n")

    recognizer.start_continuous_recognition()

    while not done:
        time.sleep(0.2)

    recognizer.stop_continuous_recognition()

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"Detected Language : {detected_language}")

    if first_partial_time:
        partial_ttft = (first_partial_time - start_time) * 1000
        print(f"TTFT Partial      : {partial_ttft:.0f} ms")
    else:
        print("TTFT Partial      : No partial response")

    if first_final_time:
        final_ttft = (first_final_time - start_time) * 1000
        print(f"TTFT Final        : {final_ttft:.0f} ms")
    else:
        print("TTFT Final        : No final response")

    print(f"Total Time        : {total_time:.2f} sec")
    print(f"Final Segments    : {len(final_results)}")

    print()
    print("=" * 60)
    print("FINAL TRANSCRIPT")
    print("=" * 60)

    full_transcript = " ".join(final_transcript)

    if full_transcript.strip():
        print(full_transcript)
    else:
        print("No speech recognized.")

    print()

    return {
        "detected_language": detected_language,
        "partial_results": partial_results,
        "final_results": final_results,
        "final_transcript": full_transcript,
        "total_time_sec": round(total_time, 2)
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    transcribe_audio_auto_detect(INPUT_AUDIO_FILE)
this script is working for me 

so now for all use cases help me add these in script incrementally and let me know how can i get the observation and
difference because i need to document that as well that what gain I am seeing after adding those metrix and by testing
from original transcript and when adding one by one every use case and then what is the improvement if any after adding 
every usecase incrementally

and i have to try out these usecases to improve azure transcriptions which i am getting from above script using AzureSDK so basically i want to improve my transcription quality without changing the spoken words by user by trying out these usecase 

Provider	Phase	Task	Description	Outcome	Owner	Status	Priority
Azure	Setup	ASR Config Finalization	Lock language/locale, audio format (telephony/app), disable unnecessary auto‑detection	Stable, predictable recognition			High
Azure	Setup	Concurrency & Quota Validation	Validate concurrency limits, rate limits, and quotas	No runtime throttling			High
Azure	Integration	Real‑Time Socket Integration	Implement and validate WebSocket/streaming ingestion	Low‑latency real‑time ASR			High
Azure	Audio	VAD Evaluation & Tuning	Evaluate built‑in VAD behavior; tune sensitivity, silence thresholds, and endpointing	Reduced truncation and false cut‑offs			
Azure	Accuracy	Word / Phrase Boosting	Boost digits, identifiers, domain terms	Improved numeric accuracy			High
Azure	Accuracy	Transcript‑Based Vocabulary Tuning	Use sample transcripts to refine vocabulary/style boosting	Domain alignment			High
Azure	Logic	Numeric Handling Validation	Validate digit‑by‑digit vs grouped digit behavior	Reduced verification failures			High
Azure	Quality	Emotion / Tone Evaluation	Assess ASR behavior under neutral vs stressed speech	Robust recognition			High
Azure	Testing	Latency & Timeout Testing	Validate response times within conversational SLA	Smooth turn‑taking			High
Azure	Testing	Load & Concurrency Testing	Validate peak concurrent real‑time streams	Stable under load			High
Azure	Monitoring	Logging & Alerts Setup	Enable error, latency, socket‑drop monitoring	Early issue detection			High
Azure	Go‑Live	Fallback Validation	Test re‑prompt / DTMF / alternate flow	Resilient failure handling			High


and i want to check if any improvement we can see in baseline transcript like number is detected at numeric handling stage properly 
or short words are coming better or not at Word / Phrase Boosting stage or not  
and every stage should compare with the transcript we are getting without doing anything using above script 
also when doing Numeric Handling don't convert "to -> 2" as well because for that understand context first before converting word to number that if it is actually digit .
and also when writing documentation script mention Which stage or using multiple stages which helped in improving transcription quality should be used for production?
also detect transcription from noisy audio and short small words 
and for every stage mention what parameters are we trying out with and if multiple parametrs at one stage mention that  and document everything as well 
like progress of every stage , parameters used for that stage like what parameter did we use/change for respective stage name and comparison of all stage and 
then which are the stages which helped improving transcript quality from the original transcript recienved from above script but without loosing the eaxct spoken audio from trasncription don't change the words in transcription 
and also for spanish language transcription don't convert digit keep it in spanish only .


and at the end i need full table of comparison of all stage , parameters used to test that stage , what are the improvements we got , and pther important parameters 
and after trying every stage one by and comparing with baseline 
combine all stage at once and then generate transcript and compare with baseline to see the imporvement or not 
also try out by combining those stage which actually will show improvement according to production perspective so generate transcript by running those stage at once also and compare with baseline script and document everything in very detailed manner . 
also when documenting don't add transcript everytime it would make it lengthy just once at end 

  
