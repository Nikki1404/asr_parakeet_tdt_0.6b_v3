import time
import subprocess
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk

# ==========================
# CONFIG
# ==========================
SPEECH_KEY = "a919211feda747e0b8fxxxx"
SPEECH_REGION = "eastus"

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

    # FINAL SUMMARY
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


if __name__ == "__main__":
    transcribe_audio_auto_detect(INPUT_AUDIO_FILE)

now i want to benchmark azure via streaming audio and benchmark similar to this way but all files from maria_audio folder 

file_name	reference_txt	parakeet-tdt-0.6b-v3					
		ttft_ms	ttfb_ms	avg_latency_ms	wer	total_time_sec	transcription
0a12a9ea-af37-41ec-905f-3babb9580e97.wav	posturas pues yo creo que es algo de vocación o sea hay artistas que sí tienen esa vocación política y otro que no yo no yo no tengo ósea si yo puedo hacer una canción como zafra negra que hablaba de	61218.5	58269.9	50571.5444	21.95	181.7987	postura, pues yo creo que es algo de vocación, o sea hay artistas que sí tienen esa vocación política y otros que no, yo no, yo no tengo o sea sí yo puedo hacer una canción como Safra Negra que hablaba de



The agenda of the discussion was to understand the client-side configurations being used across all ASRs, identify the key bottlenecks, and explore the available options to overcome those bottlenecks.

At a high level, we aligned on a common flow for language handling across Google, Azure, and DeepGram. All three ASRs start in multilingual mode, allowing detection of both English and Spanish. At the beginning of the interaction, the user is explicitly asked which language they want to continue with. Even if the user speaks in either language initially, the ASRs are capable of detecting it. Once the first response is processed, the transcript is sent to the LLM agent, which determines the final language and returns a flag (ENUS or ESUS). Based on this, the system locks the language configuration for the rest of the session across these ASRs. This ensures consistency in the conversation. For NVIDIA/Parakeet, however, this locking is not applied and it continues with auto-detection.

For Google ASR, the discussion confirmed that the setup is straightforward, with the telephony model being used. It supports both multilingual and single-language configurations. The only meaningful configuration lever here is the language setting. When multiple languages are enabled, detect_language is required, but once the system is locked to a single language, this parameter has no effect regardless of its value. No major bottlenecks were identified for Google, and it behaves as expected.

For Azure ASR, the system is running via the SDK with minimal configurability. It follows the same multilingual-to-single-language locking approach as Google. However, Azure does not provide options like model selection, and most parameters remain default (such as 16 kHz sample rate). Because of this, while Azure is stable, it offers limited flexibility for tuning, and no major configuration bottlenecks were identified—just fewer options to optimize compared to other ASRs.

For DeepGram, the main discussion point was around endpointing, which was previously identified as a bottleneck. The default endpointing of 25 ms silence was too aggressive and caused premature finalization, making it feel like the system was interrupting the speaker. This was addressed by increasing endpointing to 200 ms of silence, which improved conversational flow significantly. DeepGram also follows the same multilingual start and then locks to a single language after the LLM decision. Endpointing remains the primary tuning parameter and key area of improvement.

For NVIDIA / Parakeet (WebSocket-based streaming), the discussion highlighted a key limitation. Unlike the other ASRs, it does not support dynamic language locking during an active session without reinitialization. As a result, it continues to rely entirely on auto language detection throughout the interaction. It was also confirmed that the gRPC-based approach has been completely dropped due to higher latency and slightly worse WER, and the WebSocket (v3) implementation is now preferred for better performance. The trade-off here is reduced configurability, especially around language control. It was agreed to integrate the WebSocket approach into the required options, and Sumeet will proceed with testing using the WebSocket-based Parakeet setup and share observations post-testing.

Overall, the discussion confirmed that most configurations are centered around language handling and endpointing, with limited tuning options in Azure, resolved endpointing issues in DeepGram, stable behavior in Google, and language control limitations in NVIDIA being the primary bottlenecks identified.
