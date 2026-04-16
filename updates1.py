this is the azure benchmarking script but i just want a script using this script and use all main things to test azure transcriptions how it's coming 

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk
from azure.identity import ClientSecretCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SPEECH_ENDPOINT = "https://cxaicoe-speechservice.cognitiveservices.azure.com"
OUTPUT_FOLDER = "azure-stt-files"
LATENCY_FOLDER = "azure-latency-logs"
AUDIO_FOLDER = "audio"

# Supported audio file extensions
SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']

class BinaryFileReaderCallback(speechsdk.audio.PullAudioInputStreamCallback):
    def __init__(self, filename: str):
        super().__init__()
        self._file_h = open(filename, "rb")

    def read(self, buffer: memoryview) -> int:
        try:
            size = buffer.nbytes
            frames = self._file_h.read(size)
            buffer[:len(frames)] = frames
            return len(frames)
        except Exception as ex:
            print(f'Exception in read: {ex}')
            raise

    def close(self) -> None:
        try:
            self._file_h.close()
        except Exception as ex:
            print(f'Exception in close: {ex}')
            raise

def get_auth_token():
    """Get Azure AAD authentication token"""
    tenant_id = os.getenv("AZURE_TENANT_ID", "dafe49bc-5ac3-4310-97b4-3e44a28cbf18")
    client_id = os.getenv("AZURE_CLIENT_ID", "d51ad24a-f6ca-4338-8a38-ca9f0323bf26") 
    client_secret = os.getenv("AZURE_CLIENT_SECRET", "ixxxxxxxxxxxxxxxxRELqThRKc7Ga5A")
    
    try:
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        token_response = credential.get_token("https://cognitiveservices.azure.com/.default")
        access_token = token_response.token
        print(f"Got AAD access token (length): {len(access_token)}")
        return access_token
    except Exception as e:
        print(f"Failed to get authentication token: {e}")
        raise

def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")
    if not os.path.exists(LATENCY_FOLDER):
        os.makedirs(LATENCY_FOLDER)
        print(f"Created latency directory: {LATENCY_FOLDER}")

def get_maria_audio_files():
    """Get list of audio files that start with 'maria' but exclude 'maria2.mp3'"""
    try:
        if not os.path.exists(AUDIO_FOLDER):
            print(f"Error: Audio folder not found - {AUDIO_FOLDER}")
            return []
        
        audio_files = []
        for file in os.listdir(AUDIO_FOLDER):
            file_path = os.path.join(AUDIO_FOLDER, file)
            
            # Check if it's a file (not directory)
            if not os.path.isfile(file_path):
                continue
            
            # Get file extension and base name
            file_ext = Path(file).suffix.lower()
            file_name = Path(file).name.lower()
            
            # Filter: starts with "maria", supported extension, but not "maria2.mp3"
            if (file_name.startswith('maria') and 
                file_ext in SUPPORTED_EXTENSIONS and 
                file_name != 'maria2.mp3'):
                audio_files.append(file_path)
        
        print(f"Found {len(audio_files)} audio files starting with 'maria' (excluding maria2.mp3):")
        for i, file in enumerate(audio_files, 1):
            print(f"  {i}. {Path(file).name}")
        
        return audio_files
    
    except Exception as e:
        print(f"Error reading audio folder: {e}")
        return []

def save_transcript(input_file, transcript, language):
    """Save transcript to file with specific naming convention"""
    try:
        ensure_output_directory()
        
        # Get base filename without extension
        base_name = Path(input_file).stem
        output_filename = f"{base_name}_cgwords.txt"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Write only the transcript content (no header)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript.strip())
        
        print(f"Transcript saved to: {output_path}")
        print(f"Total characters: {len(transcript)}")
        
    except Exception as e:
        print(f"Error saving transcript: {e}")

def get_audio_duration(file_path):
    """Get audio file duration in seconds"""
    try:
        # Try to import mutagen, install if not available
        try:
            from mutagen import File
            from mutagen.mp3 import MP3
        except ImportError:
            print("Installing mutagen library for audio duration detection...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mutagen"])
            from mutagen import File
            from mutagen.mp3 import MP3
        
        if file_path.lower().endswith('.mp3'):
            audio = MP3(file_path)
            return audio.info.length if audio.info else 0
        else:
            # For other formats, use mutagen's general File function
            audio = File(file_path)
            return audio.info.length if audio and audio.info else 0
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        return 0

def save_latency_metrics(input_file, latency_data):
    """Save latency metrics to JSON file"""
    try:
        ensure_output_directory()
        
        # Get base filename without extension
        base_name = Path(input_file).stem
        latency_filename = f"{base_name}_latency.json"
        latency_path = os.path.join(LATENCY_FOLDER, latency_filename)
        
        # Write latency data as JSON
        with open(latency_path, 'w', encoding='utf-8') as f:
            json.dump(latency_data, f, indent=2, default=str)
        
        print(f"Latency metrics saved to: {latency_path}")
        
    except Exception as e:
        print(f"Error saving latency metrics: {e}")

def transcribe_from_file(input_file, language='en-US'):
    """Transcribe audio file using pull stream approach"""
    print(f"\nTranscribing file: {Path(input_file).name} with language: {language}")
    print("-" * 50)
    
    # Check if file exists
    if not Path(input_file).exists():
        print(f"Error: File not found - {input_file}")
        return False
    
    # Variable to collect the complete transcript
    complete_transcript = []
    
    # Latency tracking variables
    latency_metrics = {
        'audio_file': input_file,
        'audio_duration_sec': get_audio_duration(input_file),
        'timestamp': datetime.now().isoformat(),
        'model': 'azure-speech-service',
        'language': language,
        'timing_metrics': {},
        'latencies': [],
        'summary': {}
    }
    
    # Timing variables
    start_time = time.time()
    connection_start_time = None
    session_started_time = None
    first_response_time = None
    first_final_time = None
    response_count = 0
    total_words = 0
    total_characters = 0
    response_latencies = []
    
    try:
        # Get authentication token
        connection_start_time = time.time()
        access_token = get_auth_token()
        connection_end_time = time.time()
        
        latency_metrics['timing_metrics']['connection_time_sec'] = connection_end_time - connection_start_time
        
        # Setup speech config
        speech_config = speechsdk.SpeechConfig(endpoint=SPEECH_ENDPOINT)
        speech_config.authorization_token = access_token
        speech_config.speech_recognition_language = language
        speech_config.output_format = speechsdk.OutputFormat.Detailed
        speech_config.enable_dictation()
        
        # Create pull stream for MP3
        file_extension = Path(input_file).suffix.lower()
        
        if file_extension == '.mp3':
            # Use compressed format for MP3
            compressed_format = speechsdk.audio.AudioStreamFormat(
                compressed_stream_format=speechsdk.AudioStreamContainerFormat.MP3
            )
            callback = BinaryFileReaderCallback(input_file)
            stream = speechsdk.audio.PullAudioInputStream(
                stream_format=compressed_format, 
                pull_stream_callback=callback
            )
            audio_config = speechsdk.audio.AudioConfig(stream=stream)
        else:
            # Use file directly for WAV/other formats
            audio_config = speechsdk.audio.AudioConfig(filename=input_file)
        
        # Create speech recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )
        
        # Control flag for session
        done = False
        
        def stop_cb(evt):
            """Callback to stop continuous recognition"""
            nonlocal done
            done = True
        
        # Setup event handlers
        def recognizing_handler(evt):
            if evt.result.text:
                print(f'RECOGNIZING: {evt.result.text}')
        
        def recognized_handler(evt):
            nonlocal response_count, total_words, total_characters, first_response_time, first_final_time
            
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                if evt.result.text:
                    current_time = time.time()
                    response_count += 1
                    
                    # Count words and characters
                    words_in_response = len(evt.result.text.split())
                    chars_in_response = len(evt.result.text)
                    total_words += words_in_response
                    total_characters += chars_in_response
                    
                    print(f'RECOGNIZED: {evt.result.text}')
                    
                    # Record first response timing
                    if first_response_time is None:
                        first_response_time = current_time
                        first_final_time = current_time
                        latency_metrics['timing_metrics']['first_response_latency_sec'] = first_response_time - start_time
                        latency_metrics['timing_metrics']['first_final_latency_sec'] = first_final_time - start_time
                        latency_metrics['timing_metrics']['first_byte_latency_sec'] = first_response_time - start_time
                    
                    # Record individual response latency
                    latency_data = {
                        'response_num': response_count,
                        'latency_from_start_ms': (current_time - start_time) * 1000,
                        'latency_from_session_start_ms': (current_time - session_started_time) * 1000 if session_started_time else 0,
                        'latency_from_first_response_ms': (current_time - first_response_time) * 1000 if first_response_time else 0,
                        'is_final': True,  # Azure Speech SDK recognized events are typically final
                        'words': words_in_response,
                        'char_count': chars_in_response
                    }
                    
                    latency_metrics['latencies'].append(latency_data)
                    response_latencies.append((current_time - start_time) * 1000)
                    
                    # Add recognized text to complete transcript
                    complete_transcript.append(evt.result.text)
            else:
                print('No speech recognized in this segment')
        
        def session_started_handler(evt):
            nonlocal session_started_time
            session_started_time = time.time()
            print('SESSION STARTED')
            
            # Record timing metrics
            latency_metrics['timing_metrics']['send_duration_sec'] = session_started_time - (connection_start_time + latency_metrics['timing_metrics']['connection_time_sec'])
            latency_metrics['timing_metrics']['time_to_first_chunk_sec'] = session_started_time - start_time
        
        def session_stopped_handler(evt):
            nonlocal latency_metrics
            session_end_time = time.time()
            
            print('SESSION STOPPED')
            
            # Calculate final timing metrics
            latency_metrics['total_processing_time_sec'] = session_end_time - start_time
            
            # Calculate summary statistics
            if response_latencies:
                latency_metrics['summary'] = {
                    'total_responses': len(latency_metrics['latencies']),
                    'final_responses': len(latency_metrics['latencies']),  # All Azure responses are final
                    'interim_responses': 0,
                    'total_words': total_words,
                    'total_characters': total_characters,
                    'avg_latency_from_start_ms': sum(response_latencies) / len(response_latencies),
                    'min_latency_from_start_ms': min(response_latencies),
                    'max_latency_from_start_ms': max(response_latencies)
                }
            else:
                latency_metrics['summary'] = {
                    'total_responses': 0,
                    'final_responses': 0,
                    'interim_responses': 0,
                    'total_words': 0,
                    'total_characters': 0,
                    'avg_latency_from_start_ms': 0,
                    'min_latency_from_start_ms': 0,
                    'max_latency_from_start_ms': 0
                }
            
            # Save the complete transcript when session ends
            final_transcript = ' '.join(complete_transcript)
            if final_transcript.strip():
                save_transcript(input_file, final_transcript, language)
                print(f"Transcription completed for {Path(input_file).name}")
            else:
                # Save empty file with note if no speech detected
                save_transcript(input_file, "No speech content detected in this audio file.", language)
                print(f"No speech detected in {Path(input_file).name}")
            
            # Save latency metrics
            save_latency_metrics(input_file, latency_metrics)
            
            # Print summary
            print(f"\nLatency Summary:")
            print(f"  Total processing time: {latency_metrics['total_processing_time_sec']:.2f}s")
            print(f"  Audio duration: {latency_metrics['audio_duration_sec']:.2f}s")
            if latency_metrics['audio_duration_sec'] > 0:
                real_time_factor = latency_metrics['total_processing_time_sec'] / latency_metrics['audio_duration_sec']
                print(f"  Real-time factor: {real_time_factor:.2f}x")
            print(f"  Total responses: {latency_metrics['summary']['total_responses']}")
            if response_latencies:
                print(f"  Average latency: {latency_metrics['summary']['avg_latency_from_start_ms']:.0f}ms")
        
        def canceled_handler(evt):
            nonlocal latency_metrics
            session_end_time = time.time()
            
            print(f'CANCELED: {evt.result.cancellation_details.reason}')
            if evt.result.cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f'Error Code: {evt.result.cancellation_details.error_code}')
                print(f'Error Details: {evt.result.cancellation_details.error_details}')
            
            # Calculate final timing metrics even for canceled sessions
            latency_metrics['total_processing_time_sec'] = session_end_time - start_time
            latency_metrics['canceled'] = True
            latency_metrics['cancellation_reason'] = str(evt.result.cancellation_details.reason)
            latency_metrics['error_code'] = str(evt.result.cancellation_details.error_code) if evt.result.cancellation_details.error_code else None
            latency_metrics['error_details'] = evt.result.cancellation_details.error_details
            
            # Calculate summary statistics for partial results
            if response_latencies:
                latency_metrics['summary'] = {
                    'total_responses': len(latency_metrics['latencies']),
                    'final_responses': len(latency_metrics['latencies']),
                    'interim_responses': 0,
                    'total_words': total_words,
                    'total_characters': total_characters,
                    'avg_latency_from_start_ms': sum(response_latencies) / len(response_latencies),
                    'min_latency_from_start_ms': min(response_latencies),
                    'max_latency_from_start_ms': max(response_latencies)
                }
            else:
                latency_metrics['summary'] = {
                    'total_responses': 0,
                    'final_responses': 0,
                    'interim_responses': 0,
                    'total_words': 0,
                    'total_characters': 0,
                    'avg_latency_from_start_ms': 0,
                    'min_latency_from_start_ms': 0,
                    'max_latency_from_start_ms': 0
                }
            
            # Save partial transcript even if canceled
            partial_transcript = ' '.join(complete_transcript)
            if partial_transcript.strip():
                partial_transcript += "\n\n[Note: Recognition was canceled before completion]"
                save_transcript(input_file, partial_transcript, language)
            else:
                # Save error info if no transcript
                error_info = f"Recognition failed.\nReason: {evt.result.cancellation_details.reason}\nError Code: {evt.result.cancellation_details.error_code}\nError Details: {evt.result.cancellation_details.error_details}"
                save_transcript(input_file, error_info, language)
            
            # Save latency metrics even for canceled sessions
            save_latency_metrics(input_file, latency_metrics)
        
        # Connect event handlers
        speech_recognizer.recognizing.connect(recognizing_handler)
        speech_recognizer.recognized.connect(recognized_handler)
        speech_recognizer.session_started.connect(session_started_handler)
        speech_recognizer.session_stopped.connect(session_stopped_handler)
        speech_recognizer.canceled.connect(canceled_handler)
        
        # Connect stop callbacks
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)
        
        # Start recognition
        print("Starting continuous recognition...")
        speech_recognizer.start_continuous_recognition()
        
        # Wait until done
        try:
            while not done:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Recognition interrupted by user")
            return False
        
        # Properly stop and cleanup
        print("Stopping recognition...")
        speech_recognizer.stop_continuous_recognition()
        
        # Close stream if it's a pull stream
        if file_extension == '.mp3':
            callback.close()
        
        print("Session closed successfully")
        return True
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return False

def batch_transcribe_files(language='en-US'):
    """Process all maria audio files in batch"""
    print("=" * 60)
    print("Azure Speech-to-Text Batch Processor")
    print("Processing files starting with 'maria' (excluding maria2.mp3)")
    print("=" * 60)
    
    # Get list of audio files
    audio_files = get_maria_audio_files()
    
    if not audio_files:
        print("No audio files found to process")
        return
    
    print(f"\nStarting batch processing with language: {language}")
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*20} Processing file {i}/{len(audio_files)} {'='*20}")
        
        success = transcribe_from_file(audio_file, language)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Add delay between files to avoid rate limiting
        if i < len(audio_files):
            print("Waiting 2 seconds before next file...")
            time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETED")
    print(f"Total files processed: {len(audio_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output folder: {os.path.abspath(OUTPUT_FOLDER)}")
    print("=" * 60)

def parse_arguments():
    """Parse command line arguments"""    
    # If no arguments, run batch processing
    if len(sys.argv) < 2:
        return None, 'en-US', True  # No file, default language, batch mode
    
    # Check if first argument is a language flag
    if sys.argv[1] in ['--language', '-l']:
        if len(sys.argv) < 3:
            print("Error: Language flag requires a language code")
            sys.exit(1)
        return None, sys.argv[2], True  # No file, specified language, batch mode
    
    # First argument is a file
    input_file = sys.argv[1]
    language = 'en-US'  # Default language
    
    # Parse language argument
    for i in range(2, len(sys.argv)):
        if sys.argv[i] in ['--language', '-l'] and i + 1 < len(sys.argv):
            language = sys.argv[i + 1]
            break
    
    return input_file, language, False  # Specific file, language, single file mode

def main():
    """Main function"""
    try:
        input_file, language, batch_mode = parse_arguments()
        
        if batch_mode:
            # Batch processing mode
            batch_transcribe_files(language)
        else:
            # Single file processing mode
            transcribe_from_file(input_file, language)
            
    except KeyboardInterrupt:
        print("\nTranscription interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    # print("Usage examples:")
    # print("  python azure-transcribe.py                           # Process all maria files (excluding maria2.mp3)")
    # print("  python azure-transcribe.py --language es-ES          # Batch process with Spanish")
    # print("  python azure-transcribe.py single_file.mp3           # Process single file")
    # print("  python azure-transcribe.py single_file.mp3 -l es-ES  # Process single file with Spanish")
    # print()
    main()



and i was provided for this 
speech_key="xxxxxxxxxxxxxxxxxxx78dcc9363",
speech_region="eastus",
language=["en-US", "es-US"]



and i have to try out these things to improve azure transcriptions 
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


so first give script for trnscription testing and then one by one give suggestions to do improvements in transcription regarding given field I had just provided. if possible or not that too. 

    
    
