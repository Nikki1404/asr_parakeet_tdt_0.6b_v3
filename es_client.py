import subprocess
import wave
import riva.client

SERVER = "192.168.4.62:50051"
INPUT_MP3 = "/home/re_nikitav/audio_maria/maria1.mp3"
TEMP_WAV = "/tmp/maria1_test.wav"
SAMPLE_RATE = 16000
CHUNK_SEC = 30
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_SEC

print("Converting mp3 -> wav...")

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        INPUT_MP3,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        TEMP_WAV
    ],
    check=True
)

print("Reading wav...")

with open(TEMP_WAV, "rb") as f:
    wav_bytes = f.read()

# skip WAV header
pcm = wav_bytes[44:]

print(f"PCM bytes = {len(pcm)}")

def audio_stream():
    offset = 0
    chunk_num = 0
    while offset < len(pcm):
        chunk = pcm[offset: offset + CHUNK_BYTES]
        offset += CHUNK_BYTES
        chunk_num += 1
        print(f"SENDING CHUNK {chunk_num}")
        yield chunk

auth = riva.client.Auth(uri=SERVER)
asr_service = riva.client.ASRService(auth)

config = riva.client.StreamingRecognitionConfig(
    config=riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="es-US",
        max_alternatives=1,
        enable_automatic_punctuation=True
    ),
    interim_results=False
)

print("Starting gRPC stream...")

responses = asr_service.streaming_response_generator(
    audio_chunks=audio_stream(),
    streaming_config=config
)

for response in responses:
    print("RAW RESPONSE RECEIVED")
    if not response.results:
        print("No results in response")
        continue

    for result in response.results:
        if result.alternatives:
            print("TEXT:", result.alternatives[0].transcript)
            print("FINAL:", result.is_final)



import subprocess
import riva.client

SERVER = "192.168.4.62:50051"
INPUT_MP3 = "/home/re_nikitav/parakeet-asr-multilingual/audio_maria/maria1.mp3"
TEMP_WAV = "/tmp/maria1_30s.wav"

print("Converting first 30 sec to wav...")

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        INPUT_MP3,
        "-t", "30",
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        TEMP_WAV
    ],
    check=True
)

with open(TEMP_WAV, "rb") as f:
    wav_bytes = f.read()

pcm = wav_bytes[44:]

auth = riva.client.Auth(uri=SERVER)
asr_service = riva.client.ASRService(auth)

for lang in ["es-US", "es-ES", "es", "en-US"]:
    print(f"\nTESTING LANGUAGE = {lang}")

    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,
        language_code=lang,
        max_alternatives=1,
        enable_automatic_punctuation=False
    )

    try:
        response = asr_service.offline_recognize(
            audio_bytes=pcm,
            config=config
        )

        if not response.results:
            print("NO RESULTS")
            continue

        for i, result in enumerate(response.results, start=1):
            if result.alternatives:
                print(f"RESULT {i}: {result.alternatives[0].transcript}")

    except Exception as e:
        print("ERROR:", e)


getting this 
Converting first 30 sec to wav...
ffmpeg version 5.1.8-0+deb12u1 Copyright (c) 2000-2025 the FFmpeg developers
  built with gcc 12 (Debian 12.2.0-14+deb12u1)
  configuration: --prefix=/usr --extra-version=0+deb12u1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libglslang --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librist --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libsvtav1 --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --disable-sndio --enable-libjxl --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-libplacebo --enable-librav1e --enable-shared
  libavutil      57. 28.100 / 57. 28.100
  libavcodec     59. 37.100 / 59. 37.100
  libavformat    59. 27.100 / 59. 27.100
  libavdevice    59.  7.100 / 59.  7.100
  libavfilter     8. 44.100 /  8. 44.100
  libswscale      6.  7.100 /  6.  7.100
  libswresample   4.  7.100 /  4.  7.100
  libpostproc    56.  6.100 / 56.  6.100
[mp3 @ 0x5603d9109880] Estimating duration from bitrate, this may be inaccurate
Input #0, mp3, from '/home/re_nikitav/parakeet-asr-multilingual/audio_maria/maria1.mp3':
  Duration: 00:15:02.24, start: 0.000000, bitrate: 128 kb/s
  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 128 kb/s
Stream mapping:
  Stream #0:0 -> #0:0 (mp3 (mp3float) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to '/tmp/maria1_30s.wav':
  Metadata:
    ISFT            : Lavf59.27.100
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s
    Metadata:
      encoder         : Lavc59.37.100 pcm_s16le
size=     938kB time=00:00:30.00 bitrate= 256.0kbits/s speed=75.3x    
video:0kB audio:938kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.008125%

TESTING LANGUAGE = es-US
ERROR: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.INVALID_ARGUMENT
        details = "Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; "
        debug_error_string = "UNKNOWN:Error received from peer ipv4:192.168.4.62:50051 {grpc_status:3, grpc_message:"Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; "}"
>

TESTING LANGUAGE = es-ES
ERROR: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.INVALID_ARGUMENT
        details = "Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; "
        debug_error_string = "UNKNOWN:Error received from peer ipv4:192.168.4.62:50051 {grpc_status:3, grpc_message:"Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; "}"
>

TESTING LANGUAGE = es
ERROR: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.INVALID_ARGUMENT
        details = "Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; "
        debug_error_string = "UNKNOWN:Error received from peer ipv4:192.168.4.62:50051 {grpc_message:"Error: Unavailable model requested given these parameters: language_code=es; sample_rate=16000; type=offline; ", grpc_status:3}"
>

TESTING LANGUAGE = en-US
ERROR: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.INVALID_ARGUMENT
        details = "Error: Unavailable model requested given these parameters: language_code=en; sample_rate=16000; type=offline; "
        debug_error_string = "UNKNOWN:Error received from peer ipv4:192.168.4.62:50051 {grpc_message:"Error: Unavailable model requested given these parameters: language_code=en; sample_rate=16000; type=offline; ", grpc_status:3}"
>
(env) root@cx-asr-test:/home/re_nikitav/parakeet_test# 
