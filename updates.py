import asyncio
import websockets
import json
import base64
import subprocess
import tempfile
from pathlib import Path

WS_URL = "ws://192.168.4.62:9000/v1/realtime?intent=transcription"
INPUT_FILE = "/home/re_nikitav/audio_maria/maria1.mp3"

SAMPLE_RATE = 16000

# IMPORTANT: 80 ms realtime chunks
CHUNK_MS = 80
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000


def convert_to_wav(src_path: str, out_path: str):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            out_path
        ],
        check=True
    )


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_file = str(Path(tmpdir) / "test.wav")

        convert_to_wav(INPUT_FILE, wav_file)

        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=60,
            max_size=2**26
        ) as ws:

            print("CONNECTED")

            # minimal config
            await ws.send(json.dumps({
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_params": {
                        "sample_rate_hz": 16000,
                        "num_channels": 1
                    }
                }
            }))

            print("SESSION CONFIG SENT")

            async def sender():
                with open(wav_file, "rb") as f:
                    # skip wav header
                    f.read(44)

                    chunk_num = 0

                    while True:
                        chunk = f.read(CHUNK_BYTES)

                        if not chunk:
                            break

                        chunk_num += 1

                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                chunk
                            ).decode("utf-8")
                        }))

                        print(
                            f"SENT CHUNK {chunk_num} "
                            f"({CHUNK_MS} ms)"
                        )

                        # CRITICAL: realtime pacing
                        await asyncio.sleep(
                            CHUNK_MS / 1000
                        )

                await ws.send(json.dumps({
                    "type": "input_audio_buffer.commit"
                }))

                await ws.send(json.dumps({
                    "type": "input_audio_buffer.done"
                }))

                print("STREAM COMPLETED")

            async def receiver():
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=300
                        )

                        msg = json.loads(raw)

                        print("\nRAW RESPONSE:")
                        print(
                            json.dumps(
                                msg,
                                indent=2
                            )
                        )

                        # transcript print
                        if "transcript" in msg:
                            print(
                                "\nTRANSCRIPT:",
                                msg["transcript"]
                            )

                        if msg.get(
                            "is_last_result",
                            False
                        ):
                            print(
                                "\nFINAL RESULT RECEIVED"
                            )
                            break

                    except asyncio.TimeoutError:
                        print("NO MORE RESPONSES")
                        break

            await asyncio.gather(
                sender(),
                receiver()
            )


asyncio.run(main())



import asyncio
import websockets
import json
import base64
import subprocess
import tempfile
from pathlib import Path

WS_URL = "ws://192.168.4.38:9000/v1/realtime?intent=transcription"
INPUT_FILE = "/home/re_nikitav/audio_maria/maria1.mp3"

SAMPLE_RATE = 16000

# 80 ms audio chunk
CHUNK_MS = 80
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000

# SAFE FAST STREAMING (2x realtime)
SEND_DELAY_MS = 40


def convert_to_wav(src_path: str, out_path: str):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            out_path
        ],
        check=True
    )


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_file = str(Path(tmpdir) / "test.wav")

        convert_to_wav(INPUT_FILE, wav_file)

        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=60,
            max_size=2**26
        ) as ws:

            print("CONNECTED")

            # IMPORTANT FIX
            await ws.send(json.dumps({
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "language": "es-US"
                    },
                    "input_audio_params": {
                        "sample_rate_hz": 16000,
                        "num_channels": 1
                    }
                }
            }))

            print("SESSION CONFIG SENT")

            async def sender():
                with open(wav_file, "rb") as f:
                    # skip wav header
                    f.read(44)

                    chunk_num = 0

                    while True:
                        chunk = f.read(CHUNK_BYTES)

                        if not chunk:
                            break

                        chunk_num += 1

                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                chunk
                            ).decode("utf-8")
                        }))

                        print(
                            f"SENT CHUNK {chunk_num}"
                        )

                        # SAFE 2x SPEED
                        await asyncio.sleep(
                            SEND_DELAY_MS / 1000
                        )

                await ws.send(json.dumps({
                    "type": "input_audio_buffer.commit"
                }))

                await ws.send(json.dumps({
                    "type": "input_audio_buffer.done"
                }))

                print("STREAM COMPLETED")

            async def receiver():
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=600
                        )

                        msg = json.loads(raw)

                        print("\nRAW RESPONSE:")
                        print(
                            json.dumps(
                                msg,
                                indent=2
                            )
                        )

                        transcript = msg.get(
                            "transcript",
                            ""
                        )

                        if transcript:
                            print(
                                "\nTRANSCRIPT:",
                                transcript
                            )

                        if msg.get(
                            "is_last_result",
                            False
                        ):
                            print(
                                "\nFINAL RESULT RECEIVED"
                            )
                            break

                    except asyncio.TimeoutError:
                        print("TIMEOUT")
                        break

            await asyncio.gather(
                sender(),
                receiver()
            )


asyncio.run(main())


this i am getting after 1 chunk
SENT CHUNK 1

RAW RESPONSE:
{
  "event_id": "event_67c4125b-0c33-4736-b3d6-fbb8fa28d5e1",
  "type": "conversation.created",
  "conversation": {
    "event_id": "event_1f2919ce-d6dd-4620-afba-9d1fffb8f7da",
    "id": "conv_ddb2998e-faf2-4bf1-a0f5-a58ce658eb4b",
    "object": "realtime.conversation"
  }
}

RAW RESPONSE:
{
  "event_id": "event_881ba0db-5627-48da-a508-ac001f510ad9",
  "type": "transcription_session.updated",
  "session": {
    "modalities": [
      "text"
    ],
    "input_audio_format": "pcm16",
    "input_audio_transcription": {
      "language": "es-US",
      "model": "conformer",
      "prompt": null
    },
    "input_audio_params": {
      "sample_rate_hz": 16000,
      "num_channels": 1
    },
    "recognition_config": {
      "max_alternatives": 1,
      "enable_automatic_punctuation": false,
      "enable_word_time_offsets": false,
      "enable_profanity_filter": false,
      "enable_verbatim_transcripts": false,
      "custom_configuration": ""
    },
    "speaker_diarization": {
      "enable_speaker_diarization": false,
      "max_speaker_count": 8
    },
    "word_boosting": {
      "enable_word_boosting": false,
      "word_boosting_list": []
    },
    "endpointing_config": {
      "start_history": 0,
      "start_threshold": 0.0,
      "stop_history": 0,
      "stop_threshold": 0.0,
      "stop_history_eou": 0,
      "stop_threshold_eou": 0.0
    }
  }
}


(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# python3 transcribe_test.py 
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
[mp3 @ 0x55cf0a318840] Estimating duration from bitrate, this may be inaccurate
Input #0, mp3, from '/home/re_nikitav/audio_maria/maria1.mp3':
  Duration: 00:15:02.24, start: 0.000000, bitrate: 128 kb/s
  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 128 kb/s
Stream mapping:
  Stream #0:0 -> #0:0 (mp3 (mp3float) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to '/tmp/tmpev3nbv46/test.wav':
  Metadata:
    ISFT            : Lavf59.27.100
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s
    Metadata:
      encoder         : Lavc59.37.100 pcm_s16le
size=   28195kB time=00:15:02.24 bitrate= 256.0kbits/s speed= 577x    
video:0kB audio:28195kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.000270%
CONNECTED
SESSION CONFIG SENT
SENT CHUNK 1 (80 ms)

RAW RESPONSE:
{
  "event_id": "event_2a7fda37-8cb7-47f1-b437-db83357a10b5",
  "type": "conversation.created",
  "conversation": {
    "event_id": "event_05c76299-56a3-4fef-bd2e-0c09ffa36f92",
    "id": "conv_4a57ca66-c46a-45c3-8192-39b1ff2f8aac",
    "object": "realtime.conversation"
  }
}

RAW RESPONSE:
{
  "event_id": "event_4fc5d831-7c1c-4ffe-b724-65b8ebf7defc",
  "type": "transcription_session.updated",
  "session": {
    "modalities": [
      "text"
    ],
    "input_audio_format": "pcm16",
    "input_audio_transcription": {
      "language": "en-US",
      "model": "conformer",
      "prompt": null
    },
    "input_audio_params": {
      "sample_rate_hz": 16000,
      "num_channels": 1
    },
    "recognition_config": {
      "max_alternatives": 1,
      "enable_automatic_punctuation": false,
      "enable_word_time_offsets": false,
      "enable_profanity_filter": false,
      "enable_verbatim_transcripts": false,
      "custom_configuration": ""
    },
    "speaker_diarization": {
      "enable_speaker_diarization": false,
      "max_speaker_count": 8
    },
    "word_boosting": {
      "enable_word_boosting": false,
      "word_boosting_list": []
    },
    "endpointing_config": {
      "start_history": 0,
      "start_threshold": 0.0,
      "stop_history": 0,
      "stop_threshold": 0.0,
      "stop_history_eou": 0,
      "stop_threshold_eou": 0.0
    }
  }
}
SENT CHUNK 2 (80 ms)
SENT CHUNK 3 (80 ms)
SENT CHUNK 4 (80 ms)
SENT CHUNK 5 (80 ms)
SENT CHUNK 6 (80 ms)
SENT CHUNK 7 (80 ms)
SENT CHUNK 8 (80 ms)
SENT CHUNK 9 (80 ms)
SENT CHUNK 10 (80 ms)
SENT CHUNK 11 (80 ms)
SENT CHUNK 12 (80 ms)
SENT CHUNK 13 (80 ms)
SENT CHUNK 14 (80 ms)
SENT CHUNK 15 (80 ms)
SENT CHUNK 16 (80 ms)
SENT CHUNK 17 (80 ms)
SENT CHUNK 18 (80 ms)
SENT CHUNK 19 (80 ms)
SENT CHUNK 20 (80 ms)
SENT CHUNK 21 (80 ms)
SENT CHUNK 22 (80 ms)
SENT CHUNK 23 (80 ms)
SENT CHUNK 24 (80 ms)
SENT CHUNK 25 (80 ms)
SENT CHUNK 26 (80 ms)
SENT CHUNK 27 (80 ms)
SENT CHUNK 28 (80 ms)
SENT CHUNK 29 (80 ms)
SENT CHUNK 30 (80 ms)
SENT CHUNK 31 (80 ms)
SENT CHUNK 32 (80 ms)
SENT CHUNK 33 (80 ms)
SENT CHUNK 34 (80 ms)
SENT CHUNK 35 (80 ms)
SENT CHUNK 36 (80 ms)
SENT CHUNK 37 (80 ms)
SENT CHUNK 38 (80 ms)
SENT CHUNK 39 (80 ms)
SENT CHUNK 40 (80 ms)
SENT CHUNK 41 (80 ms)
SENT CHUNK 42 (80 ms)
SENT CHUNK 43 (80 ms)
SENT CHUNK 44 (80 ms)
SENT CHUNK 45 (80 ms)
SENT CHUNK 46 (80 ms)
SENT CHUNK 47 (80 ms)
SENT CHUNK 48 (80 ms)
SENT CHUNK 49 (80 ms)
SENT CHUNK 50 (80 ms)
SENT CHUNK 51 (80 ms)
SENT CHUNK 52 (80 ms)
SENT CHUNK 53 (80 ms)
SENT CHUNK 54 (80 ms)
SENT CHUNK 55 (80 ms)
SENT CHUNK 56 (80 ms)
SENT CHUNK 57 (80 ms)
SENT CHUNK 58 (80 ms)
SENT CHUNK 59 (80 ms)
SENT CHUNK 60 (80 ms)
SENT CHUNK 61 (80 ms)
SENT CHUNK 62 (80 ms)
SENT CHUNK 63 (80 ms)
SENT CHUNK 64 (80 ms)
SENT CHUNK 65 (80 ms)
SENT CHUNK 66 (80 ms)
SENT CHUNK 67 (80 ms)
SENT CHUNK 68 (80 ms)
SENT CHUNK 69 (80 ms)
SENT CHUNK 70 (80 ms)
SENT CHUNK 71 (80 ms)
SENT CHUNK 72 (80 ms)
SENT CHUNK 73 (80 ms)
SENT CHUNK 74 (80 ms)
SENT CHUNK 75 (80 ms)
SENT CHUNK 76 (80 ms)
SENT CHUNK 77 (80 ms)
SENT CHUNK 78 (80 ms)
SENT CHUNK 79 (80 ms)
SENT CHUNK 80 (80 ms)
SENT CHUNK 81 (80 ms)
SENT CHUNK 82 (80 ms)
SENT CHUNK 83 (80 ms)
SENT CHUNK 84 (80 ms)
SENT CHUNK 85 (80 ms)
SENT CHUNK 86 (80 ms)
SENT CHUNK 87 (80 ms)
SENT CHUNK 88 (80 ms)
SENT CHUNK 89 (80 ms)
SENT CHUNK 90 (80 ms)
SENT CHUNK 91 (80 ms)
SENT CHUNK 92 (80 ms)
SENT CHUNK 93 (80 ms)
SENT CHUNK 94 (80 ms)
SENT CHUNK 95 (80 ms)
SENT CHUNK 96 (80 ms)
SENT CHUNK 97 (80 ms)
SENT CHUNK 98 (80 ms)
