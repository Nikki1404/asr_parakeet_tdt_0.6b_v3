```python id="0jlwmh"
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
CHUNK_SEC = 5
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_SEC


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

                        print(f"SENT CHUNK {chunk_num}")

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

                        print("\nRAW RESPONSE:")
                        print(raw)

                    except asyncio.TimeoutError:
                        print("NO MORE RESPONSES")
                        break

            await asyncio.gather(
                sender(),
                receiver()
            )


asyncio.run(main())
```

but getting this 
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
[mp3 @ 0x55d14670d840] Estimating duration from bitrate, this may be inaccurate
Input #0, mp3, from '/home/re_nikitav/audio_maria/maria1.mp3':
  Duration: 00:15:02.24, start: 0.000000, bitrate: 128 kb/s
  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 128 kb/s
Stream mapping:
  Stream #0:0 -> #0:0 (mp3 (mp3float) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to '/tmp/tmpajpsn8wm/test.wav':
  Metadata:
    ISFT            : Lavf59.27.100
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s
    Metadata:
      encoder         : Lavc59.37.100 pcm_s16le
size=   28195kB time=00:15:02.24 bitrate= 256.0kbits/s speed= 559x    
video:0kB audio:28195kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.000270%
CONNECTED
SESSION CONFIG SENT
SENT CHUNK 1

RAW RESPONSE:
{"event_id":"event_b4263ff0-6ae3-48e8-af8e-45969b0a6353","type":"conversation.created","conversation":{"event_id":"event_f7b513c9-eb09-4715-9350-c189dc8ffa97","id":"conv_ef0e788c-acbb-4136-82cf-cd7bae54f209","object":"realtime.conversation"}}
SENT CHUNK 2
SENT CHUNK 3
SENT CHUNK 4
SENT CHUNK 5
SENT CHUNK 6
SENT CHUNK 7
SENT CHUNK 8
SENT CHUNK 9
SENT CHUNK 10
SENT CHUNK 11
SENT CHUNK 12
SENT CHUNK 13
SENT CHUNK 14
SENT CHUNK 15
SENT CHUNK 16
SENT CHUNK 17
SENT CHUNK 18
SENT CHUNK 19
SENT CHUNK 20
SENT CHUNK 21
SENT CHUNK 22
SENT CHUNK 23
SENT CHUNK 24
SENT CHUNK 25
SENT CHUNK 26
SENT CHUNK 27
SENT CHUNK 28
SENT CHUNK 29
SENT CHUNK 30
SENT CHUNK 31
SENT CHUNK 32
SENT CHUNK 33
SENT CHUNK 34
SENT CHUNK 35
SENT CHUNK 36
SENT CHUNK 37
SENT CHUNK 38
SENT CHUNK 39
SENT CHUNK 40
SENT CHUNK 41
SENT CHUNK 42
SENT CHUNK 43
SENT CHUNK 44
SENT CHUNK 45
SENT CHUNK 46
SENT CHUNK 47
SENT CHUNK 48
SENT CHUNK 49
SENT CHUNK 50
SENT CHUNK 51
SENT CHUNK 52
SENT CHUNK 53
SENT CHUNK 54
SENT CHUNK 55
SENT CHUNK 56
SENT CHUNK 57
SENT CHUNK 58
SENT CHUNK 59
SENT CHUNK 60
SENT CHUNK 61
SENT CHUNK 62
SENT CHUNK 63
SENT CHUNK 64
SENT CHUNK 65
SENT CHUNK 66
SENT CHUNK 67
SENT CHUNK 68
SENT CHUNK 69
SENT CHUNK 70
SENT CHUNK 71
SENT CHUNK 72
SENT CHUNK 73
SENT CHUNK 74
SENT CHUNK 75
SENT CHUNK 76
SENT CHUNK 77
SENT CHUNK 78
SENT CHUNK 79
SENT CHUNK 80
SENT CHUNK 81
SENT CHUNK 82
SENT CHUNK 83
SENT CHUNK 84
SENT CHUNK 85
SENT CHUNK 86
SENT CHUNK 87
SENT CHUNK 88
SENT CHUNK 89
SENT CHUNK 90
SENT CHUNK 91
SENT CHUNK 92
SENT CHUNK 93
SENT CHUNK 94
SENT CHUNK 95
SENT CHUNK 96
SENT CHUNK 97
SENT CHUNK 98
SENT CHUNK 99
SENT CHUNK 100
SENT CHUNK 101
SENT CHUNK 102
SENT CHUNK 103
SENT CHUNK 104
SENT CHUNK 105
SENT CHUNK 106
SENT CHUNK 107
SENT CHUNK 108
SENT CHUNK 109
SENT CHUNK 110
SENT CHUNK 111
SENT CHUNK 112
SENT CHUNK 113
SENT CHUNK 114
SENT CHUNK 115
SENT CHUNK 116
SENT CHUNK 117
SENT CHUNK 118
SENT CHUNK 119
SENT CHUNK 120
SENT CHUNK 121
SENT CHUNK 122
SENT CHUNK 123
SENT CHUNK 124
SENT CHUNK 125
SENT CHUNK 126
SENT CHUNK 127
SENT CHUNK 128
SENT CHUNK 129
SENT CHUNK 130
SENT CHUNK 131
SENT CHUNK 132
SENT CHUNK 133
SENT CHUNK 134
SENT CHUNK 135
SENT CHUNK 136
SENT CHUNK 137
SENT CHUNK 138
SENT CHUNK 139
SENT CHUNK 140
SENT CHUNK 141
SENT CHUNK 142
SENT CHUNK 143
SENT CHUNK 144
SENT CHUNK 145
SENT CHUNK 146
SENT CHUNK 147
SENT CHUNK 148
SENT CHUNK 149
SENT CHUNK 150
SENT CHUNK 151
SENT CHUNK 152
SENT CHUNK 153
SENT CHUNK 154
SENT CHUNK 155
SENT CHUNK 156
SENT CHUNK 157
SENT CHUNK 158
SENT CHUNK 159
SENT CHUNK 160
SENT CHUNK 161
SENT CHUNK 162
SENT CHUNK 163
SENT CHUNK 164
SENT CHUNK 165
SENT CHUNK 166
SENT CHUNK 167
SENT CHUNK 168
SENT CHUNK 169
SENT CHUNK 170
SENT CHUNK 171
SENT CHUNK 172
SENT CHUNK 173
SENT CHUNK 174
SENT CHUNK 175
SENT CHUNK 176
SENT CHUNK 177
SENT CHUNK 178
SENT CHUNK 179
SENT CHUNK 180
SENT CHUNK 181
STREAM COMPLETED

RAW RESPONSE:
{"event_id":"event_4955794a-c2ae-4968-92e0-1f4a17a2ebd1","type":"transcription_session.updated","session":{"modalities":["text"],"input_audio_format":"pcm16","input_audio_transcription":{"language":"en-US","model":"conformer","prompt":null},"input_audio_params":{"sample_rate_hz":16000,"num_channels":1},"recognition_config":{"max_alternatives":1,"enable_automatic_punctuation":false,"enable_word_time_offsets":false,"enable_profanity_filter":false,"enable_verbatim_transcripts":false,"custom_configuration":""},"speaker_diarization":{"enable_speaker_diarization":false,"max_speaker_count":8},"word_boosting":{"enable_word_boosting":false,"word_boosting_list":[]},"endpointing_config":{"start_history":0,"start_threshold":0.0,"stop_history":0,"stop_threshold":0.0,"stop_history_eou":0,"stop_threshold_eou":0.0}}}

RAW RESPONSE:
{"event_id":"event_0000","type":"input_audio_buffer.committed","previous_item_id":"msg_0000","item_id":"msg_0001"}

RAW RESPONSE:
{"event_id":"event_259da9f6-96c5-46b9-a2ff-a4d1b9ed7f61","type":"error","error":{"type":"transcription_error","code":"StatusCode.INTERNAL","message":"Internal error from Core","param":null,"event_id":"56735ada-5de0-4300-9675-1ef52b83b173"}}

RAW RESPONSE:
{"event_id":"event_0000","type":"conversation.item.input_audio_transcription.completed","item_id":"msg_0000","content_index":0,"transcript":"","words_info":{"words":[]},"vad_states":{"vad_states":[]},"is_last_result":true}
