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

# 80 ms audio chunks
CHUNK_MS = 80
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000

# FAST send delay
SEND_DELAY_MS = 5


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

            # minimal config only
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
                            f"SENT CHUNK {chunk_num}"
                        )

                        # FAST BATCH MODE
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

                        msg_type = msg.get(
                            "type",
                            ""
                        )

                        transcript = msg.get(
                            "transcript",
                            ""
                        )

                        # print raw
                        print("\nRAW RESPONSE:")
                        print(
                            json.dumps(
                                msg,
                                indent=2
                            )
                        )

                        # print transcript only
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

why getting this 
SENT CHUNK 11119
SENT CHUNK 11120
SENT CHUNK 11121
SENT CHUNK 11122
SENT CHUNK 11123
SENT CHUNK 11124
SENT CHUNK 11125
SENT CHUNK 11126
SENT CHUNK 11127
SENT CHUNK 11128
SENT CHUNK 11129
SENT CHUNK 11130
SENT CHUNK 11131
SENT CHUNK 11132
SENT CHUNK 11133
SENT CHUNK 11134
SENT CHUNK 11135
SENT CHUNK 11136
SENT CHUNK 11137
SENT CHUNK 11138
SENT CHUNK 11139
SENT CHUNK 11140
SENT CHUNK 11141
SENT CHUNK 11142
SENT CHUNK 11143
SENT CHUNK 11144
SENT CHUNK 11145
SENT CHUNK 11146
SENT CHUNK 11147
SENT CHUNK 11148
SENT CHUNK 11149
SENT CHUNK 11150
SENT CHUNK 11151
SENT CHUNK 11152
SENT CHUNK 11153
SENT CHUNK 11154
SENT CHUNK 11155
SENT CHUNK 11156
SENT CHUNK 11157
SENT CHUNK 11158
SENT CHUNK 11159
SENT CHUNK 11160
SENT CHUNK 11161
SENT CHUNK 11162
SENT CHUNK 11163
SENT CHUNK 11164
SENT CHUNK 11165
SENT CHUNK 11166
SENT CHUNK 11167
SENT CHUNK 11168
SENT CHUNK 11169
SENT CHUNK 11170
SENT CHUNK 11171
SENT CHUNK 11172
SENT CHUNK 11173
SENT CHUNK 11174
SENT CHUNK 11175
SENT CHUNK 11176
SENT CHUNK 11177
SENT CHUNK 11178
SENT CHUNK 11179
SENT CHUNK 11180
SENT CHUNK 11181
SENT CHUNK 11182
SENT CHUNK 11183
SENT CHUNK 11184
SENT CHUNK 11185
SENT CHUNK 11186
SENT CHUNK 11187
SENT CHUNK 11188
SENT CHUNK 11189
SENT CHUNK 11190
SENT CHUNK 11191
SENT CHUNK 11192
SENT CHUNK 11193
SENT CHUNK 11194
SENT CHUNK 11195
SENT CHUNK 11196
SENT CHUNK 11197
SENT CHUNK 11198
SENT CHUNK 11199
SENT CHUNK 11200
SENT CHUNK 11201
SENT CHUNK 11202
SENT CHUNK 11203
SENT CHUNK 11204
SENT CHUNK 11205
SENT CHUNK 11206
SENT CHUNK 11207
SENT CHUNK 11208
SENT CHUNK 11209
SENT CHUNK 11210
SENT CHUNK 11211
SENT CHUNK 11212
SENT CHUNK 11213
SENT CHUNK 11214
SENT CHUNK 11215
SENT CHUNK 11216
SENT CHUNK 11217
SENT CHUNK 11218
SENT CHUNK 11219
SENT CHUNK 11220
SENT CHUNK 11221
SENT CHUNK 11222
SENT CHUNK 11223
SENT CHUNK 11224
SENT CHUNK 11225
SENT CHUNK 11226
SENT CHUNK 11227
SENT CHUNK 11228
SENT CHUNK 11229
SENT CHUNK 11230
SENT CHUNK 11231
SENT CHUNK 11232
SENT CHUNK 11233
SENT CHUNK 11234
SENT CHUNK 11235
SENT CHUNK 11236
SENT CHUNK 11237
SENT CHUNK 11238
SENT CHUNK 11239
SENT CHUNK 11240
SENT CHUNK 11241
SENT CHUNK 11242
SENT CHUNK 11243
SENT CHUNK 11244
SENT CHUNK 11245
SENT CHUNK 11246
SENT CHUNK 11247
SENT CHUNK 11248
SENT CHUNK 11249
SENT CHUNK 11250
SENT CHUNK 11251
SENT CHUNK 11252
SENT CHUNK 11253
SENT CHUNK 11254
SENT CHUNK 11255
SENT CHUNK 11256
SENT CHUNK 11257
SENT CHUNK 11258
SENT CHUNK 11259
SENT CHUNK 11260
SENT CHUNK 11261
SENT CHUNK 11262
SENT CHUNK 11263
SENT CHUNK 11264
SENT CHUNK 11265
SENT CHUNK 11266
SENT CHUNK 11267
SENT CHUNK 11268
SENT CHUNK 11269
SENT CHUNK 11270
SENT CHUNK 11271
SENT CHUNK 11272
SENT CHUNK 11273
SENT CHUNK 11274
SENT CHUNK 11275
SENT CHUNK 11276
SENT CHUNK 11277
SENT CHUNK 11278
SENT CHUNK 11279
STREAM COMPLETED

RAW RESPONSE:
{
  "event_id": "event_0000",
  "type": "input_audio_buffer.committed",
  "previous_item_id": "msg_0000",
  "item_id": "msg_0001"
}

RAW RESPONSE:
{
  "event_id": "event_87912d99-681f-4f84-9794-93048198f36c",
  "type": "error",
  "error": {
    "type": "transcription_error",
    "code": "StatusCode.INTERNAL",
    "message": "Internal error from Core",
    "param": null,
    "event_id": "e8f04004-aca4-490f-92d2-3ae442df22b7"
  }
}

RAW RESPONSE:
{
  "event_id": "event_0000",
  "type": "conversation.item.input_audio_transcription.completed",
  "item_id": "msg_0000",
  "content_index": 0,
  "transcript": "",
  "words_info": {
    "words": []
  },
  "vad_states": {
    "vad_states": []
  },
  "is_last_result": true
}

FINAL RESULT RECEIVED
didn't get transcription 
