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

# 80 ms chunk
CHUNK_MS = 80
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000

# safe 2x speed
SEND_DELAY_MS = 40

MODEL_NAME = (
    "parakeet-rnnt-1.1b-unified-ml-cs-"
    "universal-multi-asr-streaming-"
    "silero-vad-sortformer"
)


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

            # FINAL FIX
            await ws.send(json.dumps({
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "language": "es-US",
                        "model": MODEL_NAME
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

                        # 2x realtime
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





import riva.client
import time
import json
import subprocess
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_FOLDER = Path("/home/re_nikitav/audio_maria")
OUTPUT_FOLDER = Path("/home/re_nikitav/parakeet_0.6b_es_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

SERVER = "192.168.4.62::50051"
LANGUAGE = "es-US"

SAMPLE_RATE = 16000
CHUNK_MS = 80
CHUNK_SIZE = SAMPLE_RATE * 2 * CHUNK_MS // 1000

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}


# =========================
# CONVERT TO WAV
# =========================
def convert_to_wav(input_file: Path, output_file: Path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(output_file)
        ],
        check=True
    )


# =========================
# TRANSCRIBE ONE FILE
# =========================
def transcribe_file(asr_service, file_path: Path):
    print(f"\nSTARTING -> {file_path.name}")

    wav_path = OUTPUT_FOLDER / f"{file_path.stem}.wav"

    convert_to_wav(file_path, wav_path)

    with open(wav_path, "rb") as f:
        audio_data = f.read()

    chunks = [
        audio_data[i:i + CHUNK_SIZE]
        for i in range(0, len(audio_data), CHUNK_SIZE)
    ]

    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=LANGUAGE,
            max_alternatives=1,
            enable_automatic_punctuation=True
        ),
        interim_results=True
    )

    chunk_latency = []
    final_parts = []

    start_time = time.time()
    first_transcript_time = None

    responses = asr_service.streaming_response_generator(
        audio_chunks=chunks,
        streaming_config=config
    )

    chunk_num = 0

    for response in responses:
        recv_time = time.time()

        for result in response.results:
            transcript = result.alternatives[0].transcript

            chunk_num += 1

            latency_ms = (
                recv_time - start_time
            ) * 1000

            chunk_latency.append({
                "chunk_num": chunk_num,
                "latency_ms": round(latency_ms, 2),
                "is_final": result.is_final,
                "text": transcript
            })

            if transcript:
                print(
                    f"{file_path.name} | "
                    f"CHUNK {chunk_num} | "
                    f"{transcript}"
                )

            if transcript and first_transcript_time is None:
                first_transcript_time = latency_ms

            if result.is_final and transcript:
                final_parts.append(transcript)

    total_time = time.time() - start_time

    # =========================
    # SAVE TRANSCRIPT
    # =========================
    transcript_file = (
        OUTPUT_FOLDER /
        f"{file_path.stem}_transcription.txt"
    )

    transcript_text = "\n".join(final_parts)

    transcript_file.write_text(
        transcript_text,
        encoding="utf-8"
    )

    # =========================
    # SAVE LATENCY
    # =========================
    latency_file = (
        OUTPUT_FOLDER /
        f"{file_path.stem}_latency.json"
    )

    latency_json = {
        "audio_file": str(file_path),
        "total_time_sec": round(total_time, 2),
        "ttft_ms": round(first_transcript_time, 2)
        if first_transcript_time
        else None,
        "total_chunks": len(chunk_latency),
        "chunk_latency": chunk_latency
    }

    latency_file.write_text(
        json.dumps(latency_json, indent=2),
        encoding="utf-8"
    )

    print(f"COMPLETED -> {file_path.name}")


# =========================
# MAIN
# =========================
def main():
    auth = riva.client.Auth(uri=SERVER)
    asr_service = riva.client.ASRService(auth)

    files = sorted([
        f for f in INPUT_FOLDER.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    print(f"TOTAL FILES = {len(files)}")

    for file in files:
        transcribe_file(asr_service, file)

    print("\nALL FILES COMPLETED")


if __name__ == "__main__":
    main()



{
  "audio_file": "audio\\maria1.mp3",
  "audio_duration_sec": 902.2432,
  "total_processing_time_sec": 725.1283068656921,
  "timestamp": "2026-04-01T23:44:26.203884",
  "model": "nova-3",
  "language": "multi",
  "timing_metrics": {
    "connection_time_sec": 1.2218022346496582,
    "send_duration_sec": 2.5307729244232178,
    "first_byte_latency_sec": 5.35716438293457,
    "first_response_latency_sec": 5.35716438293457,
    "first_final_latency_sec": 5.35716438293457,
    "time_to_first_chunk_sec": 1.2289783954620361
  },
  "latencies": [
    {
      "response_num": 1,
      "latency_from_start_ms": 5357.16438293457,
      "latency_from_send_start_ms": 4129.21929359436,
      "latency_from_first_chunk_ms": 4128.185987472534,
      "latency_from_send_end_ms": 1598.4463691711426,
      "is_final": true,
      "words": 8,
      "char_count": 49
    },
    {
      "response_num": 2,
      "latency_from_start_ms": 8366.647958755493,
      "latency_from_send_start_ms": 7138.702869415283,
      "latency_from_first_chunk_ms": 7137.669563293457,
      "latency_from_send_end_ms": 4607.929944992065,
      "is_final": true,
      "words": 11,
      "char_count": 55
    },
    {
      "response_num": 3,
      "latency_from_start_ms": 10218.449115753174,
      "latency_from_send_start_ms": 8990.504026412964,
      "latency_from_first_chunk_ms": 8989.470720291138,
      "latency_from_send_end_ms": 6459.731101989746,
      "is_final": true,
      "words": 9,
      "char_count": 35
    },
    {
      "response_num": 4,
      "latency_from_start_ms": 11160.973310470581,
      "latency_from_send_start_ms": 9933.028221130371,
      "latency_from_first_chunk_ms": 9931.994915008545,
      "latency_from_send_end_ms": 7402.255296707153,
      "is_final": true,
      "words": 1,
      "char_count": 5
    },
    {
      "response_num": 5,
      "latency_from_start_ms": 15154.016971588135,
      "latency_from_send_start_ms": 13926.071882247925,
      "latency_from_first_chunk_ms": 13925.038576126099,
      "latency_from_send_end_ms": 11395.298957824707,
      "is_final": true,
      "words": 19,
      "char_count": 79
    },
    {
      "response_num": 6,
      "latency_from_start_ms": 16479.26616668701,
      "latency_from_send_start_ms": 15251.321077346802,
      "latency_from_first_chunk_ms": 15250.287771224976,
      "latency_from_send_end_ms": 12720.548152923584,
      "is_final": true,
      "words": 2,
      "char_count": 12
    },
    {
      "response_num": 7,
      "latency_from_start_ms": 19625.90265274048,
      "latency_from_send_start_ms": 18397.95756340027,
      "latency_from_first_chunk_ms": 18396.924257278442,
      "latency_from_send_end_ms": 15867.18463897705,
      "is_final": true,
      "words": 12,
      "char_count": 57
    },
    {
      "response_num": 8,
      "latency_from_start_ms": 23592.215299606323,
      "latency_from_send_start_ms": 22364.270210266113,
      "latency_from_first_chunk_ms": 22363.236904144287,
      "latency_from_send_end_ms": 19833.497285842896,
      "is_final": true,
      "words": 8,
      "char_count": 40
    },
    {
      "response_num": 9,
      "latency_from_start_ms": 26238.208055496216,
      "latency_from_send_start_ms": 25010.262966156006,
      "latency_from_first_chunk_ms": 25009.22966003418,
      "latency_from_send_end_ms": 22479.490041732788,
      "is_final": true,
      "words": 18,
      "char_count": 91
    },
    {
      "response_num": 10,
      "latency_from_start_ms": 29024.53565597534,
      "latency_from_send_start_ms": 27796.590566635132,
      "latency_from_first_chunk_ms": 27795.557260513306,
      "latency_from_send_end_ms": 25265.817642211914,
      "is_final": true,
      "words": 13,
      "char_count": 67
    },
    {
      "response_num": 11,
      "latency_from_start_ms": 29629.070043563843,
      "latency_from_send_start_ms": 28401.124954223633,
      "latency_from_first_chunk_ms": 28400.091648101807,
      "latency_from_send_end_ms": 25870.352029800415,
      "is_final": true,
      "words": 2,
      "char_count": 9
    },
    {
      "response_num": 12,
      "latency_from_start_ms": 33644.31428909302,
      "latency_from_send_start_ms": 32416.369199752808,
      "latency_from_first_chunk_ms": 32415.33589363098,
      "latency_from_send_end_ms": 29885.59627532959,
      "is_final": true,
      "words": 12,
      "char_count": 70
    },
    {
      "response_num": 14,
      "latency_from_start_ms": 37111.42110824585,
      "latency_from_send_start_ms": 35883.47601890564,
      "latency_from_first_chunk_ms": 35882.44271278381,
      "latency_from_send_end_ms": 33352.70309448242,
      "is_final": true,
      "words": 6,
      "char_count": 28
    },
    {
      "response_num": 15,
      "latency_from_start_ms": 41146.053314208984,
      "latency_from_send_start_ms": 39918.108224868774,
      "latency_from_first_chunk_ms": 39917.07491874695,
      "latency_from_send_end_ms": 37387.33530044556,
      "is_final": true,
      "words": 11,
      "char_count": 56
    },
    {
      "response_num": 16,
      "latency_from_start_ms": 43587.95881271362,
      "latency_from_send_start_ms": 42360.01372337341,
      "latency_from_first_chunk_ms": 42358.98041725159,
      "latency_from_send_end_ms": 39829.240798950195,
      "is_final": true,
      "words": 16,
      "char_count": 84
    },
    {
      "response_num": 17,
      "latency_from_start_ms": 44339.16425704956,
      "latency_from_send_start_ms": 43111.21916770935,
      "latency_from_first_chunk_ms": 43110.185861587524,
      "latency_from_send_end_ms": 40580.44624328613,
      "is_final": true,
      "words": 1,
      "char_count": 3
    },
    {
      "response_num": 18,
      "latency_from_start_ms": 47525.39300918579,
      "latency_from_send_start_ms": 46297.44791984558,
      "latency_from_first_chunk_ms": 46296.414613723755,
      "latency_from_send_end_ms": 43766.67499542236,
      "is_final": true,
      "words": 3,
      "char_count": 12
    },
    {
      "response_num": 19,
      "latency_from_start_ms": 49212.24308013916,
      "latency_from_send_start_ms": 47984.29799079895,
      "latency_from_first_chunk_ms": 47983.264684677124,
      "latency_from_send_end_ms": 45453.52506637573,
      "is_final": true,
      "words": 3,
      "char_count": 14
    },
    {
      "response_num": 26,
      "latency_from_start_ms": 67657.81950950623,
      "latency_from_send_start_ms": 66429.87442016602,
      "latency_from_first_chunk_ms": 66428.84111404419,
      "latency_from_send_end_ms": 63899.1014957428,
      "is_final": true,
      "words": 8,
      "char_count": 35
    },
    {
      "response_num": 27,
      "latency_from_start_ms": 68804.40855026245,
      "latency_from_send_start_ms": 67576.46346092224,
      "latency_from_first_chunk_ms": 67575.43015480042,
      "latency_from_send_end_ms": 65045.69053649902,
      "is_final": true,
      "words": 3,
      "char_count": 22
    },
    {
      "response_num": 28,
      "latency_from_start_ms": 72758.25095176697,
      "latency_from_send_start_ms": 71530.30586242676,
      "latency_from_first_chunk_ms": 71529.27255630493,
      "latency_from_send_end_ms": 68999.53293800354,
      "is_final": true,
      "words": 1,
      "char_count": 6
    },
    {
      "response_num": 29,
      "latency_from_start_ms": 73619.63844299316,
      "latency_from_send_start_ms": 72391.69335365295,
      "latency_from_first_chunk_ms": 72390.66004753113,
      "latency_from_send_end_ms": 69860.92042922974,
      "is_final": true,
      "words": 12,
      "char_count": 54
    },
    {
      "response_num": 30,
      "latency_from_start_ms": 76087.5825881958,
      "latency_from_send_start_ms": 74859.63749885559,
      "latency_from_first_chunk_ms": 74858.60419273376,
      "latency_from_send_end_ms": 72328.86457443237,
      "is_final": true,
      "words": 9,
      "char_count": 42
    },
    {
      "response_num": 31,
      "latency_from_start_ms": 79594.0170288086,
      "latency_from_send_start_ms": 78366.07193946838,
      "latency_from_first_chunk_ms": 78365.03863334656,
      "latency_from_send_end_ms": 75835.29901504517,
      "is_final": true,
      "words": 10,
      "char_count": 48
    },
    {
      "response_num": 33,
      "latency_from_start_ms": 83827.15034484863,
      "latency_from_send_start_ms": 82599.20525550842,
      "latency_from_first_chunk_ms": 82598.1719493866,
      "latency_from_send_end_ms": 80068.4323310852,
      "is_final": true,
      "words": 4,
      "char_count": 20
    },
    {
      "response_num": 34,
      "latency_from_start_ms": 87847.91469573975,
      "latency_from_send_start_ms": 86619.96960639954,
      "latency_from_first_chunk_ms": 86618.93630027771,
      "latency_from_send_end_ms": 84089.19668197632,
      "is_final": true,
      "words": 14,
      "char_count": 62
    },
    {
      "response_num": 35,
      "latency_from_start_ms": 89694.06628608704,
      "latency_from_send_start_ms": 88466.12119674683,
      "latency_from_first_chunk_ms": 88465.087890625,
      "latency_from_send_end_ms": 85935.34827232361,
      "is_final": true,
      "words": 6,
      "char_count": 30
    },
    {
      "response_num": 36,
      "latency_from_start_ms": 93734.89022254944,
      "latency_from_send_start_ms": 92506.94513320923,
      "latency_from_first_chunk_ms": 92505.9118270874,
      "latency_from_send_end_ms": 89976.17220878601,
      "is_final": true,
      "words": 20,
      "char_count": 93
    },
    {
      "response_num": 37,
      "latency_from_start_ms": 95985.13507843018,
      "latency_from_send_start_ms": 94757.18998908997,
      "latency_from_first_chunk_ms": 94756.15668296814,
      "latency_from_send_end_ms": 92226.41706466675,
      "is_final": true,
      "words": 9,
      "char_count": 41
    },
    {
      "response_num": 38,
      "latency_from_start_ms": 99283.65540504456,
      "latency_from_send_start_ms": 98055.71031570435,
      "latency_from_first_chunk_ms": 98054.67700958252,
      "latency_from_send_end_ms": 95524.93739128113,
      "is_final": true,
      "words": 8,
      "char_count": 43
    },
    {
      "response_num": 39,
      "latency_from_start_ms": 102645.26057243347,
      "latency_from_send_start_ms": 101417.31548309326,
      "latency_from_first_chunk_ms": 101416.28217697144,
      "latency_from_send_end_ms": 98886.54255867004,
      "is_final": true,
      "words": 10,
      "char_count": 58
    },
    {
      "response_num": 40,
      "latency_from_start_ms": 106658.10203552246,
      "latency_from_send_start_ms": 105430.15694618225,
      "latency_from_first_chunk_ms": 105429.12364006042,
      "latency_from_send_end_ms": 102899.38402175903,
      "is_final": true,
      "words": 11,
      "char_count": 59
    },
    {
      "response_num": 41,
      "latency_from_start_ms": 106942.01827049255,
      "latency_from_send_start_ms": 105714.07318115234,
      "latency_from_first_chunk_ms": 105713.03987503052,
      "latency_from_send_end_ms": 103183.30025672913,
      "is_final": true,
      "words": 5,
      "char_count": 20
    },
    {
      "response_num": 42,
      "latency_from_start_ms": 109529.61993217468,
      "latency_from_send_start_ms": 108301.67484283447,
      "latency_from_first_chunk_ms": 108300.64153671265,
      "latency_from_send_end_ms": 105770.90191841125,
      "is_final": true,
      "words": 11,
      "char_count": 52
    },
    {
      "response_num": 43,
      "latency_from_start_ms": 113549.46041107178,
      "latency_from_send_start_ms": 112321.51532173157,
      "latency_from_first_chunk_ms": 112320.48201560974,
      "latency_from_send_end_ms": 109790.74239730835,
      "is_final": true,
      "words": 9,
      "char_count": 50
    },
    {
      "response_num": 44,
      "latency_from_start_ms": 117163.38968276978,
      "latency_from_send_start_ms": 115935.44459342957,
      "latency_from_first_chunk_ms": 115934.41128730774,
      "latency_from_send_end_ms": 113404.67166900635,
      "is_final": true,
      "words": 20,
      "char_count": 115
    },
    {
      "response_num": 46,
      "latency_from_start_ms": 119352.22578048706,
      "latency_from_send_start_ms": 118124.28069114685,
      "latency_from_first_chunk_ms": 118123.24738502502,
      "latency_from_send_end_ms": 115593.50776672363,
      "is_final": true,
      "words": 6,
      "char_count": 23
    },
    {
      "response_num": 47,
      "latency_from_start_ms": 122765.30385017395,
      "latency_from_send_start_ms": 121537.35876083374,
      "latency_from_first_chunk_ms": 121536.32545471191,
      "latency_from_send_end_ms": 119006.58583641052,
      "is_final": true,
      "words": 4,
      "char_count": 25
    },
    {
      "response_num": 48,
      "latency_from_start_ms": 126738.58213424683,
      "latency_from_send_start_ms": 125510.63704490662,
      "latency_from_first_chunk_ms": 125509.60373878479,
      "latency_from_send_end_ms": 122979.8641204834,
      "is_final": true,
      "words": 4,
      "char_count": 17
    },
    {
      "response_num": 49,
      "latency_from_start_ms": 129998.61454963684,
      "latency_from_send_start_ms": 128770.66946029663,
      "latency_from_first_chunk_ms": 128769.6361541748,
      "latency_from_send_end_ms": 126239.89653587341,
      "is_final": true,
      "words": 11,
      "char_count": 54
    },
    {
      "response_num": 50,
      "latency_from_start_ms": 133347.43523597717,
      "latency_from_send_start_ms": 132119.49014663696,
      "latency_from_first_chunk_ms": 132118.45684051514,
      "latency_from_send_end_ms": 129588.71722221375,
      "is_final": true,
      "words": 15,
      "char_count": 81
    },
    {
      "response_num": 51,
      "latency_from_start_ms": 134202.46362686157,
      "latency_from_send_start_ms": 132974.51853752136,
      "latency_from_first_chunk_ms": 132973.48523139954,
      "latency_from_send_end_ms": 130443.74561309814,
      "is_final": true,
      "words": 2,
      "char_count": 7
    },
    {
      "response_num": 52,
      "latency_from_start_ms": 138228.4917831421,
      "latency_from_send_start_ms": 137000.54669380188,
      "latency_from_first_chunk_ms": 136999.51338768005,
      "latency_from_send_end_ms": 134469.77376937866,
      "is_final": true,
      "words": 14,
      "char_count": 70
    },
    {
      "response_num": 53,
      "latency_from_start_ms": 139581.12287521362,
      "latency_from_send_start_ms": 138353.1777858734,
      "latency_from_first_chunk_ms": 138352.1444797516,
      "latency_from_send_end_ms": 135822.4048614502,
      "is_final": true,
      "words": 6,
      "char_count": 34
    },
    {
      "response_num": 54,
      "latency_from_start_ms": 142442.312002182,
      "latency_from_send_start_ms": 141214.3669128418,
      "latency_from_first_chunk_ms": 141213.33360671997,
      "latency_from_send_end_ms": 138683.59398841858,
      "is_final": true,
      "words": 2,
      "char_count": 9
    },
    {
      "response_num": 55,
      "latency_from_start_ms": 145931.75172805786,
      "latency_from_send_start_ms": 144703.80663871765,
      "latency_from_first_chunk_ms": 144702.77333259583,
      "latency_from_send_end_ms": 142173.03371429443,
      "is_final": true,
      "words": 15,
      "char_count": 74
    },
    {
      "response_num": 56,
      "latency_from_start_ms": 149119.85969543457,
      "latency_from_send_start_ms": 147891.91460609436,
      "latency_from_first_chunk_ms": 147890.88129997253,
      "latency_from_send_end_ms": 145361.14168167114,
      "is_final": true,
      "words": 6,
      "char_count": 29
    },
    {
      "response_num": 57,
      "latency_from_start_ms": 150849.82776641846,
      "latency_from_send_start_ms": 149621.88267707825,
      "latency_from_first_chunk_ms": 149620.84937095642,
      "latency_from_send_end_ms": 147091.10975265503,
      "is_final": true,
      "words": 1,
      "char_count": 2
    },
    {
      "response_num": 58,
      "latency_from_start_ms": 154797.74236679077,
      "latency_from_send_start_ms": 153569.79727745056,
      "latency_from_first_chunk_ms": 153568.76397132874,
      "latency_from_send_end_ms": 151039.02435302734,
      "is_final": true,
      "words": 1,
      "char_count": 3
    },
    {
      "response_num": 59,
      "latency_from_start_ms": 155997.46251106262,
      "latency_from_send_start_ms": 154769.5174217224,
      "latency_from_first_chunk_ms": 154768.4841156006,
      "latency_from_send_end_ms": 152238.7444972992,
      "is_final": true,
      "words": 14,
      "char_count": 55
    },
    {
      "response_num": 60,
      "latency_from_start_ms": 158096.82822227478,
      "latency_from_send_start_ms": 156868.88313293457,
      "latency_from_first_chunk_ms": 156867.84982681274,
      "latency_from_send_end_ms": 154338.11020851135,
      "is_final": true,
      "words": 9,
      "char_count": 32
    },
    {
      "response_num": 61,
      "latency_from_start_ms": 160524.35111999512,
      "latency_from_send_start_ms": 159296.4060306549,
      "latency_from_first_chunk_ms": 159295.37272453308,
      "latency_from_send_end_ms": 156765.6331062317,
      "is_final": true,
      "words": 3,
      "char_count": 12
    },
    {
      "response_num": 63,
      "latency_from_start_ms": 164806.902885437,
      "latency_from_send_start_ms": 163578.9577960968,
      "latency_from_first_chunk_ms": 163577.92448997498,
      "latency_from_send_end_ms": 161048.18487167358,
      "is_final": true,
      "words": 3,
      "char_count": 12
    },
    {
      "response_num": 65,
      "latency_from_start_ms": 169935.19043922424,
      "latency_from_send_start_ms": 168707.24534988403,
      "latency_from_first_chunk_ms": 168706.2120437622,
      "latency_from_send_end_ms": 166176.47242546082,
      "is_final": true,
      "words": 8,
      "char_count": 39
    },
    {
      "response_num": 66,
      "latency_from_start_ms": 173965.65198898315,
      "latency_from_send_start_ms": 172737.70689964294,
      "latency_from_first_chunk_ms": 172736.67359352112,
      "latency_from_send_end_ms": 170206.93397521973,
      "is_final": true,
      "words": 4,
      "char_count": 20
    },
    {
      "response_num": 67,
      "latency_from_start_ms": 175860.75401306152,
      "latency_from_send_start_ms": 174632.8089237213,
      "latency_from_first_chunk_ms": 174631.7756175995,
      "latency_from_send_end_ms": 172102.0359992981,
      "is_final": true,
      "words": 19,
      "char_count": 88
    },
    {
      "response_num": 68,
      "latency_from_start_ms": 179814.13626670837,
      "latency_from_send_start_ms": 178586.19117736816,
      "latency_from_first_chunk_ms": 178585.15787124634,
      "latency_from_send_end_ms": 176055.41825294495,
      "is_final": true,
      "words": 18,
      "char_count": 89
    },
    {
      "response_num": 69,
      "latency_from_start_ms": 181993.97468566895,
      "latency_from_send_start_ms": 180766.02959632874,
      "latency_from_first_chunk_ms": 180764.9962902069,
      "latency_from_send_end_ms": 178235.25667190552,
      "is_final": true,
      "words": 2,
      "char_count": 6
    },
    {
      "response_num": 70,
      "latency_from_start_ms": 184064.26668167114,
      "latency_from_send_start_ms": 182836.32159233093,
      "latency_from_first_chunk_ms": 182835.2882862091,
      "latency_from_send_end_ms": 180305.54866790771,
      "is_final": true,
      "words": 10,
      "char_count": 45
    },
    {
      "response_num": 71,
      "latency_from_start_ms": 186557.03496932983,
      "latency_from_send_start_ms": 185329.08987998962,
      "latency_from_first_chunk_ms": 185328.0565738678,
      "latency_from_send_end_ms": 182798.3169555664,
      "is_final": true,
      "words": 1,
      "char_count": 6
    },
    {
      "response_num": 72,
      "latency_from_start_ms": 188637.16745376587,
      "latency_from_send_start_ms": 187409.22236442566,
      "latency_from_first_chunk_ms": 187408.18905830383,
      "latency_from_send_end_ms": 184878.44944000244,
      "is_final": true,
      "words": 9,
      "char_count": 37
    },
    {
      "response_num": 73,
      "latency_from_start_ms": 192353.60503196716,
      "latency_from_send_start_ms": 191125.65994262695,
      "latency_from_first_chunk_ms": 191124.62663650513,
      "latency_from_send_end_ms": 188594.88701820374,
      "is_final": true,
      "words": 1,
      "char_count": 8
    },
    {
      "response_num": 75,
      "latency_from_start_ms": 196732.55014419556,
      "latency_from_send_start_ms": 195504.60505485535,
      "latency_from_first_chunk_ms": 195503.57174873352,
      "latency_from_send_end_ms": 192973.83213043213,
      "is_final": true,
      "words": 4,
      "char_count": 22
    },
    {
      "response_num": 79,
      "latency_from_start_ms": 209623.9161491394,
      "latency_from_send_start_ms": 208395.9710597992,
      "latency_from_first_chunk_ms": 208394.93775367737,
      "latency_from_send_end_ms": 205865.19813537598,
      "is_final": true,
      "words": 8,
      "char_count": 29
    },
    {
      "response_num": 80,
      "latency_from_start_ms": 212148.8094329834,
      "latency_from_send_start_ms": 210920.8643436432,
      "latency_from_first_chunk_ms": 210919.83103752136,
      "latency_from_send_end_ms": 208390.09141921997,
      "is_final": true,
      "words": 3,
      "char_count": 19
    },
    {
      "response_num": 85,
      "latency_from_start_ms": 225741.96529388428,
      "latency_from_send_start_ms": 224514.02020454407,
      "latency_from_first_chunk_ms": 224512.98689842224,
      "latency_from_send_end_ms": 221983.24728012085,
      "is_final": true,
      "words": 3,
      "char_count": 15
    },
    {
      "response_num": 88,
      "latency_from_start_ms": 233586.4918231964,
      "latency_from_send_start_ms": 232358.5467338562,
      "latency_from_first_chunk_ms": 232357.51342773438,
      "latency_from_send_end_ms": 229827.77380943298,
      "is_final": true,
      "words": 14,
      "char_count": 76
    },
    {
      "response_num": 89,
      "latency_from_start_ms": 237448.1053352356,
      "latency_from_send_start_ms": 236220.1602458954,
      "latency_from_first_chunk_ms": 236219.12693977356,
      "latency_from_send_end_ms": 233689.38732147217,
      "is_final": true,
      "words": 11,
      "char_count": 66
    },
    {
      "response_num": 90,
      "latency_from_start_ms": 238096.94576263428,
      "latency_from_send_start_ms": 236869.00067329407,
      "latency_from_first_chunk_ms": 236867.96736717224,
      "latency_from_send_end_ms": 234338.22774887085,
      "is_final": true,
      "words": 1,
      "char_count": 4
    },
    {
      "response_num": 91,
      "latency_from_start_ms": 244031.03232383728,
      "latency_from_send_start_ms": 242803.08723449707,
      "latency_from_first_chunk_ms": 242802.05392837524,
      "latency_from_send_end_ms": 240272.31431007385,
      "is_final": true,
      "words": 16,
      "char_count": 79
    },
    {
      "response_num": 92,
      "latency_from_start_ms": 248024.04808998108,
      "latency_from_send_start_ms": 246796.10300064087,
      "latency_from_first_chunk_ms": 246795.06969451904,
      "latency_from_send_end_ms": 244265.33007621765,
      "is_final": true,
      "words": 10,
      "char_count": 36
    },
    {
      "response_num": 93,
      "latency_from_start_ms": 248291.0840511322,
      "latency_from_send_start_ms": 247063.138961792,
      "latency_from_first_chunk_ms": 247062.10565567017,
      "latency_from_send_end_ms": 244532.36603736877,
      "is_final": true,
      "words": 7,
      "char_count": 31
    },
    {
      "response_num": 94,
      "latency_from_start_ms": 252290.7361984253,
      "latency_from_send_start_ms": 251062.79110908508,
      "latency_from_first_chunk_ms": 251061.75780296326,
      "latency_from_send_end_ms": 248532.01818466187,
      "is_final": true,
      "words": 17,
      "char_count": 76
    },
    {
      "response_num": 95,
      "latency_from_start_ms": 255733.2100868225,
      "latency_from_send_start_ms": 254505.2649974823,
      "latency_from_first_chunk_ms": 254504.23169136047,
      "latency_from_send_end_ms": 251974.49207305908,
      "is_final": true,
      "words": 11,
      "char_count": 58
    },
    {
      "response_num": 96,
      "latency_from_start_ms": 256819.4193840027,
      "latency_from_send_start_ms": 255591.47429466248,
      "latency_from_first_chunk_ms": 255590.44098854065,
      "latency_from_send_end_ms": 253060.70137023926,
      "is_final": true,
      "words": 8,
      "char_count": 33
    },
    {
      "response_num": 97,
      "latency_from_start_ms": 258267.97246932983,
      "latency_from_send_start_ms": 257040.02737998962,
      "latency_from_first_chunk_ms": 257038.9940738678,
      "latency_from_send_end_ms": 254509.2544555664,
      "is_final": true,
      "words": 4,
      "char_count": 14
    },
    {
      "response_num": 98,
      "latency_from_start_ms": 259929.82053756714,
      "latency_from_send_start_ms": 258701.87544822693,
      "latency_from_first_chunk_ms": 258700.8421421051,
      "latency_from_send_end_ms": 256171.1025238037,
      "is_final": true,
      "words": 7,
      "char_count": 31
    },
    {
      "response_num": 101,
      "latency_from_start_ms": 266849.54500198364,
      "latency_from_send_start_ms": 265621.59991264343,
      "latency_from_first_chunk_ms": 265620.5666065216,
      "latency_from_send_end_ms": 263090.8269882202,
      "is_final": true,
      "words": 3,
      "char_count": 25
    },
    {
      "response_num": 102,
      "latency_from_start_ms": 270901.6592502594,
      "latency_from_send_start_ms": 269673.7141609192,
      "latency_from_first_chunk_ms": 269672.68085479736,
      "latency_from_send_end_ms": 267142.941236496,
      "is_final": true,
      "words": 26,
      "char_count": 106
    },
    {
      "response_num": 103,
      "latency_from_start_ms": 273451.11417770386,
      "latency_from_send_start_ms": 272223.16908836365,
      "latency_from_first_chunk_ms": 272222.1357822418,
      "latency_from_send_end_ms": 269692.39616394043,
      "is_final": true,
      "words": 13,
      "char_count": 66
    },
    {
      "response_num": 104,
      "latency_from_start_ms": 277447.0684528351,
      "latency_from_send_start_ms": 276219.1233634949,
      "latency_from_first_chunk_ms": 276218.09005737305,
      "latency_from_send_end_ms": 273688.35043907166,
      "is_final": true,
      "words": 7,
      "char_count": 43
    },
    {
      "response_num": 105,
      "latency_from_start_ms": 280107.0022583008,
      "latency_from_send_start_ms": 278879.0571689606,
      "latency_from_first_chunk_ms": 278878.02386283875,
      "latency_from_send_end_ms": 276348.28424453735,
      "is_final": true,
      "words": 22,
      "char_count": 108
    },
    {
      "response_num": 106,
      "latency_from_start_ms": 283492.99597740173,
      "latency_from_send_start_ms": 282265.0508880615,
      "latency_from_first_chunk_ms": 282264.0175819397,
      "latency_from_send_end_ms": 279734.2779636383,
      "is_final": true,
      "words": 14,
      "char_count": 60
    },
    {
      "response_num": 107,
      "latency_from_start_ms": 287158.84256362915,
      "latency_from_send_start_ms": 285930.89747428894,
      "latency_from_first_chunk_ms": 285929.8641681671,
      "latency_from_send_end_ms": 283400.1245498657,
      "is_final": true,
      "words": 7,
      "char_count": 35
    },
    {
      "response_num": 109,
      "latency_from_start_ms": 293568.0844783783,
      "latency_from_send_start_ms": 292340.1393890381,
      "latency_from_first_chunk_ms": 292339.10608291626,
      "latency_from_send_end_ms": 289809.36646461487,
      "is_final": true,
      "words": 10,
      "char_count": 44
    },
    {
      "response_num": 110,
      "latency_from_start_ms": 294567.52943992615,
      "latency_from_send_start_ms": 293339.58435058594,
      "latency_from_first_chunk_ms": 293338.5510444641,
      "latency_from_send_end_ms": 290808.8114261627,
      "is_final": true,
      "words": 5,
      "char_count": 27
    },
    {
      "response_num": 112,
      "latency_from_start_ms": 303227.2756099701,
      "latency_from_send_start_ms": 301999.3305206299,
      "latency_from_first_chunk_ms": 301998.29721450806,
      "latency_from_send_end_ms": 299468.55759620667,
      "is_final": true,
      "words": 31,
      "char_count": 134
    },
    {
      "response_num": 113,
      "latency_from_start_ms": 307237.53929138184,
      "latency_from_send_start_ms": 306009.5942020416,
      "latency_from_first_chunk_ms": 306008.5608959198,
      "latency_from_send_end_ms": 303478.8212776184,
      "is_final": true,
      "words": 1,
      "char_count": 3
    },
    {
      "response_num": 114,
      "latency_from_start_ms": 307440.81830978394,
      "latency_from_send_start_ms": 306212.8732204437,
      "latency_from_first_chunk_ms": 306211.8399143219,
      "latency_from_send_end_ms": 303682.1002960205,
      "is_final": true,
      "words": 3,
      "char_count": 14
    },
    {
      "response_num": 116,
      "latency_from_start_ms": 311915.81892967224,
      "latency_from_send_start_ms": 310687.87384033203,
      "latency_from_first_chunk_ms": 310686.8405342102,
      "latency_from_send_end_ms": 308157.1009159088,
      "is_final": true,
      "words": 2,
      "char_count": 16
    },
    {
      "response_num": 117,
      "latency_from_start_ms": 313696.8810558319,
      "latency_from_send_start_ms": 312468.9359664917,
      "latency_from_first_chunk_ms": 312467.9026603699,
      "latency_from_send_end_ms": 309938.1630420685,
      "is_final": true,
      "words": 3,
      "char_count": 13
    },
    {
      "response_num": 119,
      "latency_from_start_ms": 320153.33008766174,
      "latency_from_send_start_ms": 318925.38499832153,
      "latency_from_first_chunk_ms": 318924.3516921997,
      "latency_from_send_end_ms": 316394.6120738983,
      "is_final": true,
      "words": 1,
      "char_count": 3
    },
    {
      "response_num": 120,
      "latency_from_start_ms": 320898.4842300415,
      "latency_from_send_start_ms": 319670.5391407013,
      "latency_from_first_chunk_ms": 319669.50583457947,
      "latency_from_send_end_ms": 317139.7662162781,
      "is_final": true,
      "words": 9,
      "char_count": 40
    },
    {
      "response_num": 121,
      "latency_from_start_ms": 324790.0056838989,
      "latency_from_send_start_ms": 323562.0605945587,
      "latency_from_first_chunk_ms": 323561.0272884369,
      "latency_from_send_end_ms": 321031.2876701355,
      "is_final": true,
      "words": 7,
      "char_count": 35
    },
    {
      "response_num": 125,
      "latency_from_start_ms": 333857.430934906,
      "latency_from_send_start_ms": 332629.4858455658,
      "latency_from_first_chunk_ms": 332628.45253944397,
      "latency_from_send_end_ms": 330098.7129211426,
      "is_final": true,
      "words": 1,
      "char_count": 3
    },
    {
      "response_num": 126,
      "latency_from_start_ms": 336316.58244132996,
      "latency_from_send_start_ms": 335088.63735198975,
      "latency_from_first_chunk_ms": 335087.6040458679,
      "latency_from_send_end_ms": 332557.8644275665,
      "is_final": true,
      "words": 11,
      "char_count": 58
    },
    {
      "response_num": 127,
      "latency_from_start_ms": 340145.96152305603,
      "latency_from_send_start_ms": 338918.0164337158,
      "latency_from_first_chunk_ms": 338916.983127594,
      "latency_from_send_end_ms": 336387.2435092926,
      "is_final": true,
      "words": 19,
      "char_count": 82
    },
    {
      "response_num": 128,
      "latency_from_start_ms": 344177.5190830231,
      "latency_from_send_start_ms": 342949.57399368286,
      "latency_from_first_chunk_ms": 342948.54068756104,
      "latency_from_send_end_ms": 340418.80106925964,
      "is_final": true,
      "words": 19,
      "char_count": 85
    },
    {
      "response_num": 129,
      "latency_from_start_ms": 347258.8698863983,
      "latency_from_send_start_ms": 346030.9247970581,
      "latency_from_first_chunk_ms": 346029.8914909363,
      "latency_from_send_end_ms": 343500.1518726349,
      "is_final": true,
      "words": 12,
      "char_count": 53
    },
    {
      "response_num": 130,
      "latency_from_start_ms": 351219.5007801056,
      "latency_from_send_start_ms": 349991.5556907654,
      "latency_from_first_chunk_ms": 349990.52238464355,
      "latency_from_send_end_ms": 347460.78276634216,
      "is_final": true,
      "words": 11,
      "char_count": 59
    },
    {
      "response_num": 131,
      "latency_from_start_ms": 353392.83537864685,
      "latency_from_send_start_ms": 352164.89028930664,
      "latency_from_first_chunk_ms": 352163.8569831848,
      "latency_from_send_end_ms": 349634.1173648834,
      "is_final": true,
      "words": 10,
      "char_count": 46
    },
    {
      "response_num": 133,
      "latency_from_start_ms": 359318.85719299316,
      "latency_from_send_start_ms": 358090.91210365295,
      "latency_from_first_chunk_ms": 358089.8787975311,
      "latency_from_send_end_ms": 355560.13917922974,
      "is_final": true,
      "words": 12,
      "char_count": 55
    },
    {
      "response_num": 134,
      "latency_from_start_ms": 361386.7599964142,
      "latency_from_send_start_ms": 360158.814907074,
      "latency_from_first_chunk_ms": 360157.78160095215,
      "latency_from_send_end_ms": 357628.04198265076,
      "is_final": true,
      "words": 8,
      "char_count": 37
    },
    {
      "response_num": 135,
      "latency_from_start_ms": 364724.5590686798,
      "latency_from_send_start_ms": 363496.6139793396,
      "latency_from_first_chunk_ms": 363495.5806732178,
      "latency_from_send_end_ms": 360965.8410549164,
      "is_final": true,
      "words": 11,
      "char_count": 55
    },
    {
      "response_num": 136,
      "latency_from_start_ms": 368705.00683784485,
      "latency_from_send_start_ms": 367477.06174850464,
      "latency_from_first_chunk_ms": 367476.0284423828,
      "latency_from_send_end_ms": 364946.2888240814,
      "is_final": true,
      "words": 16,
      "char_count": 84
    },
    {
      "response_num": 137,
      "latency_from_start_ms": 372218.5535430908,
      "latency_from_send_start_ms": 370990.6084537506,
      "latency_from_first_chunk_ms": 370989.5751476288,
      "latency_from_send_end_ms": 368459.8355293274,
      "is_final": true,
      "words": 15,
      "char_count": 58
    },
    {
      "response_num": 138,
      "latency_from_start_ms": 373574.506521225,
      "latency_from_send_start_ms": 372346.56143188477,
      "latency_from_first_chunk_ms": 372345.52812576294,
      "latency_from_send_end_ms": 369815.78850746155,
      "is_final": true,
      "words": 5,
      "char_count": 28
    },
    {
      "response_num": 139,
      "latency_from_start_ms": 377514.91260528564,
      "latency_from_send_start_ms": 376286.96751594543,
      "latency_from_first_chunk_ms": 376285.9342098236,
      "latency_from_send_end_ms": 373756.1945915222,
      "is_final": true,
      "words": 10,
      "char_count": 56
    },
    {
      "response_num": 140,
      "latency_from_start_ms": 379182.57546424866,
      "latency_from_send_start_ms": 377954.63037490845,
      "latency_from_first_chunk_ms": 377953.5970687866,
      "latency_from_send_end_ms": 375423.85745048523,
      "is_final": true,
      "words": 10,
      "char_count": 45
    },
    {
      "response_num": 141,
      "latency_from_start_ms": 383161.00311279297,
      "latency_from_send_start_ms": 381933.05802345276,
      "latency_from_first_chunk_ms": 381932.02471733093,
      "latency_from_send_end_ms": 379402.28509902954,
      "is_final": true,
      "words": 15,
      "char_count": 74
    },
    {
      "response_num": 142,
      "latency_from_start_ms": 384700.67048072815,
      "latency_from_send_start_ms": 383472.72539138794,
      "latency_from_first_chunk_ms": 383471.6920852661,
      "latency_from_send_end_ms": 380941.9524669647,
      "is_final": true,
      "words": 9,
      "char_count": 41
    },
    {
      "response_num": 143,
      "latency_from_start_ms": 388286.43012046814,
      "latency_from_send_start_ms": 387058.48503112793,
      "latency_from_first_chunk_ms": 387057.4517250061,
      "latency_from_send_end_ms": 384527.7121067047,
      "is_final": true,
      "words": 12,
      "char_count": 72
    },
    {
      "response_num": 144,
      "latency_from_start_ms": 392322.4844932556,
      "latency_from_send_start_ms": 391094.5394039154,
      "latency_from_first_chunk_ms": 391093.5060977936,
      "latency_from_send_end_ms": 388563.7664794922,
      "is_final": true,
      "words": 15,
      "char_count": 65
    },
    {
      "response_num": 145,
      "latency_from_start_ms": 394933.76660346985,
      "latency_from_send_start_ms": 393705.82151412964,
      "latency_from_first_chunk_ms": 393704.7882080078,
      "latency_from_send_end_ms": 391175.0485897064,
      "is_final": true,
      "words": 11,
      "char_count": 54
    },
    {
      "response_num": 146,
      "latency_from_start_ms": 397566.4668083191,
      "latency_from_send_start_ms": 396338.5217189789,
      "latency_from_first_chunk_ms": 396337.48841285706,
      "latency_from_send_end_ms": 393807.74879455566,
      "is_final": true,
      "words": 6,
      "char_count": 43
    },
    {
      "response_num": 147,
      "latency_from_start_ms": 401550.5533218384,
      "latency_from_send_start_ms": 400322.60823249817,
      "latency_from_first_chunk_ms": 400321.57492637634,
      "latency_from_send_end_ms": 397791.83530807495,
      "is_final": true,
      "words": 12,
      "char_count": 60
    },
    {
      "response_num": 148,
      "latency_from_start_ms": 402061.6579055786,
      "latency_from_send_start_ms": 400833.7128162384,
      "latency_from_first_chunk_ms": 400832.6795101166,
      "latency_from_send_end_ms": 398302.9398918152,
      "is_final": true,
      "words": 7,
      "char_count": 31
    },
    {
      "response_num": 149,
      "latency_from_start_ms": 406091.98236465454,
      "latency_from_send_start_ms": 404864.03727531433,
      "latency_from_first_chunk_ms": 404863.0039691925,
      "latency_from_send_end_ms": 402333.2643508911,
      "is_final": true,
      "words": 17,
      "char_count": 81
    },
    {
      "response_num": 150,
      "latency_from_start_ms": 408823.6575126648,
      "latency_from_send_start_ms": 407595.7124233246,
      "latency_from_first_chunk_ms": 407594.67911720276,
      "latency_from_send_end_ms": 405064.93949890137,
      "is_final": true,
      "words": 8,
      "char_count": 39
    },
    {
      "response_num": 152,
      "latency_from_start_ms": 413235.3594303131,
      "latency_from_send_start_ms": 412007.4143409729,
      "latency_from_first_chunk_ms": 412006.3810348511,
      "latency_from_send_end_ms": 409476.6414165497,
      "is_final": true,
      "words": 5,
      "char_count": 20
    },
    {
      "response_num": 153,
      "latency_from_start_ms": 417148.52380752563,
      "latency_from_send_start_ms": 415920.5787181854,
      "latency_from_first_chunk_ms": 415919.5454120636,
      "latency_from_send_end_ms": 413389.8057937622,
      "is_final": true,
      "words": 10,
      "char_count": 62
    },
    {
      "response_num": 154,
      "latency_from_start_ms": 419395.02692222595,
      "latency_from_send_start_ms": 418167.08183288574,
      "latency_from_first_chunk_ms": 418166.0485267639,
      "latency_from_send_end_ms": 415636.3089084625,
      "is_final": true,
      "words": 13,
      "char_count": 66
    },
    {
      "response_num": 156,
      "latency_from_start_ms": 423942.8508281708,
      "latency_from_send_start_ms": 422714.90573883057,
      "latency_from_first_chunk_ms": 422713.87243270874,
      "latency_from_send_end_ms": 420184.13281440735,
      "is_final": true,
      "words": 7,
      "char_count": 44
    },
    {
      "response_num": 157,
      "latency_from_start_ms": 427903.9270877838,
      "latency_from_send_start_ms": 426675.9819984436,
      "latency_from_first_chunk_ms": 426674.9486923218,
      "latency_from_send_end_ms": 424145.2090740204,
      "is_final": true,
      "words": 13,
      "char_count": 60
    },
    {
      "response_num": 158,
      "latency_from_start_ms": 431755.8341026306,
      "latency_from_send_start_ms": 430527.8890132904,
      "latency_from_first_chunk_ms": 430526.8557071686,
      "latency_from_send_end_ms": 427997.1160888672,
      "is_final": true,
      "words": 11,
      "char_count": 62
    },
    {
      "response_num": 159,
      "latency_from_start_ms": 434134.49907302856,
      "latency_from_send_start_ms": 432906.55398368835,
      "latency_from_first_chunk_ms": 432905.5206775665,
      "latency_from_send_end_ms": 430375.78105926514,
      "is_final": true,
      "words": 9,
      "char_count": 40
    },
    {
      "response_num": 160,
      "latency_from_start_ms": 436901.4503955841,
      "latency_from_send_start_ms": 435673.5053062439,
      "latency_from_first_chunk_ms": 435672.47200012207,
      "latency_from_send_end_ms": 433142.7323818207,
      "is_final": true,
      "words": 20,
      "char_count": 91
    },
    {
      "response_num": 161,
      "latency_from_start_ms": 437318.85528564453,
      "latency_from_send_start_ms": 436090.9101963043,
      "latency_from_first_chunk_ms": 436089.8768901825,
      "latency_from_send_end_ms": 433560.1372718811,
      "is_final": true,
      "words": 1,
      "char_count": 4
    },
    {
      "response_num": 162,
      "latency_from_start_ms": 440853.4116744995,
      "latency_from_send_start_ms": 439625.4665851593,
      "latency_from_first_chunk_ms": 439624.4332790375,
      "latency_from_send_end_ms": 437094.6936607361,
      "is_final": true,
      "words": 18,
      "char_count": 85
    },
    {
      "response_num": 163,
      "latency_from_start_ms": 443157.88221359253,
      "latency_from_send_start_ms": 441929.9371242523,
      "latency_from_first_chunk_ms": 441928.9038181305,
      "latency_from_send_end_ms": 439399.1641998291,
      "is_final": true,
      "words": 4,
      "char_count": 26
    },
    {
      "response_num": 164,
      "latency_from_start_ms": 446407.60946273804,
      "latency_from_send_start_ms": 445179.6643733978,
      "latency_from_first_chunk_ms": 445178.631067276,
      "latency_from_send_end_ms": 442648.8914489746,
      "is_final": true,
      "words": 15,
      "char_count": 73
    },
    {
      "response_num": 165,
      "latency_from_start_ms": 450454.14447784424,
      "latency_from_send_start_ms": 449226.199388504,
      "latency_from_first_chunk_ms": 449225.1660823822,
      "latency_from_send_end_ms": 446695.4264640808,
      "is_final": true,
      "words": 18,
      "char_count": 85
    },
    {
      "response_num": 166,
      "latency_from_start_ms": 454268.4757709503,
      "latency_from_send_start_ms": 453040.5306816101,
      "latency_from_first_chunk_ms": 453039.4973754883,
      "latency_from_send_end_ms": 450509.7577571869,
      "is_final": true,
      "words": 4,
      "char_count": 15
    },
    {
      "response_num": 167,
      "latency_from_start_ms": 454702.6436328888,
      "latency_from_send_start_ms": 453474.6985435486,
      "latency_from_first_chunk_ms": 453473.66523742676,
      "latency_from_send_end_ms": 450943.92561912537,
      "is_final": true,
      "words": 13,
      "char_count": 56
    },
    {
      "response_num": 173,
      "latency_from_start_ms": 468441.0500526428,
      "latency_from_send_start_ms": 467213.1049633026,
      "latency_from_first_chunk_ms": 467212.0716571808,
      "latency_from_send_end_ms": 464682.3320388794,
      "is_final": true,
      "words": 2,
      "char_count": 16
    },
    {
      "response_num": 175,
      "latency_from_start_ms": 473035.4177951813,
      "latency_from_send_start_ms": 471807.47270584106,
      "latency_from_first_chunk_ms": 471806.43939971924,
      "latency_from_send_end_ms": 469276.69978141785,
      "is_final": true,
      "words": 7,
      "char_count": 43
    },
    {
      "response_num": 176,
      "latency_from_start_ms": 476272.39656448364,
      "latency_from_send_start_ms": 475044.45147514343,
      "latency_from_first_chunk_ms": 475043.4181690216,
      "latency_from_send_end_ms": 472513.6785507202,
      "is_final": true,
      "words": 9,
      "char_count": 43
    },
    {
      "response_num": 178,
      "latency_from_start_ms": 481128.81875038147,
      "latency_from_send_start_ms": 479900.87366104126,
      "latency_from_first_chunk_ms": 479899.84035491943,
      "latency_from_send_end_ms": 477370.10073661804,
      "is_final": true,
      "words": 6,
      "char_count": 29
    },
    {
      "response_num": 180,
      "latency_from_start_ms": 488372.3244667053,
      "latency_from_send_start_ms": 487144.3793773651,
      "latency_from_first_chunk_ms": 487143.3460712433,
      "latency_from_send_end_ms": 484613.6064529419,
      "is_final": true,
      "words": 4,
      "char_count": 21
    },
    {
      "response_num": 183,
      "latency_from_start_ms": 496755.2127838135,
      "latency_from_send_start_ms": 495527.26769447327,
      "latency_from_first_chunk_ms": 495526.23438835144,
      "latency_from_send_end_ms": 492996.49477005005,
      "is_final": true,
      "words": 4,
      "char_count": 20
    },
    {
      "response_num": 184,
      "latency_from_start_ms": 497675.52971839905,
      "latency_from_send_start_ms": 496447.58462905884,
      "latency_from_first_chunk_ms": 496446.551322937,
      "latency_from_send_end_ms": 493916.8117046356,
      "is_final": true,
      "words": 1,
      "char_count": 6
    },
    {
      "response_num": 185,
      "latency_from_start_ms": 501291.76139831543,
      "latency_from_send_start_ms": 500063.8163089752,
      "latency_from_first_chunk_ms": 500062.7830028534,
      "latency_from_send_end_ms": 497533.043384552,
      "is_final": true,
      "words": 1,
      "char_count": 6
    },
    {
      "response_num": 190,
      "latency_from_start_ms": 513452.4304866791,
      "latency_from_send_start_ms": 512224.48539733887,
      "latency_from_first_chunk_ms": 512223.45209121704,
      "latency_from_send_end_ms": 509693.71247291565,
      "is_final": true,
      "words": 3,
      "char_count": 16
    },
    {
      "response_num": 191,
      "latency_from_start_ms": 514870.7616329193,
      "latency_from_send_start_ms": 513642.8165435791,
      "latency_from_first_chunk_ms": 513641.7832374573,
      "latency_from_send_end_ms": 511112.0436191559,
      "is_final": true,
      "words": 3,
      "char_count": 16
    },
    {
      "response_num": 192,
      "latency_from_start_ms": 518868.3407306671,
      "latency_from_send_start_ms": 517640.3956413269,
      "latency_from_first_chunk_ms": 517639.3623352051,
      "latency_from_send_end_ms": 515109.6227169037,
      "is_final": true,
      "words": 11,
      "char_count": 52
    },
    {
      "response_num": 193,
      "latency_from_start_ms": 520892.98152923584,
      "latency_from_send_start_ms": 519665.03643989563,
      "latency_from_first_chunk_ms": 519664.0031337738,
      "latency_from_send_end_ms": 517134.2635154724,
      "is_final": true,
      "words": 12,
      "char_count": 63
    },
    {
      "response_num": 199,
      "latency_from_start_ms": 535873.8803863525,
      "latency_from_send_start_ms": 534645.9352970123,
      "latency_from_first_chunk_ms": 534644.9019908905,
      "latency_from_send_end_ms": 532115.1623725891,
      "is_final": true,
      "words": 11,
      "char_count": 52
    },
    {
      "response_num": 205,
      "latency_from_start_ms": 550606.7018508911,
      "latency_from_send_start_ms": 549378.7567615509,
      "latency_from_first_chunk_ms": 549377.7234554291,
      "latency_from_send_end_ms": 546847.9838371277,
      "is_final": true,
      "words": 4,
      "char_count": 17
    },
    {
      "response_num": 218,
      "latency_from_start_ms": 581265.6590938568,
      "latency_from_send_start_ms": 580037.7140045166,
      "latency_from_first_chunk_ms": 580036.6806983948,
      "latency_from_send_end_ms": 577506.9410800934,
      "is_final": true,
      "words": 3,
      "char_count": 20
    },
    {
      "response_num": 224,
      "latency_from_start_ms": 594994.6899414062,
      "latency_from_send_start_ms": 593766.744852066,
      "latency_from_first_chunk_ms": 593765.7115459442,
      "latency_from_send_end_ms": 591235.9719276428,
      "is_final": true,
      "words": 1,
      "char_count": 5
    },
    {
      "response_num": 233,
      "latency_from_start_ms": 617176.1107444763,
      "latency_from_send_start_ms": 615948.1656551361,
      "latency_from_first_chunk_ms": 615947.1323490143,
      "latency_from_send_end_ms": 613417.3927307129,
      "is_final": true,
      "words": 4,
      "char_count": 18
    },
    {
      "response_num": 239,
      "latency_from_start_ms": 632620.9948062897,
      "latency_from_send_start_ms": 631393.0497169495,
      "latency_from_first_chunk_ms": 631392.0164108276,
      "latency_from_send_end_ms": 628862.2767925262,
      "is_final": true,
      "words": 8,
      "char_count": 39
    },
    {
      "response_num": 244,
      "latency_from_start_ms": 646254.7838687897,
      "latency_from_send_start_ms": 645026.8387794495,
      "latency_from_first_chunk_ms": 645025.8054733276,
      "latency_from_send_end_ms": 642496.0658550262,
      "is_final": true,
      "words": 15,
      "char_count": 82
    },
    {
      "response_num": 245,
      "latency_from_start_ms": 648125.6942749023,
      "latency_from_send_start_ms": 646897.7491855621,
      "latency_from_first_chunk_ms": 646896.7158794403,
      "latency_from_send_end_ms": 644366.9762611389,
      "is_final": true,
      "words": 5,
      "char_count": 29
    },
    {
      "response_num": 247,
      "latency_from_start_ms": 653904.9768447876,
      "latency_from_send_start_ms": 652677.0317554474,
      "latency_from_first_chunk_ms": 652675.9984493256,
      "latency_from_send_end_ms": 650146.2588310242,
      "is_final": true,
      "words": 13,
      "char_count": 63
    },
    {
      "response_num": 248,
      "latency_from_start_ms": 656136.1520290375,
      "latency_from_send_start_ms": 654908.2069396973,
      "latency_from_first_chunk_ms": 654907.1736335754,
      "latency_from_send_end_ms": 652377.434015274,
      "is_final": true,
      "words": 6,
      "char_count": 22
    },
    {
      "response_num": 250,
      "latency_from_start_ms": 661334.8302841187,
      "latency_from_send_start_ms": 660106.8851947784,
      "latency_from_first_chunk_ms": 660105.8518886566,
      "latency_from_send_end_ms": 657576.1122703552,
      "is_final": true,
      "words": 9,
      "char_count": 44
    },
    {
      "response_num": 256,
      "latency_from_start_ms": 676900.2830982208,
      "latency_from_send_start_ms": 675672.3380088806,
      "latency_from_first_chunk_ms": 675671.3047027588,
      "latency_from_send_end_ms": 673141.5650844574,
      "is_final": true,
      "words": 16,
      "char_count": 82
    },
    {
      "response_num": 258,
      "latency_from_start_ms": 683148.0550765991,
      "latency_from_send_start_ms": 681920.1099872589,
      "latency_from_first_chunk_ms": 681919.0766811371,
      "latency_from_send_end_ms": 679389.3370628357,
      "is_final": true,
      "words": 13,
      "char_count": 70
    },
    {
      "response_num": 261,
      "latency_from_start_ms": 690900.2125263214,
      "latency_from_send_start_ms": 689672.2674369812,
      "latency_from_first_chunk_ms": 689671.2341308594,
      "latency_from_send_end_ms": 687141.494512558,
      "is_final": true,
      "words": 4,
      "char_count": 27
    },
    {
      "response_num": 272,
      "latency_from_start_ms": 719020.7757949829,
      "latency_from_send_start_ms": 717792.8307056427,
      "latency_from_first_chunk_ms": 717791.7973995209,
      "latency_from_send_end_ms": 715262.0577812195,
      "is_final": true,
      "words": 2,
      "char_count": 10
    },
    {
      "response_num": 274,
      "latency_from_start_ms": 723315.2232170105,
      "latency_from_send_start_ms": 722087.2781276703,
      "latency_from_first_chunk_ms": 722086.2448215485,
      "latency_from_send_end_ms": 719556.5052032471,
      "is_final": true,
      "words": 1,
      "char_count": 3
    }
  ],
  "summary": {
    "total_responses": 162,
    "final_responses": 162,
    "interim_responses": 0,
    "total_words": 1422,
    "total_characters": 6921,
    "avg_latency_from_send_start_ms": 288198.8338423364,
    "min_latency_from_send_start_ms": 4129.21929359436,
    "max_latency_from_send_start_ms": 722087.2781276703,
    "avg_latency_from_send_end_ms": 285668.06091791316,
    "min_latency_from_send_end_ms": 1598.4463691711426,
    "max_latency_from_send_end_ms": 719556.5052032471
  }
}
