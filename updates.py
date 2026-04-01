getting this 
(venv) PS C:\Users\re_nikitav\Documents\parakeet-asr-multilingual> python benchmarking_client.py --host 34.118.200.125 --port 8001 --speed 3.0
================================================================================
TOTAL FILES = 56
================================================================================

[1/56] STARTING herring1.mp3
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\benchmarking_client.py", line 366, in <module>
    asyncio.run(
    ~~~~~~~~~~~^
        run_batch(
        ^^^^^^^^^^
    ...<4 lines>...
        )
        ^
    )
    ^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\benchmarking_client.py", line 330, in run_batch
    result = await benchmark_file(uri, file, speed)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\benchmarking_client.py", line 240, in benchmark_file
    pcm = load_audio_as_16k_pcm(str(filepath))
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\benchmarking_client.py", line 42, in load_audio_as_16k_pcm
    .run(
     ~~~^
        capture_stdout=True,
        ^^^^^^^^^^^^^^^^^^^^
        capture_stderr=True,
        ^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\venv\Lib\site-packages\ffmpeg\_run.py", line 313, in run
    process = run_async(
        stream_spec,
    ...<5 lines>...
        overwrite_output=overwrite_output,
    )
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\venv\Lib\site-packages\ffmpeg\_run.py", line 284, in run_async
    return subprocess.Popen(
           ~~~~~~~~~~~~~~~~^
        args, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Program Files\Python313\Lib\subprocess.py", line 1038, in __init__
    self._execute_child(args, executable, preexec_fn, close_fds,
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass_fds, cwd, env,
                        ^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
                        gid, gids, uid, umask,
                        ^^^^^^^^^^^^^^^^^^^^^^
                        start_new_session, process_group)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\subprocess.py", line 1550, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
                       ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
                             # no special security
                             ^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
                             cwd,
                             ^^^^
                             startupinfo)
                             ^^^^^^^^^^^^
FileNotFoundError: [WinError 2] The system cannot find the file specified

for this 
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import ffmpeg
import websockets

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2
DEFAULT_AUDIO_DIR = "c:/Users/re_nikitav/Downloads/audios/audios"
OUTPUT_BASE_DIR = "benchmark_results"
MODEL_NAME = "parakeet-tdt-0.6b-v3"


# =========================
# AUDIO LOADER USING FFMPEG PYTHON
# =========================
def load_audio_as_16k_pcm(path: str) -> bytes:
    """
    Convert any audio file to 16k mono PCM16 LE.
    """
    try:
        out, _ = (
            ffmpeg
            .input(path)
            .output(
                "pipe:",
                format="s16le",
                acodec="pcm_s16le",
                ac=1,
                ar=SAMPLE_RATE,
            )
            .run(
                capture_stdout=True,
                capture_stderr=True,
            )
        )

        return out

    except ffmpeg.Error as e:
        raise RuntimeError(
            f"FFmpeg failed for {path}\n"
            f"{e.stderr.decode()}"
        )


# =========================
# AUDIO DURATION
# =========================
def get_audio_duration(path: str) -> float:
    try:
        probe = ffmpeg.probe(path)
        return float(probe["format"]["duration"])
    except Exception:
        return 0.0


# =========================
# LATENCY ENTRY
# =========================
class LatencyEntry:
    def __init__(self, response_num, latency_ms, is_final, text):
        self.response_num = response_num
        self.latency_ms = latency_ms
        self.is_final = is_final
        self.words = len(text.split()) if text else 0

    def to_dict(self):
        return {
            "response_num": self.response_num,
            "latency_ms": self.latency_ms,
            "is_final": self.is_final,
            "words": self.words,
        }


# =========================
# RESULT OBJECT
# =========================
class BenchmarkResult:
    def __init__(self, file_name, duration_sec):
        self.file_name = file_name
        self.audio_duration_sec = duration_sec
        self.timestamp = datetime.now().isoformat()

        self.latencies = []
        self.transcript_parts = []

        self.ttfb_ms = None
        self.ttft_ms = None

        self.total_processing_time_sec = 0
        self.detected_language = "unknown"
        self.error = None

    @property
    def full_transcript(self):
        return " ".join(self.transcript_parts)

    def to_json(self):
        vals = [x.latency_ms for x in self.latencies]

        return {
            "audio_file": self.file_name,
            "audio_duration_sec": self.audio_duration_sec,
            "total_processing_time_sec": self.total_processing_time_sec,
            "timestamp": self.timestamp,
            "model": MODEL_NAME,
            "detected_language": self.detected_language,
            "ttfb_ms": self.ttfb_ms,
            "ttft_ms": self.ttft_ms,
            "latencies": [x.to_dict() for x in self.latencies],
            "summary": {
                "total_responses": len(self.latencies),
                "final_responses": len(
                    [x for x in self.latencies if x.is_final]
                ),
                "avg_latency_ms": sum(vals) / len(vals) if vals else 0,
                "min_latency_ms": min(vals) if vals else 0,
                "max_latency_ms": max(vals) if vals else 0,
            },
        }


# =========================
# RECEIVER
# =========================
async def receiver(ws, send_start, result):
    response_num = 0
    first_response = True
    first_text = True

    print("[RECEIVER] Waiting for transcripts...")

    try:
        async for raw in ws:
            now = time.perf_counter()
            latency_ms = (now - send_start) * 1000

            if first_response:
                result.ttfb_ms = latency_ms
                print(f"[TTFB] {latency_ms:.0f} ms")
                first_response = False

            try:
                msg = json.loads(raw)
            except Exception as e:
                print(f"[JSON ERROR] {e}")
                continue

            text = msg.get("text", "").strip()
            msg_type = msg.get("type", "")

            if "language" in msg:
                result.detected_language = msg["language"]

            if not text:
                continue

            if first_text:
                result.ttft_ms = latency_ms
                print(f"[TTFT] {latency_ms:.0f} ms")
                first_text = False

            response_num += 1
            is_final = msg_type == "transcript"

            print(
                f"[TRANSCRIPT #{response_num}] "
                f"{latency_ms:.0f} ms | {text[:100]}"
            )

            result.latencies.append(
                LatencyEntry(
                    response_num,
                    latency_ms,
                    is_final,
                    text,
                )
            )

            if is_final:
                result.transcript_parts.append(text)

    except Exception as e:
        print(f"[RECEIVER ERROR] {e}")


# =========================
# SENDER
# =========================
async def sender(ws, pcm, speed):
    chunk_delay = (CHUNK_MS / 1000) / speed
    offset = 0

    total_chunks = (len(pcm) + CHUNK_BYTES - 1) // CHUNK_BYTES

    print(f"[SENDER] Total chunks: {total_chunks}")

    send_start = time.perf_counter()
    chunk_num = 0

    while offset < len(pcm):
        chunk = pcm[offset: offset + CHUNK_BYTES]

        await ws.send(chunk)

        offset += CHUNK_BYTES
        chunk_num += 1

        if chunk_num % 100 == 0 or chunk_num == total_chunks:
            pct = (chunk_num / total_chunks) * 100
            print(
                f"[SENDER] {chunk_num}/{total_chunks} "
                f"({pct:.1f}%)"
            )

        await asyncio.sleep(chunk_delay)

    print("[SENDER] Sending flush")
    await ws.send(json.dumps({"cmd": "flush"}))

    return send_start


# =========================
# BENCHMARK SINGLE FILE
# =========================
async def benchmark_file(uri, filepath, speed):
    pcm = load_audio_as_16k_pcm(str(filepath))
    duration_sec = get_audio_duration(str(filepath))

    result = BenchmarkResult(str(filepath), duration_sec)

    start_wall = time.perf_counter()

    try:
        print(f"[CONNECTING] {uri}")

        async with websockets.connect(
            uri,
            open_timeout=120,
            ping_timeout=120,
            max_size=2**24,
        ) as ws:

            print("[CONNECTED]")

            send_start = await sender(ws, pcm, speed)

            await asyncio.wait_for(
                receiver(ws, send_start, result),
                timeout=duration_sec + 300,
            )

    except Exception as e:
        result.error = str(e)
        print(f"[ERROR] {e}")

    result.total_processing_time_sec = (
        time.perf_counter() - start_wall
    )

    return result


# =========================
# SAVE OUTPUT
# =========================
def save_output(result, output_dir):
    stem = Path(result.file_name).stem
    folder = output_dir / stem
    folder.mkdir(parents=True, exist_ok=True)

    latency_path = folder / f"{stem}_latency.json"
    transcript_path = folder / f"{stem}_transcript.txt"

    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(
            result.to_json(),
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(transcript_path, "w", encoding="utf-8") as f:
        if result.error:
            f.write(result.error)
        else:
            f.write(result.full_transcript)

    return latency_path, transcript_path


# =========================
# BATCH RUNNER
# =========================
async def run_batch(host, port, audio_dir, speed):
    uri = f"ws://{host}:{port}/ws"

    audio_dir = Path(audio_dir)
    output_dir = Path(OUTPUT_BASE_DIR)
    output_dir.mkdir(exist_ok=True)

    files = sorted(audio_dir.glob("*.mp3"))

    if not files:
        print("[ERROR] No MP3 files found")
        sys.exit(1)

    total = len(files)

    print("=" * 80)
    print(f"TOTAL FILES = {total}")
    print("=" * 80)

    for idx, file in enumerate(files, 1):
        print(f"\n[{idx}/{total}] STARTING {file.name}")

        result = await benchmark_file(uri, file, speed)

        latency_path, transcript_path = save_output(
            result,
            output_dir,
        )

        print(f"[{idx}/{total}] DONE")

        if result.error:
            print(f"ERROR: {result.error}")
        else:
            print(f"TTFB: {result.ttfb_ms:.0f} ms")
            print(f"TTFT: {result.ttft_ms:.0f} ms")

        print(f"SAVED: {latency_path}")
        print(f"SAVED: {transcript_path}")
        print("-" * 80)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="34.118.200.125")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--audio-dir",
        default=DEFAULT_AUDIO_DIR,
    )
    parser.add_argument("--speed", type=float, default=3.0)

    args = parser.parse_args()

    asyncio.run(
        run_batch(
            args.host,
            args.port,
            args.audio_dir,
            args.speed,
        )
    )
