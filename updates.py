import argparse
import asyncio
import json
import time
import statistics
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import websockets


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2

INPUT_FOLDER = Path("downloads/audios/audios")
OUTPUT_FOLDER = Path("transcription_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)


# =========================
# AUDIO LOADER (NO FFMPEG)
# =========================
def load_mp3_as_pcm16(filepath):
    """
    Load MP3 file as 16k mono PCM16 bytes
    No ffmpeg required
    """
    print(f"Loading audio -> {filepath.name}")

    audio, sr = librosa.load(
        str(filepath),
        sr=SAMPLE_RATE,
        mono=True
    )

    duration_sec = len(audio) / SAMPLE_RATE

    pcm16 = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16).tobytes()

    print(f"Loaded {filepath.name} | Duration: {duration_sec/60:.1f} min")

    return pcm16, duration_sec


# =========================
# SAVE RESULTS
# =========================
def save_results(filepath, transcript_text, result_json):
    output_dir = OUTPUT_FOLDER / filepath.stem
    output_dir.mkdir(exist_ok=True)

    transcript_path = output_dir / f"{filepath.stem}_transcript.txt"
    latency_path = output_dir / f"{filepath.stem}_latency.json"

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    print(f"SAVED -> {filepath.name}")


# =========================
# TRANSCRIBE ONE FILE
# =========================
async def transcribe_file(filepath, host, port, speed):
    uri = f"ws://{host}:{port}/ws"

    print(f"\nSTARTING -> {filepath.name}")

    pcm, duration_sec = load_mp3_as_pcm16(filepath)

    transcript_parts = []
    latencies = []

    start_time = time.time()

    first_response_time = None
    first_final_time = None
    response_num = 0

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**24
        ) as ws:

            async def sender():
                offset = 0
                chunk_delay = (CHUNK_MS / 1000) / speed

                while offset < len(pcm):
                    chunk = pcm[offset: offset + CHUNK_BYTES]
                    offset += CHUNK_BYTES

                    if len(chunk) < CHUNK_BYTES:
                        chunk += bytes(CHUNK_BYTES - len(chunk))

                    await ws.send(chunk)
                    await asyncio.sleep(chunk_delay)

                await asyncio.sleep(0.5)

                await ws.send(json.dumps({"cmd": "flush"}))

            async def receiver():
                nonlocal first_response_time
                nonlocal first_final_time
                nonlocal response_num

                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=60
                        )

                        now = time.time()

                        msg = json.loads(raw)

                        text = msg.get("text", "")
                        msg_type = msg.get("type", "")

                        latency_ms = (now - start_time) * 1000

                        response_num += 1

                        if first_response_time is None:
                            first_response_time = latency_ms

                        is_final = msg_type == "transcript"

                        if is_final and first_final_time is None:
                            first_final_time = latency_ms

                        if text:
                            transcript_parts.append(text)

                        latencies.append({
                            "response_num": response_num,
                            "latency_ms": latency_ms,
                            "is_final": is_final,
                            "words": len(text.split())
                        })

                        if is_final:
                            print(
                                f"{filepath.name} | "
                                f"FINAL {response_num} | "
                                f"TTFT {latency_ms:.0f} ms"
                            )

                    except asyncio.TimeoutError:
                        print(f"{filepath.name} | receiver timeout")
                        break

                    except websockets.exceptions.ConnectionClosed:
                        print(f"{filepath.name} | connection closed")
                        break

            await asyncio.gather(sender(), receiver())

    except Exception as e:
        print(f"FAILED -> {filepath.name}")
        print(str(e))
        traceback.print_exc()
        return

    total_time = time.time() - start_time

    transcript_text = "\n".join(transcript_parts)

    latency_values = [x["latency_ms"] for x in latencies]

    result_json = {
        "audio_file": str(filepath),
        "audio_duration_sec": duration_sec,
        "total_processing_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
        "model": "parakeet-tdt-0.6b-v3",
        "ttfb_ms": first_response_time,
        "ttft_ms": first_final_time,
        "latencies": latencies,
        "summary": {
            "total_responses": len(latencies),
            "final_responses": sum(x["is_final"] for x in latencies),
            "avg_latency_ms": (
                statistics.mean(latency_values)
                if latency_values else 0
            ),
            "min_latency_ms": (
                min(latency_values)
                if latency_values else 0
            ),
            "max_latency_ms": (
                max(latency_values)
                if latency_values else 0
            )
        }
    }

    save_results(filepath, transcript_text, result_json)

    print(
        f"COMPLETED -> {filepath.name} | "
        f"TTFB: {first_response_time:.0f} ms | "
        f"TTFT: {first_final_time:.0f} ms"
    )


# =========================
# BATCH RUNNER
# =========================
async def run_batch(host, port, workers, speed):
    files = sorted(INPUT_FOLDER.glob("*.mp3"))

    total_files = len(files)

    print("=" * 80)
    print(f"TOTAL FILES = {total_files}")
    print(f"WORKERS = {workers}")
    print(f"SPEED = {speed}x")
    print("=" * 80)

    semaphore = asyncio.Semaphore(workers)

    async def process_file(idx, file):
        async with semaphore:
            print(f"\n[{idx}/{total_files}] PROCESSING {file.name}")
            await transcribe_file(file, host, port, speed)

    tasks = [
        asyncio.create_task(process_file(idx, file))
        for idx, file in enumerate(files, start=1)
    ]

    await asyncio.gather(*tasks)

    print("\nALL FILES COMPLETED")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8001)

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Recommended: 2"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="Recommended: 4x"
    )

    return parser.parse_args()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        run_batch(
            args.host,
            args.port,
            args.workers,
            args.speed
        )
    )


why getting partial transcript as well 
Well she was telling me about this thing.
Yeah.
Yeah, she's a new age thing.
That she's a new ancient thing and she's a bug and she has a big
Yeah, she's a new age thing and she's a book and she has a video blog and she's promoting
Yeah, she's a new age thing and she's a book and she has a video blog and she's promoting her book after her show. It's weird. I'm gonna go online to see.
Yeah, she's a new age thing and she's a book and she has a video blog and she's promoting her book after her show. It's weird. I'm gonna go online to see what it's like New Age thing. Yeah.
Yeah, she's a new age thing and she's a book and she has a video blog and she's promoting her book after her show. It's weird. I'm gonna go online to see what it's like New Age thing. Yeah. I've heard of those
Yeah, she's a new age thing and she has a book and she has a video blog and she's promoting her book after her show. It's weird. I'm gonna go online to see what it's like new age thing. Yeah. I've heard of those things. She's into them, I guess.
Yeah, she's a new age thing and she has a book and she has a video blog and she's promoting her book after her show. It's weird. I'm gonna go online to see what it's like. Yeah. Your own God. I've heard of those things. She's into them, I guess. You know, and the other she has a new show.
Yeah, she's a new age thing and she has a book and she has a video blog and she's promoting her book after her show. It's weird. I'm gonna go online to see what it's a new age thing. Yeah. Your own God. I've heard of those things. Get into them, I guess. You know, and the other day she has a new show about she gets like ten people.
Yeah, she's a new age thing and she has a book and she has a video blog and she's promoting her book after her show. It's weird. I'm gonna go online to see what it's a new age thing. Yeah. Like your own computer, your own God. I've heard of those things. She's into them. I guess. You know and the other day she has a new show about she gets like ten people and they have to I don't know what it is.
Yeah, she's a new age thing and she has a book and she has a video blog and she's promoting her book after her show. It's weird. I'm gonna go online to see what it's a new age thing. Yeah. Like your own computer, your own God. I've heard of those things. She's into them? I guess. You know, and the other day she has a new show about she gets like ten people and they have to I don't know what it is.
It's about basically she giv she's giving them money.
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice.
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, You just wanna
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, You just trying to show and promote that you're trying to donate so much.
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, You just trying to show and promote that you're trying to donate so much millions of dollars. Why do you have to promote it and make a thing?
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, You just trying to show and promote that you're trying to donate so much millions of dollars, why do you have to promote it and make a thing out of it? And then people who to
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, You just trying to show and promote that you're trying to donate so much millions of dollars. Why do you have to promote it and make a thing out of it? And then the people who to able are able to raise the thing that the most
It it's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, You just trying to show and promote that you're trying to donate so much millions of dollars, why do you have to promote it and make a thing out of it? And then the people who t ab are able to raise the thing that the most funds and donate it, she gives
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, You just trying to show and promote that you're trying to donate so much millions of dollars. Why do you have to promote it and make a thing out of it? And then the people who t ab are able to raise the thing that the most funds and donate it, she gives them like ten million dollars, yeah.
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, You just try to show and promote that you're trying to donate so much millions of dollars. Why do you have to promote it and make a thing out of it? And then the people who t ab are able to raise the thing that the most funds and donate it, she gives them like ten million dollars, yeah. No. Like that I'm not interested in that work, you know. I don't know.
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, you just want to show and promote that you're trying to donate so much millions of dollars. Why do you have to promote it and make a thing out of it? And then the people who t ab are able to raise the thing that the most funds and donate it, she gives them like ten million dollars. No. Like that I'm not interested in that work, you know. I don't know, I think it's getting so bad direction.
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, You just want to show and promote that you're trying to donate so much millions of dollars. Why do you have to promote it and make a thing out of it? And then the people who t ab are able to raise the thing that the most funds and donate it, she gives them like ten million dollars, yeah. No. Like that I'm not interested in that work, you know. I don't know, I think it's getting so bad that she's like I
It's about basically she giv she's giving them money or they get millions of dollars to donate to a charity of the choice. But that's the whole show and I'm like, You just want to show and promote that you're trying to donate so much millions of dollars. Why do you have to promote it and make a thing out of it? And then the people who to ab are able to raise the thing that the most funds and donate it, she gives them like ten million dollars, yeah. No. Like that I'm not interested in that work, you know. I don't know, I think it's getting so bad that she is like I think what she has
Great.
Great. Yeah.
Yeah. At all. I've never
Great bottom. Yeah. At all. I've never watched open.
Great bottle. Yeah. At all. I've never watched open. Yeah, and then
Great bond. Yeah. At all. I've never watched open. Yeah, and then it's good. It's good. And maybe not.
Great bottle. Yeah. At all. I've never watched open. Yeah, and then it's good. It was good. And leaving on Friday. Oh wait, okay.
Great bottom. Yeah. At all. I've never watched open. Yeah. And then it's good. It's good. I'm leaving on Friday. Oh wait, okay. Please don't leave the pictures, yes.
Great bottle. Yeah. At all. I've never watched open. Yeah. And then it's good. I'm leaving on Friday. Oh wait, okay. Please don't leave the pictures, yes. Already they can email it to you, right? You can email it.
Great bottle or something. Yeah. At all. I've never watched open. And then it's good. It was good. And leaving on Friday. Oh wait, okay. Please don't leave the pictures, yes. Already they can email it to you, right? You think I email Niganga? I'd rather yeah, email it.
Great bottle or something. Yeah. At all. I've never watched open. And then it's good. I'm leaving on Friday. Oh wait, okay. Please don't leave the channel. Already they can email it to you, right? You think I email from Niganga? I'd rather, yeah, email it to me. Okay, I'll email it. Sorry, I keep ready.
They're on a drawer or something. Yeah. At all. I've never watched open. And then it's good. I'm leaving on Friday. Oh wait, okay. Please don't leave the pictures, yes. Already they can email it to you, right? You think I email from Niganga? I'd rather yeah, email it to me. Okay, I'll email it. Sorry, I keep and I saw it today because I you know I use it to put my my
They're like bottle or something. Yeah. At all. I've never watched open by the content is good. I'm leaving on Friday. Oh wait, okay. Please don't leave the pictures, yes. Already they can email it to you, right? You think they email from Niganga? I'd rather yeah, email it to me. Okay, I'll email it. Sorry, I keep and I saw it today because I you know I use it to pull my my my shoes off. So I don't I I don't think my grandma's gonna get it right.
Great, they're like bottle. Yeah. At all. I've never watched open by the biggest. I'm leaving on Friday. Oh wait, okay. Please don't leave the pictures. Yes. Already they can email it to you, right? You think I email from Niganga? I'd rather, yeah, email it to me. Okay, I'll email it. Sorry, I keep and I saw it today because I you know I use it to pull my my my shoes off. I I don't think my grandma's gonna get it right. I showed it to her.
They're like bottle. Yeah. At all. I've never watched open by the content. I'm leaving on Friday. Oh wait, okay. Please don't leave the pictures. Yes. Already they can email it to you, right? You think I email from Niganga? I'd rather, yeah, email it to me. Okay, I'll email it. Sorry, I keep and I saw it today because I you know I use it to pull my my my shoes off. I I don't think my grandma's gonna get it right. I showed it to her. She went to you.
It It''ssinging the the
Oh my god.
Oh my god. So what do you tell your grandmother? Because I remember when she did it the first time, your grandmother was here.
So what do you tell your grandmother? Grandmother was here visiting. You know, we used to live with my dad over there.
And
Okay Okay....
And and then we went to go see them. Your grandmother's like, oh todos se miran las manos, la vejez se mira a mano. Because it's true, you like you see our faces off and
A Ayyoo...................... W Whoho,, the the v vestest....
And then we went to go see them. And your grandmother's like, Oh, todos se mira la mano, la vejez se mira la mano'cause it's true, you like you see her face is off and your hands age. So la cara no, so I don't know. I don't wanna i if she gets me one, fine, then I'll use you know.
And and then we went to go see them. And your grandmother's like, Oh, todos se mira la mano, la vejez se mira la mano. Because it's true, you like you see her face is off and your hands age. So la cara no, so I don't know. I don't wanna if she gets me one, fine, then I'll use you know. But I trust you with more.
And then we went to go see them. Your grandmother's like, oh, todos se miran las manos, la vejez se mira la mano. Because it's true, you like you see her face is off and your hands age. So la cara no, so I don't know. I don't wanna if she gets me one, fine, then I'll use you know. But I trust you more.
And then we went to go see them. And your grandmother's like, Oh, todos se mira la mano, la vejez se mira la mano because it's true, you like you see her face is off and your hands age. So I don't know. I don't wanna if she gets me one, fine, then I'll use you know. But I trust you with more. You can only give one. Yeah, I think it's like you know how to like just
And then we went to go see them. And your grandmother's like, Oh, todos se mira la mano, la vejez se mira la mano. Because it's true, you like you see her face is off and your hands age. So I don't know. I don't wanna if she gets me one, fine, then I'll use you know. But I trust you more. You can only give one. Yeah, like you know how they like just I mean on the bottom fine but
And then we went to go see them. And your grandmother's like, Oh, todos se mira la mano, la vejez se mira la mano'cause it's true, you like you see her face is off and your hands age. So I don't know. I don't wanna if she gets me one, fine, then I'll use you know. But I trust you more. You can only give one. Yeah, like you know how they like just I mean on the bottom of it, fine, but look at them. You know they're so patriotic and they
And and then we went to go see them. And your grandmother's like, Oh, todo se mira, la vejez se mira la mano. Because it's true, you like you see her face is off and your hands age. So I don't know. I don't wanna if she gets me one, fine, then I'll use you know. But I trust you more. You can only give up one. Yeah, that's it. Like you know how they like just I mean on the bottom of it, fine, but don't look at them. You know they're so patriotic and they Yeah, yeah, I know. I'm sorry.
And then we went to go see them. And your grandmother's like, Oh, todo se mira, la vejez se mira la mano'cause it's true, you like you see her face is off and your hands age. So I don't know. I don't wanna if she gets me one, fine, then I'll use you know. But I trust you more. You can only give it one. Yeah, that's it. Like you know how they like just I mean on the bottom of it, fine, but don't look at them. You know they're so patriotic and they Yeah, yeah. I know. I'm sorry.
Just
Just emails y hacer la.
Just email ya se lo mandó o email it closed now email it open.
Just email. Or email it closed now, email it open. Yeah. So you can see how it opens.
Just email uh or email it closed now or email it open. Yeah. So you can see how it folds.
Just email uh Yocalamandor. Or mail it closed now or email it open. So you can see how it folds. Maybe it's good, it's moving.
Just email uh Yasalamando. Or mail it close now or email it open. Yeah. So you can see how it falls. Where? So
Just email uh Yasalamandol. I'll email it close now or email it open. So you can see how it folds. Where? To to uh where his mouth with his
Just email uh salamandol. Or email it closed now or email it open. Yeah. So you can see how it folds. Where? To To uh With his mom with his mom or with his sister? No, with his mom.
Just email uh se la mandarla. I'll email it close now or email it open. Yeah. So you can see how it folds. But he's good. He's moving. Where? To to uh with his mom with his he listen with his mom or his sister. No, with his mom. He's moving with his uh they're moving from
Just email uh email it closed now or email it open. Yeah. So you can see how it folds. But he's good. He's moving. Where? To to uh with his m with his he listen with his mom with his sister. No, with his mom. He's moving with his uh they're moving from from the house there, right?
Just email uh se lo mandó. So you can see how it folds. But he's good, he's moving. Where? To to uh with his m with his he is with his mom with his sister. No, with his mom. He's moving with his uh they're moving from from the house they are right in right now, porque it's
Just email uh se lo mandó. So you can see how it folds. But he's good. He's moving. Where? To to uh with his m with his he listen with his mom with his sister. No, with his mom. He's moving with his uh they're moving from from the house they are right in right now porque it's a lot cheaper. It's like
Almost.
Almost fifty dollars cheaper.
Almost fifty dollars cheaper. But they sold their house, but they sold their house.
Almost fifty dollars cheaper. But they sold their house, but they sold their house. But the thing is that
Almost fifty dollars cheaper. But they sold their house, but they sold their house. But the thing is that this lived with him, like I don't know how like
almost fifty dollars cheaper. But they sold their house, but they sold their house. But the thing is that the sutillo that lived with him like I don't know like yeah, several years.
almost fifty dollars cheaper. But they sold their house, but they sold their house. But the thing is that the sutillo that lived with him like I don't know like yeah, several years, he didn't do anything.
almost fifty dollars cheaper. But they sold their house, madam. They sold their house. But the thing is that the sutillo that lived with him like I don't know, like yeah, several years, he didn't do anything. He no pagan nada en la casa.
almost fifty dollars cheaper. But they sold their house, but they sold their house. But the thing is that the sutillo that lived with him like I don't know like yeah several years, he doesn't do anything. He no finding la casa. Like he doesn't not even water
H Haveave
Almost fifty dollars cheaper. But they sold their house, but then they sold their house. But the thing is that the sutillo that lived with him like I don't know, like yeah, several years, he doesn't do anything. He no pagan nada en la casa. Like he doesn't even water and he he lives there with his he lives there with his wife and his daughter and I understand see you know
Almost fifty dollars cheaper. But they sold their house, but then they sold their house. But the thing is that the sutillo that lived with him like I don't know, like yeah, several years, he doesn't do anything. He no faja nada en la casa. Like he doesn't even water, and he he lives there with his he lives there with his wife and his daughter, and I understand, see you know, if you don't have enough money, at least
almost fifty dollars cheaper. But they sold their house, but then they sold their house. But the thing is that the sutillo that lived with him like I don't know, like yeah, several years, he doesn't do anything. He no fanada en la casa. Like he doesn't even water and he he lives there with his he lives there with his wife and his daughter and I understand, see you know, if you don't have enough money, at least I go, you know? Yeah.
almost fifty dollars cheaper. But they sold their house, but then they sold their house. But the thing is that the sutillo that lived with him like I don't know, like yeah, several years, he doesn't do anything. He no pagan nada in la casa. Like he doesn't even water and he he lives there with his he lives there with his wife and his daughter and I understand, see you know, if you don't have enough money, at least I go, you know? Yeah. But nothing.
Eh Eh bu buenoeno so so che che
It's his wife and everybody it's not just
So he's not like I can I can't mantener su familia anymore. It's his wife and everybody, it's not just Yeah, you know.
So his mom's like I can't I can't mantle your wife It's his wife and everybody, it's not just Yeah. You know? And she's retired.
So his mom's like I can't I can't mantle anymore. It's his wife and everybody, it's not just Yeah, uh you know and she's retired. Okay, well
So his mom's like I can't I can't maintain su familia anymore. Your wife has to be It's his wife and everybody, it's not just Yeah, uh you know. And she's retired. S she's responsible for the you know, her mom.
So his mom's like I can't I can't maintain anymore. Your wife has It's his wife and everybody, it's not just Yeah. You know? And she's retired. She's responsible for the you know, her mom and her her older daughter.
So his mom's like I can't I can't maintain anymore. Your wife has to be It's his wife and everybody, it's not just Yeah, uh you know and she's retired. Okay, well she she's responsible for the you know her mom and her her older daughter, but A AI has me.
So his mom's like I can't I can't maintain anymore. Your wife has It's his wife and everybody, it's not just Yeah, uh you know. And she's retired. S she's responsible for the, you know, her mom and her her older daughter, but AI AI has anything. Yeah.
So his mom's like I can't I can't mantia anymore your wife. It's his wife and everybody, it's not just Yeah. You know? And she's retired. Okay, well she she's responsible for the, you know, her mom and her her older daughter, but AI AI has anything. Yeah. So Orlando's like fed up with it.
So his mom's like, I can't I can't maintain anymore. The wife It's his wife and everybody, it's not just Yeah. You know, and she's retired. Okay, well she she's responsible for the you know her mom and her her older daughter, but AI AI has anything. Yeah. So Orlando's like fed up with it. Well the thing is that they strive to
So his mom's like, I can't I can't maintain anymore. The wife has It's his wife and everybody, it's not just Yeah. You know, and she's retired. Okay, well she she's responsible for the you know her mom and her her older daughter, but AI AI has anything. Yeah. So Orlando's like fed up with it. Well the thing is that they started looking for a new house.
So his mom's like, I can't I can't mantle anymore than I can have. It's his wife and everybody, it's not just Yeah. You know? And she's retired. Okay, well she she's responsible for the, you know, her mom and her her older daughter, but AI AI has anything. Yeah. So Orlando's like fed up with it. Well the thing is that they started looking for a new house.
Um
Uh and his and a grandmother.
Uh-huh. And his grandmother? And and if she wants to go with him.
de dezaza
Uh-huh. And his and the grandmother? And and if she wants to go with him. Because she was. Yeah, she lived from one.
Uh-huh. And his and the grandmother? And if she wants to go with him. Because she was. Yeah, she lived from one. But the grandmother was taking, you know
Uh-huh. And his grandmother? And if she wants to go with him. Because she was. Yeah, she lived from one. But the grandmother was taking, you know, her son's side.
Uh-huh. And his grandmother? And if she wants to go with him. Because she was. Yeah, she lived somewhere. But the grandmother was taking, you know, her son's side. The thing.
Uh-huh. And his and the grandmother? And and if she wants to go with him. Because she was. Yeah, she lived somewhere. But the grandmother was taking, you know, her son's side. The thing is. Yeah. It's a little
Uh-huh and his and the grandmother? And if she wants to go with him. The sister, the one that was married, lived somewhere else. Yeah, she lived somewhere. But the grandmother was taking, you know, her son's side. The thing is. Yeah. It's a little bit more.
Uh-huh. And his and the grandmother? And and if she wants to go with him. Because she was. Yeah, she lived somewhere. But the grandmother was taking, you know, her son's side. The thing is. Yeah. It's a little with his mom, que tora conde. Like even
Uh-huh. And his and the grandmother? And if she wants to go with him. Because she was sister, the one that was married, lived somewhere else. Yeah, she lived somewhere. But the grandmother was taking, you know, her son's side. The thing is. Siempre no cumbre. Yeah. It's a little bit more. Like even some me, like, they favor them so much.
Uh-huh and his and the grandmother? And if she wants to go with him. Because she was sister, the one that was married lived somewhere else. Yeah, she didn't say one. But the grandmother was taking, you know, her son's side. The thing is. Yeah. It's a little bit more. Like even some me, like, they favor them so much. Yeah. I don't know if it's like a Hispanic thing.
Uh-huh. And his grandmother? And if she wants to go with him. Because she was the one that was married lived somewhere else. Yeah, she didn't say one. But the grandmother was taking, you know, her son's side. The thing is. Yeah. It's a little bit more. Like even some favor them so much. Yeah. I don't know if it's like the Hispanic thing that the men are no matter
Me take dinero with my mom and my own.
What my mom and my uncle, you know my mom feather.
She's always favored my uncle.
No No somet somethinghing somet somethinghing....
grandma.
Like even to this day, even to this day, she'll give me came back to like, you know.
Like even to this day, even to this day, she'll give me back to like you know, money for pay the rental.
Even to this day, she'll give him back for like you know, money for pay the rent or like what because he does the work. I'm like, one or one.
Like even to this day, even to this day, she'll give him back to like you know, money for pay the rent or like what'cause he does the work. I'm like, one or one kid or you're calling sick.
Even to this day, she'll give me back to like you know, money pay the rent. So I gave what because he used to work. I'm like, one of the kids, so he'll call in six. So to make up the difference, my grandma gave it.
Even to this day, she'll give me back to like, you know, money pay the rent. So I gave what because he used to work. I'm like, so he'll call in six. So to make up the difference, my grandma will give it some money.
Yeah.
Que jodían esa.
That's wow. The end is like stupid nickaravage.
That's wow. It's like stubborn rabbit. It's not really this in general.
That's wow. The end is like stubborn. It's not really in general. I think it's like aw. I mean, I used to help in 18.
That's wow. It's like stubborn. It's not great. It's in general. I think it's like aw. I mean, I understand helping a child, helping your kid if they're an 18, you know.
That's wow. It's like it's too many karaves. It's not great. It's in general. I think it's like aw. I mean, I understand helping a child, helping your kid if they're an a you know they're in a crisis situation, but that's
That's wow. It's like it's too many karaves. It's not great in general. I think it's like aw. I mean, I understand helping a child, helping your kid if they're an age, you know, they're in a crisis situation, but that's okay.
That's wow. It's like it's two mini karavish. It's not great in general. I think it's like aw. I mean I understand helping a child, helping your kid if they're an a you know they're in a crisis situation, but that's okay. Wow. Well the thing is like they found this case.
Ye Yeahah...
That's wow. It's not really just in general. I think it's like aw. I mean I understand helping a child, helping your kid if they're in a crisis situation, but that's okay. Wow. Well the thing is like they found this place, right? But the house comes with a little efficiency.
Aside within the terror, you know, within the area. And so Ornando was like, Well, just tell my uncle that he.
aside within the the terror, you know, within the area. And so Orlando was like, well, just tell my uncle that he wants to go live with us, that the
aside within the the territory within the area and so Ornando was like well just tell my uncle that he wants to go live with us that he can rent the apartment out
Un
aside within the the terrace you know within the area and so ornandoz like well just tell my uncle that he wants to go live with us that the he can rent the apartment out don't don't yeah don't let him know that the that it comes with the house
Aside within the the terrace, you know, within the area. And so Orlando's like, well, just tell my uncle that he wants to go live with us, that the he can rent the apartment out. Don't don't rent, yeah. Don't let him know that the that it comes with the house. The winner. So they're like, Well, if you want to move in next.
Aside within the the terrace, you know, within the area. And so Orlando's like, well, just tell my uncle that he wants to go live with us, that the he can rent the apartment out. Don't don't rent, yeah. Don't let him know that the that it comes with the house. The winner. So they're like, Well, if you want to move in next to us, it's cheap, it's $50.
Aside within the terrace, you know, within the area. And so Orlando's like, well, just tell my uncle that he wants to go live with us, that the he can rent the apartment out. Don't don't yeah, don't let him know that the that it comes with the house. The winner. So they're like, Well, if you want to move in next to us, it's cheap, it's $50, and you're not gonna get cheaper than that.
Aside within the terrace, you know, within the area. And so Orlando's like, well, just tell my uncle that he wants to go live with us, that the he can rent the apartment out. Don't don't yeah, don't let him know that the that it comes with the house. The winner. So they're like, Well, if you want to move in next to us, it's cheap, it's $50, and you're not gonna get cheaper than that anywhere.
He's gonna pay for the fifty dollars.
He's gonna pay for the $50, his water, his life. He thinks that he
He's gonna pay for the $50, his water, his life. He thinks that he he's that he's just
He's gonna pay for the $50, his water, his life. He thinks that he he's like he's just helping out.
He's gonna pay for the $50, his water, his life. He thinks that he he's that he's just helping out. It's kind of it's you know fatality.
He's gonna pay for the $50, his water, his life. He thinks that he he's yeah that he's just helping out. It's kind of it's you know, but that was the only way.
He's gonna pay for the $50, his water, his life. He thinks that he he's that he's just helping out. It's kind of it's you know, but that was the only way to make him, you know.
He's gonna pay for the $50, his water, his life. He thinks that he's just helping out. So it's kind of it's you know, but that was the only way to make him, you know. But why do they have to live with him? Why can't he live with his wife and kids in another place?
He's gonna pay for the $50, his water, his life. He thinks that he's just helping out. So it's kind of it's you know, but that was the only way to make him, you know. But why do they have to live with him? Why can't he live with his wife and kids in another place? Because apparently.
He's gonna pay for the $50, his water, his life. He thinks that he's just helping out. It's kind of it's you know, but that was the only way to make him, you know. But why do they have to live with him? Why can't he live with his wife and kids in another place? Because apparently he can't afford he can't afford live.
He's gonna pay for the $50, his water, his life. He thinks that he's just helping out. It's kind of it's you know, but that was the only way to make him, you know. But why do they have to live with him? Why can't he live with his wife and kids in another place? Because apparently he can't afford living on his own.
He's gonna pay for the $50, his water, his life. He thinks that he's just helping out. It's kind of it's you know, but that was the only way to make him, you know. But why do they have to live with him? Why can't he live with his wife and kids in another place? Because apparently he can't afford living on his own.
I know the situation's difficult.
I know the situation's difficult, I thought, and but
I know the situation's difficult, and but a lot of any works.
I know the situation is difficult, y to, and but a lot of Ani works, and his wife works también.
I know the situation is difficult, y to, and but a lot of and he works, and his wife works también.
And so.
And some, how many do they have?
Yeah.
One kid, and they can't afford rent.
One kid? One and they can't afford rent or anything. Not even
One kid? One and they can't over not even and Orlando's mom retired and Orlando are sustaining all of them. I mean that's
Okay Okay or or are are just just the theVVss....
One kid? One and they can't avoid rent or anything not even and Orlando Tom Retire and Orlando are sustaining all of them. I mean it's like crazy. Pero bueno but I'm going.
One kid? One avoid rent earnings not even and Orlando Tom Retired and Orlando are sustaining all of them. I mean it's like crazy. But bueno But I'm going on Friday
Tähän laan.
For how long? Ten days? Take pictures.
how long 10 days? Yeah. Take pictures. Oh exciting.
how long 10 days yeah take pictures oh how exciting yeah i can't i can't wait to get the needle
How long 10 days? Yeah. Take pictures. Oh exciting! Yeah, I can't I can't wait to get the new spirit flights Nikanav from, huh?
how long 10 days yeah take pictures oh exciting yeah i can't i i can't wait to get the new the spirit fight in nicaragua from huh spirit airline so they fight in Nicaragua from for all of them orlando
How long 10 days? Yeah. Take pictures. Oh exciting! Yeah, I can't I can't wait to get a new Spirit flight in Nicaragua from huh? Spirit Airlines. So they fight in Nicaragua from Fort Lauderdale. Orlando. My friends, my friends, ones at work.
How long 10 days? Yeah, take pictures. Oh exciting! Yeah, I can't I can't wait to get a new Spirit Flights Nicaragua. So they fight in Nicaragua from Fort Laura. Orlando. My friends, my friends, ones that work. Don't know something about right now, but the whole
How long 10 days? Yeah. Take pictures. Oh exciting! Yeah, I can't I can't wait to get a new. The Spirit Fights Nicaragua Spirit Airlines. So they fight in Nicaragua from Fortale. Orlando. My friends, my friends, ones that work, don't know something about right now, but the whole drama thing and everything. They join the Spirit Air.
How long 10 days? Yeah. Take pictures. Oh exciting! Yeah, I can't I can't wait to get a new. The Spirit Fight Nicaragua from huh? Spirit Airlines. So they fight in Nicaragua from Fortale. Orlando. My friends, my friends, ones that work, don't know something about right now, but the whole drama thing and everything. They join the Spirit Airfare thing here for nine years.
How long 10 days? Yeah, take pictures. Oh exciting! Yeah, I can't wait to get a new Spirit Fights Nicaragua from huh? Spirit Airlines. So they fight to Nicaragua from Fortale. Orlando. My friends, my friends, ones that work, don't know something about right now, but the whole drama thing and everything. They join the Spirit Airfare thing here for $9. They're going, they
How long 10 days? Yeah. Take pictures. Oh exciting! Yeah, I can't I can't wait to get a new. The Spirit Fights Nicaragua from huh? Spirit Airlines. So they fight to Nicaragua from Fort Laurade. Orlando. My friends, my friends, ones that work, don't know Salah right now, but the whole drama thing and everything. They join the Spirit Airfare thing area for $1. They will go to New York for $36 now.
For how long 10 days? Yeah. Take pictures. Oh, exciting! Yeah, I can't wait to get a new one. The Spirit fight to Nicaragua from huh? Spirit Airlines. So they fight to Nicaragua from Fort Laura. Orlando. My friends, my friends, ones that work, don't know something about right now, but the whole drama thing and everything. They joined the Spirit Airfare thing area for $9. They're going, they're going to New York for $36 long trip. Orlando joined and he's been telling me to join. I told Jason.
For how long 10 days? Yeah. Take pictures. Oh, exciting! Yeah, I can't wait to get a new one. The Spirit fight to Nicaragua from huh? Spirit Airlines. So they fight to Nicaragua from Fort Laura. Orlando. My friends, my friends, ones that work, don't know something about right now, but the whole drama thing and everything. They joined the Spirit Airfare thing area for $9. They were going to New York for $36 long trip. Orlando joined and he's been telling me to join. I told Jason, I'm like, I think I'm gonna do.
Okay.
Well you pay nine dollars for the first two months and then.
You pay $9 for the first three months and then after that $30 for the whole year.
You pay $9 for the first three months, and then after that, it's $30 for the whole year. So even if you don't have a buying an airfare, you still stay
You pay $9 for the first three months, and then after that, it's $30 for the whole year. So, even if you don't want to buy an airfare, you still save what you would yeah, and it's just our
You pay $9 for the first three months, and then after that, $30 for the whole year. So, even if you don't want to buy an airfare, you still save what you would say you are going to travel at least on one day.
It you pay $9 for the first three months, and then after that, $30 for the whole year. So, even if you don't buy an airfare, you still save what you would yeah, and in the far you are going to travel at least to one area. And so, we have another frame.
It you pay $9 for the first three months, and then after that, it's $30 for the whole year. So, even if you don't buy an airfare, you still save what you would yeah. And in the way, you are going to travel at least to one area. And so, we have another firm, our architect firm, like next to our
It you pay $9 for the first three months, and then after that, it's $30 for the whole year. So, even if you don't want to buy an airfare, you still save what you would yeah. And it's not sorry, you are going to travel at least to one area. And so we have another firm, our architect firm, like next to our office, and the reception there.
You pay $9 for the first three months, and then after that, it's $30 for the whole year. So, even if you don't buy an airfare, you still save what you would say you are going to travel at least to one place. And so, the girl, we have another firm, our architecture firm, like next to our office, and the receptionaire and the girl, the office manager, in our in our office.
It you pay $9 for the first three months, and then after that, it's $30 for the whole year. So, even if you don't buy an airfare, you still save what you would yeah, and it's not sorry, you are going to travel at least to one place. And so the girl, we have another firm, our architecture firm, like next to our office, and the receptionaire and the girl, the office manager in our in our office are going everywhere. Like they once you need.
You pay $9 for the first three months, and then after that, it's $30 for the whole year. So, even if you don't buy an airfare, you still save what you would say you are going to travel at least to one place. And so, we have another firm, our architecture firm, like next to our office. And the receptionaire and the office manager in our office are going everywhere. Like they went to New York and they did a whole thing in New York for 24 hours.
You pay $9 for the first three months, and then after that, it's $30 for the whole year. So even if you don't buy an airfare, you still save what you would say you are going to travel at least to one place. And so we have another firm, our architecture firm, like next to our office, and the receptioner and the girl, the office manager and our in our office are going everywhere. Like they went to New York and did a whole thing in New York for 24 hours. They flew in like at 6 in the morning and left at 6 in the morning the next day. They were up the whole day.
Basta.
International they're doing the same thing from here, but they're coming on a fighting until.
This time around they're doing the same thing for an airport, they're coming on a Friday and even on Saturday to Boston.
This time around they're doing the same thing for an airport, they're coming on a Friday, they're going on Sally. Then they're going to Boston. They give them a list, they're going to Boston.
This time around they do the same thing for an airport, they're coming on a Friday Negro Sally. Then they're going to Boston. They give a list. They go to Boston, going to Washington,
This time around they're doing the same thing for an airport, they're coming on a Friday, they're going on Sally. Then they're going to Boston. They give them a list. They go to Boston, going to Washington, because the buttons like Y.
This time around they're doing the same thing for an airport, they're coming on a Friday, they're going on Sally. Then they're going to Boston, they give a list, they're going to Boston, going to Washington, because the buttons like yik? Okay. Like, you know, like after a while, like.
This time around they're doing the same thing for an airport, they're coming on a Friday, they're going on Sally. Then they're going to Boston. They give them a list. They go to Boston, going to Washington, because the buttons like yik? Okay. Like, you know, like after a while, like hello. Yeah, you're gonna travel your hope.
This time around they're doing the same thing for New York, they're coming on a Friday, they're going on Sally. Then they're going to Boston, they give a list, going to Boston, going to Washington, because the buttons like yik? Okay, like, you know, like after a while, like Hello! Yeah, you're gonna travel your whole country with Jackie because you're not Jackie though.
This time around they're doing the same thing for New York, they're coming on a Friday, they're going on Sally. Then they're going to Boston. They give a list. Going to Boston, going to Washington. The buttons like yik? Okay. Like, you know, like after a while, like hello! Yeah, you're gonna travel the whole country with Jackie because you're not Jackie then. So it's really good.
This time around they're doing the same thing for New York, they're coming on a Friday, they're going on Sally. Then they're going to Boston. They give a list. Going to Boston, going to Washington. The buttons like yik? Okay. Like, you know, like after a while, like Hello! Yeah, you're gonna travel the whole country with Jackie because you're not Jackie then. So it's really good and so Jason and I'm like, they kind of joined. The only thing is they don't fight.
This time around they're doing the same thing for an airport, they're coming on a Friday, they're going on Sally. Then they're going to Boston. They give them a list, they're going to Boston, going to Washington, and the button's like, yik? You okay, like, you know? Like after how like Hello! Yeah, you're gonna travel the whole country with Jackie because you're not Jackie then. So it's really good and so Jason and I'm like, they don't want to join. The only thing is that they don't fight your cannabis, I'm like. Yeah.
Então
Con Nicaragua, Pipe.
from Nicaragua he paid I think it was like one.
From Nicaragua he paid I think it was like one hundred two or something like
From Nicaragua he paid, I think it was like 102 or something like that so much. To make a round-trip ticket to Kingston.
From Nicaragua he paid I think it was like one hundred and two or something like that so make a round trip ticket to Kingston where Jason says. What on I
From Nicaragua, he paid, I think it was like 102 or something like that. To make a round-trip ticket to Kingston where Jason says 109. So
From Nicaragua, he paid, I think it was like 102 or something like decimal to make a round-trip ticket to Kingston, where Jason says. 109. So we'll have it work.
From Nicaragua he paid I think it was like 102 or something like decimal to make a round-trip ticket to Kingston where Jason says. 109. So we'll have it at work.
From Nicaragua, he paid, I think it was like 102 or something like decimal to make a round-trip ticket to Kingston, where Jason says. Why don't I so happen at work? He said, like I'm I'm with Yenin Cansara.
No No they they have have list list the theururcipcipinging it it''ss in in the the th thyy t tickick it it''ss you you know know you you know know premiumum
Um
What?
What Thursday.
What Thursday, one of the girls like grew her fed up because
What Thursday, one of the girls like, we were fed up because it's like this odorless.
What Thursday, one of the girls, like we were fed up because it's like this, all of us are, you know, we put all
What Thursday, one of the girls, like, we were fed up because it's like all of us are, you know, we put all our work and all our efforts, and we try to peace or whatever, and and
Thursday, one of the girls, like, we were fed up because it's like all of us are, you know, we put all our work and all our efforts, and we try to please her and whatever, and and we really like what we're doing. I like my
What Thursday, one of the girls, like, we were fed up because it's like all of us are, you know, we put all our work and all our efforts and we try to please her and whatever, and and we really like what we're doing. I like peace. I was the cousin of the boss.
What Thursday, one of the girls, like, we were fed up because it's like, all of us are, you know, we put all our work and all our efforts and we try to peace and whatever, and and we really like what we're doing. I like peace. I was the cousin of the of the boss, the same old lady who've been uh
Yeah.
Yeah, I mean there she's there.
Yeah, I mean, she's there. You can't do anything about it. The thing is that...
Yeah, I mean, she's there. You can't do anything about it. The thing is that.
I I''mm su surere have have sam samee?? It It''ss with with it it''ss that that''ss ah ah bu buenoeno pero pero
One of the girls, she went into her office Thursday and
And
Told her to put Small White on a few.
And told her to slow white um if you don't agree with something or you don't like what
And told your lift with SlongWite. If you don't agree with something or you don't like what we're doing, let us know!
And told your lift with Slong White. If you don't agree with something or you don't like what we're doing, let us know that every time you go on a
And told her this was long white. Um, if you don't agree with something or you don't like what we're doing, let us know that every time you go on a trip and you come back, are you are you
And told her this was long white. If you don't agree with something or you don't like what we're doing, let us know that every time you go on a trip and you come back, all you know all you say is
And told your report, Strong White. If you don't agree with something or you don't like what we're doing, let us know that every time you go on a trip and you come back, all you say is: I don't like what you guys are doing, you guys are wasting your time.
And told her therefore Strong White: if you don't agree with something or you don't like what we're doing, let us know that every time you go on a trip and you come back, all you say is: I don't like what you guys are doing, you guys are wasting your time, that she's a constructor.
And told her therefore Strong White: if you don't agree with something or you don't like what we're doing, let us know that every time you go on a trip and you come back, all you say is, I don't like what you guys are doing, you guys are wasting your time. She's not constructive with her critic.
Per enci en
atrizes aitorais da Virginia Borde de Heileg.
Beatriz de todo este tiempo y hallar una cara como que no entendía lo que lo estaban diciendo.
La bicatriz de todo este tiempo, Harley una cara como que no entendía lo que lo estaban diciendo a ella.
and then you know Beatrice.
and then you know Beatrich and Maria Sarah.
And then you know Beatrice finished, and Maria's like, What are you talking about? I've never said that about you. Where are
And then Beatrice finished, and Maria's like, what are you talking about? I've never said that about. Where are you hearing? And she's like, well, you tell Andrea.
And then Beatrice finished, and Maria's like, what are you talking about? I've never said that about. Where are you hearing this? And she's like, well, you t Andrea. You know, and Andrea ist her supervisor. And she
And then Beatrice finished, and Maria's like, what are you talking about? I've never said that about. Where are you hearing this? And she's like, well, you tell Andrea. You know, and Andrea is our supervisor, and she tells us all this feedback.
And then Beatrice finished, and Maria is like, what are you talking about? I've never said that about. Where are you hearing this? And she's like, well, you tell Andrea. And Andrea ist her supervisor, and she tells all this feedback. And she's like, I've never said that.
Right.
Right.
A la.
I've Never Said That, dice Olivia en Marinette.
A NeverSed Dad, mista Olivia en marinesta è Never Sedd.
En
And um if you want to
And if you wanted to confront her, we'll have a meeting.
And if you wanted, could confront her. We'll have a meeting and I'll tell her that I've never said that in front.
Ha Ha....
And if you want, I could confront her. We'll have a meeting and I'll tell her that I've never said that in front of you. And Biaf says like, no, I don't want that.
Nor Normalmal....
And if you want, I could confront her. We'll have a meeting, and I'll tell her that I've never said that in front of you. And be at this like, no, I don't want that. You know, that's the last thing I want. Just let us know if you disagree with them. La cosa es que that was Thursday and yesterday.
And if you want, I could confront her. We'll have a meeting, and I'll tell her that I've never said that in front of you, da da da. And be at least like, no, I don't want that. You know, that's the last thing I want. Just let us know if you disagree with them. La cosa es que that was Thursday. And yesterday I had to run some errands for the past.
And if you want, I could confront her. We'll have a meeting, and I'll tell her that I've never said that in front of you, da da da. And be at least like, no, I don't want that. You know, that's the last thing I want. Just know if you disagree with me. La cosa es que that was Thursday. And yesterday I had to run some errands for the passport thing.
E Erovrovetaeta....
And if you want, I could confront her. We'll have a meeting, and I'll tell her that I've never said that in front of you, da da da. And be at Nisa's like, no, I don't want that. You know, that's the last thing I want. Just know if you disagree with me. Like Ocefe, that was Thursday. And yesterday I had to run some errands for the passport thing. And I called and I was like, oh, I'm gonna be a few minutes late.
And if you want, I could confront her. We'll have a meeting and I'll tell her that I've never said that in front of you, da da da. And be at Nisa's like, no, I don't want that. You know, I that's the last thing I want. Just know if you disagree with them. Like O sea, that was Thursday. Then yesterday I had to run some errands for the passport thing. I called and I was like, oh, I'm gonna be a few minutes late.
Yeah.
Are you leaving somewhere again? Well
My
My passport, the visa
my passport the visa expires in Jules so I have to remember
My passport, the visa expires in June, so I have to remove the writing. Do it now because
My passport visa expires in June, so I have to renue the writing. Do it now because oh, you're fine to Nicaragua? Oh, it's not, okay.
My passport visa expires in June, so I have to renovative writings. Do it now because you're friends in Nicaragua? Oh, it's not okay.
Yeah.
Buena multicara, esa demanda.
Bueno, te cagan eso de madre. Yeah, el password.
The thing is, when I went to Canaan I saw de madre. Yeah, it has to be a lot of.
In December, December before that,
In December, December before that, you were to Canada or Mexico, you didn't eat?
In December before that, you were to Canada or Mexico, you didn't need a passport. That December, they
In December, December before that, you were to Canada or Mexico, you didn't need a passport. And in that December, they passed a law, so, like, there was two weeks before.
In December before that, you were to Canada or Mexico, you didn't need a passport. And in that December, they passed a law. So, like, there was two weeks before the trip, and I'm like, oh, do you need a passport?
In December, December before that, you were to Canada or Mexico, you didn't need a passport. And in that December, they passed a law. So, like, where was two weeks before the trip, and I'm like, well, do you need a passport? I don't know. I have mine up, you know, up to date.
In December, December before that, you were to Canada or Mexico, you didn't need a passport. And in that December, they passed a law. So, like, where was two weeks before the trip, and I'm like, well, do you need a passport? I don't know. Because I have mine up, you know, up to date and my oops.
In December, December before that, you were to Canada or Mexico, you didn't need a passport. And in that December, they passed a law. So, like, where was two weeks before the trip, and I'm like, well, do you need a passport? I don't know. Because I have mine up, you know, up to date and my oops. I also, lucky, you know, in downtown as well.
In December, December before that, you were to Canada or Mexico, you didn't need a passport. And in that December, they passed a law. So, like, where was two weeks before the trip, and I'm like, well, do you need a passport? I don't know. Because I have mine up, you know, up to date and my oops. I also, look, you know, in downtown, there's a passport building that there's
In In the the De Deberber be beforfor Can Can or or you you did did a aspsproro they they was was two two tri tripp I I''mm like like well well do do need needss like like I I don don''tt because because I I up up to to da dayy I know know in intoto pas passese we we''rerewworkork at at C Clolopsps
In December, December before that, you were to Canada or Mexico, you didn't need a passport. And in that December, they passed a law. So, like, where I was two weeks before the trip, and I'm like, well, do you need a passport? You need like, I don't know. Because I have mine up, you know, up to date and my oops. I also, lucky, you know, in downtown there's a passport build, but there's a courthouse in Professor Street. We were working at the courthouse, and I would go through my lunchtime, Tengara.
In December, December before that, you were in Canada or Mexico, you didn't need a passport. And in that December, they passed a law, so like, where I was two weeks before the trip, and I'm like, well, do you need a passport? I don't know. Because I have mine up, you know, up to date and my oops. I also, lucky, you know, in downtown there's a passport building. There's a courthouse in Professor Street. We were working at the courthouse. And I would go through my lunchtime. Thank God I was able to find up.
In December, December before that, you were in Canada or Mexico, you didn't need a passport. And in that December, they passed a law. So, like, where I was two weeks before the trip, and I'm like, well, do you need a passport? I don't know. Because I have mine up, you know, up to date and my oops. I also, lucky, you know, in downtown, there's a passport build, there's a courthouse across the street. We were working at the courthouse, and I would go through my lunchtime. Thank God I was able to find a make an appointment and then I had to say
In December, December before that, you were in Canada or Mexico, you didn't need a passport. And in that December, they passed a law, so like, where I was two weeks before the trip, and I'm like, well, do you need a passport? I don't know. Because I have mine up, you know, up to date and my oops. I also, lucky, you know, in downtown, there's a passport build, but there's a courthouse across the street where we were working at the courthouse, and I would go during my lunchtime. Thank God I was able to find a where I had to make an appointment, and then I had to say that I was taking doctor appointments.
C'est un
and rating for hacking out two hours.
And everything are hacking out two hours in line. I have sort of 100 lava.
And narratives are hacked two hours in line, and I thought 100 lobby. Two an hour became. And then the thing is that I.
I was two hours in line. I thought 100 lava. Two an hour became. And then the thing is that I went and they told me that I didn't have.
And everything I had to go, I was two hours online and I saw two an hour became. And then the thing is that I went and they told me that I didn't have to get a new passport so
And everything I had to go two hours online. I saw two now became. And then the thing is that I went and they told me that I didn't have I had to get a new passport. So I'm like, oh my god, I have like I literally.
And everything I had to go I was two hours online. I saw alaby came. And then the thing is that I went and they told me that I didn't have I had to get a new passport. So I'm like, oh my god, I have like I literally had a week I have a week. Well, how long?
And everything I had to go. I was two hours online. I saw alabo. And then the thing is that I went and they told me that I didn't have I had to get a new passport. So I'm like, oh my god, I have like I literally had a we have a weapon. Well, how were we to the last picture? Because
And everything I had to go two hours in line. I saw almond now became. And then the thing is that I went and they told me that I didn't have I had to get a new passport. So I'm like, oh my god, I have like I literally had a we have a weapon. Well, how will we took the last picture? Because it doesn't, it doesn't import.
And everything I had to go two hours online and I saw almond allowed. And then the thing is that I went and they told me that I didn't have I had to get a new passport. So I'm like, oh my god, I have like I literally had a we have a weapon. Well, how were we to the last picture? Because it doesn't importer. The thing is that my passport
And everything I had to go two hours online and I saw it now became. And then the thing is that I went and they told me that I didn't have I had to get a new passport. So I'm like, oh my god, I have like I literally had a we have a week. Well, how were we to the last picture? Because it doesn't importer. The thing is that my passport, AMA Luante Valida.
And everything I had to go two hours online and I saw 100 lobby. And then the thing is that I went and they told me that I didn't have I had to get a new passport. So I'm like, oh my god, I have like I literally had a we have a week. Well, how will we took the last picture? Because it doesn't importer. The thing is that my passport, AMA, validado los bases, and now it
And everything I had to go I was two hours in line. And then the thing is that I went and they told me that I didn't have I had to get a new passport. So I'm like, oh my god, I have like I literally had a we I have a week. Well, how will we took the last picture? Because it doesn't it doesn't importer. The thing is that my passport, a manual validado, dos bases, and now it's law, you know.
And everything I had to go. I was two hours in line. I saw 100 lobby. And then the thing is that I went and they told me that I didn't have I had to get a new passport. So I'm like, oh my god, I have like I literally had a we I have a week. Well, how will we took the last picture? Because it doesn't, it doesn't importa. The thing is that my passport, a manuante validado, dos bases, and now it's law, you know, in the
programs whatever
for our interns or whatever. Get
Yeah.
Ya te lo han revalidado dos veces ya sea.
Even
Even if I have like clean you know
even if I have like cle you know sheets available.
even if I have like cle you know sheets available, little pages available.
A Aguguaa!!
Even if I have like little pages available. But you get your US passport, you're gonna keep an agreement passport too? I I I don't know. With a US passport, you don't have to do that.
Even if I have like cle, you know sheets available, little pages available. And you get your US passport, you're gonna keep an agreement passport too? I I I don't know. With a US passport, you don't have to do that? No, I think my parents
I think Karen does
I think Karen does. I don't know why. But Karen was a mourn there.
I think Ken does. I don't know why. But Karen was a moiner. She liked more on a sun. But she has an
Don't have it anymore. I think Karen does. I don't know why. But Karen was a moiner. Like more on a sound but she has a when she lived over there, I think that's when she
I think Karen does. I don't know why. But Karen was a moiner. Like more- when she lived over there, I think that's when she got it. Oh really? Oh okay.
I think Karen does. I don't know why. But Karen wasn't born there. But she has a crap. When she lived over there, I think that's when she got it. Oh, really? Oh, okay. D'accord say it? Your boss was going for citizenship too, or no?
I think Karen does. I don't know why. But Karen wasn't born there. But she has a coro. When she lived over there, I think that's when she got it. Oh really? Oh okay. D'accord say it? Your bus was going for citizenship too, or no? Yeah, I hope to try to.
I think Karen does. I don't know why. But Karen wasn't born there. But she has a coro. When she lived over there, I think that's when she got it. Oh, really? Oh, okay. D'accord say it? Your bus is going for citizenship too, or no? Yeah, I hope to try. Thank you, it's not a minor.
I think Karen does. I don't know why. But Karen wasn't born there. But she has a car. When she lived over there, I think that's when she got it. Oh really? Oh, okay. D'accord say it? Your boss is going for citizens too, or no? Yeah, I hope to try. Oh, thank you. He's old.
I think Karen does. I don't know why Karen wasn't born there. But she has a crap. When she lived over there, I think that's when she got it. Oh really? Oh okay. Diacosa? Your birth uh is going for citizenship too, or not? Yeah, I hope to try it. Oh, thank you. He's old. Thank you. I forgot how old he is.
Thank you.
Thank you.
Thank you.
Thank you.
Thank you very much.
Anything else and against some masters you have to catch up for a second
Anything else and again some laughters you have charter for a second distribution, things, and more case masters.
Anything else, remediation laptops, you have chat operation as we should have more questions afterwards. Yes, yes, thank you.
Anything else, formation laptops, you have chat operation, just promotion after. Yes, yes, thank you.
Anything else, remediation naptions, you have catch a fresh restriction. Things promotion naptions. Yes, yes, thank you. And what I think. So after I calmer and then she comes in,
Anything else, and we get some napkins, you have to actually put a presentation. Thanks. Yes, yes, thank you. So after I call Marie and then she calls me about it and she's like, Leah, we're having a meeting.
Anything else, let me get some napkins to help catch operations, resting from a case nap. Yes, yes, naked. And what's okay? So afterwards, I call Marie and then she calls her back, and she's like, Leah, we're having a meeting. Other designers, I need a committee.
Anything else, let me get some napkins, you have to catch operations, resting things. Things promotion laughs. Yes, yes, naked. And what's okay? So afterwards, I call Marie and then she calls her about it and she's like, Leah, we're having a meeting. All the designers, I need to clear things up, blah, blah, blah. And I'm like, Maria's the name of the boss, the passage.
Anything else, let me get some napkins or help catch operations, just meeting things. I'm a case nap. Yes, yes, thank you. And we're okay. So after I call Marie and then she calls her about it and she's like, Leah, we're having a meeting. All the designers, I need to clear things up, et da da da da, and I'm like Maria's the name of the boss. Yeah. The Passa. Yeah. Our supervisor.
A Anytnyt,, we we get get some some hel helppersers.... Com Comococasease L Lapapeses and and so so af afterter.... then then she she back back,, she she''ss like like,, all all des desee,, and and Mar Mariaia.... Le Le P Passaassa........
Anything else? Let me get some napkins, you have to actually operate the things. Thanks. Promote laughter. Yes, yes, thank you. And we're okay. So after I call Maria and then she calls her about it and she's like, Leah, we're having a meeting. All the designers, I need to clear things up, blah, blah, blah. And I'm like, Maria's the name of the boss. Yeah. Yeah. Our supervisor is Andrea. Andrea is her cousin. No, Andrea.
Anything else? Let me get some napkins to help catch up for the success. Thanks. Promo face napkins. Yes, yes, thank you. And we're okay. So after I call Maria and then she calls her about it and she's like, Leah, we're having a meeting. All the designers, I need to clear things up, blah, blah, blah. And I'm like, Maria's the name of the boss. Yeah. Yeah. Our supervisor is Andrea. Andrea is her cousin. No, Andrea is just a girl she's always worked with.
Anything else, let me get some napkins to help catch up for the successful thing. Thanks. Promo face napkins. Yes, yes, thank you. And we're okay. So after I call Maria and then she calls her about it and she's like, Leah, we're having a meeting. All the designers, I need to clean up, blah, blah, blah. And I'm like, Maria's the name of the boss. Yeah. Yeah. Our supervisor is Andrea. Andrea is her cousin. No, Andrea is just a girl she's always worked with.
Про Просиси......
She's not above her. No, she's
She's not above her. No, she's below her. She's just above us. Okay.
She's not above her. No, she's below her, she's just above us. Okay.
pero si hay una mala
But she has like my left hand. To me, she's always
Pero she has like my left. To me, she's always been nice. Ooh Andrea? Yeah.
Fero she has like my left to me, she's always been nice, but she've told
Fellow, she has like my left. To me's always been nice, but she know they've told me several.
Fo she has like malefacts. To me she's always been nice, but she Andrea? Yeah. You know they've told me several gente por ella, stuff like that.
Food, she has like malefactions. To me she's always been nice, but she Andrea? Yeah. You know they've told me several hente porrea, stuff like that. Oh really?
than the thing.
But then the thing is Maria wanted Super Funny
So then the thing is Maria wanted to confront everybody and clear everything out.
So then the thing is, Maria wanted to converge and clear out and find that that wasn't true.
La cosa es que
She had me on Fold Conference and
She had me on Phone Conference
She had me on phone conference, and she practically told Andrea everything that she said she was taking
She had me on phone conference and she practically told Andrea everything was tell her, and she wasn't really.
She had me on phone conference and she practically told Andrea everything that she said she was telling her, and she wasn't referring to us and
Whatever.
And
And I the thing is that I get to the office and that there's like
Almost mine. She's like that's not true.
Almost mine. She's like, that's not true. Jonathan Mentipoids My Word Against Her.
Yeah.
Yeah. She's like really.
She's like, believe me. That's what she says. I'm not lying.
She's like believe me, that's what she says, I'm not lying.
I don't know what it is.
So I don't know what this is boha reputational.
So, I don't know what they're smile, you know? Both are reputational. Yeah, both of them.
So, I don't know what refutational. Yeah, both of them, I don't know good actors.
So, I don't know what's going to be, you know? This is both are refutational. Yeah, both of them. So it's like there's that I don't know. Some people are good acts too. Oh yeah.
I don't know, you know? This is both have refutational. Yeah both of them. So it's like there's that I don't know. Some people were good acts too. Oh yeah and then Maria
So, I don't know, you know? This is both refutational. Yeah both of them. So it's like there's that I don't know. Some people were good acts too. Oh yeah and then Maria
Bleep
Blein it on Ela.
Blamed it on Elena Elena left the office, so she got to
Blamed it on Elena Elena left the office, she got transferred to New York so, you know she kind of found
Blamed it on Elena Elena left the office, she got transferred to New York so, you know, she kind of found: okay, well, let me blame it on Elena because
She kind of found: okay, well, let me blame it on Elena because she's not in the office
Ja, en natuurlijk.
Yeah, but and not to you know.
But in the end, if this problem gets
But in the end, if this problem gets bigger,
But in the end, if this problem gets bigger and Andrea says, boy, Leno is the one gossip.
But in the end, if this problem gets big Andrea says, Boy, Elena is the one gossip. You know, and she does, Elena's like, she's
what's going on.
I am.
It's a
I'm just like
What was it
What was it beaming on that?
Да.
that she was the one that told me
that she was the one that told Maria
Some that told Maria what Andrea was saying that
That she was the one that told Maria what Andrea was saying to us. That's how Maria found out that.
Some that told Maria what Andrea was saying to us. That's how Maria found out that
telling us that she wasn't happy
Bravo!
If it's not happy with the work, fine, you know, she can- I mean you can-
But if she's not happy with the work, fine. I mean, you can understand that everybody has their own level of expectation. Why you can just sit and be like, look,
Bro, but if she's not in the work, fine. You know, she can I mean you can understand that, you know, everybody has their own level of expectation boy, you can just sit and be like no, you guys, I would prefer you guys telenational.
But if she's not happy work, fine. You can understand that. Everybody has their own level of exploitation, you can just sit down be like, look, you guys. I would prefer you guys. Tellina, she never said.
Já.
I don't know who was lying and on
That I don't know who is lying, and honestly, I'm tired of it, and I don't care, you know?
I don't know who is lying. And honestly, I'm tired of it. And I don't care. You know? Yeah.
I don't know who is lying. And honestly, I'm tired of it. And I don't care. You me tienen cansada.
Muchas gracias.
Considering the
In the
In a situation, the truth always conserves. Always.
In a situation, the truth always conserved. Always. Consum made from you were splept
How annoying the Psych High School
How annoying the psych high school.
Thank you.
I get along with everybody, but I do
Gracias.
minutes.
And it's distracting.
So
They gave me insensations.
They gave me intersections to the movies, or else?
They gave me tickets to the movies. Are valing?
Asmov.
Mach!
Bird.
Oh, Festation uh
Hello.
Hello.
Mm.
Fine.
Dankjewel.
I'm Hierodonius and my cousin.
Waiting breakfast.
Lines.
¿Quieres
Yeah.
Niente.
Sí.
Very cute. I think you have tennis. But this is no more sweets.
Very cute. I thought you need so linium. But that is, no more sweets. This is the under there, I promise. No six three, no six.
Ма.
My valence.
Эффать.
Yazer.
Go Googleogle..................
Her lungs are swollen, and they put her in antibiotics for two weeks. She's had me take two kills a day, and I have to give her 15 drops of some fluid.
Her lungs are swollen и puttern antibiotics for two weeks. Сhe's have two kills a day and have her fifteen drops of some fluid thing
I told him what happened. He started making
I told him what happened, he started making noise.
I told him what happened, he started making noise. I'm like, yes, that was exactly, and so then he did an S her long.
I told him what happened. He started making noise. I'm like, yes, that was exactly, and so then he did an extra launch. He asked me, has she been an accident?
I told him what happened. He started making an noise. I'm like, yes, it was exactly and so then he did an extra on her lawn. He asked me, has she been an accident or something? No.
He started making an noise. He asked me has she been an accident or something? No. Has she gone into the plug?
I told him what happened. He started making a noise. I'm like, yes, that was exactly, and so then he did an issue on her lawn. He asked me: has she been an accident or something? Like no. Zick, has she gone into the plug? He asked me if she swallowed water or something. Like, no, like she hates the water.
I told him what happened. He started making a noise. I'm like, yes, that was exactly, and so then he did an extra on her lungs. He asked me: has she been an accident or something? No. Has she gone into the plug? He asked me if she swallowed water or something. Like, no, she hates the water. I'm like, I bet even when I bait her, it's like no.
So, I don't know if maybe it's from before.
So I don't know if maybe it's from before that she's hiding, that she's exhumating, sitting there. I mean, I don't know if maybe one day when I took her to the
So I don't know if maybe it's from before that she's hiding, she's accumulating and sitting there. I mean, I don't know maybe one day when I took her to the beach.
Menos falta
And as Walton Water, I mean, I don't know where it came from.
So hitomido. Hám.
So hit omidop
En die mensen.
No, no, no. I'm like, it's curable, it's curable.
No, no, no. I'm like, it's curable, it's creable. Oh, yeah, yeah, yeah. Exactly.
No, no, no. I'm like, it's curable, it's cutable. It's like, oh, yeah, yeah, yeah. It's just like her show me the crazy, a part of her. It's so fascinating. I Will Want to Be a Back, etc.
No, no, no. I'm like, it's curable, it's treatable. He's like, oh, yeah, yeah, yeah. It's just like her, part of her. It's so fascinating. I Will Wanna Be a Bad, except I don't like biology. Oh, and
No, no, no. I'm like, it's curable, it's treatable. It's like, oh, yeah, yeah, yeah. It's just her He Show Me the Ecrays, like a part of her lot. It's so fascinating. I Would Want to Be except. I don't like biology. Oh, and the Hedges for a Lot of Catches each today. I'll tell you about the
No, no, no. I'm like, it's curable, it's treatable. It's like, oh, yeah, yeah, yeah. It's just her He Show Me the Ecrays, a part of her lot. It's so fascinating. I Would Wanted, except I don't like biology. Oh, and the Hedge for a Lot of Catches each today. I'll tell you about that later. That's just so sad.
This is what
This is why it couldn't be a vet.
Anyways.
Anyways, no visual me a porte-bollante.
Anyways, no, V show me a part of black. Like the dose was very dark because were swollen.
Anyways, um, no, Fish Me a Parte Vollance that in the x ray they came out black, like the dose was very dark because they're spolen. Exactly, it shouldn't be that dark, it should be more
Anyways, um, no Fichome Porter Vollance that in the X ray that came out black, like the dose was very dark because they were spolen. Exactly, it shouldn't be that dark, it should be more like this. And you compared it to like a lighter
So he's like we'll put our antibiotics and then compact
So he's put on antibiotics and then come back in two weeks.
He's like
He's not sure.
He's not sure. He's like he's like you know put tried ticket by Alex
Eso.
And so d'
Лааййфферер.....
What? Her long pin swollen ask.
Her lumping swollen? Адо нотин аскем. И значит.
Her lump being swollen? А не аскрет и.
Her lump being swollen? Адит. He's not sure what it is. He told her the medication and then see if it happens in two weeks, and two weeks, and he'll.
Her lunch and swollen, I didn't ask him. He's not sure what it is. He told me to give her the medication and then see if it happens in two weeks, and if it continues in two weeks, he'll see what else it is.
Her lump being swollen А, един аск. И значит the medication и see it happens in two weeks.
So
So he gave her shot today.
He gave her shot today, so I'm not gonna give her any medication substitution.
I'm still.
En die Cleaner Glends en de
Any cleaner glands and then I told about the fleeting, he gave me some pills for the fleas. It's a pill you give for one pill every 10 days and it kills the fleas within 30 minutes.
Any cleaner glands, and then I told about the fleeting, he gave me some pills for the fleas. It's a pill you give her one pill every ten days, and it kills the fleas within 30 minutes of pertaking the pill. So I gave her the pillar when I got home.
He just solmit to, you know not to be her now, but just in the house or whatever, like, you know.
use
So other than that she's okay.
I'm not.
Henel!
He should ask me where it was.
He asked me where it came from, and I told him I didn't know, so.
He asked me where it came from, and I told him I didn't know, so I know.
I don't
Okay.
Summer
Yeah, yeah.
Der K.
The Coffin. Yeah, he's like the It's Polycus her long.
The copy, yeah, it's probably
Either
I just thought about the sneezy
Does needs in I really think it's because she could.
The sneezy nibulating speaker she hits her nose.
The sneezy nib relating is because she hits her nose. She's always longing herself into the nose.
The sneezing I really think is because she hits her nose. When I hit my nose, I sneeze. She's always herself into the carpet and hitting herself with the paws.
She's always launching herself into the carpet and hitting herself with the por. This thing has to do with that.
When I hit my nose, I sneeze. Сила, hitting herself with the por. This thing has to do with that. I mentioned, episode is like severely sneeze.
Sneezing I really think is because she hits her nose. I mean, when I hit my nose, I sneeze. Си'ось launching herself into the corpet and hitting herself with the portion. This thing has to do with that. I mentioned episode is like severely the solution in the senior.
The sneezing I really think is because she hits her nose. I mean, when I hit my nose, I sneeze. Си ось launching herself into the corpet and hitting herself with the pod. This thing has to do with diabetes. Episode tennis like severely then tuition. Detoga temperature, she has um.
Certo.
Cetohydrend temperature in her normal temperature is 101.
Shetohydrigen temperature temperature was 101, so it was a little bit high.
Per esempio.
Así que...
Schigina tu kan.
Ok, what are we gonna do?
Ok, what are we gonna do? We be perhaps.
Ан.
Yeah.
A băi.
Oké.
Okej,
I'm not sure.
Děkuji se dělá.
You know, I have to, well, I need to.
So yesterday, Jasonai started telling me like he's
Yesterday, Jason draw details because the was being constructed.
I need to help me draw detail because the was being constructed foundations and the puddings,
I need to help me draw detail because the was be constructed and the puddings, I know the actual construction and what the
I need him to help me draw details the was being constructed foundations and the puddings, рейдену до construction, what the materials are called, and how the technically put them on the drawings. He does, obviously.
Cartese.
I understand but it's like look,
I understand, but it's like look, I have things to do. You know, last week and my take
I understand, but it's like look like I have this do you know last week and I had to take work home for my boss.
aquí aquí
I understand, but it's like: look, I have things to do. You know, last weekend I had to take work home for my boss. My boss. So it's like, I don't suit that every weekend all the time to do it.
have that exam to study for us.
I'm not sure.
I want you to go to Africa.
I want you to go to what are you going to do afterwards
Why should go to one to afterwards?
Mentor.
What had to get on Christina's daughter, something?
What happened to O'Neav to get Christina's daughter something
How can we find something there on sale?
I just spent $140 in the back, so
Era.
AJ has to go with carried
To buy something for, somebody.
To by something for somebody, one of our friends, Birthdays as you know.
To buy something for one of our friends Birthdays as tonight.
Кивер.
Aioran panic!
Yeah, it's in the halls. I could get together.
Um
Um
naar word to do.
wag
White.
It was yesterday, oh, with the whole passp
It was yesterday, oh, with the whole passport thing. Uh huh. I had to do the ladies, like in the morning, she told me she like you had to do
So I told the co-workers and my
So I'm like, oh, I told my co workers and Maria, I'm like, I'm not too nice, I have to go.
And
I'm like out of home.
Tom and D will be behind.
Thank you.
Dev
Do they pay more for taking work at home or no?
Corizație interium!
Ineiro?
And I don't solve them when I do.
And I don't solve them when I do because it's a laser costume.
Ja.
Don't expect me to do it.
They expect Niji to weak hands.
The expect needs to maybe meet candles.
Volunteer.
I'm not!
And I think she's like she's kind of normally in the
And I think she's like she's kind of neurologic that all of us are just bad up but her.
And I think she's kind of noticing that all of us are just made up with the whole situation.
And I think she's like she's kind of noticing that all of us are just made up with the whole situation. Who that really? Yeah.
And I think she's kind of noticing that all of us are just made up with the whole situation. And that we're just sick of it. There's so much trauma on you.
And I think she's kind of noticing that all of us are just made up with the whole situation. And that we're just sick of it. There's so much trauma. You don't need that much trauma with all the stress and deadlines.
And I think she's kind of noticing that all of us are just made up with the whole situation. And that we're just sick of that. There's so much trauma. You don't need that much trauma with all the stress and deadlines, you don't need this noise.
Chasse!
The jobs transform now, we don't need to add to it.
without yourself, die.
Ciao!
So I get we go down.
So I get we we build out our like review
Like two weeks ago.
Yes.
Excuse and skin part bonuses.
¡Es la vida!
Dar nu vi vedeți.
A
I think we do. I don't wonder.
You know.
You know, by the normal.
You know, by the neurobasket. No, it's animal.
Vă.
Which is bipolar.
No, he's still there.
Сида!
They haven't mentioned it again.
Adiós.
Because when he's around, she's like, you know
Because when he's around, she's like, you know, inabatable. She doesn't, yeah, she doesn't bother.
Because when he's around, she's like, you know. She doesn't bother us.
On our back, every five minutes. Is it dunya, is it dunya.
That's annoying. It turned one day, and now it wasn't me. We went, we
That's annoying. It's one day, and I wasn't me. We went, we had a morning meeting whatever, we went.
That's annoying. It's one day, and I wasn't me. We went, we had a morning me, whatever. We went to the office to do
You know, she had an assignment that night.
Мъжа!
She goes, she's like yeah, well. They don't know what you do, worries for the ask so much and they
No.
No.
Contrive, I'm not
I can try, but I'm not
As becomes
It's because they either maybe once did it or
That's because they either maybe once did it, or now they forgot, you know, take so long how time consumers, or two, they never don't.
The tries told your one day. Monday you should just sit with one of you know
The 3 soldier was one day. You should just sit with one of us and you know, sit down, and actually take your mind. See how
The three soldier would be one day. You should just sit with one of us, and you know, sit down and actually take your mind and see how long it takes to draw some
Telegó.
Ale bombás, cis
The People of Action, the parties
Ik heb
I had this name for the Fathums, right?
I had this name for the paths, right? And I just like trial. And yes, this sounds very easy.
I had this in for the paths, right? And I just like trial. And yes, this sound very easy, but it's a lot of thank you for me because
I had this in for like the paths, right? And I just like trial. And yes, this sound very easy, but it's a lot of tanky for me because sharp feels like
Is weten.
It's very time to make as a give you a board with different price range.
Est-ce que vous
One, two, three, four, five, five minimals expensive.
in the specs, which is part of your content.
In the specs like you, which is part of your contract, you told
In the specs, which is part of your contract, you told them you know, prices one and the cheapest, you're going to need
Five percent.
A.
So I have to make sure that what I'm using is being a scientist.
So I have to make sure that what I'm using is being assigned to the batum and that personage. You know how long I'm missing?
So I have to make sure that what I'm using is being assigned to the datum and that personage. You know how long that's a long. And then, in addition to coordinate that with
So I have to make sure that what I'm using is being assigned to the batum and that personage, you know, in this corner with coin, the other interior finishes.
We have to make sure that one of music is all in corn that with coin the other interior finishes a lot of work.
Литар.
No, no, shit, I'm not, you know.
So when I tell my boss and I can fly, I know signal not to resist. I'm just, you know, lady gagging, but shit, no, no, shit, I'm not, you know, I'm not questioning your, you know, just whatever.
So when I tell my boss and I can fight, I know signal not to resist I'm just you know lady gagging, but shit, no, no, shit, I'm not you know, I'm not questioning your, you know, just whatever. I want to get on the right once because
So when I tell my boss and I can find it, I know it's in a lot of resistance. I'm just, you know, lady gagging, but shit, no, no, shit. I'm not, you know, I'm not questioning your, you know, just whatever. But I want to get on the right once because we think I want we submitted the joints.
So when I told my boss and I can find it, I know it's in a lot of resistance. I'm just, you know, lady gagging, but shit, no, no, shit, I'm not, you know, I'm not questioning your, you know, just whatever. But I want to get them right once because we think that when we submitted the joints, we not so get so yet, but we gave them
So when I tell my boss and I can find it, I know it's in a lot of services. I'm just, you know, lady gagging, but shit, no, no, shit, I'm not, you know, I'm not questioning your, you know, just whatever. But I want to get them right once because we think out when we submitted the joints, we not subside. But we gave them in the spec what we were going to press out, and the contract we did.
So when I told my boss and I compliment, I know it's in a lot of resistance. I'm just, you know, lady gagging, but shit, no, no, shit. I'm not, you know, I'm not questioning your, you know, just whatever. But I want to get them right once because we think that when we submitted the joints, we not subside. But we gave them in the spec what we were gonna price out, and the contract didn't bid it out, didn't price it. So the
So when I told my boss and I compliment, I know it's in a lot of resistance. I'm just, you know, lady gagging, but shit, no, no, shit. I'm not, you know, I'm not questioning your, you know, just whatever. But I want to get them right once because we think that when we submitted the joins, we not subside. But we gave them in the spec what we were gonna price out, and the contract didn't bid it out, didn't price it. So now when we're submitting these new drains,
So when I told my boss and I compliment, I know it's in a long time. I'm just, you know, lady gagging, but shit, no, no, shit, I'm not, you know, I'm not questioning your, you know, just whatever. But I want to get them right once because we think that when we submitted the joins, we not subside out yet. But we gave them in the spec what we were going to price out, and the contractor didn't bid it out, didn't price it. So the so now we're submitting these new drains, the joint, the work is coming out more expensive now.
So when I told my boss and I compliment, I know it's in a lot of resistance. I'm just, you know, lady gagging, but shit, no, no, shit. I'm not, you know, I'm not questioning your, you know, just whatever. But I want to get them right once because we think that when we submitted the joins, we not subside out yet. But we gave them in the spec what we were going to price out, and the contract didn't bid it out, didn't price it. So the so now we're submitting these new drains, the joint, the workers coming out more expensive now. But it wasn't our fault, it was that he.
So when I told my boss and I compliment, I know it's in a lot of resistance. I'm just, you know, lady gagging, but shit, no, no, shit. I'm not, you know, I'm not questioning your, you know, just whatever. But I want to get them right once because we think that when we submitted the joins, we not submit out yet. But we gave them in the spec what we were going to price out, and the contract didn't bid it out, didn't price it. So now when we're submitting these new drains, the joint work was coming out more expensive now. But it wasn't our fault, it was that he, you
He embedded correctly constood.
He didn't bid it correctly, he comes to the contract
He didn't bid it correctly. He comes, he comes to the contradictions to us and he's like, Oh, you know, you guys use this and that's now like no.
He didn't bid it correctly he comes, he comes to the contract he's like, Oh, you know, you guys use this now this now like no, I suppose right then and there, and I print all the jars. I'm like, I know.
He didn't bid it correctly he comes, he comes to the contract comes to us and he's like, oh, you know, you guys use this in that's now like no, I sometimes right then and there, and I print all the jars. I'm like, I know what I did, I worked on this for a month and a half, you know I know
He didn't bid it correctly he comes, he comes to the contract comes to us and he's like, Oh, you know, you guys use this in that's now like no, I sometimes right then and there, and I print all the jars. I'm like, I know what I did, I worked on this for a month and a half, you know I mean I memorize the color line and the numbers and everything.
He didn't bid it correctly he comes, he comes to the contract comes to us and he's like, oh, you know, you guys use this in that's now like no, I sometimes right then and there, and I printed all the jards, I'm like, I know what I did, I worked on this for a month and a half, you know. I know, I mean, I memorize the color line and the numbers and everything. It's
And at the end we were right. So I'm like I'm glad
And at the end we were right, so I'm like I'm glad that I tick so much with that because now at the end.
And at the end we were right. So I'm glad that I tick so much time with that because now at the end avoids err.
O OKKosos
Can you expect us? I mean no I mean we put it off
They expect us, I mean, no, I mean we put it off to do things like
They expect us, I mean, no, I mean we put it off a long home to do things like Project and Lia.
Amen.
And then but I told her.
And then but I told her. Yeah, where's like a little
And then but I told her. Yeah, it was like a little flesh hop.
Tá dele.
They're gonna fix the building
They're gonna fix it. Are you building? In my office
C'est pas là.
We were like some connection and we were like
We were like, and we are just like
We were like, and we had that were like unsure this is like the chinaho.
Demolispe!
on A-line last Friday.
Etlaï la Fra.
Airline last Fri.
Nike eighty five dollar presus that they have if you want factoring $40.
А-ля!
Most name.
The most unemployed works there makes their stoolow.
I met when I went.
I was like, I met I met when I went in China.
Momento.
We went to this place to get myself.
Wir winnen zu places with stuff and one of the guys
We Weininessess for the the mas masss and and one one doll doll
So, you remember that name?
The tannes business. Install with the auditions, his bio business,
The Tanish Tan shoot business. He stopped all the shoes, he like look as Nike, shavers, and used to watch them and seemed
The Tanish Tan shoot business. He stopped all the shoes, he looked as naked, shaved. And he sort of seemed like tennis shoes
The Tanish Tanchus business. He stopped all the shoes, he looked as naqui, shaved. And he s what sim thing. And he play tennis
The same things, a glow cows can book you at the check.
No?
Westrias len
Some jurioz.
Entonces
I think jury as Len against Gorbis and the Pedinome in Sio tienen Ladono
Emergences and things that we have about from Chic.
And rechasses and things that we have bought from Chigos. Like 45 dollar, you know
And remember chests and things that we have bought from Chigos. Like $45
A B.
Each other like you compare two sounds, or you automatically know.
Each other like you compare two sounds or you automatically know.
Steam, steel, something like that, and just a lot of
Stain steel, something like that, and just custom jury and get something.
Stain steel, something like that, and just like custom jury, yeah, and get something you bar
Something you want to see if it has that and you
Know somebody wanna see if it has that and usually has like a heavier sound.
And the
And the sound is very light.
And we tested muddynesses like MDC and she
Of gold.
We had B and Sinti and
It's hell.
One of letters, cheaper materials.
Chimber
Pudi
Dénben
Was the given girls Pasi Pan design, she could
I was gonna give the girl the Plastic Fen design
Yeah, so she had to hold the plastic
Né!
You the
Mai stoken Godov
She can only wear
A
