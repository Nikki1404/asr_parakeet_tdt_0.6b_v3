import requests

def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    url = (
        "https://eastus2.api.cognitive.microsoft.com"
        "/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
    )
    headers = {"Ocp-Apim-Subscription-Key": "5s3pFV0dpevEwzemTJFSFdaGtMI4uUtaANYUkVd"}

    # ✅ Add both locales — Azure will auto-detect per utterance
    definition = '{"locales":["en-US", "es-ES"],"profanityFilterMode":"None"}'

    files = {
        "audio": (filename, audio_bytes, _mime_type(filename)),
        "definition": (None, definition, "application/json"),
    }
    resp = requests.post(url, headers=headers, files=files, timeout=60)
    data = resp.json()

    combined = " ".join(
        p.get("text", "") for p in data.get("combinedPhrases", [])
    ).strip()

    return {"success": True, "transcript": combined, "raw": data}

def _mime_type(filename):
    if filename.endswith(".wav"):  return "audio/wav"
    elif filename.endswith(".mp3"): return "audio/mpeg"
    elif filename.endswith(".m4a"): return "audio/mp4"
    return "application/octet-stream"

with open("a.wav", "rb") as f:
    audio_bytes = f.read()

res = transcribe_audio(audio_bytes, "a.wav")
print(res["transcript"])



import requests

def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    url = (
        "https://eastus2.api.cognitive.microsoft.com"
        "/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
    )
    headers = {"Ocp-Apim-Subscription-Key": "5s3pFV0dpevEwzemTJFSFdaGtMI4uUtaANYUkVd"}

    # ✅ Single locale = no language detection overhead, just transcribe
    definition = '{"locales":["es-ES"],"profanityFilterMode":"None"}'

    files = {
        "audio": (filename, audio_bytes, _mime_type(filename)),
        "definition": (None, definition, "application/json"),
    }

    resp = requests.post(url, headers=headers, files=files, timeout=60)
    data = resp.json()

    print("STATUS:", resp.status_code)
    print("RAW:", data)  # ← keep this until it works

    combined = " ".join(
        p.get("text", "") for p in data.get("combinedPhrases", [])
    ).strip()

    if not combined:
        combined = " ".join(
            p.get("text", "") for p in data.get("phrases", [])
        ).strip()

    return {"success": True, "transcript": combined, "raw": data}

def _mime_type(filename):
    if filename.endswith(".wav"):   return "audio/wav"
    elif filename.endswith(".mp3"): return "audio/mpeg"
    elif filename.endswith(".m4a"): return "audio/mp4"
    return "application/octet-stream"

# ✅ Filename matches what you're reading
with open("spanish.wav", "rb") as f:
    audio_bytes = f.read()

res = transcribe_audio(audio_bytes, "spanish.wav")
print(res["transcript"])


import requests

def transcribe_spanglish(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    url = (
        "https://eastus2.api.cognitive.microsoft.com"
        "/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
    )
    headers = {"Ocp-Apim-Subscription-Key": "5s3pFV0dpevEwzemTJFSFdaGtMI4uUtaANYUkVd"}

    # Pass both locales — Azure detects per phrase
    definition = '{"locales":["en-US","es-ES"],"profanityFilterMode":"None"}'

    files = {
        "audio": (filename, audio_bytes, _mime_type(filename)),
        "definition": (None, definition, "application/json"),
    }

    resp = requests.post(url, headers=headers, files=files, timeout=60)
    data = resp.json()

    phrases = data.get("phrases", [])

    transcript_parts = []
    for p in phrases:
        text     = p.get("text", "").strip()
        locale   = p.get("locale", "unknown")
        offset   = p.get("offsetMilliseconds", 0)
        duration = p.get("durationMilliseconds", 0)

        print(f"[{locale}] [{offset}ms -> {offset+duration}ms] {text}")
        transcript_parts.append(text)

    full_transcript = " ".join(transcript_parts).strip()

    return {
        "success": True,
        "transcript": full_transcript,
        "phrases": phrases,  # each has locale, text, offset
        "raw": data
    }

def _mime_type(filename):
    if filename.endswith(".wav"):   return "audio/wav"
    elif filename.endswith(".mp3"): return "audio/mpeg"
    elif filename.endswith(".m4a"): return "audio/mp4"
    return "application/octet-stream"

with open("spanglish.wav", "rb") as f:
    audio_bytes = f.read()

res = transcribe_spanglish(audio_bytes, "spanglish.wav")

print("\n--- Full Transcript ---")
print(res["transcript"])
#azuresdk

import azure.cognitiveservices.speech as speechsdk

# Use your credentials
speech_key = "5s3pFV0dpevEwzemTJFxxxxxxxxxxxxx"
service_region = "eastus2"

def main():
    # 1. Setup config
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    
    # 2. Tell Azure to use the new MAI model
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestTranscriptionEnhancedMode, "true")
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_ProxyHostName, "mai-transcribe-1")
    
    # 3. Connect to Microphone
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Listening... Speak into your microphone.")

    # 4. Handle results
    def recognized_cb(evt):
        print(f"RECOGNIZED: {evt.result.text}")

    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.start_continuous_recognition()

    # Keep the program running
    import time
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        speech_recognizer.stop_continuous_recognition()

if __name__ == "__main__":
    main()


[en-US] [960ms -> 8160ms] My sister-in-law, Kiera Gaga, you know, she's a linguist, and so she wants to sort of track, you know, Spanglish.
[en-US] [9120ms -> 11440ms] So she asked me if I would wear it at work.
[en-US] [13560ms -> 18160ms] So if you see me with this thing, I look kind of goofy.
[en-US] [19440ms -> 25280ms] Yeah, and it records it on Comolo Chips de la Camarita, you know, the tiny little one, and then it, you know...
[en-US] [26800ms -> 30480ms] She has some sort of program because she's doing, you know, she's got like a grant or something.
[en-US] [31280ms -> 45920ms] Yeah, and then they transcribe it and then I guess they figure out, you know, whatever she does, something to do with transcribing in Spanglish and, you know, and everything.
[en-US] [46880ms -> 48800ms] Right, right.
[en-US] [48840ms -> 52240ms] But I don't know, you know, personal or something.
[en-US] [56080ms -> 56640ms] Is it me?
[en-US] [80420ms -> 80980ms] Hello?
[en-US] [80980ms -> 81420ms] Is it me?
[en-US] [81900ms -> 82380ms] Hello?
[en-US] [82460ms -> 83540ms] Is it a four-wheel drive?
[en-US] [86060ms -> 86620ms] Nieve.
[en-US] [87220ms -> 88460ms] Mi hermana viva in Georgia.
[en-US] [88460ms -> 92660ms] No, you don't have a nieve.
[en-US] [94780ms -> 95980ms] I'm a montana.
[en-US] [96660ms -> 102460ms] Yeah, muerta que viva.
[en-US] [110720ms -> 112400ms] It's okay.
[en-US] [112960ms -> 114560ms] No, no, no.
[en-US] [115600ms -> 120720ms] It's a horse country.
[en-US] [120720ms -> 121680ms] It's a horse country.
[en-US] [123440ms -> 125960ms] It's really pretty actually when you, I think it's before Gainesville.
[en-US] [125960ms -> 127920ms] I think it's before Gainesville.
[en-US] [128160ms -> 131360ms] When you, when you drive on 75, I think it's 75.
[en-US] [132000ms -> 132840ms] It's real pretty.
[en-US] [132840ms -> 136880ms] I mean, you can see the big, the pieces of property.
[en-US] [137840ms -> 141360ms] In Virginia, where they have the horse, it's horse country.
[en-US] [141360ms -> 145440ms] You see the big, because has a little bit of hills.
[en-US] [145760ms -> 146720ms] Not like a lot of hills.
[en-US] [149920ms -> 150800ms] 75, right.
[en-US] [150800ms -> 154800ms] No, no, no.
[en-US] [156240ms -> 158560ms] And it's like horse country.
[en-US] [158560ms -> 163840ms] So you see the big prop properties with, you know, the trees and, you know, you see the horses.
[en-US] [163840ms -> 168880ms] It's very pretty, actually, amazing.
[en-US] [170240ms -> 170760ms] Perfect.
[en-US] [171040ms -> 171920ms] Good job, dear.
[en-US] [177080ms -> 182920ms] All right, it's so cute.
[en-US] [183000ms -> 183760ms] It's very cute.
[en-US] [189760ms -> 208880ms] okay very cute
[en-US] [213680ms -> 227680ms] But, you know, you know, we go to the north side so it doesn't get any sun.
[en-US] [258080ms -> 258090ms] .
[en-US] [258080ms -> 258090ms] .
[en-US] [258080ms -> 258090ms] .
[en-US] [258080ms -> 276710ms] Bueno, too bad.
[en-US] [285840ms -> 286040ms] No.
[en-US] [286080ms -> 289680ms] Right, right, right, right, right, right, right, right, right.
[en-US] [291280ms -> 293000ms] Okay, do it.
[en-US] [293040ms -> 300000ms] I'm Michelle.
[en-US] [303520ms -> 303850ms] Boy, .
[en-US] [303840ms -> 306400ms] This is a sentiuno.
[en-US] [306400ms -> 306410ms] .
[en-US] [313200ms -> 313610ms] Boy, .
[en-US] [313600ms -> 313610ms] .
[en-US] [313600ms -> 313610ms] .
[en-US] [332200ms -> 334520ms] If we pay you, I mean, you're a friend.
[en-US] [334760ms -> 337120ms] You may say, like, you know, don't.
[en-US] [337200ms -> 338680ms] Commentary, you guys see you.
[en-US] [340240ms -> 341160ms] This is.
[en-US] [341920ms -> 344160ms] It is.
[en-US] [344160ms -> 347520ms] Two degrees because it used to be 61.
[en-US] [348000ms -> 348480ms] Yeah.
[en-US] [348480ms -> 350880ms] 74.
[en-US] [351120ms -> 351680ms] Alison.
[en-US] [354720ms -> 356160ms] Alison.
[en-US] [362800ms -> 362810ms] .
[en-US] [362800ms -> 362810ms] .
[en-US] [362800ms -> 362810ms] .
[en-US] [362800ms -> 362810ms] .
[en-US] [362800ms -> 362810ms] .
[en-US] [362800ms -> 374960ms] but again on the north side, we don't ever get direct sun.
[en-US] [375040ms -> 389680ms] We get no direct sun, we get.
[en-US] [389680ms -> 389690ms] .
[en-US] [389680ms -> 389690ms] .
[en-US] [389680ms -> 389690ms] .
[en-US] [389680ms -> 389690ms] .
[en-US] [389680ms -> 389690ms] .
[en-US] [389680ms -> 389690ms] .
[en-US] [389680ms -> 389690ms] .
[en-US] [389680ms -> 389690ms] .
[en-US] [396080ms -> 428740ms] yeah warm up the he doesn't miss it in other words no that's what they said about five years super because I had no milk thermostat you know you hit it he said what it is 36
[en-US] [435600ms -> 437840ms
