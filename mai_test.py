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
