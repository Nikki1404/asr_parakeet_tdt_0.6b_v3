curl --location 'https://<YourServiceRegion>.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15' \
--header 'Content-Type: multipart/form-data' \
--header 'Ocp-Apim-Subscription-Key: <YourSpeechResourceKey>' \
--form 'audio=@"YourAudioFile.wav"' \
--form 'definition={
  "locales": ["en"],
  "enhancedMode": {
    "enabled": true,
    "model":"mai-transcribe-1"
  }
}'

curl --location 'https://eastus.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15' --header 'Ocp-Apim-Subscription-Key: YOUR_KEY_HERE' --form 'audio=@"C:\path\to\yourfile.wav"' --form 'definition={"locales":["en"],"enhancedMode":{"enabled":true,"model":"mai-transcribe-1"}}'
curl --location "https://eastus2.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15" --header "Ocp-Apim-Subscription-Key: 5s3pFV0dpevEwzemTJFSxxxxxxxxxxACHYHv6XJ3w3AAAAACOGzVQQ" --form "audio=@C:/Users/YourName/Desktop/audio/maria1.wav" --form "definition={\"locales\":[\"en-US\",\"es-MX\"],\"enhancedMode\":{\"enabled\":true,\"model\":\"mai-transcribe-1\"}}"
curl.exe -k --location "https://eastus2.stt.speech.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15" --header "Ocp-Apim-Subscription-Key: YOUR_REAL_KEY" --form "audio=@C:/Users/re_nikitav/Downloads/a.wav" --form 'definition={"locales":["en-US"],"enhancedMode":{"enabled":true,"model":"mai-transcribe-1"}}'
curl.exe -k -v --location "https://ankit-m9l63a79-eastus2.cognitiveservices.azure.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15" --header "Ocp-Apim-Subscription-Key: 5s3pFV0dpevEwzemTJFSFdaGtMI4uUtaANYUkVdfH2o4RdY89CbtJQQJ99BDA" --form "audio=@C:/Users/re_nikitav/Downloads/a.wav" --form 'definition={"locales":["en-US"],"enhancedMode":{"enabled":true,"model":"mai-transcribe-1"}}'


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
