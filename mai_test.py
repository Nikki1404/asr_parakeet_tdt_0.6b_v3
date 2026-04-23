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
