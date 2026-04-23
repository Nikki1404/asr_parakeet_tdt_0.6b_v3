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
PS C:\Users\re_nikitav> curl.exe --location "https://eastus2.stt.speech.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15" --header "Ocp-Apim-Subscription-Key: 5s3pFV0dpevEwzemTJFSFdaGtxxxxxxx" --form "audio=@C:/Users/re_nikitav/Downloads/a.wav" --form "definition={\"locales\":[\"en-US\"],\"enhancedMode\":{\"enabled\":true,\"model\":\"mai-transcribe-1\"}}"
curl: (35) schannel: next InitializeSecurityContext failed: CRYPT_E_NO_REVOCATION_CHECK (0x80092012) - The revocation function was unable to check revocation for the certificate.
curl: (3) bad range specification in URL position 11:
locales\:[\en-US\],\enhancedMode\:{\enabled\:true,\model\:\mai-transcribe-1\}}
