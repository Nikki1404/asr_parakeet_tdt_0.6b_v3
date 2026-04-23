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
PS C:\Users\re_nikitav> curl.exe -k -v --location "https://eastus2.stt.speech.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15" --header "Ocp-Apim-Subscription-Key: 5s3pFV0dpevEwzemTJFSFdaGxxxxxx3w3AAAAACOGzVQQ" --form "audio=@C:/Users/re_nikitav/Downloads/a.wav" --form 'definition={"locales":["en-US"],"enhancedMode":{"enabled":true,"model":"mai-transcribe-1"}}'
* Host eastus2.stt.speech.microsoft.com:443 was resolved.
* IPv6: (none)
* IPv4: 9.234.135.167
*   Trying 9.234.135.167:443...
* schannel: disabled automatic use of client certificate
* ALPN: curl offers http/1.1
* ALPN: server did not agree on a protocol. Uses default.
* Established connection to eastus2.stt.speech.microsoft.com (9.234.135.167 port 443) from 192.168.29.110 port 53466
* using HTTP/1.x
> POST /speechtotext/transcriptions:transcribe?api-version=2025-10-15 HTTP/1.1
> Host: eastus2.stt.speech.microsoft.com
> User-Agent: curl/8.18.0
> Accept: */*
> Ocp-Apim-Subscription-Key: 5s3pFV0dpevxxxxxxxAACOGzVQQ
> Content-Length: 3783739
> Content-Type: multipart/form-data; boundary=------------------------lzJ8NF5rH3r4pY3O4YaQdY
> Expect: 100-continue
>
* schannel: remote party requests renegotiation
* schannel: renegotiating SSL/TLS connection
* schannel: SSL/TLS connection renegotiated
* schannel: remote party requests renegotiation
* schannel: renegotiating SSL/TLS connection
* schannel: SSL/TLS connection renegotiated
< HTTP/1.1 100 Continue
<
* upload completely sent off: 3783739 bytes
< HTTP/1.1 404 Not Found
< Content-Length: 0
< Date: Thu, 23 Apr 2026 12:13:32 GMT
< Server: Kestrel
< Strict-Transport-Security: max-age=31536000; includeSubDomains
<
* Connection #0 to host eastus2.stt.speech.microsoft.com:443 left intact
PS C:\Users\re_nikitav>
