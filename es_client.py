(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# python transcribe_parakeet_es_bathc.py   --base-url http://localhost:9000   --input-folder /home/re_nikitav/audio_maria   --output-folder /home/re_nikitav/parakeet_es_results

[1/15]

STARTING -> maria1.mp3
Traceback (most recent call last):
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_parakeet_es_bathc.py", line 421, in <module>
    asyncio.run(
  File "/usr/lib/python3.11/asyncio/runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/asyncio/base_events.py", line 653, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_parakeet_es_bathc.py", line 385, in run_batch
    await transcribe_file(
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_parakeet_es_bathc.py", line 147, in transcribe_file
    ws_url = create_transcription_session(base_url)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/re_nikitav/parakeet-asr-multilingual/transcribe_parakeet_es_bathc.py", line 97, in create_transcription_session
    raise ValueError(
ValueError: Could not find websocket url in response: {'modalities': ['text'], 'input_audio_format': 'pcm16', 'input_audio_transcription': {'language': 'es-US', 'model': 'parakeet-0.6b-unified-ml-cs-es-US-asr-streaming-silero-vad-sortformer', 'prompt': None}, 'input_audio_params': {'sample_rate_hz': 16000, 'num_channels': 1}, 'recognition_config': {'max_alternatives': 1, 'enable_automatic_punctuation': False, 'enable_word_time_offsets': False, 'enable_profanity_filter': False, 'enable_verbatim_transcripts': False, 'custom_configuration': ''}, 'speaker_diarization': {'enable_speaker_diarization': False, 'max_speaker_count': 8}, 'word_boosting': {'enable_word_boosting': False, 'word_boosting_list': []}, 'endpointing_config': {'start_history': 0, 'start_threshold': 0.0, 'stop_history': 0, 'stop_threshold': 0.0, 'stop_history_eou': 0, 'stop_threshold_eou': 0.0}, 'id': 'sess_216c5d6e-bab3-4b7b-8545-a0af9b76966d', 'object': 'realtime.transcription_session', 'client_secret': None}
(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# 


use this 
(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# curl -X POST http://localhost:9000/v1/audio/transcriptions
{"error":{"message":"file: Field required","type":"BadRequestError","code":400}}(env) root@cx-asr-test:/home/re_nikitav/parakeet-asr-multilingual# ^C
