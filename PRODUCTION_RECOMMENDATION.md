# Azure STT Production Recommendation

Generated: 2026-04-21T17:26:09.271856

## Key Rule

- Production transcript must preserve exact spoken words.
- This evaluation does not paraphrase or rewrite transcript meaning.
- Numeric interpretation is analysis-only and context-aware.

## Recommended Production Stages

- Recommended stage combination: Stage 1, Stage 2, Stage 3, Stage 5, Stage 6

## Best Single Stage

- Stage 1 — asr_config
- Quality score: 64.3
- Helped in: punctuation/readability, VAD / truncation handling

## Why These Stages Help

Recommended production path is usually stages 1, 2, 3, 5, 6 because they improve transcription quality while keeping exact spoken meaning preserved.

## Unsafe Stages / Caution

- No unsafe stages detected based on meaning-preservation checks

## Stage-by-Stage Progress

### Stage 0 — baseline
- Phase: Baseline
- Task: Original working script
- Quality score: 64.3
- Meaning preserved safe: True
- Meaning risk score: 10
- Key help areas: punctuation/readability, VAD / truncation handling
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - baseline_features: before=N/A | after=No improvement features enabled | why=Reference-only baseline

### Stage 1 — asr_config
- Phase: Setup
- Task: ASR Config Finalization
- Quality score: 64.3
- Meaning preserved safe: True
- Meaning risk score: 10
- Key help areas: punctuation/readability, VAD / truncation handling
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - candidate_languages: before=Implicit / untracked | after=['en-US', 'es-US'] | why=Restrict language detection space

### Stage 2 — vad_tuning
- Phase: Audio
- Task: VAD Evaluation & Tuning
- Quality score: 64.3
- Meaning preserved safe: True
- Meaning risk score: 10
- Key help areas: punctuation/readability, VAD / truncation handling
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - SpeechServiceConnection_EndSilenceTimeoutMs: before=800 | after=1500 | why=Reduce premature cut-off
  - SpeechServiceConnection_InitialSilenceTimeoutMs: before=5000 | after=5000 | why=Keep startup silence tolerance unchanged
  - Speech_SegmentationSilenceTimeoutMs: before=800 | after=800 | why=Keep segmentation unchanged except end silence

### Stage 3 — phrase_boost
- Phase: Accuracy
- Task: Word / Phrase Boosting
- Quality score: 64.3
- Meaning preserved safe: True
- Meaning risk score: 10
- Key help areas: punctuation/readability, VAD / truncation handling
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - PhraseListGrammar: before=Disabled | after=Enabled | why=Bias recognition toward expected digits and domain terms
  - phrase_count: before=0 | after=38 | why=Provide domain lexicon hints

### Stage 4 — vocab_tuning
- Phase: Accuracy
- Task: Transcript-Based Vocabulary Tuning
- Quality score: 63.02
- Meaning preserved safe: True
- Meaning risk score: 10
- Key help areas: punctuation/readability, VAD / truncation handling
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - auto_mined_terms: before=Disabled | after=['right', 'because', "that's", 'porque', 'tiene', 'horse', "don't", "didn't", 'little', 'something', 'country', 'cuando', "she's", 'thing', 'pretty', 'actually', 'think', 'madre', 'sensor', 'great', 'spanglish', 'guess', 'hello', 'drive', 'nieve', 'llama', 'estado', 'before', 'hills', 'tambi', 'tengo', 'sabes', 'calefacci', "doesn't", 'bueno', 'fresco', 'maybe', 'mucho', 'allison', 'menos', 'direct', 'years', 'carro', 'takes', 'throws', 'fashioned', 'engines', 'first', 'weird', 'morning'] | why=Use recurring transcript vocabulary for biasing

### Stage 5 — numeric_handling
- Phase: Logic
- Task: Numeric Handling Validation
- Quality score: 64.3
- Meaning preserved safe: True
- Meaning risk score: 10
- Key help areas: punctuation/readability, VAD / truncation handling
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - Detailed_JSON_analysis: before=Not parsed | after=Parsed ITN/Lexical/Display fields | why=Analyze number behavior without rewriting transcript

### Stage 6 — dictation_mode
- Phase: Accuracy
- Task: Dictation Mode
- Quality score: 44.6
- Meaning preserved safe: True
- Meaning risk score: 30
- Key help areas: punctuation/readability
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - dictation_mode: before=Disabled | after=Enabled | why=Allow better punctuation handling

### Stage 7 — emotion_tone
- Phase: Quality
- Task: Emotion / Tone Evaluation
- Quality score: 44.6
- Meaning preserved safe: True
- Meaning risk score: 30
- Key help areas: punctuation/readability
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - tone_proxy_analysis: before=Disabled | after=Enabled | why=Quality analysis only

### Stage 8 — latency_testing
- Phase: Testing
- Task: Latency & Timeout Testing
- Quality score: 44.6
- Meaning preserved safe: True
- Meaning risk score: 30
- Key help areas: punctuation/readability
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - latency_multi_run: before=Single run | after=Three runs | why=Measure latency consistency

### Stage 9 — realtime_socket
- Phase: Integration
- Task: Real-Time Socket Integration
- Quality score: 46.1
- Meaning preserved safe: True
- Meaning risk score: 30
- Key help areas: punctuation/readability
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - ingestion_method: before=AudioConfig(filename=...) | after=PushAudioInputStream | why=Simulate streaming/real-time ingestion
  - stream_chunk_ms: before=N/A | after=40 | why=Simulate low-latency streaming chunks

### Stage 10 — concurrency
- Phase: Testing
- Task: Load & Concurrency Testing
- Quality score: 44.6
- Meaning preserved safe: True
- Meaning risk score: 30
- Key help areas: punctuation/readability
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - concurrency_levels: before=Single stream only | after=[1, 3, 5, 10] | why=Load testing

### Stage 11 — logging_alerts
- Phase: Monitoring
- Task: Logging & Alerts Setup
- Quality score: 44.6
- Meaning preserved safe: True
- Meaning risk score: 30
- Key help areas: punctuation/readability
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - structured_alerting: before=Disabled | after=Enabled | why=Operational monitoring

### Stage 12 — fallback
- Phase: Go-Live
- Task: Fallback Validation
- Quality score: 44.6
- Meaning preserved safe: True
- Meaning risk score: 30
- Key help areas: punctuation/readability
- Parameters used/changed:
  - audio_conversion: before=Original input format | after=WAV PCM 16kHz mono 16-bit | why=Standardize input for stable Azure STT behavior
  - meaning_preservation_policy: before=Not explicitly documented | after=Do not paraphrase or rewrite transcript text | why=Preserve exact spoken content
  - fallback_chain: before=Not simulated | after=['recognition', 're-prompt', 'language_retry', 'dtmf', 'agent_escalation'] | why=Production resiliency testing
