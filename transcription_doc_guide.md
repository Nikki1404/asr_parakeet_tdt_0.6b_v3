# Azure STT Transcription Quality Lab — Documentation Guide

> **Audio:** `audio/maria1.mp3`  |  **Generated:** 2026-04-22 19:59:48  |  **Region:** `eastus`
> **Languages Tested:** ['en-US', 'es-US']  |  **Latency flag:** >20% regression vs baseline  |  **Re-prompt threshold:** confidence < 0.6

---

## Quick Reference: Stage → Requirements Table Mapping

| Script Stage | Phase | Task | Priority |
|---|---|---|---|
| Stage 0 | Reference | Baseline (original script) | — |
| Stage 1 / 1b | Setup | **ASR Config Finalization** — lock lang/locale, audio format, disable auto-detect | High |
| Stage 2 | Setup | **Concurrency & Quota Validation** — validate limits, rates, quotas | High |
| Stage 3 | Integration | **Real-Time Socket Integration** — WebSocket/streaming ingestion | High |
| Stage 4a / 4b / 4c | Audio | **VAD Evaluation & Tuning** — silence thresholds, endpointing | — |
| Stage 5 | Accuracy | **Word / Phrase Boosting** — boost digits, identifiers, domain terms | High |
| Stage 6 | Accuracy | **Transcript-Based Vocabulary Tuning** — sample transcript vocab | High |
| Stage 7a / 7b / 7c | Logic | **Numeric Handling Validation** — digit-by-digit vs grouped | High |
| Stage 8 | Quality | **Emotion / Tone Evaluation** — neutral vs stressed speech | High |
| Stage 9 | Testing | **Latency & Timeout Testing** — response times, SLA | High |
| Stage 10 | Testing | **Load & Concurrency Testing** — peak concurrent streams | High |
| Stage 11 | Monitoring | **Logging & Alerts Setup** — error, latency, socket-drop | High |
| Stage 12 | Go-Live | **Fallback Validation** — re-prompt, DTMF, alternate flow | High |
| Stage C1 | Production | Combined Best ← **USE THIS IN PRODUCTION** | — |
| Stage C2 | Production | Combined All | — |

---

## 1. Setup

```bash
pip install azure-cognitiveservices-speech

# FFmpeg (for audio conversion)
winget install ffmpeg          # Windows
brew install ffmpeg            # macOS
sudo apt install ffmpeg        # Ubuntu/Linux

# Edit config at top of script
SPEECH_KEY    = "YOUR_AZURE_SPEECH_KEY"
SPEECH_REGION = "eastus"
INPUT_AUDIO_FILE = "audio/your_file.mp3"
```

Customize `DOMAIN_PHRASES` with terms specific to your domain (product names, IVR menus, identifiers).

---

## 2. How to Run

```bash
# All 12 stages (recommended for full documentation)
python transcription_stages_lab.py

# Single stage
python transcription_stages_lab.py --stage stage_5

# Custom audio file
python transcription_stages_lab.py --file audio/call2.mp3

# Override max concurrency for Stages 2 & 10
python transcription_stages_lab.py --concurrency 10

# Skip audio conversion (if already 16kHz PCM WAV)
python transcription_stages_lab.py --skip-wav
```

**Output files generated:**

| File | Contents |
|------|----------|
| `transcription_report.md` | Full per-stage documentation, comparison table, transcripts |
| `transcription_doc_guide.md` | This documentation guide with actual results filled in |
| `results.json` | Machine-readable metrics for all stages |
| `transcription_audit.log` | Structured JSON log records (one per line) |
| Console | Live partials + finals + stage summaries |

---

## 3. Stage-by-Stage: What It Does, Parameters, Actual Results

### Stage 0 — Baseline

**Requirements phase:** Reference  

**What it does:** Your original script, unchanged. Auto-detects language between en-US and es-ES. All Azure default settings. This is the reference every other stage compares against.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `locked_language` | `None (auto-detect en-US + es-ES)` |
| `recognition_mode` | `conversation (default)` |
| `profanity` | `masked (default)` |
| `end_silence_ms` | `800` |
| `initial_silence_ms` | `5000` |
| `seg_silence_ms` | `600` |
| `output_format` | `detailed` |
| `phrase_list` | `none` |
| `numeric_pp` | `False` |
| `audio_format` | `16kHz PCM WAV` |

**Actual Results from This Run:**

- Segments: **93**
- Words: **1254**
- Digit tokens: **19**
- Short words: **650**
- TTFB: **2384.7 ms**
- TTFT Partial: **2384.7 ms**
- TTFT Final: **2875.8 ms**
- Total time: **443.22 s**
- Confidence avg/min: **0.767 / 0.069**
- Low-conf segments: **16**

**What to observe:** Segment count, word count, digit tokens, short words, TTFT.

---

### Stage 1 — ASR Config Finalization

**Requirements phase:** Setup  

**What it does:** Locks language to whatever Stage 0 detected. Sets profanity to `raw` so no words are masked. Eliminates auto-detect overhead per utterance.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `locked_language` | `None (auto-detect preserved)` |
| `candidate_languages` | `['en-US', 'es-US']` |
| `recognition_mode` | `conversation` |
| `profanity` | `raw ← CHANGED` |
| `end_silence_ms` | `800` |
| `initial_silence_ms` | `5000` |
| `seg_silence_ms` | `600` |
| `output_format` | `detailed` |
| `phrase_list` | `none` |
| `numeric_pp` | `False` |
| `audio_format` | `16kHz PCM WAV` |

**Stage 1b** runs the same config on 8kHz (telephony) audio to compare formats.

**Actual Results from This Run:**

- Segments: **93**
- Words: **1254**
- Digit tokens: **19**
- Short words: **650**
- TTFB: **2609.0 ms** (▲9% vs baseline)
- TTFT Partial: **2609.0 ms** (▲9% vs baseline)
- TTFT Final: **3087.1 ms** (▲7% vs baseline)
- Total time: **443.03 s**
- Confidence avg/min: **0.767 / 0.069**
- Low-conf segments: **16**
- vs Baseline: **100.0% similar** (0.0% different)

**What to observe:** Check TTFT vs Stage 0. Ensure English + Spanish both remain accurate.

---

### Stage 1b — ASR Config: Telephony 8kHz Format

**Requirements phase:** Setup  

**What it does:** Same locked-language config as Stage 1, but audio is downsampled to 8kHz. Tests whether your source audio is better matched to telephony or broadband format.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `locked_language` | `None (auto-detect preserved)` |
| `candidate_languages` | `['en-US', 'es-US']` |
| `audio_format` | `8kHz PCM WAV` |

If Stage 1b word count ≥ Stage 1 → source is phone call audio → use 8kHz in production.

**Actual Results from This Run:**

- Segments: **97**
- Words: **1241**
- Digit tokens: **19**
- Short words: **643**
- TTFB: **2339.8 ms** (▼2% vs baseline)
- TTFT Partial: **2339.8 ms** (▼2% vs baseline)
- TTFT Final: **2792.5 ms** (▼3% vs baseline)
- Total time: **442.66 s**
- Confidence avg/min: **0.734 / 0.051**
- Low-conf segments: **20**
- vs Baseline: **33.4% similar** (66.6% different)

**What to observe:** Compare Stage 1 vs Stage 1b transcripts.

---

### Stage 2 — Concurrency & Quota Validation

**Requirements phase:** Setup  

**What it does:** Runs [1, 3, 5] simultaneous recognition sessions. Tests whether your Azure subscription tier handles concurrent streams without throttling (HTTP 429).

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `base_config` | `Stage 1 (bilingual auto-detect)` |

This stage validates infrastructure limits, not transcription quality.

**Actual Results from This Run:**

- TTFT Partial: **5965.7 ms** (▲150% vs baseline ⚠️ regression)
- TTFT Final: **5965.7 ms** (▲107% vs baseline ⚠️ regression)
- Total time: **1330.58 s**

**Concurrency Results:**

| Sessions | Success% | Throttled | TTFT Avg | P50 | P95 | Max |
|----------|----------|-----------|----------|-----|-----|-----|
| 1 | 100.0% | 0 | 2819.2 ms | 2819.2 ms | 2819.2 ms | 2819.2 ms |
| 3 | 100.0% | 0 | 3046.2 ms | 2991.7 ms | 3167.4 ms | 3167.4 ms |
| 5 | 100.0% | 0 | 5965.7 ms | 5942.2 ms | 6182.2 ms | 6182.2 ms |

**What to observe:** Concurrency failures.

---

### Stage 3 — Real-Time Socket Integration

**Requirements phase:** Integration  

**What it does:** Uses `PushAudioInputStream` to push audio in 100ms real-time chunks, simulating a live WebSocket or microphone feed. Validates streaming latency vs file-based.

Watch for chunk-boundary artifacts — words split across 100ms boundaries.

**Actual Results from This Run:**

- Segments: **94**
- Words: **1279**
- Digit tokens: **19**
- Short words: **665**
- TTFB: **4845.4 ms** (▲103% vs baseline ⚠️ regression)
- TTFT Partial: **4845.4 ms** (▲103% vs baseline ⚠️ regression)
- TTFT Final: **5438.9 ms** (▲89% vs baseline ⚠️ regression)
- Total time: **910.87 s**
- vs Baseline: **67.4% similar** (32.6% different)

**What to observe:** TTFT partial latency.

---

### Stage 4a — VAD: Default (800ms)

**Requirements phase:** Audio  

**What it does:** Same as Stage 1. Isolates VAD behaviour at the default 800ms end-silence. Reference point for VAD comparison.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `end_silence_ms` | `800  (default)` |
| `initial_silence_ms` | `5000 (default)` |
| `seg_silence_ms` | `600  (default)` |

**Actual Results from This Run:**

- Segments: **93**
- Words: **1254**
- Digit tokens: **19**
- Short words: **650**
- TTFB: **2582.9 ms** (▲8% vs baseline)
- TTFT Partial: **2582.9 ms** (▲8% vs baseline)
- TTFT Final: **3028.3 ms** (▲5% vs baseline)
- Total time: **536.69 s**
- Confidence avg/min: **0.767 / 0.069**
- Low-conf segments: **16**
- vs Baseline: **100.0% similar** (0.0% different)

**What to observe:** Segment count. Are any sentences truncated mid-speech?

---

### Stage 4b — VAD: Conservative (1200ms)

**Requirements phase:** Audio  

**What it does:** Increases end-silence to 1200ms (+50%). Best for speakers who pause mid-sentence or read numbers slowly. Reduces false cut-offs.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `end_silence_ms` | `1200  ← INCREASED (was: 800)` |
| `initial_silence_ms` | `8000  ← INCREASED (was: 5000)` |
| `seg_silence_ms` | `1000  ← INCREASED (was: 600)` |

**Decision rule:** If 4b segment count < 4a AND word count ≥ 4a → use 4b.

**Actual Results from This Run:**

- Segments: **81**
- Words: **1254**
- Digit tokens: **18**
- Short words: **657**
- TTFB: **2565.4 ms** (▲8% vs baseline)
- TTFT Partial: **2565.4 ms** (▲8% vs baseline)
- TTFT Final: **3735.7 ms** (▲30% vs baseline ⚠️ regression)
- Total time: **442.81 s**
- Confidence avg/min: **0.737 / 0.053**
- Low-conf segments: **19**
- vs Baseline: **39.0% similar** (61.0% different)

**What to observe:** Segment count vs 4a (should decrease). Word count vs 4a (should increase or equal).

---

### Stage 4c — VAD: Aggressive (2000ms)

**Requirements phase:** Audio  

**What it does:** End-silence 2000ms. Maximum pause tolerance for very slow/deliberate speakers.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `end_silence_ms` | `2000  ← INCREASED (was: 1200)` |
| `initial_silence_ms` | `15000 ← INCREASED (was: 8000)` |
| `seg_silence_ms` | `1500  ← INCREASED (was: 1000)` |

**Risk:** If speaker pauses > 2s between sentences, 4c may merge them. If segment count drops drastically → stick with 4b.

**Actual Results from This Run:**

- Segments: **65**
- Words: **1286**
- Digit tokens: **19**
- Short words: **675**
- TTFB: **2403.9 ms** (▲1% vs baseline)
- TTFT Partial: **2403.9 ms** (▲1% vs baseline)
- TTFT Final: **5161.5 ms** (▲79% vs baseline ⚠️ regression)
- Total time: **443.05 s**
- Confidence avg/min: **0.749 / 0.053**
- Low-conf segments: **13**
- vs Baseline: **33.7% similar** (66.3% different)

**What to observe:** If segment count drops drastically → utterances are merging (avoid 4c).

---

### Stage 5 — Word / Phrase Boosting

**Requirements phase:** Accuracy  

**What it does:** Adds `PhraseListGrammar` with 28 domain-specific entries. Soft vocabulary injection — increases recognizer's prior for these phrases when acoustic evidence is ambiguous.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `locked_language` | `en-US` |
| `recognition_mode` | `conversation` |
| `profanity` | `raw` |
| `end_silence_ms` | `1200` |
| `initial_silence_ms` | `8000` |
| `seg_silence_ms` | `1000` |
| `phrase_list` | `28 entries  ← ADDED` |
| `phrase_categories` | `digit-sequences, identifiers, short-words, IVR-menus` |
| `numeric_pp` | `False` |

Phrase categories: digit sequences, identifiers (account/PIN/ZIP), short words (ID/OK), IVR patterns (press one/two).

**Actual Results from This Run:**

- Segments: **80**
- Words: **1264**
- Digit tokens: **18**
- Short words: **660**
- TTFB: **2360.8 ms** (▼1% vs baseline)
- TTFT Partial: **2360.8 ms** (▼1% vs baseline)
- TTFT Final: **3516.9 ms** (▲22% vs baseline ⚠️ regression)
- Total time: **442.57 s**
- Confidence avg/min: **0.743 / 0.071**
- Low-conf segments: **19**
- vs Baseline: **42.3% similar** (57.7% different)

**What to observe:** digit_token_count vs Stage 0. short_word_count vs Stage 0.

---

### Stage 6 — Transcript-Based Vocabulary Tuning

**Requirements phase:** Accuracy  

**What it does:** Extracts words that appeared ≥2 times in the baseline transcript and adds them to the phrase list on top of Stage 5. Self-bootstraps vocabulary from your domain audio.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `locked_language` | `en-US` |
| `recognition_mode` | `conversation` |
| `profanity` | `raw` |
| `end_silence_ms` | `1200` |
| `initial_silence_ms` | `8000` |
| `seg_silence_ms` | `1000` |
| `phrase_list` | `Stage5(28) + baseline(99)  ← ADDED` |
| `numeric_pp` | `False` |

Best used when domain has unusual proper nouns or technical terms.

**Actual Results from This Run:**

- Segments: **85**
- Words: **1310**
- Digit tokens: **19**
- Short words: **667**
- TTFB: **2427.0 ms** (▲2% vs baseline)
- TTFT Partial: **2427.0 ms** (▲2% vs baseline)
- TTFT Final: **3578.3 ms** (▲24% vs baseline ⚠️ regression)
- Total time: **442.83 s**
- Confidence avg/min: **0.717 / 0.065**
- Low-conf segments: **22**
- vs Baseline: **39.0% similar** (61.0% different)

**What to observe:** Check if any word mis-recognised in Stage 0 is now correct. Compare similarity_pct to Stage 5.

---

### Stage 7a — Numeric: Conversation Mode (Azure native)

**Requirements phase:** Logic  

**What it does:** Measures how Azure natively outputs numbers in conversation mode without any post-processing. Reference for numeric comparison.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `recognition_mode` | `conversation` |
| `numeric_pp` | `False` |
| `phrase_list` | `28 entries` |

**Actual Results from This Run:**

- Segments: **80**
- Words: **1264**
- Digit tokens: **18**
- Short words: **660**
- TTFB: **2731.9 ms** (▲15% vs baseline)
- TTFT Partial: **2731.9 ms** (▲15% vs baseline)
- TTFT Final: **3916.8 ms** (▲36% vs baseline ⚠️ regression)
- Total time: **442.61 s**
- Confidence avg/min: **0.743 / 0.071**
- Low-conf segments: **19**
- vs Baseline: **42.3% similar** (57.7% different)

**What to observe:** digit_token_count. How many numbers appear as words vs digits?

---

### Stage 7b — Numeric: Dictation Mode

**Requirements phase:** Logic  

**What it does:** Switches to Azure dictation mode, which is trained to output spoken numbers as digit tokens natively. No post-processing added yet.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `recognition_mode` | `dictation  ← CHANGED (was: conversation)` |
| `numeric_pp` | `False` |
| `phrase_list` | `28 entries` |

Check: does `"I need to go"` still say `"to"` (not `"2"`)?

**Actual Results from This Run:**

- Segments: **80**
- Words: **1264**
- Digit tokens: **18**
- Short words: **660**
- TTFB: **3516.0 ms** (▲47% vs baseline ⚠️ regression)
- TTFT Partial: **3516.0 ms** (▲47% vs baseline ⚠️ regression)
- TTFT Final: **4657.8 ms** (▲62% vs baseline ⚠️ regression)
- Total time: **442.35 s**
- Confidence avg/min: **0.743 / 0.071**
- Low-conf segments: **19**
- vs Baseline: **23.1% similar** (76.9% different)

**What to observe:** digit_token_count vs 7a. Check: does 'I need to go' still say 'to' (not '2')?

---

### Stage 7c — Numeric: Dictation + Context-Aware Post-Processor

**Requirements phase:** Logic  

**What it does:** Dictation mode + context-aware post-processor that converts remaining word-numbers to digits only in clear numeric context.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `recognition_mode` | `dictation` |
| `numeric_pp` | `True  ← ADDED` |
| `never_convert` | `'to','for','a','an','won','ate' + full list` |
| `spanish_handling` | `pass-through (no conversion)` |
| `phrase_list` | `28 entries` |

**Safety rules:**

| Word | Converted? | Rule |
|------|-----------|------|
| `"to"` | ❌ NEVER | Preposition |
| `"for"` | ❌ NEVER | Preposition |
| `"a"` / `"an"` | ❌ NEVER | Article |
| `"won"` | ❌ NEVER | Past tense |
| `"ate"` | ❌ NEVER | Past tense |
| `"one"` → `"1"` | ✅ numeric context | After `number`, `press`, etc. |
| Spanish words | ❌ NEVER | Pass-through |

**Examples:**
```
"I need to go"                  → "I need to go"        (to: NEVER)
"press one for English"          → "press 1 for English" (after 'press')
"account number one two three"   → "account number 1 2 3"
"there are two people"           → "there are 2 people"  ('people' = quantity)
Spanish: "presione dos"          → "presione dos"        (pass-through)
```

**Actual Results from This Run:**

- Segments: **80**
- Words: **1264**
- Digit tokens: **18**
- Short words: **660**
- TTFB: **2475.8 ms** (▲4% vs baseline)
- TTFT Partial: **2475.8 ms** (▲4% vs baseline)
- TTFT Final: **3630.0 ms** (▲26% vs baseline ⚠️ regression)
- Total time: **442.61 s**
- Confidence avg/min: **0.743 / 0.071**
- Low-conf segments: **19**
- vs Baseline: **23.1% similar** (76.9% different)

**What to observe:** digit_token_count vs 7b (should be ≥). Verify 'to'/'for'/'a' stayed as words.

---

### Stage 8 — Emotion / Tone Evaluation

**Requirements phase:** Quality  

**What it does:** Parses per-segment confidence scores and NBest alternatives from Azure's Detailed output. Low confidence indicates uncertainty from noise, stressed speech, accent variation.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `recognition_mode` | `conversation` |
| `output_format` | `detailed  (parses Confidence + NBest)` |
| `confidence_threshold` | `0.6` |
| `profanity` | `raw` |
| `end_silence_ms` | `1200` |
| `phrase_list` | `28 entries` |
| `numeric_pp` | `False` |

Azure confidence ranges 0.0–1.0. Clean call-centre audio: 0.85–0.97. Values < 0.6 → human review.

**Actual Results from This Run:**

- Segments: **80**
- Words: **1264**
- Digit tokens: **18**
- Short words: **660**
- TTFB: **2723.4 ms** (▲14% vs baseline)
- TTFT Partial: **2723.4 ms** (▲14% vs baseline)
- TTFT Final: **3877.9 ms** (▲35% vs baseline ⚠️ regression)
- Total time: **443.02 s**
- Confidence avg/min: **0.743 / 0.071**
- Low-conf segments: **19**
- vs Baseline: **42.3% similar** (57.7% different)

**What to observe:** confidence_avg and confidence_min. low_conf_segments count (segments below 0.6).

---

### Stage 9 — Latency & Timeout Testing

**Requirements phase:** Testing  

**What it does:** Runs 3 recognition passes and collects P50/P95 TTFT statistics. Flags any run >20% slower than Stage 0 baseline. Also tests with tight timeout (500ms) to check truncation risk.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `latency_regression_pct` | `20` |
| `runs_for_stats` | `3` |
| `tight_end_silence` | `500` |
| `normal_end_silence` | `1200` |
| `phrase_list` | `28 entries` |
| `numeric_pp` | `False` |

Alert fires in `transcription_audit.log` if any run exceeds SLA.

**Actual Results from This Run:**

- TTFB: **2511.2 ms** (▲5% vs baseline)
- TTFT Partial: **2511.2 ms** (▲5% vs baseline)
- TTFT Final: **3814.5 ms** (▲33% vs baseline ⚠️ regression)
- Total time: **442.81 s**

**Latency Statistics (vs Stage 0):** P50=3814.5ms | P95=3886.2ms

| Run | TTFB (ms) | TTFT-P (ms) | TTFT-F (ms) | Total (s) | vs Baseline |
|-----|-----------|------------|------------|-----------|-------------|
| 1 | 2487.5 | 2487.5 | 3659.3 | 442.99 | ✅ +4% |
| 2 | 2685.4 | 2685.4 | 3886.2 | 443.01 | ✅ +13% |
| 3 | 2511.2 | 2511.2 | 3814.5 | 442.81 | ✅ +5% |
| tight-500ms | 4167.5 | 4167.5 | 5273.5 | 442.85 | ⚠️ +75% |

**What to observe:** P50 and P95 TTFT across runs. Does tight timeout (500ms) cause truncation vs normal (1200ms)?

---

### Stage 10 — Load & Concurrency Testing

**Requirements phase:** Testing  

**What it does:** Runs multiple concurrent sessions at increasing levels via `ThreadPoolExecutor`. Tests system stability under peak load. Azure Free tier: 1 concurrent; S0 tier: 20 concurrent.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `concurrency_levels` | `[1, 3, 5]` |
| `latency_regression_pct` | `20` |
| `output_format` | `simple (reduces payload under load)` |
| `phrase_list` | `none (reduces setup time under load)` |

Use `wall_time_sec` to calculate throughput (sessions/second).

**Actual Results from This Run:**

- TTFT Final: **6283.7 ms** (▲119% vs baseline ⚠️ regression)
- Total time: **1331.13 s**

**Concurrency Results:**

| Sessions | Success% | Throttled | TTFT Avg | P50 | P95 | Max |
|----------|----------|-----------|----------|-----|-----|-----|
| 1 | 100.0% | 0 | 3880.5 ms | 3880.5 ms | 3880.5 ms | 3880.5 ms |
| 3 | 100.0% | 0 | 3203.2 ms | 3088.2 ms | 3433.2 ms | 3433.2 ms |
| 5 | 100.0% | 0 | 6071.3 ms | 6006.6 ms | 6283.7 ms | 6283.7 ms |

**What to observe:** At which concurrency level do throttle errors appear? P95 TTFT degradation as concurrency increases.

---

### Stage 11 — Logging & Alerts Setup

**Requirements phase:** Monitoring  

**What it does:** Runs Combined Best config while structured JSON logging is active. Every event (start, segment, complete, error) is logged to `transcription_audit.log`.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `log_file` | `transcription_audit.log` |
| `log_format` | `JSON (one record per line)` |
| `alert_regression_pct` | `20` |
| `alert_empty` | `True` |
| `alert_error` | `True` |
| `base_config` | `Combined Best (Stage C1)` |

**Alert thresholds:**

| Alert | Trigger |
|-------|---------|
| `HIGH_LATENCY` | TTFB or TTFT-P >20% slower than Stage 0 |
| `EMPTY_TRANSCRIPT` | `segment_count == 0` |
| `RECOGNITION_ERROR` | `error_code` in cancellation details |

**Log record format:**
```json
{"ts": "2025-01-15T10:23:45.123Z", "stage": "stage_11",
 "event": "COMPLETE", "segment_count": 7,
 "ttft_final_ms": 412.5, "total_time_sec": 18.3}
```

**Actual Results from This Run:**

- Segments: **80**
- Words: **1264**
- Digit tokens: **18**
- Short words: **660**
- TTFB: **2709.5 ms** (▲14% vs baseline)
- TTFT Partial: **2709.5 ms** (▲14% vs baseline)
- TTFT Final: **3972.3 ms** (▲38% vs baseline ⚠️ regression)
- Total time: **442.66 s**
- Confidence avg/min: **0.743 / 0.071**
- Low-conf segments: **19**
- vs Baseline: **23.1% similar** (76.9% different)

**What to observe:** transcription_audit.log record count. Any alerts triggered?

---

### Stage 12 — Fallback Validation

**Requirements phase:** Go-Live  

**What it does:** Validates fallback handling. Segments with confidence < 0.6 are flagged for re-prompt. Empty transcripts trigger DTMF fallback.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `reprompt_threshold` | `0.6` |
| `dtmf_fallback` | `Triggered when transcript is empty or all-silence` |
| `re-prompt_trigger` | `Any segment with confidence < 0.6` |
| `base_config` | `Combined Best` |

**Fallback logic:**
```
IF transcript == '' or segment_count == 0  → DTMF fallback
IF any segment confidence < 0.6     → flag for re-prompt
IF error_code present                      → log + alert + fallback
```

**Actual Results from This Run:**

- Segments: **80**
- Words: **1264**
- Digit tokens: **18**
- Short words: **660**
- TTFB: **2767.3 ms** (▲16% vs baseline)
- TTFT Partial: **2767.3 ms** (▲16% vs baseline)
- TTFT Final: **4132.0 ms** (▲44% vs baseline ⚠️ regression)
- Total time: **442.88 s**
- Confidence avg/min: **0.743 / 0.071**
- Low-conf segments: **19**
- vs Baseline: **23.1% similar** (76.9% different)

**Fallback Report:**
- Low-confidence segments: **19**
- Re-prompt flagged: **True**
- DTMF fallback triggered: **False**

**What to observe:** low_conf_segments count. fallback_report.reprompt_flagged. fallback_report.dtmf_fallback.

---

### Stage C1 — Combined Best  ✅ PRODUCTION RECOMMENDATION

**Requirements phase:** Production  

**What it does:** Combines all stages that show measurable improvement: locked language, raw profanity, conservative VAD, phrase boosting, dictation mode, numeric PP.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `locked_language` | `None` |
| `candidate_languages` | `['en-US', 'es-US']` |
| `recognition_mode` | `dictation  (Stage7)` |
| `profanity` | `raw  (Stage1)` |
| `end_silence_ms` | `1200  (Stage4b)` |
| `initial_silence_ms` | `8000  (Stage4b)` |
| `seg_silence_ms` | `1000  (Stage4b)` |
| `phrase_list` | `28 entries  (Stage5)` |
| `numeric_pp` | `True  (Stage7c)` |

| Component | From Stage | Config |
|-----------|-----------|--------|
| Locked language | Stage 1 | `speech_recognition_language` |
| Profanity raw | Stage 1 | `ProfanityOption.Raw` |
| Conservative VAD | Stage 4b | `end_silence_ms=1200` |
| Phrase boosting | Stage 5 | `PhraseListGrammar` |
| Dictation mode | Stage 7b | `enable_dictation()` |
| Numeric PP | Stage 7c | `apply_numeric_pp=True` |

**Actual Results from This Run:**

- Segments: **80**
- Words: **1264**
- Digit tokens: **18**
- Short words: **660**
- TTFB: **2649.8 ms** (▲11% vs baseline)
- TTFT Partial: **2649.8 ms** (▲11% vs baseline)
- TTFT Final: **3848.6 ms** (▲34% vs baseline ⚠️ regression)
- Total time: **443.21 s**
- Confidence avg/min: **0.743 / 0.071**
- Low-conf segments: **19**
- vs Baseline: **23.1% similar** (76.9% different)

**What to observe:** similarity_pct vs Stage 0. digit_token_count (expect highest). TTFT (expect ≤ Stage 0). word_count (expect ≥ Stage 0).

---

### Stage C2 — Combined All Stages

**Requirements phase:** Production  

**What it does:** C1 + aggressive VAD (Stage 4c) + extended vocabulary (Stage 6).

**Parameters:**

| Parameter | Value |
|-----------|-------|
| `locked_language` | `None` |
| `candidate_languages` | `['en-US', 'es-US']` |
| `recognition_mode` | `dictation` |
| `profanity` | `raw` |
| `end_silence_ms` | `2000  (Stage4c)` |
| `initial_silence_ms` | `15000 (Stage4c)` |
| `seg_silence_ms` | `1500  (Stage4c)` |
| `phrase_list` | `Stage5(28) + Stage6(99)` |
| `numeric_pp` | `True` |

**Use C2 over C1 when:** Audio has very long pauses (>1.5s within a sentence) or Stage C2 word count > C1.

**Stick with C1 when:** Stage C2 segment count << C1 (utterances merging).

**Actual Results from This Run:**

- Segments: **65**
- Words: **1327**
- Digit tokens: **19**
- Short words: **679**
- TTFB: **2923.5 ms** (▲23% vs baseline ⚠️ regression)
- TTFT Partial: **2923.5 ms** (▲23% vs baseline ⚠️ regression)
- TTFT Final: **11561.0 ms** (▲302% vs baseline ⚠️ regression)
- Total time: **443.08 s**
- Confidence avg/min: **0.722 / 0.042**
- Low-conf segments: **17**
- vs Baseline: **27.3% similar** (72.7% different)

**What to observe:** If segment_count same as C1 → no benefit from aggressive VAD. If word_count higher than C1 → C2 recovered words → use C2.

---

## 4. Parameters Reference

| Parameter | Azure Property / Method | Default | Stage C1 Value | Effect |
|-----------|------------------------|---------|----------------|--------|
| `locked_language` | `speech_config.speech_recognition_language` | `None` | `detected` | Removes auto-detect latency |
| `recognition_mode` | `enable_dictation()` | `conversation` | `dictation` | Better native digit output |
| `profanity` | `set_profanity(ProfanityOption.Raw)` | `masked` | `raw` | No words censored |
| `end_silence_ms` | `SpeechServiceConnection_EndSilenceTimeoutMs` | `800` | `1200` | Tolerate natural pauses |
| `initial_silence_ms` | `SpeechServiceConnection_InitialSilenceTimeoutMs` | `5000` | `8000` | More time before speech |
| `seg_silence_ms` | `Speech_SegmentationSilenceTimeoutMs` | `600` | `1000` | Mid-sentence pause tolerance |
| `phrase_list` | `PhraseListGrammar.addPhrase()` | `none` | `30+ entries` | Soft vocabulary boost |
| `apply_numeric_pp` | Python post-processor | `False` | `True` | Word→digit (contextual) |
| `output_format` | `OutputFormat.Detailed` | `Detailed` | `Detailed` | Confidence + NBest parsing |

---

## 5. Results Summary Table

> Auto-filled from this run. Use to compare improvements across stages.

| Stage | Phase | Seg | Words | Digits | Short | TTFB | TTFT-P | TTFT-F | Conf | vs BL | Key Finding |
|-------|-------|-----|-------|--------|-------|------|--------|--------|------|-------|-------------|
| stage_0 | Referenc | 93 | 1254 | 19 | 650 | 2384.7 | 2384.7 | 2875.8 | 0.77 | — | Baseline reference |
| stage_1 | Setup | 93 | 1254 | 19 | 650 | 2609.0 | 2609.0 | 3087.1 | 0.77 | 100.0% | No change vs baseline |
| stage_1b | Setup | 97 | 1241 | 19 | 643 | 2339.8 | 2339.8 | 2792.5 | 0.73 | 33.4% | 66.6% change; -13 words |
| stage_2 | Setup | None | None | None | None | — | 5965.7 | 5965.7 | — | — | No throttle errors |
| stage_3 | Integrat | 94 | 1279 | 19 | 665 | 4845.4 | 4845.4 | 5438.9 | — | 67.4% | 32.6% change; +25 words |
| stage_4a | Audio | 93 | 1254 | 19 | 650 | 2582.9 | 2582.9 | 3028.3 | 0.77 | 100.0% | No change vs baseline |
| stage_4b | Audio | 81 | 1254 | 18 | 657 | 2565.4 | 2565.4 | 3735.7 | 0.74 | 39.0% | 61.0% change; same words |
| stage_4c | Audio | 65 | 1286 | 19 | 675 | 2403.9 | 2403.9 | 5161.5 | 0.75 | 33.7% | 66.3% change; +32 words |
| stage_5 | Accuracy | 80 | 1264 | 18 | 660 | 2360.8 | 2360.8 | 3516.9 | 0.74 | 42.3% | 57.7% change; +10 words |
| stage_6 | Accuracy | 85 | 1310 | 19 | 667 | 2427.0 | 2427.0 | 3578.3 | 0.72 | 39.0% | 61.0% change; +56 words |
| stage_7a | Logic | 80 | 1264 | 18 | 660 | 2731.9 | 2731.9 | 3916.8 | 0.74 | 42.3% | 57.7% change; +10 words |
| stage_7b | Logic | 80 | 1264 | 18 | 660 | 3516.0 | 3516.0 | 4657.8 | 0.74 | 23.1% | 76.9% change; +10 words |
| stage_7c | Logic | 80 | 1264 | 18 | 660 | 2475.8 | 2475.8 | 3630.0 | 0.74 | 23.1% | 76.9% change; +10 words |
| stage_8 | Quality | 80 | 1264 | 18 | 660 | 2723.4 | 2723.4 | 3877.9 | 0.74 | 42.3% | 57.7% change; +10 words |
| stage_9 | Testing | — | — | — | — | 2511.2 | 2511.2 | 3814.5 | — | — | P95=3886.2ms — SLA ✅ |
| stage_10 | Testing | None | None | None | None | — | None | 6283.7 | — | — | No throttle errors |
| stage_11 | Monitori | 80 | 1264 | 18 | 660 | 2709.5 | 2709.5 | 3972.3 | 0.74 | 23.1% | 76.9% change; +10 words |
| stage_12 | Go-Live | 80 | 1264 | 18 | 660 | 2767.3 | 2767.3 | 4132.0 | 0.74 | 23.1% | 76.9% change; +10 words |
| stage_c1 | Producti | 80 | 1264 | 18 | 660 | 2649.8 | 2649.8 | 3848.6 | 0.74 | 23.1% | 76.9% change; +10 words |
| stage_c2 | Producti | 65 | 1327 | 19 | 679 | 2923.5 | 2923.5 | 11561.0 | 0.72 | 27.3% | 72.7% change; +73 words |

---

## 6. Identifying Improvement at Each Stage

| Stage | ✅ Improvement means... | ❌ No improvement means... |
|-------|----------------------|--------------------------|
| Stage 1 | TTFT-P lower than Stage 0 | TTFT same — network latency dominates |
| Stage 1b | Word count ≥ Stage 1 | Word count < Stage 1 — 8kHz loses quality |
| Stage 2 | 100% success at expected concurrency | Any 429 errors at expected peak |
| Stage 3 | Transcript ≈ Stage 1 quality | Words dropped at chunk boundaries |
| Stage 4b | Fewer segments, same/more words | Same segments & words — VAD was fine |
| Stage 5 | More digit tokens, short words preserved | Same as Stage 4b — phrases not in audio |
| Stage 6 | Mis-recognised words now correct | Identical to Stage 5 |
| Stage 7c | More digit tokens than 7b | Same as 7b — PP had nothing to convert |
| Stage 8 | Low `low_conf_segments` | Many low-conf segs → noisy/stressed audio |
| Stage 9 | P95 TTFT < SLA threshold | P95 > SLA → consider closer Azure region |
| Stage C1 | All gains from above compounded | Identical to Stage 0 → audio already optimal |

**Interpreting `vs BL` (similarity to baseline):**

| Similarity | Meaning |
|-----------|---------|
| `100%` | Transcript unchanged — stage had no effect on this audio |
| `95–99%` | Minor wording changes (a few words) |
| `85–94%` | Noticeable changes (digit conversions, recovered words) |
| `< 85%` | Major changes — verify whether they are improvements |

---

## 7. Production Configuration (Copy-Paste)

```python
import azure.cognitiveservices.speech as speechsdk

SPEECH_KEY    = "YOUR_KEY"
SPEECH_REGION = "eastus"

def build_production_recognizer(wav_file: str, language: str = "en-US"):
    """Stage C1 — Production configuration."""
    sc = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)

    # Stage 1: Lock language, raw profanity
    sc.speech_recognition_language = language
    sc.set_profanity(speechsdk.ProfanityOption.Raw)

    # Stage 7b: Dictation mode for better digit output
    sc.enable_dictation()

    # Detailed output for confidence scores (Stage 8/12)
    sc.output_format = speechsdk.OutputFormat.Detailed

    # Stage 4b: Conservative VAD
    sc.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,     "1200")
    sc.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "8000")
    sc.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,             "1000")

    audio_cfg  = speechsdk.audio.AudioConfig(filename=wav_file)
    recognizer = speechsdk.SpeechRecognizer(speech_config=sc, audio_config=audio_cfg)

    # Stage 5: Phrase boosting
    phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
    for phrase in DOMAIN_PHRASES:
        phrase_list.addPhrase(phrase)

    return recognizer


# After recognition, apply Stage 7c numeric post-processor:
# final_text = numeric_postprocess(raw_transcript, language=detected_language)
```

---

## 8. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `AuthenticationFailure` | Wrong SPEECH_KEY or SPEECH_REGION | Check Azure portal |
| `FileNotFoundError` | Audio file path wrong | Check `INPUT_AUDIO_FILE` |
| `FFmpeg conversion failed` | FFmpeg not installed | `winget install ffmpeg` |
| Empty transcript | Network issue or long silence at start | Increase `initial_silence_ms`, check connectivity |
| Transcript identical across all stages | Audio already well-handled by Azure defaults | Test with noisier/faster audio |
| `"to"` converted to `"2"` | Bug in `_NEVER_CONVERT` | Verify you are using latest script version |
| Stage 2/10 throttle errors | Azure tier limit reached | Upgrade to S0+ or add retry-with-backoff |
| P95 TTFT-P showing regression vs Stage 0 | Network latency to Azure region | Switch to closer region (`westus`, `westeurope`) |
| Low confidence on all segments | Very noisy audio | Consider audio pre-processing (noise reduction) |
| Stage 3 chunk boundary artifacts | Chunk too small | Increase `chunk_ms` from 100 to 200 |

---

## ⚠️ Alerts Triggered During This Run

| Alert Type | Stage | Detail |
|-----------|-------|--------|
| `HIGH_LATENCY` | stage_2 | metric=ttft_partial_ms, value_ms=5371.5, baseline_ms=2384.7, regression_pct=125.2, threshold_pct=20 |
| `HIGH_LATENCY` | stage_2 | metric=ttft_final_ms, value_ms=5848.7, baseline_ms=2875.8, regression_pct=103.4, threshold_pct=20 |
| `HIGH_LATENCY` | stage_2 | metric=ttft_partial_ms, value_ms=5416.4, baseline_ms=2384.7, regression_pct=127.1, threshold_pct=20 |
| `HIGH_LATENCY` | stage_2 | metric=ttft_final_ms, value_ms=5942.2, baseline_ms=2875.8, regression_pct=106.6, threshold_pct=20 |
| `HIGH_LATENCY` | stage_2 | metric=ttft_partial_ms, value_ms=5688.3, baseline_ms=2384.7, regression_pct=138.5, threshold_pct=20 |
| `HIGH_LATENCY` | stage_2 | metric=ttft_final_ms, value_ms=6182.2, baseline_ms=2875.8, regression_pct=115.0, threshold_pct=20 |
| `HIGH_LATENCY` | stage_2 | metric=ttft_partial_ms, value_ms=5440.9, baseline_ms=2384.7, regression_pct=128.2, threshold_pct=20 |
| `HIGH_LATENCY` | stage_2 | metric=ttft_final_ms, value_ms=5943.6, baseline_ms=2875.8, regression_pct=106.7, threshold_pct=20 |
| `HIGH_LATENCY` | stage_2 | metric=ttft_partial_ms, value_ms=5440.5, baseline_ms=2384.7, regression_pct=128.1, threshold_pct=20 |
| `HIGH_LATENCY` | stage_2 | metric=ttft_final_ms, value_ms=5912.0, baseline_ms=2875.8, regression_pct=105.6, threshold_pct=20 |
| `HIGH_LATENCY` | stage_4b | metric=ttft_final_ms, value_ms=3735.7, baseline_ms=2875.8, regression_pct=29.9, threshold_pct=20 |
| `HIGH_LATENCY` | stage_4c | metric=ttft_final_ms, value_ms=5161.5, baseline_ms=2875.8, regression_pct=79.5, threshold_pct=20 |
| `HIGH_LATENCY` | stage_5 | metric=ttft_final_ms, value_ms=3516.9, baseline_ms=2875.8, regression_pct=22.3, threshold_pct=20 |
| `HIGH_LATENCY` | stage_6 | metric=ttft_final_ms, value_ms=3578.3, baseline_ms=2875.8, regression_pct=24.4, threshold_pct=20 |
| `HIGH_LATENCY` | stage_7a | metric=ttft_final_ms, value_ms=3916.8, baseline_ms=2875.8, regression_pct=36.2, threshold_pct=20 |
| `HIGH_LATENCY` | stage_7b | metric=ttft_partial_ms, value_ms=3516.0, baseline_ms=2384.7, regression_pct=47.4, threshold_pct=20 |
| `HIGH_LATENCY` | stage_7b | metric=ttft_final_ms, value_ms=4657.8, baseline_ms=2875.8, regression_pct=62.0, threshold_pct=20 |
| `HIGH_LATENCY` | stage_7c | metric=ttft_final_ms, value_ms=3630.0, baseline_ms=2875.8, regression_pct=26.2, threshold_pct=20 |
| `HIGH_LATENCY` | stage_8 | metric=ttft_final_ms, value_ms=3877.9, baseline_ms=2875.8, regression_pct=34.8, threshold_pct=20 |
| `HIGH_LATENCY` | stage_9 | metric=ttft_final_ms, value_ms=3659.3, baseline_ms=2875.8, regression_pct=27.2, threshold_pct=20 |
| `HIGH_LATENCY` | stage_9 | metric=ttft_final_ms, value_ms=3886.2, baseline_ms=2875.8, regression_pct=35.1, threshold_pct=20 |
| `HIGH_LATENCY` | stage_9 | metric=ttft_final_ms, value_ms=3814.5, baseline_ms=2875.8, regression_pct=32.6, threshold_pct=20 |
| `HIGH_LATENCY` | stage_9_tight | metric=ttft_partial_ms, value_ms=4167.5, baseline_ms=2384.7, regression_pct=74.8, threshold_pct=20 |
| `HIGH_LATENCY` | stage_9_tight | metric=ttft_final_ms, value_ms=5273.5, baseline_ms=2875.8, regression_pct=83.4, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_partial_ms, value_ms=3447.7, baseline_ms=2384.7, regression_pct=44.6, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_final_ms, value_ms=3880.5, baseline_ms=2875.8, regression_pct=34.9, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_partial_ms, value_ms=2869.3, baseline_ms=2384.7, regression_pct=20.3, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_partial_ms, value_ms=5706.6, baseline_ms=2384.7, regression_pct=139.3, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_final_ms, value_ms=6283.7, baseline_ms=2875.8, regression_pct=118.5, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_partial_ms, value_ms=5388.7, baseline_ms=2384.7, regression_pct=126.0, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_final_ms, value_ms=5913.0, baseline_ms=2875.8, regression_pct=105.6, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_partial_ms, value_ms=5449.6, baseline_ms=2384.7, regression_pct=128.5, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_final_ms, value_ms=5894.8, baseline_ms=2875.8, regression_pct=105.0, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_partial_ms, value_ms=5721.0, baseline_ms=2384.7, regression_pct=139.9, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_final_ms, value_ms=6258.4, baseline_ms=2875.8, regression_pct=117.6, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_partial_ms, value_ms=5435.7, baseline_ms=2384.7, regression_pct=127.9, threshold_pct=20 |
| `HIGH_LATENCY` | stage_10 | metric=ttft_final_ms, value_ms=6006.6, baseline_ms=2875.8, regression_pct=108.9, threshold_pct=20 |
| `HIGH_LATENCY` | stage_11 | metric=ttft_final_ms, value_ms=3972.3, baseline_ms=2875.8, regression_pct=38.1, threshold_pct=20 |
| `HIGH_LATENCY` | stage_12 | metric=ttft_final_ms, value_ms=4132.0, baseline_ms=2875.8, regression_pct=43.7, threshold_pct=20 |
| `HIGH_LATENCY` | stage_c1 | metric=ttft_final_ms, value_ms=3848.6, baseline_ms=2875.8, regression_pct=33.8, threshold_pct=20 |
| `HIGH_LATENCY` | stage_c2 | metric=ttft_partial_ms, value_ms=2923.5, baseline_ms=2384.7, regression_pct=22.6, threshold_pct=20 |
| `HIGH_LATENCY` | stage_c2 | metric=ttft_final_ms, value_ms=11561.0, baseline_ms=2875.8, regression_pct=302.0, threshold_pct=20 |

---

*Azure STT Transcription Quality Lab — Documentation Guide  |  Generated: 2026-04-22 19:59:48  |  Audio: `audio/maria1.mp3`*
