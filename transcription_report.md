# Azure STT Transcription Quality Lab — Full Report

**Audio:** `audio/maria1.mp3`  **Generated:** 2026-04-22 19:59:48  **Region:** eastus

**Candidate Languages:** ['en-US', 'es-US']  **Latency Regression:** >20% vs baseline  **Re-prompt Threshold:** confidence < 0.6

---

## Stage-by-Stage Analysis

### Stage 0 — Baseline

**Phase:** Reference  |  **Task:** Original script — no modifications

**Description:** Exact copy of the working script. Auto-detect en-US/es-ES. All default Azure settings. This is the reference for all comparisons.

**Parameters Changed:** `None — reference state`

#### Parameters Used

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

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `93` |
| Word Count | `1254` |
| Digit Tokens | `19` |
| Short Words (1–3 chars) | `650` |
| TTFB — Time to First Byte (ms) | `2384.7 ms` |
| TTFT Partial — First Partial Result (ms) | `2384.7 ms` |
| TTFT Final — First Finalised Segment (ms) | `2875.8 ms` |
| Total Time (sec) | `443.22` |
| Confidence Avg | `0.767` |
| Confidence Min | `0.069` |
| Low-Conf Segments | `16` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Reference transcript. All stages compared against this.

**What to Observe:** Segment count, word count, digit tokens, short words, TTFT.

---

### Stage 1 — ASR Config Finalization

**Phase:** Setup  |  **Task:** Preserve bilingual auto-detect + improve ASR config

**Description:** For Spanglish audio we DO NOT lock language. Azure auto-detect remains enabled for en-US + es-ES. Only profanity is changed to RAW and config is optimized.

**Parameters Changed:** `profanity (masked→raw), auto-detect preserved`

#### Parameters Used

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

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `93` |
| Word Count | `1254` |
| Digit Tokens | `19` |
| Short Words (1–3 chars) | `650` |
| TTFB — Time to First Byte (ms) | `2609.0 ms` `(▲9% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2609.0 ms` `(▲9% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3087.1 ms` `(▲7% vs baseline)` |
| Total Time (sec) | `443.03` |
| Similarity to Baseline | `100.0%` |
| Confidence Avg | `0.767` |
| Confidence Min | `0.069` |
| Low-Conf Segments | `16` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Better fidelity without breaking bilingual recognition.

**What to Observe:** Check TTFT vs Stage 0. Ensure English + Spanish both remain accurate.

> ℹ️  Transcript **identical** to baseline — no change for this audio.

---

### Stage 1b — ASR Config: Telephony 8kHz Format

**Phase:** Setup  |  **Task:** Test telephony audio format

**Description:** Same bilingual config as Stage 1 but tested using 8kHz telephony audio.

**Parameters Changed:** `audio_format: 16kHz → 8kHz`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `locked_language` | `None (auto-detect preserved)` |
| `candidate_languages` | `['en-US', 'es-US']` |
| `audio_format` | `8kHz PCM WAV` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `97` |
| Word Count | `1241` |
| Digit Tokens | `19` |
| Short Words (1–3 chars) | `643` |
| TTFB — Time to First Byte (ms) | `2339.8 ms` `(▼2% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2339.8 ms` `(▼2% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `2792.5 ms` `(▼3% vs baseline)` |
| Total Time (sec) | `442.66` |
| Similarity to Baseline | `33.4%` |
| Confidence Avg | `0.734` |
| Confidence Min | `0.051` |
| Low-Conf Segments | `20` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Check if telephony audio performs better.

**What to Observe:** Compare Stage 1 vs Stage 1b transcripts.

> 🔄  Transcript **66.6% different** from baseline (similarity: 33.4%).

---

### Stage 2 — Concurrency & Quota Validation

**Phase:** Setup  |  **Task:** Validate concurrency and quotas

**Description:** Infrastructure validation only.

**Parameters Changed:** `N/A`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `base_config` | `Stage 1 (bilingual auto-detect)` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| TTFT Partial — First Partial Result (ms) | `5965.7 ms` `(▲150% vs baseline ⚠️ regression)` |
| TTFT Final — First Finalised Segment (ms) | `5965.7 ms` `(▲107% vs baseline ⚠️ regression)` |
| Total Time (sec) | `1330.58` |
| Numeric PP Applied | `False` |

#### Concurrency Results

| Sessions | Success% | Throttled | TTFT Avg | P50 | P95 | P Max |
|----------|----------|-----------|----------|-----|-----|-------|
| 1 | 100.0% | 0 | 2819.2 ms | 2819.2 ms | 2819.2 ms | 2819.2 ms |
| 3 | 100.0% | 0 | 3046.2 ms | 2991.7 ms | 3167.4 ms | 3167.4 ms |
| 5 | 100.0% | 0 | 5965.7 ms | 5942.2 ms | 6182.2 ms | 6182.2 ms |

**Expected Outcome:** No throttling.

**What to Observe:** Concurrency failures.

---

### Stage 3 — Real-Time Socket Integration

**Phase:** Integration  |  **Task:** Streaming ASR

**Description:** PushAudioInputStream real-time chunks.

**Parameters Changed:** `audio_input: file → stream`

#### Parameters Used

| Parameter | Value |
|-----------|-------|

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `94` |
| Word Count | `1279` |
| Digit Tokens | `19` |
| Short Words (1–3 chars) | `665` |
| TTFB — Time to First Byte (ms) | `4845.4 ms` `(▲103% vs baseline ⚠️ regression)` |
| TTFT Partial — First Partial Result (ms) | `4845.4 ms` `(▲103% vs baseline ⚠️ regression)` |
| TTFT Final — First Finalised Segment (ms) | `5438.9 ms` `(▲89% vs baseline ⚠️ regression)` |
| Total Time (sec) | `910.87` |
| Similarity to Baseline | `67.4%` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Low-latency partials.

**What to Observe:** TTFT partial latency.

> 🔄  Transcript **32.6% different** from baseline (similarity: 67.4%).

---

### Stage 4a — VAD: Default (800ms)

**Phase:** Audio  |  **Task:** VAD baseline — built-in default settings

**Description:** Same as Stage 1. Isolates VAD behaviour at default 800ms.

**Parameters Changed:** `None from Stage 1 — VAD baseline`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `end_silence_ms` | `800  (default)` |
| `initial_silence_ms` | `5000 (default)` |
| `seg_silence_ms` | `600  (default)` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `93` |
| Word Count | `1254` |
| Digit Tokens | `19` |
| Short Words (1–3 chars) | `650` |
| TTFB — Time to First Byte (ms) | `2582.9 ms` `(▲8% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2582.9 ms` `(▲8% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3028.3 ms` `(▲5% vs baseline)` |
| Total Time (sec) | `536.69` |
| Similarity to Baseline | `100.0%` |
| Confidence Avg | `0.767` |
| Confidence Min | `0.069` |
| Low-Conf Segments | `16` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Same as Stage 1. Segment count reference for VAD comparison.

**What to Observe:** Segment count. Are any sentences truncated mid-speech?

> ℹ️  Transcript **identical** to baseline — no change for this audio.

---

### Stage 4b — VAD: Conservative (1200ms)

**Phase:** Audio  |  **Task:** VAD conservative — reduce truncation and false cut-offs

**Description:** Increases end-silence to 1200ms (+50%). Best for speakers who pause mid-sentence.

**Parameters Changed:** `end_silence_ms: 800→1200, seg_silence_ms: 600→1000, initial_silence_ms: 5000→8000`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `end_silence_ms` | `1200  ← INCREASED (was: 800)` |
| `initial_silence_ms` | `8000  ← INCREASED (was: 5000)` |
| `seg_silence_ms` | `1000  ← INCREASED (was: 600)` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `81` |
| Word Count | `1254` |
| Digit Tokens | `18` |
| Short Words (1–3 chars) | `657` |
| TTFB — Time to First Byte (ms) | `2565.4 ms` `(▲8% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2565.4 ms` `(▲8% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3735.7 ms` `(▲30% vs baseline ⚠️ regression)` |
| Total Time (sec) | `442.81` |
| Similarity to Baseline | `39.0%` |
| Confidence Avg | `0.737` |
| Confidence Min | `0.053` |
| Low-Conf Segments | `19` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Fewer mid-sentence truncations.

**What to Observe:** Segment count vs 4a (should decrease). Word count vs 4a (should increase or equal).

> 🔄  Transcript **61.0% different** from baseline (similarity: 39.0%).

---

### Stage 4c — VAD: Aggressive (2000ms)

**Phase:** Audio  |  **Task:** VAD aggressive — maximum pause tolerance

**Description:** End-silence 2000ms. Risk: may merge two utterances if speaker pauses < 2s.

**Parameters Changed:** `end_silence_ms: 1200→2000, seg_silence_ms: 1000→1500, initial_silence_ms: 8000→15000`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `end_silence_ms` | `2000  ← INCREASED (was: 1200)` |
| `initial_silence_ms` | `15000 ← INCREASED (was: 8000)` |
| `seg_silence_ms` | `1500  ← INCREASED (was: 1000)` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `65` |
| Word Count | `1286` |
| Digit Tokens | `19` |
| Short Words (1–3 chars) | `675` |
| TTFB — Time to First Byte (ms) | `2403.9 ms` `(▲1% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2403.9 ms` `(▲1% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `5161.5 ms` `(▲79% vs baseline ⚠️ regression)` |
| Total Time (sec) | `443.05` |
| Similarity to Baseline | `33.7%` |
| Confidence Avg | `0.749` |
| Confidence Min | `0.053` |
| Low-Conf Segments | `13` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Maximum pause tolerance. Best for slow/hesitant speakers.

**What to Observe:** If segment count drops drastically → utterances are merging (avoid 4c).

> 🔄  Transcript **66.3% different** from baseline (similarity: 33.7%).

---

### Stage 5 — Word / Phrase Boosting

**Phase:** Accuracy  |  **Task:** Boost digits, identifiers, domain terms via PhraseListGrammar

**Description:** Adds 28 domain-specific phrases to PhraseListGrammar.

**Parameters Changed:** `phrase_list: none → 28 entries`

#### Parameters Used

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

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `80` |
| Word Count | `1264` |
| Digit Tokens | `18` |
| Short Words (1–3 chars) | `660` |
| TTFB — Time to First Byte (ms) | `2360.8 ms` `(▼1% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2360.8 ms` `(▼1% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3516.9 ms` `(▲22% vs baseline ⚠️ regression)` |
| Total Time (sec) | `442.57` |
| Similarity to Baseline | `42.3%` |
| Confidence Avg | `0.743` |
| Confidence Min | `0.071` |
| Low-Conf Segments | `19` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Improved numeric accuracy. Short words (ID, OK) less dropped.

**What to Observe:** digit_token_count vs Stage 0. short_word_count vs Stage 0.

> 🔄  Transcript **57.7% different** from baseline (similarity: 42.3%).

---

### Stage 6 — Transcript-Based Vocabulary Tuning

**Phase:** Accuracy  |  **Task:** Use sample transcripts to refine vocabulary/style boosting

**Description:** Extracts words ≥2 appearances from baseline (99 found) and adds to phrase list on top of Stage 5.

**Parameters Changed:** `phrase_list: Stage5 list + 99 baseline-extracted phrases`

#### Parameters Used

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

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `85` |
| Word Count | `1310` |
| Digit Tokens | `19` |
| Short Words (1–3 chars) | `667` |
| TTFB — Time to First Byte (ms) | `2427.0 ms` `(▲2% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2427.0 ms` `(▲2% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3578.3 ms` `(▲24% vs baseline ⚠️ regression)` |
| Total Time (sec) | `442.83` |
| Similarity to Baseline | `39.0%` |
| Confidence Avg | `0.717` |
| Confidence Min | `0.065` |
| Low-Conf Segments | `22` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Domain-specific words from your audio get a recognition boost.

**What to Observe:** Check if any word mis-recognised in Stage 0 is now correct. Compare similarity_pct to Stage 5.

> 🔄  Transcript **61.0% different** from baseline (similarity: 39.0%).

---

### Stage 7a — Numeric: Conversation Mode (Azure native)

**Phase:** Logic  |  **Task:** Validate digit-by-digit vs grouped digit behavior — baseline

**Description:** Measures how Azure natively outputs numbers in conversation mode without post-processing.

**Parameters Changed:** `None from Stage 5 — numeric baseline`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `recognition_mode` | `conversation` |
| `numeric_pp` | `False` |
| `phrase_list` | `28 entries` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `80` |
| Word Count | `1264` |
| Digit Tokens | `18` |
| Short Words (1–3 chars) | `660` |
| TTFB — Time to First Byte (ms) | `2731.9 ms` `(▲15% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2731.9 ms` `(▲15% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3916.8 ms` `(▲36% vs baseline ⚠️ regression)` |
| Total Time (sec) | `442.61` |
| Similarity to Baseline | `42.3%` |
| Confidence Avg | `0.743` |
| Confidence Min | `0.071` |
| Low-Conf Segments | `19` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Mixed output — some words, some digits.

**What to Observe:** digit_token_count. How many numbers appear as words vs digits?

> 🔄  Transcript **57.7% different** from baseline (similarity: 42.3%).

---

### Stage 7b — Numeric: Dictation Mode

**Phase:** Logic  |  **Task:** Test dictation mode for improved digit-by-digit output

**Description:** Switches to Azure dictation mode, optimised to output spoken numbers as digit tokens.

**Parameters Changed:** `recognition_mode: conversation → dictation  ← CHANGED`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `recognition_mode` | `dictation  ← CHANGED (was: conversation)` |
| `numeric_pp` | `False` |
| `phrase_list` | `28 entries` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `80` |
| Word Count | `1264` |
| Digit Tokens | `18` |
| Short Words (1–3 chars) | `660` |
| TTFB — Time to First Byte (ms) | `3516.0 ms` `(▲47% vs baseline ⚠️ regression)` |
| TTFT Partial — First Partial Result (ms) | `3516.0 ms` `(▲47% vs baseline ⚠️ regression)` |
| TTFT Final — First Finalised Segment (ms) | `4657.8 ms` `(▲62% vs baseline ⚠️ regression)` |
| Total Time (sec) | `442.35` |
| Similarity to Baseline | `23.1%` |
| Confidence Avg | `0.743` |
| Confidence Min | `0.071` |
| Low-Conf Segments | `19` |
| Numeric PP Applied | `False` |

**Expected Outcome:** More digit tokens in transcript.

**What to Observe:** digit_token_count vs 7a. Check: does 'I need to go' still say 'to' (not '2')?

> 🔄  Transcript **76.9% different** from baseline (similarity: 23.1%).

---

### Stage 7c — Numeric: Dictation + Context-Aware Post-Processor

**Phase:** Logic  |  **Task:** Full numeric handling: dictation + context-aware word-to-digit PP

**Description:** Dictation mode + post-processor. SAFETY: 'to/for/a/an/won/ate' NEVER converted. Spanish: pass-through.

**Parameters Changed:** `numeric_pp: False → True  ← ADDED (on top of Stage 7b)`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `recognition_mode` | `dictation` |
| `numeric_pp` | `True  ← ADDED` |
| `never_convert` | `'to','for','a','an','won','ate' + full list` |
| `spanish_handling` | `pass-through (no conversion)` |
| `phrase_list` | `28 entries` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `80` |
| Word Count | `1264` |
| Digit Tokens | `18` |
| Short Words (1–3 chars) | `660` |
| TTFB — Time to First Byte (ms) | `2475.8 ms` `(▲4% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2475.8 ms` `(▲4% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3630.0 ms` `(▲26% vs baseline ⚠️ regression)` |
| Total Time (sec) | `442.61` |
| Similarity to Baseline | `23.1%` |
| Confidence Avg | `0.743` |
| Confidence Min | `0.071` |
| Low-Conf Segments | `19` |
| Numeric PP Applied | `True` |

**Expected Outcome:** Maximum digit output. 'to' never becomes '2'.

**What to Observe:** digit_token_count vs 7b (should be ≥). Verify 'to'/'for'/'a' stayed as words.

> 🔄  Transcript **76.9% different** from baseline (similarity: 23.1%).

---

### Stage 8 — Emotion / Tone Evaluation

**Phase:** Quality  |  **Task:** Assess ASR behavior under neutral vs stressed speech

**Description:** Parses per-segment confidence scores and NBest alternatives. Flags segments below 0.6 for re-prompt.

**Parameters Changed:** `output_format: detailed (confidence + NBest parsing enabled)`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `recognition_mode` | `conversation` |
| `output_format` | `detailed  (parses Confidence + NBest)` |
| `confidence_threshold` | `0.6` |
| `profanity` | `raw` |
| `end_silence_ms` | `1200` |
| `phrase_list` | `28 entries` |
| `numeric_pp` | `False` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `80` |
| Word Count | `1264` |
| Digit Tokens | `18` |
| Short Words (1–3 chars) | `660` |
| TTFB — Time to First Byte (ms) | `2723.4 ms` `(▲14% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2723.4 ms` `(▲14% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3877.9 ms` `(▲35% vs baseline ⚠️ regression)` |
| Total Time (sec) | `443.02` |
| Similarity to Baseline | `42.3%` |
| Confidence Avg | `0.743` |
| Confidence Min | `0.071` |
| Low-Conf Segments | `19` |
| Numeric PP Applied | `False` |

**Expected Outcome:** Robust recognition. Confidence > threshold for most segments.

**What to Observe:** confidence_avg and confidence_min. low_conf_segments count (segments below 0.6).

> 🔄  Transcript **57.7% different** from baseline (similarity: 42.3%).

---

### Stage 9 — Latency & Timeout Testing

**Phase:** Testing  |  **Task:** Validate response times within conversational SLA

**Description:** Runs 3 times, collects P50/P95 TTFT. Flags runs >20% slower than Stage 0. Also tests tight timeout (500ms).

**Parameters Changed:** `3 runs for statistical stability; tight-timeout sub-test`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `latency_regression_pct` | `20` |
| `runs_for_stats` | `3` |
| `tight_end_silence` | `500` |
| `normal_end_silence` | `1200` |
| `phrase_list` | `28 entries` |
| `numeric_pp` | `False` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| TTFB — Time to First Byte (ms) | `2511.2 ms` `(▲5% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2511.2 ms` `(▲5% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3814.5 ms` `(▲33% vs baseline ⚠️ regression)` |
| Total Time (sec) | `442.81` |
| Numeric PP Applied | `False` |

#### Latency Runs (vs Stage 0 baseline)

| Run | TTFB (ms) | TTFT-P (ms) | TTFT-F (ms) | Total (s) | vs Baseline |
|-----|-----------|------------|------------|-----------|-------------|
| 1 | 2487.5 | 2487.5 | 3659.3 | 442.99 | ✅ +4% |
| 2 | 2685.4 | 2685.4 | 3886.2 | 443.01 | ✅ +13% |
| 3 | 2511.2 | 2511.2 | 3814.5 | 442.81 | ✅ +5% |
| tight-500ms | 4167.5 | 4167.5 | 5273.5 | 442.85 | ⚠️ +75% |

**Expected Outcome:** TTFT-P and TTFB within 20% of Stage 0 baseline.

**What to Observe:** P50 and P95 TTFT across runs. Does tight timeout (500ms) cause truncation vs normal (1200ms)?

---

### Stage 10 — Load & Concurrency Testing

**Phase:** Testing  |  **Task:** Validate peak concurrent real-time streams

**Description:** Runs [1, 3, 5] concurrent sessions via ThreadPoolExecutor.

**Parameters Changed:** `concurrent sessions: 1 → multiple (ThreadPoolExecutor)`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `concurrency_levels` | `[1, 3, 5]` |
| `latency_regression_pct` | `20` |
| `output_format` | `simple (reduces payload under load)` |
| `phrase_list` | `none (reduces setup time under load)` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| TTFT Final — First Finalised Segment (ms) | `6283.7 ms` `(▲119% vs baseline ⚠️ regression)` |
| Total Time (sec) | `1331.13` |
| Numeric PP Applied | `False` |

#### Concurrency Results

| Sessions | Success% | Throttled | TTFT Avg | P50 | P95 | P Max |
|----------|----------|-----------|----------|-----|-----|-------|
| 1 | 100.0% | 0 | 3880.5 ms | 3880.5 ms | 3880.5 ms | 3880.5 ms |
| 3 | 100.0% | 0 | 3203.2 ms | 3088.2 ms | 3433.2 ms | 3433.2 ms |
| 5 | 100.0% | 0 | 6071.3 ms | 6006.6 ms | 6283.7 ms | 6283.7 ms |

**Expected Outcome:** Stable under load. No throttle errors at expected peak concurrency.

**What to Observe:** At which concurrency level do throttle errors appear? P95 TTFT degradation as concurrency increases.

---

### Stage 11 — Logging & Alerts Setup

**Phase:** Monitoring  |  **Task:** Enable error, latency, socket-drop monitoring

**Description:** Runs Combined Best config while JSON logging is active. Alerts: TTFB/TTFT-P >20% slower than baseline → HIGH_LATENCY; empty → EMPTY_TRANSCRIPT; error → RECOGNITION_ERROR.

**Parameters Changed:** `Logging + alerts layer enabled. No ASR config changes.`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `log_file` | `transcription_audit.log` |
| `log_format` | `JSON (one record per line)` |
| `alert_regression_pct` | `20` |
| `alert_empty` | `True` |
| `alert_error` | `True` |
| `base_config` | `Combined Best (Stage C1)` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `80` |
| Word Count | `1264` |
| Digit Tokens | `18` |
| Short Words (1–3 chars) | `660` |
| TTFB — Time to First Byte (ms) | `2709.5 ms` `(▲14% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2709.5 ms` `(▲14% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3972.3 ms` `(▲38% vs baseline ⚠️ regression)` |
| Total Time (sec) | `442.66` |
| Similarity to Baseline | `23.1%` |
| Confidence Avg | `0.743` |
| Confidence Min | `0.071` |
| Low-Conf Segments | `19` |
| Numeric PP Applied | `True` |

**Expected Outcome:** Audit log populated. Alerts file shows triggered thresholds.

**What to Observe:** transcription_audit.log record count. Any alerts triggered?

> 🔄  Transcript **76.9% different** from baseline (similarity: 23.1%).

---

### Stage 12 — Fallback Validation

**Phase:** Go-Live  |  **Task:** Test re-prompt / DTMF / alternate flow

**Description:** Validates fallback: low-conf segments flagged for re-prompt (threshold: 0.6); empty → DTMF fallback.

**Parameters Changed:** `Fallback logic layer (post-processing). No ASR config changes.`

#### Parameters Used

| Parameter | Value |
|-----------|-------|
| `reprompt_threshold` | `0.6` |
| `dtmf_fallback` | `Triggered when transcript is empty or all-silence` |
| `re-prompt_trigger` | `Any segment with confidence < 0.6` |
| `base_config` | `Combined Best` |

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `80` |
| Word Count | `1264` |
| Digit Tokens | `18` |
| Short Words (1–3 chars) | `660` |
| TTFB — Time to First Byte (ms) | `2767.3 ms` `(▲16% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2767.3 ms` `(▲16% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `4132.0 ms` `(▲44% vs baseline ⚠️ regression)` |
| Total Time (sec) | `442.88` |
| Similarity to Baseline | `23.1%` |
| Confidence Avg | `0.743` |
| Confidence Min | `0.071` |
| Low-Conf Segments | `19` |
| Numeric PP Applied | `True` |

#### Fallback Report

- Low-confidence segments: **19**
- Re-prompt flagged: **True**
- DTMF fallback triggered: **False**

**Expected Outcome:** Resilient failure handling. No silent failures.

**What to Observe:** low_conf_segments count. fallback_report.reprompt_flagged. fallback_report.dtmf_fallback.

> 🔄  Transcript **76.9% different** from baseline (similarity: 23.1%).

---

### Stage C1 — Combined Best  ✅ PRODUCTION RECOMMENDATION

**Phase:** Production  |  **Task:** All effective stages combined

**Description:** Stage1 (locked lang + raw profanity) + Stage 4b (conservative VAD) + Stage 5 (phrase boosting) + Stage 7c (dictation + numeric PP).

**Parameters Changed:** `All effective stages applied together`

#### Parameters Used

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

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `80` |
| Word Count | `1264` |
| Digit Tokens | `18` |
| Short Words (1–3 chars) | `660` |
| TTFB — Time to First Byte (ms) | `2649.8 ms` `(▲11% vs baseline)` |
| TTFT Partial — First Partial Result (ms) | `2649.8 ms` `(▲11% vs baseline)` |
| TTFT Final — First Finalised Segment (ms) | `3848.6 ms` `(▲34% vs baseline ⚠️ regression)` |
| Total Time (sec) | `443.21` |
| Similarity to Baseline | `23.1%` |
| Confidence Avg | `0.743` |
| Confidence Min | `0.071` |
| Low-Conf Segments | `19` |
| Numeric PP Applied | `True` |

**Expected Outcome:** Best overall accuracy. All individual improvements compounded.

**What to Observe:** similarity_pct vs Stage 0. digit_token_count (expect highest). TTFT (expect ≤ Stage 0). word_count (expect ≥ Stage 0).

> 🔄  Transcript **76.9% different** from baseline (similarity: 23.1%).

---

### Stage C2 — Combined All Stages

**Phase:** Production  |  **Task:** Every stage combined for maximum coverage

**Description:** C1 + Stage 4c (aggressive VAD) + Stage 6 (extended vocab).

**Parameters Changed:** `Stage C1 + aggressive VAD + extended vocab`

#### Parameters Used

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

#### Metrics

| Metric | Value |
|--------|-------|
| Detected Language | `en-US` |
| Segments | `65` |
| Word Count | `1327` |
| Digit Tokens | `19` |
| Short Words (1–3 chars) | `679` |
| TTFB — Time to First Byte (ms) | `2923.5 ms` `(▲23% vs baseline ⚠️ regression)` |
| TTFT Partial — First Partial Result (ms) | `2923.5 ms` `(▲23% vs baseline ⚠️ regression)` |
| TTFT Final — First Finalised Segment (ms) | `11561.0 ms` `(▲302% vs baseline ⚠️ regression)` |
| Total Time (sec) | `443.08` |
| Similarity to Baseline | `27.3%` |
| Confidence Avg | `0.722` |
| Confidence Min | `0.042` |
| Low-Conf Segments | `17` |
| Numeric PP Applied | `True` |

**Expected Outcome:** Maximum coverage. Compare to C1 to check if aggressive VAD helps.

**What to Observe:** If segment_count same as C1 → no benefit from aggressive VAD. If word_count higher than C1 → C2 recovered words → use C2.

> 🔄  Transcript **72.7% different** from baseline (similarity: 27.3%).

---

## 📊 Full Comparison Table

> Latency columns: raw value + (▲/▼ % vs Stage 0). ⚠ = >20% regression vs baseline.

| Stage | Phase | Seg | Words | Digits | TTFB | TTFT-P | TTFT-F | Time(s) | Conf-Avg | vs BL |
|-------|-------|-----|-------|--------|------|--------|--------|---------|----------|-------|
| stage_0 | Referenc | 93 | 1254 | 19 | 2384.7 (▼0%) | 2384.7 (▼0%) | 2875.8 (▼0%) | 443.22 | 0.77 | — |
| stage_1 | Setup | 93 | 1254 | 19 | 2609.0 (▲9%) | 2609.0 (▲9%) | 3087.1 (▲7%) | 443.03 | 0.77 | 100.0% |
| stage_1b | Setup | 97 | 1241 | 19 | 2339.8 (▼2%) | 2339.8 (▼2%) | 2792.5 (▼3%) | 442.66 | 0.73 | 33.4% |
| stage_2 | Setup | None | None | None | — | 5965.7 (▲150% ⚠) | 5965.7 (▲107% ⚠) | 1330.58 | — | — |
| stage_3 | Integrat | 94 | 1279 | 19 | 4845.4 (▲103% ⚠) | 4845.4 (▲103% ⚠) | 5438.9 (▲89% ⚠) | 910.87 | — | 67.4% |
| stage_4a | Audio | 93 | 1254 | 19 | 2582.9 (▲8%) | 2582.9 (▲8%) | 3028.3 (▲5%) | 536.69 | 0.77 | 100.0% |
| stage_4b | Audio | 81 | 1254 | 18 | 2565.4 (▲8%) | 2565.4 (▲8%) | 3735.7 (▲30% ⚠) | 442.81 | 0.74 | 39.0% |
| stage_4c | Audio | 65 | 1286 | 19 | 2403.9 (▲1%) | 2403.9 (▲1%) | 5161.5 (▲79% ⚠) | 443.05 | 0.75 | 33.7% |
| stage_5 | Accuracy | 80 | 1264 | 18 | 2360.8 (▼1%) | 2360.8 (▼1%) | 3516.9 (▲22% ⚠) | 442.57 | 0.74 | 42.3% |
| stage_6 | Accuracy | 85 | 1310 | 19 | 2427.0 (▲2%) | 2427.0 (▲2%) | 3578.3 (▲24% ⚠) | 442.83 | 0.72 | 39.0% |
| stage_7a | Logic | 80 | 1264 | 18 | 2731.9 (▲15%) | 2731.9 (▲15%) | 3916.8 (▲36% ⚠) | 442.61 | 0.74 | 42.3% |
| stage_7b | Logic | 80 | 1264 | 18 | 3516.0 (▲47% ⚠) | 3516.0 (▲47% ⚠) | 4657.8 (▲62% ⚠) | 442.35 | 0.74 | 23.1% |
| stage_7c | Logic | 80 | 1264 | 18 | 2475.8 (▲4%) | 2475.8 (▲4%) | 3630.0 (▲26% ⚠) | 442.61 | 0.74 | 23.1% |
| stage_8 | Quality | 80 | 1264 | 18 | 2723.4 (▲14%) | 2723.4 (▲14%) | 3877.9 (▲35% ⚠) | 443.02 | 0.74 | 42.3% |
| stage_9 | Testing | ? | ? | ? | 2511.2 (▲5%) | 2511.2 (▲5%) | 3814.5 (▲33% ⚠) | 442.81 | — | — |
| stage_10 | Testing | None | None | None | — | — | 6283.7 (▲119% ⚠) | 1331.13 | — | — |
| stage_11 | Monitori | 80 | 1264 | 18 | 2709.5 (▲14%) | 2709.5 (▲14%) | 3972.3 (▲38% ⚠) | 442.66 | 0.74 | 23.1% |
| stage_12 | Go-Live | 80 | 1264 | 18 | 2767.3 (▲16%) | 2767.3 (▲16%) | 4132.0 (▲44% ⚠) | 442.88 | 0.74 | 23.1% |
| stage_c1 | Producti | 80 | 1264 | 18 | 2649.8 (▲11%) | 2649.8 (▲11%) | 3848.6 (▲34% ⚠) | 443.21 | 0.74 | 23.1% |
| stage_c2 | Producti | 65 | 1327 | 19 | 2923.5 (▲23% ⚠) | 2923.5 (▲23% ⚠) | 11561.0 (▲302% ⚠) | 443.08 | 0.72 | 27.3% |

## ⚠️ Alerts Triggered

| Alert | Stage | Session | Detail |
|-------|-------|---------|--------|
| HIGH_LATENCY | stage_2 | stage_2_s0_1776857875263 | metric=ttft_partial_ms | value_ms=5371.5 | baseline_ms=2384.7 | regression_pct=125.2 | threshold_pct=20 |
| HIGH_LATENCY | stage_2 | stage_2_s0_1776857875263 | metric=ttft_final_ms | value_ms=5848.7 | baseline_ms=2875.8 | regression_pct=103.4 | threshold_pct=20 |
| HIGH_LATENCY | stage_2 | stage_2_s4_1776857875267 | metric=ttft_partial_ms | value_ms=5416.4 | baseline_ms=2384.7 | regression_pct=127.1 | threshold_pct=20 |
| HIGH_LATENCY | stage_2 | stage_2_s4_1776857875267 | metric=ttft_final_ms | value_ms=5942.2 | baseline_ms=2875.8 | regression_pct=106.6 | threshold_pct=20 |
| HIGH_LATENCY | stage_2 | stage_2_s1_1776857875264 | metric=ttft_partial_ms | value_ms=5688.3 | baseline_ms=2384.7 | regression_pct=138.5 | threshold_pct=20 |
| HIGH_LATENCY | stage_2 | stage_2_s1_1776857875264 | metric=ttft_final_ms | value_ms=6182.2 | baseline_ms=2875.8 | regression_pct=115.0 | threshold_pct=20 |
| HIGH_LATENCY | stage_2 | stage_2_s3_1776857875266 | metric=ttft_partial_ms | value_ms=5440.9 | baseline_ms=2384.7 | regression_pct=128.2 | threshold_pct=20 |
| HIGH_LATENCY | stage_2 | stage_2_s3_1776857875266 | metric=ttft_final_ms | value_ms=5943.6 | baseline_ms=2875.8 | regression_pct=106.7 | threshold_pct=20 |
| HIGH_LATENCY | stage_2 | stage_2_s2_1776857875265 | metric=ttft_partial_ms | value_ms=5440.5 | baseline_ms=2384.7 | regression_pct=128.1 | threshold_pct=20 |
| HIGH_LATENCY | stage_2 | stage_2_s2_1776857875265 | metric=ttft_final_ms | value_ms=5912.0 | baseline_ms=2875.8 | regression_pct=105.6 | threshold_pct=20 |
| HIGH_LATENCY | stage_4b | stage_4b_1776859767041 | metric=ttft_final_ms | value_ms=3735.7 | baseline_ms=2875.8 | regression_pct=29.9 | threshold_pct=20 |
| HIGH_LATENCY | stage_4c | stage_4c_1776860210220 | metric=ttft_final_ms | value_ms=5161.5 | baseline_ms=2875.8 | regression_pct=79.5 | threshold_pct=20 |
| HIGH_LATENCY | stage_5 | stage_5_1776860653572 | metric=ttft_final_ms | value_ms=3516.9 | baseline_ms=2875.8 | regression_pct=22.3 | threshold_pct=20 |
| HIGH_LATENCY | stage_6 | stage_6_1776861096508 | metric=ttft_final_ms | value_ms=3578.3 | baseline_ms=2875.8 | regression_pct=24.4 | threshold_pct=20 |
| HIGH_LATENCY | stage_7a | stage_7a_1776861539643 | metric=ttft_final_ms | value_ms=3916.8 | baseline_ms=2875.8 | regression_pct=36.2 | threshold_pct=20 |
| HIGH_LATENCY | stage_7b | stage_7b_1776861982587 | metric=ttft_partial_ms | value_ms=3516.0 | baseline_ms=2384.7 | regression_pct=47.4 | threshold_pct=20 |
| HIGH_LATENCY | stage_7b | stage_7b_1776861982587 | metric=ttft_final_ms | value_ms=4657.8 | baseline_ms=2875.8 | regression_pct=62.0 | threshold_pct=20 |
| HIGH_LATENCY | stage_7c | stage_7c_1776862425236 | metric=ttft_final_ms | value_ms=3630.0 | baseline_ms=2875.8 | regression_pct=26.2 | threshold_pct=20 |
| HIGH_LATENCY | stage_8 | stage_8_1776862868148 | metric=ttft_final_ms | value_ms=3877.9 | baseline_ms=2875.8 | regression_pct=34.8 | threshold_pct=20 |
| HIGH_LATENCY | stage_9 | stage_9_run1 | metric=ttft_final_ms | value_ms=3659.3 | baseline_ms=2875.8 | regression_pct=27.2 | threshold_pct=20 |
| HIGH_LATENCY | stage_9 | stage_9_run2 | metric=ttft_final_ms | value_ms=3886.2 | baseline_ms=2875.8 | regression_pct=35.1 | threshold_pct=20 |
| HIGH_LATENCY | stage_9 | stage_9_run3 | metric=ttft_final_ms | value_ms=3814.5 | baseline_ms=2875.8 | regression_pct=32.6 | threshold_pct=20 |
| HIGH_LATENCY | stage_9_tight | stage_9_tight_1776864641454 | metric=ttft_partial_ms | value_ms=4167.5 | baseline_ms=2384.7 | regression_pct=74.8 | threshold_pct=20 |
| HIGH_LATENCY | stage_9_tight | stage_9_tight_1776864641454 | metric=ttft_final_ms | value_ms=5273.5 | baseline_ms=2875.8 | regression_pct=83.4 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s0_1776865084627 | metric=ttft_partial_ms | value_ms=3447.7 | baseline_ms=2384.7 | regression_pct=44.6 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s0_1776865084627 | metric=ttft_final_ms | value_ms=3880.5 | baseline_ms=2875.8 | regression_pct=34.9 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s2_1776865528176 | metric=ttft_partial_ms | value_ms=2869.3 | baseline_ms=2384.7 | regression_pct=20.3 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s0_1776865972225 | metric=ttft_partial_ms | value_ms=5706.6 | baseline_ms=2384.7 | regression_pct=139.3 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s0_1776865972225 | metric=ttft_final_ms | value_ms=6283.7 | baseline_ms=2875.8 | regression_pct=118.5 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s1_1776865972225 | metric=ttft_partial_ms | value_ms=5388.7 | baseline_ms=2384.7 | regression_pct=126.0 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s1_1776865972225 | metric=ttft_final_ms | value_ms=5913.0 | baseline_ms=2875.8 | regression_pct=105.6 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s4_1776865972228 | metric=ttft_partial_ms | value_ms=5449.6 | baseline_ms=2384.7 | regression_pct=128.5 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s4_1776865972228 | metric=ttft_final_ms | value_ms=5894.8 | baseline_ms=2875.8 | regression_pct=105.0 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s2_1776865972226 | metric=ttft_partial_ms | value_ms=5721.0 | baseline_ms=2384.7 | regression_pct=139.9 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s2_1776865972226 | metric=ttft_final_ms | value_ms=6258.4 | baseline_ms=2875.8 | regression_pct=117.6 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s3_1776865972227 | metric=ttft_partial_ms | value_ms=5435.7 | baseline_ms=2384.7 | regression_pct=127.9 | threshold_pct=20 |
| HIGH_LATENCY | stage_10 | stage_10_s3_1776865972227 | metric=ttft_final_ms | value_ms=6006.6 | baseline_ms=2875.8 | regression_pct=108.9 | threshold_pct=20 |
| HIGH_LATENCY | stage_11 | stage_11_1776866415757 | metric=ttft_final_ms | value_ms=3972.3 | baseline_ms=2875.8 | regression_pct=38.1 | threshold_pct=20 |
| HIGH_LATENCY | stage_12 | stage_12_1776866858772 | metric=ttft_final_ms | value_ms=4132.0 | baseline_ms=2875.8 | regression_pct=43.7 | threshold_pct=20 |
| HIGH_LATENCY | stage_c1 | stage_c1_1776867301950 | metric=ttft_final_ms | value_ms=3848.6 | baseline_ms=2875.8 | regression_pct=33.8 | threshold_pct=20 |
| HIGH_LATENCY | stage_c2 | stage_c2_1776867745521 | metric=ttft_partial_ms | value_ms=2923.5 | baseline_ms=2384.7 | regression_pct=22.6 | threshold_pct=20 |
| HIGH_LATENCY | stage_c2 | stage_c2_1776867745521 | metric=ttft_final_ms | value_ms=11561.0 | baseline_ms=2875.8 | regression_pct=302.0 | threshold_pct=20 |
## ✅ Production Recommendation


**Use Stage C1 (Combined Best)** for production.

| Component | Config | Reason |
|-----------|--------|--------|
| Language | Locked (Stage 1) | Eliminates auto-detect latency |
| Profanity | Raw (Stage 1) | Full transcript fidelity |
| VAD | EndSilence=1200ms (Stage 4b) | Natural pause tolerance |
| Phrase Boost | 30+ domain phrases (Stage 5) | Domain accuracy |
| Numeric | Dictation + PP (Stage 7c) | Digit output, 'to' never '2' |
| Spanish | Pass-through | Spanish words preserved |

**Stage C2** if audio has very long pauses (>1.5s within a sentence).
---

## 📝 Final Transcripts

> Shown once here. 'processed' = after numeric PP (where applied).

### Stage 0 — Baseline

```
My sister-in-law, you know, she, she's a linguist. And so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. El cambio de un idioma al otro. So if you see me this thing, i look kind of goofy. Records it on como los chips de la camarida. You know the tiny little One. And then it, you know. She has some sort of Program because she's doing, you know, she's got like a Grant or something. And then I guess they figure out, you know. Whatever she does something to do with transcribing and Spanglish and you know, OK and everything. Right, right. I don't know. Personal or something so. Y tu amiga. Y tú no puedes salir en esa vaina. Hello, sigue. 4 wheel drive. Nieve mi hermana vive en George, no digo nada de nieve. Yo no solo está en la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta que vivas. ¿Pero no cara a qué hora te llamo a las 8:00 de la mañana? ¿Pero cómo sí Ah anoche? ¿La mitad del este? No, no, no, eso están en medio del Estado, estado al norte o al sur de gameloft. It's a horse country. It's a horse country. It's really pretty actually when you, I think it's before Greensboro, I think it's before Gainesville. When you, when you drive on 75, I think it's 75. It's real pretty. I mean, you can see the big. Que parece que estuvieras en Virginia. Where they have the horse, his horse country. You see the big because. Has a little bit of hills, not like a lot of hills. 75 right? No, no, no. Entonces. Like horse country. So you see the big prop properties with, you know, you know, the trees and you know, you see the horses. It's very pretty actually. I mean. Ella dice que lo tenían puesto, ayer también y salieron los 2 que están perfectos. Good Job, dear. Que como una vez que y no el el tamaño no lo pudiste escoger mejor porque no es posible. Right. It's so cute. It's very cute. Todo bien que a mi.se me parece a la madre. Si tú ves la la si tú ves la foto, la madre la. There it. Very cute. Coño tengo hace frío, pero yo no, yo no oigo el calor, tú sabes, digo lo huelo, tú sabes como tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. Because of the windows hidden Northside so it doesn't get any sun. Bueno. ¿Que es la o no? Cóbrale. Qué fresco fresco. A los mandas solo y él no viene. El que fresco una. Boy, not too bad. Right, right, right, right, right, right, right, right. Porque estuviera escrito en lenguaje de todo El Mundo, maybe, pero. Pues sicomo.se llama Michelle yo me paro y yo coño qué frío hace. En el 1661, lejano. Yo lo siento mucho, hace demasiado frío. Yo sé que ya para las a cuarto de la tarde no va a hacer frío. ¿Voy, vamos a prender la calefacción y me dice Allison, tú crees que tiene? Cuando.se entró Michel. Y fue para allá y me miraneo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no tengo nada que ver. Si ha sido un comentario yo voy a decir yo hacía mucho frío. Se levanta los 5 minutos lizeth y dice, ahora dice 63. Le digo, bueno, por lo menos subió 2 degrees because it used to be 61. Y acá decía 74 lego. Eso lo tocó él porque Allison y yo puse. Ale señor, lo pusimos en 70. Nos ha quitado la chaqueta, así que también tiene frío. Y eso quería entrar. Por lo menos les cae el sol. Pero aquí en el north side. We don't ever get Direct Sun. We get no Direct Sun we get. 61. ¿Cuándo regresa? Guys, will I warm up the Las angres a leap? He doesn't miss it, in other words. Hace como 5 años. That's what i said about 5 years. Mañana que salí de la casa que tenía que ir al súper que o sea no milk mi carro tiene un thermasta que te dice you know, you hit it y te dice what it is. 36 yo man cago en la madre cuando ya regresé. De grosse decía 41, pero cuando i dropped a groceries and i let decía 39. Maybe no, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. The only way I can turn the Heat is I have to tell it auto and turn the temperature up because then the sensor will know that it's cold outside and it throws hot air like you don't you can have and it takes time versus the Old fashioned you turn the knob to the red when the core is warm, it throws hot air. Because your engine, your engines, your engine is too cold. You know, the first time I went to use this heat a few years ago, I'm like, where's where does it say heat? You know, and then right, because it's auto electrical, right? See that chance that you can hit on and you turn the you know you the. I don't like that, that it's kind of weird. I don't like it. I mean, I like the old fashioned knobs, you know? All right, I mean, it warms up, but it's weird. It's it's, it's, it's not, it's counterintuitive because you're used to saying, OK, I want warm. Déjame decirte mi carro tiene. Nunca lo he usado. Ayer que me mondé con María, fuimos a comer. Ella nos prendió y dije, ay, coño, María, tengo el fuaz calientito. Me dice, no, es que tienen el el hirington. So when i got in the car yesterday i went, oh, mine, has it too click. I turned it on. It's i didn't turn it on this morning 'cause i was so bundled up, but i didn't eat it. Right. And it's supposedly. So I was like, cool the first time I ever use it. But you know what? I didn't use it this morning because I was so bundled up I didn't need it. No, I didn't even turn the heat on because I figured I was going to like. The Walter. Gordi es muy buen marido. Me acuerdo que tú me dijiste. ¿El enfermero no? Oh, that's not fair. That's not Fair. Look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? That's not fair. Do something about it and say that they. De todo. Right, right, right. Dad thought you can't do that. Right. Right, right, right. That'll give you a whole lot. Yo no sé cómo la gente vive de unemployment. Yo no sé cómo la gente vive de un employee porque es un país. Llama la única vez que. Is that coming here, Jim? Can I draw? That's good. No gets along with him. That's great. Right E for Kristen is good because. One of the endo Papa, which could be a little could be a little challenging. That's great. That's great. Which one? I want to play.
```

### Stage 1 — ASR Config Finalization

```
My sister-in-law, you know, she, she's a linguist. And so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. El cambio de un idioma al otro. So if you see me this thing, i look kind of goofy. Records it on como los chips de la camarida. You know the tiny little One. And then it, you know. She has some sort of Program because she's doing, you know, she's got like a Grant or something. And then I guess they figure out, you know. Whatever she does something to do with transcribing and Spanglish and you know, OK and everything. Right, right. I don't know. Personal or something so. Y tu amiga. Y tú no puedes salir en esa vaina. Hello, sigue. 4 wheel drive. Nieve mi hermana vive en George, no digo nada de nieve. Yo no solo está en la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta que vivas. ¿Pero no cara a qué hora te llamo a las 8:00 de la mañana? ¿Pero cómo sí Ah anoche? ¿La mitad del este? No, no, no, eso están en medio del Estado, estado al norte o al sur de gameloft. It's a horse country. It's a horse country. It's really pretty actually when you, I think it's before Greensboro, I think it's before Gainesville. When you, when you drive on 75, I think it's 75. It's real pretty. I mean, you can see the big. Que parece que estuvieras en Virginia. Where they have the horse, his horse country. You see the big because. Has a little bit of hills, not like a lot of hills. 75 right? No, no, no. Entonces. Like horse country. So you see the big prop properties with, you know, you know, the trees and you know, you see the horses. It's very pretty actually. I mean. Ella dice que lo tenían puesto, ayer también y salieron los 2 que están perfectos. Good Job, dear. Que como una vez que y no el el tamaño no lo pudiste escoger mejor porque no es posible. Right. It's so cute. It's very cute. Todo bien que a mi.se me parece a la madre. Si tú ves la la si tú ves la foto, la madre la. There it. Very cute. Coño tengo hace frío, pero yo no, yo no oigo el calor, tú sabes, digo lo huelo, tú sabes como tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. Because of the windows hidden Northside so it doesn't get any sun. Bueno. ¿Que es la o no? Cóbrale. Qué fresco fresco. A los mandas solo y él no viene. El que fresco una. Boy, not too bad. Right, right, right, right, right, right, right, right. Porque estuviera escrito en lenguaje de todo El Mundo, maybe, pero. Pues sicomo.se llama Michelle yo me paro y yo coño qué frío hace. En el 1661, lejano. Yo lo siento mucho, hace demasiado frío. Yo sé que ya para las a cuarto de la tarde no va a hacer frío. ¿Voy, vamos a prender la calefacción y me dice Allison, tú crees que tiene? Cuando.se entró Michel. Y fue para allá y me miraneo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no tengo nada que ver. Si ha sido un comentario yo voy a decir yo hacía mucho frío. Se levanta los 5 minutos lizeth y dice, ahora dice 63. Le digo, bueno, por lo menos subió 2 degrees because it used to be 61. Y acá decía 74 lego. Eso lo tocó él porque Allison y yo puse. Ale señor, lo pusimos en 70. Nos ha quitado la chaqueta, así que también tiene frío. Y eso quería entrar. Por lo menos les cae el sol. Pero aquí en el north side. We don't ever get Direct Sun. We get no Direct Sun we get. 61. ¿Cuándo regresa? Guys, will I warm up the Las angres a leap? He doesn't miss it, in other words. Hace como 5 años. That's what i said about 5 years. Mañana que salí de la casa que tenía que ir al súper que o sea no milk mi carro tiene un thermasta que te dice you know, you hit it y te dice what it is. 36 yo man cago en la madre cuando ya regresé. De grosse decía 41, pero cuando i dropped a groceries and i let decía 39. Maybe no, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. The only way I can turn the Heat is I have to tell it auto and turn the temperature up because then the sensor will know that it's cold outside and it throws hot air like you don't you can have and it takes time versus the Old fashioned you turn the knob to the red when the core is warm, it throws hot air. Because your engine, your engines, your engine is too cold. You know, the first time I went to use this heat a few years ago, I'm like, where's where does it say heat? You know, and then right, because it's auto electrical, right? See that chance that you can hit on and you turn the you know you the. I don't like that, that it's kind of weird. I don't like it. I mean, I like the old fashioned knobs, you know? All right, I mean, it warms up, but it's weird. It's it's, it's, it's not, it's counterintuitive because you're used to saying, OK, I want warm. Déjame decirte mi carro tiene. Nunca lo he usado. Ayer que me mondé con María, fuimos a comer. Ella nos prendió y dije, ay, coño, María, tengo el fuaz calientito. Me dice, no, es que tienen el el hirington. So when i got in the car yesterday i went, oh, mine, has it too click. I turned it on. It's i didn't turn it on this morning 'cause i was so bundled up, but i didn't eat it. Right. And it's supposedly. So I was like, cool the first time I ever use it. But you know what? I didn't use it this morning because I was so bundled up I didn't need it. No, I didn't even turn the heat on because I figured I was going to like. The Walter. Gordi es muy buen marido. Me acuerdo que tú me dijiste. ¿El enfermero no? Oh, that's not fair. That's not Fair. Look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? That's not fair. Do something about it and say that they. De todo. Right, right, right. Dad thought you can't do that. Right. Right, right, right. That'll give you a whole lot. Yo no sé cómo la gente vive de unemployment. Yo no sé cómo la gente vive de un employee porque es un país. Llama la única vez que. Is that coming here, Jim? Can I draw? That's good. No gets along with him. That's great. Right E for Kristen is good because. One of the endo Papa, which could be a little could be a little challenging. That's great. That's great. Which one? I want to play.
```

### Stage 1b — ASR Config: Telephony 8kHz Format

```
My sister-in-law, you know, she, she's a linguist. And so she wants to sort of track, you know, Spanglish. Maybe I would wear it at work. Cambio de un idioma al otro. Records. Como los chips de la camarita. She has some sort of Program because she's doing, you know, she's got like a Grant or something. And then I guess they figure out, you know. Whatever she does something to do with transcribing and Spanglish and you know, OK and everything. Right, right. But i don't know. Si, si. Yo sé que estamos un gino todo mundo hablando inglés o como dice mi hermano, si tú no quieres que nadie, oye, no personal something. Y tu amiga. Y tú no puedes salir en esa vaina. Hello. Sigue en 4 wheel drive. Mi mi hermana vive en Georgia y no dio nada de nieve. Bueno, yo no soy la casa de ella. Bueno, puede que si en las montañas. Más muerta que viva. Pero no cara a qué hora te llamo a las 8:00 de la mañana, pero no como.se llama Ah noche. ¿Eso queda como por la mitad del este? No, no, no, eso están en medio del Estado. Estado al norte o al sur de gamezo. It's a horse country. It's a horse country. It's really pretty actually when you, I think it's before going to, but I think it's before Gainesville when you when you drive on 75, I think it's 75. It's real pretty. I mean, you can see the big. ¿Con que parece que estuvieras en Virginia? Where they have the horse, it's horse country. You see the big because has a little bit of hills, not like a lot of hills. I'm 75, right? No no, no, no. Entonces, it's like horse country, so you see the big prop properties with, you know, the trees and you know, you see the horses. It's very pretty actually, I mean. Dice que lo tenían puesto ayer también y salieron los 2 que están. Perfect. Good Job, dear. Que como una vez que y no el tamaño no lo pudiste escoger mejor porque no es posible. Right. It's so cute. It's very cute. Bien que a mi.se me parece a la madre, si tu ves la la si tu ves la foto la madre las. Very cute. Coño tengo hace frío, pero yo no, yo no oigo el calor, tú sabes, digo lo huelo, tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. Hidden Northside so it doesn't get any sun. Bueno. ¿Qué lado o no? Cóbrale. Que fresco. Delincuen. Manda solo y él no viene. Que frescura. Boy, not too bad. Say right, right, right, right, right, right, right, right, right. Porque estuvieran escrito en lenguaje de todo El Mundo, maybe, pero. Pues sicomo.se llama Michelle yo me paro y yo coño qué frío hace. En el 1661 lejano, yo lo siento mucho, hace demasiado frío. ¿Yo sé que ya para las a cuarto de la tarde no va a hacer frío, voy vamos a prender la calefacción y me dice, Allison, tú crees que tiene? Que no hay colección. Lo más probable que no haya. En el centro de Michelle. Y fue para allá y me mira neo y porque yo le dije que yo lo he aprendido y los hicimos yo me hice la que yo no tengo nada que ver si ha sido un comentario yo voy a ver si si yo hacía mucho frío. Se levanta los 5 minutos lizeth y dice ahora dice 63. Le digo, bueno, por lo menos subió 2 degrees que.se used to be 61 y acá decía 70 for lego. A eso lo tocó él porque Allison y yo puse. Ale señor, lo pusimos en 70. No se ha quitado la chaqueta, así que también tiene frío. Y eso quería entrar. Por lo menos les cae el sol. Pero aquí en el north side. We don't never get Direct Sun. We get no Direct Sun. 61. ¿Cuándo regresa? Local. Yeah, as well, I warm up the La Sangre de Labo. He doesn't miss it, in other words. Hace como 5 años. Que salí de la casa, que tenía que ir al súper, que es mi carro. Tiene un thermasta que te dice y te dice what it is 36. Cuando ya regresé de. De grosse decía 41, pero cuando groceries and i let decía 39. Maybe Nokate has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. Lor, tú lo mueves a rojito. The only way i can turn the Heat is i have to tell it auto and turn the temperature up because then the sensor will know that it's cold outside and it throws hot air like you don't. You can have and it takes time versus the Old fashioned. You turn the knob to the red. When the cores are warm, it throws hot air. You're concerned. Your engine, your engines, your engines to pull. You know, the first time I went to use this heat a few years ago, I'm like, where's where does it say heat? And then right, because it's auto electrical, right? That chance that you can hit on and you turn the you know you. I don't like that that, that it's kind of weird. I don't like it. I mean, I like the old fashioned knobs, you know? All right, I mean, it warms up, but it's weird. It's it's, it's, it's not, it's counterintuitive because you're used to saying, OK, I want warm. Déjame decirte mi carro tiene. Que lo he usado. Ayer que me mondé con María. Fuimos a comer. Ella nos prendió y dije, ay, coño, María, tengo el fuaz calientito. Me dice, no, es que tienen el. So when i got in the car yesterday, i went, oh, mine has a true click. I turned it on. It's i didn't turn it on this morning because i was so bundled up, but i didn't need it. Right. And it's supposedly. So I was like, cool the first time I ever use it. But you know what? I didn't use it this morning because I was so bundled up I didn't need it. No, I didn't even turn the heat on because I figured I was going to like. Kristen. The Walter is driving. Gordi es muy buen marido. Me acuerdo que tú me dijiste. ¿El enfermero no? That's not fair. That's not fair. That's not fair. Say hello, look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? That's not fair. Do something about it and say that they. De todo un poco. Right, right, right. Dad thought you can't do that. Right, right, right. Let me give you a whole lot. Yo no sé cómo la gente vive de unemployment. Yo no sé cómo la gente vive de un employee porque es un país. 6. Is that coming here, Jim? Can I draw? That's good. No gets along with him. That's great. Right AE for Kristen is good because. One of the end of which could be a little could be a little challenging. That's great. That's great. Which one? What's going on from?
```

### Stage 2 — Concurrency & Quota Validation

```
(empty)
```

### Stage 3 — Real-Time Socket Integration

```
My sister-in-law, you know, she, she's a linguist. And so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. El cambio de un idioma al otro. So if you see me with this thing, i look kind of goofy. Records it on como los chips de la camarida. You know the tiny little One. And then it, you know. She has some sort of Program because she's doing, you know, she's got like a Grant or something. Yeah. And then they transcribe it and then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, OK and everything. Right, right. But no. You know. Inglés o como dice mi hermano, si tú no quieres que nadie. Oye, no personal or something. So. Y tu amiga. Y tú no puedes salir en esa vaina. Hello, sigue. 4 wheel drive. Nieve mi hermana vive en George, no digo nada de nieve. Y yo no solo hasta la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta que vivas. ¿Pero en okara a qué hora te llamo a las 8:00 de la mañana? Perocomo.se llama Ah anoche. ¿La mitad del este? No, no, no, eso están en medio del Estado, estado al norte o al sur de Games. It's a horse country. It's a horse country. It's really pretty actually when you, I think it's before Greensboro. I think it's before Gainesville when you when you drive on 75, I think it's 75. It's real pretty. I mean, you can see the big. Que parece que estuvieras en Virginia. Where they have the horse, his horse, country. You see the big cousin. Has a little bit of hills. A lot of hills. 75 right? No, no, no. Entonces. It's like horse country. So you see the big prop properties with, you know, you know the trees and you know, you see the horses. It's very pretty actually. I mean. Ella dice que lo tenían puesto. Ayer también y salieron los 2 que están. Perfect. Good Job, dear. Que como una vez que y no el el tamaño no lo pudiste escoger mejor porque no es posible. It's so cute. It's very cute. Qué. Todo bien que a mi.se me parece a la madre. Si tú ves la la si tú ves la foto, la madre la. Unfair. Very cute. Coño tengo hace frío, pero yo no, yo no oigo el calor, tú sabes, digo lo huelo, tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. Because of the windows hidden Northside so it doesn't get any sun. Bueno. ¿Que el lado o no? Cóbrale. Qué fresco fresco. A lo manda solo y él no viene. El que fresco una. Boy, not too bad. Right, right, right, right, right, right, right, right. Porque estuviera escrito en lenguaje de todo El Mundo, maybe, pero. Pues sicomo.se llama Michelle yo me paro y yo coño qué frío hace. En el 61 hace demasiado frío. Yo sé que ya para las a cuarto de la tarde no va a hacer frío. ¿Voy, vamos a prender la calefacción y me dice Allison, tú crees que tiene? Lo más probable que no haya. Cuando centro de Michel. Y fue para allá y me miraneo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no tengo nada que ver. Si ha sido un comentario yo voy a decir si yo hacía mucho frío. Se levanta los 5 minutos lizeth y dice, ahora dice 63. Le digo, bueno, por lo menos subió 2 degrees because it used to be 61. Y acá decía 74 lego. Eso lo tocó él porque Allison y yo puse. Ale señor, lo pusimos en 70. No se ha quitado la chaqueta, así que también tiene frío. Y eso quería entrar. Por lo menos les cae el sol. Pero aquí en el north side. We don't ever get Direct Sun. We get no Direct Sun we get. 61. ¿Cuándo regresa? Guys, will I warm up the Las angres a leap? He doesn't miss it, in other words. Hace como 5 años. That's what i said about 5 years. Mañana que salí de la casa que tenía que ir al súper que o sea no milk. Mi carro tiene un thomasta que te dice you know, you hit it y te dice what it is. 36 yo man cago en la madre fue cuando ya regresé. De grosse. Decía 41, pero cuando i dropped a groceries and i let decía 39 coño. Maybe no, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. The only way I can turn the Heat is I have to tell it auto and turn the temperature up because in the sensor will know that it's cold outside and it throws hot air like you don't. You can and it takes time versus the Old fashioned you turn the knob to the red when the core is warm it throws hot air. Because your engine, your engines, your engine's too cold. You know, the first time I went to use this heat a few years ago, I'm like, where's where does it say heat? You know, and then right, because it's auto electrical, right? That chance that you can hit on and you turn the you know you. I don't like that, that it's kind of weird. I don't like it. I mean, I like the old fashioned knobs, you know? All right, I mean, it warms up, but it's weird. It's it's, it's, it's not, it's counterintuitive because you're used to saying, OK, I want warm. Déjame decirte mi carro tiene. Que lo he usado. Ayer que me mondé con María. Fuimos a comer. Ella nos prendió y dije, ay, coño, María, tengo el fuaz calientito. Me dice, no, es que tienen el el heating. So when i got in the car yesterday, i went, oh, mine has it too click. I turned it on. It's i didn't turn it on this morning 'cause i was so bundled up i didn't need it. Right. And it's supposedly. So I was like, cool the first time I ever use it. But you know what? I didn't use it this morning because I was so bundled up I didn't need it. No, I didn't even turn the heat on because I figured I was going to like. The Walter is stopping in the drive. Gordi es muy buen marido. Me acuerdo que tú me dijiste. ¿El enfermero no? Oh, that's not fair. That's not Fair. Look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? That's not fair. Do something about it and say that they. De todo. Right, right, right. Dad thought you can't do that. Right. Right, right, right. That'll give you a whole lot. Yo no sé cómo la gente vive de unemployment. Yo no sé cómo la gente vive de un employee porque es un país. Llama la única vez que. Is that coming here, Jim? Can I draw? That's good. No gets along with him. That's great. Right E for Kristen is good because. One of the endo Papa, which could be a little could be a little challenging. That's great. That's great. Which one? I want to play.
```

### Stage 4a — VAD: Default (800ms)

```
My sister-in-law, you know, she, she's a linguist. And so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. El cambio de un idioma al otro. So if you see me this thing, i look kind of goofy. Records it on como los chips de la camarida. You know the tiny little One. And then it, you know. She has some sort of Program because she's doing, you know, she's got like a Grant or something. And then I guess they figure out, you know. Whatever she does something to do with transcribing and Spanglish and you know, OK and everything. Right, right. I don't know. Personal or something so. Y tu amiga. Y tú no puedes salir en esa vaina. Hello, sigue. 4 wheel drive. Nieve mi hermana vive en George, no digo nada de nieve. Yo no solo está en la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta que vivas. ¿Pero no cara a qué hora te llamo a las 8:00 de la mañana? ¿Pero cómo sí Ah anoche? ¿La mitad del este? No, no, no, eso están en medio del Estado, estado al norte o al sur de gameloft. It's a horse country. It's a horse country. It's really pretty actually when you, I think it's before Greensboro, I think it's before Gainesville. When you, when you drive on 75, I think it's 75. It's real pretty. I mean, you can see the big. Que parece que estuvieras en Virginia. Where they have the horse, his horse country. You see the big because. Has a little bit of hills, not like a lot of hills. 75 right? No, no, no. Entonces. Like horse country. So you see the big prop properties with, you know, you know, the trees and you know, you see the horses. It's very pretty actually. I mean. Ella dice que lo tenían puesto, ayer también y salieron los 2 que están perfectos. Good Job, dear. Que como una vez que y no el el tamaño no lo pudiste escoger mejor porque no es posible. Right. It's so cute. It's very cute. Todo bien que a mi.se me parece a la madre. Si tú ves la la si tú ves la foto, la madre la. There it. Very cute. Coño tengo hace frío, pero yo no, yo no oigo el calor, tú sabes, digo lo huelo, tú sabes como tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. Because of the windows hidden Northside so it doesn't get any sun. Bueno. ¿Que es la o no? Cóbrale. Qué fresco fresco. A los mandas solo y él no viene. El que fresco una. Boy, not too bad. Right, right, right, right, right, right, right, right. Porque estuviera escrito en lenguaje de todo El Mundo, maybe, pero. Pues sicomo.se llama Michelle yo me paro y yo coño qué frío hace. En el 1661, lejano. Yo lo siento mucho, hace demasiado frío. Yo sé que ya para las a cuarto de la tarde no va a hacer frío. ¿Voy, vamos a prender la calefacción y me dice Allison, tú crees que tiene? Cuando.se entró Michel. Y fue para allá y me miraneo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no tengo nada que ver. Si ha sido un comentario yo voy a decir yo hacía mucho frío. Se levanta los 5 minutos lizeth y dice, ahora dice 63. Le digo, bueno, por lo menos subió 2 degrees because it used to be 61. Y acá decía 74 lego. Eso lo tocó él porque Allison y yo puse. Ale señor, lo pusimos en 70. Nos ha quitado la chaqueta, así que también tiene frío. Y eso quería entrar. Por lo menos les cae el sol. Pero aquí en el north side. We don't ever get Direct Sun. We get no Direct Sun we get. 61. ¿Cuándo regresa? Guys, will I warm up the Las angres a leap? He doesn't miss it, in other words. Hace como 5 años. That's what i said about 5 years. Mañana que salí de la casa que tenía que ir al súper que o sea no milk mi carro tiene un thermasta que te dice you know, you hit it y te dice what it is. 36 yo man cago en la madre cuando ya regresé. De grosse decía 41, pero cuando i dropped a groceries and i let decía 39. Maybe no, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. The only way I can turn the Heat is I have to tell it auto and turn the temperature up because then the sensor will know that it's cold outside and it throws hot air like you don't you can have and it takes time versus the Old fashioned you turn the knob to the red when the core is warm, it throws hot air. Because your engine, your engines, your engine is too cold. You know, the first time I went to use this heat a few years ago, I'm like, where's where does it say heat? You know, and then right, because it's auto electrical, right? See that chance that you can hit on and you turn the you know you the. I don't like that, that it's kind of weird. I don't like it. I mean, I like the old fashioned knobs, you know? All right, I mean, it warms up, but it's weird. It's it's, it's, it's not, it's counterintuitive because you're used to saying, OK, I want warm. Déjame decirte mi carro tiene. Nunca lo he usado. Ayer que me mondé con María, fuimos a comer. Ella nos prendió y dije, ay, coño, María, tengo el fuaz calientito. Me dice, no, es que tienen el el hirington. So when i got in the car yesterday i went, oh, mine, has it too click. I turned it on. It's i didn't turn it on this morning 'cause i was so bundled up, but i didn't eat it. Right. And it's supposedly. So I was like, cool the first time I ever use it. But you know what? I didn't use it this morning because I was so bundled up I didn't need it. No, I didn't even turn the heat on because I figured I was going to like. The Walter. Gordi es muy buen marido. Me acuerdo que tú me dijiste. ¿El enfermero no? Oh, that's not fair. That's not Fair. Look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? That's not fair. Do something about it and say that they. De todo. Right, right, right. Dad thought you can't do that. Right. Right, right, right. That'll give you a whole lot. Yo no sé cómo la gente vive de unemployment. Yo no sé cómo la gente vive de un employee porque es un país. Llama la única vez que. Is that coming here, Jim? Can I draw? That's good. No gets along with him. That's great. Right E for Kristen is good because. One of the endo Papa, which could be a little could be a little challenging. That's great. That's great. Which one? I want to play.
```

### Stage 4b — VAD: Conservative (1200ms)

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. So if you see me with this thing, I look kind of goofy. Yeah, and records it on the Comodo Chips de la Camarita, you know, the tiny little one. And then it, you know. She has some sort of program because she's doing, you know, she's got like a grant or something. Yeah. And then they transcribe it and then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, OK and everything. Or something. Y Tomé. Hello 4 wheel drive. Nieves, mi hermana. Vive en jordanieve. Bueno, yo no solo está la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta que vivas. ¿Pero en okara a qué hora tenemos a las 8:00 de la mañana, pero no como si Ah anoche eso queda como por la mitad del estedo? ¿No, no, no, eso están en medio del Estado, estado al norte o al sur de Games? Country. It's a horse country. Actually, when you, I think it's before Greensboro, I think it's before Gainesville. When you, when you drive on 75, I think it's 75. It's real pretty. I mean, you can see the big, the pieces of property in Virginia where they have the horse. It's horse country. You see the big cause has a little bit of hills, not like a lot of hills. I-75, right? No, no, no. Entonces. Like horse country. So you see the big prop purities with, you know, you know the trees and you know, you see the horses. It's very pretty actually. I mean. Los 2 ya dice que lo tenían puesto, ayer también y salieron los 2 que están Perfect. Good Job, dear. Una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible. Cute and very cute. Pero todo bien que a mi.se me parece a la madre. Si tú ves las las si tú ves la foto la madre. Very cute. Frío pero yo no, yo no oigo el calor, tú sabes, digo lo huelo, tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. The north side so it doesn't get any sun. Bueno. Con el que vive aquel lado. Cóbrale. Qué fresco fresco. OK. A lo manda solo y él no viene. Oye que fresco una. Boy, not too bad. Right, right, right, right, right, right, right, right. Porque te hubieran escrito en lenguaje de todo El Mundo, maybe. Pues sicomo.se llama Michelle yo me paro y yo coño qué frío hace voy y veo y dice 61 lejano, yo lo siento mucho, hace demasiado frío. ¿Yo sé que ya para las a cuarto de la tarde no va a hacer frío, voy vamos a prender la calefacción y me dice, Allison, tú crees que tiene y? Que no hay calección, lo más probable que no haya. Centro de Michelle. Y fue para allá y me mira a neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no doy una que ves si ha sido un comentario, yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63. Le digo, bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego. Eso lo tocó él porque Allison y yo. Al señor lo pusimos en 70. Sí, pero no, no se ha quitado la chaqueta, así que también tiene frío. Y eso quería entrar. Por lo menos les cae el sol. Pero aquí en el north side. We don't ever get Direct Sun. We get no Direct Sun we get. 61. ¿Cuándo regresa? Él está loco. Yeah, actually, I warm up the Los Angeles. He doesn't miss it, in other words. No. Hace como 5 años. Yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thomasta que te dice you know, you hit it y te dice what it is. 36 yo man cago en la madre fue cuando ya regresé de. 21 pero cuando groceries. And i left. Maybe no, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. Lo mueves a rojito. The only way i can turn the Heat is i have to tell it auto and turn the temperature up because in the sensor will know that it's cold outside and it throws hot air like you don't you can. It takes time versus the Old fashioned you turn the knob to the red. When the core is warm, it throws hot air. It takes time because your engine, your engines, your engines too cold better. The first time I went to use this Heat a few years ago, I'm like, where's where does it say Heat? And then right because it's auto electrical, right? Pero el aire acondicionado si te da chance. You can hit on and you turn the you know you. I don't like, I don't like that. It's kind of weird. I don't like it. I mean, I like the Old fashioned knobs. Right. I mean, it warms up, but it's weird. It's it's, it's, it's not, it's counterintuitive because you're used to saying, OK, I want warm Rojo. Déjame, cite me, carro tiene. Heated seeds. Ayer que me mondé con María, fuimos a comer. Ella nos prendió y dije, ay, coño, María, tengo el fuascoalientino me dice, no, es que tienen el el hearing thing. So when i got in the car yesterday, i went, oh, mine has it too click. I turned it on. It's i didn't turn it on this morning 'cause i was so bundled up i didn't need it. Right. And it's supposedly para la espalda. So i was like, cool the first time i ever use it. But you know what, I didn't use it this morning cuz I was so bundle up I didn't need it. No, I didn't even turn the heat on because I figured I was going to like. Where's Kristen? Walter is stabbing in the driveway and we went Marito. Me acuerdo que tú me dijiste. ¿El enfermero no? Oh, that's not fair, right? That's not fair. That's not fair. Look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? That's not fair. But can't he do something about it and say that they you know? De todo un poco. Right, right. Mm-hmm. Dad thought you can't do that. Right, right, right. Right, right, right. That'll give you a whole lot. Yo no sé cómo la gente vive de unemployment porque es un 10. Yo no sé cómo la gente vive de un employee porque es un país. Yo me hago la única vez que. Is that coming here, Jim? Can I draw? That's good. No gets along with him. That's great. Right E for Kristen is good because. Which could be a little could be a little challenging. That's great. That's great. Which one? Oh.
```

### Stage 4c — VAD: Aggressive (2000ms)

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. I look kind of goofy. Yeah. It records it on a Comodo Chips de la Camarita, you know, that tiny little one. And then it, you know, she has some sort of program because she's doing, you know, she's got like, a grant or something. Yeah. And then they transcribe it. And then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, OK and everything. Right, right. But I don't know, I love the lobby, you know, you know, Tamil Nadu and English or Gomez, you do not get it, you know, personal or something so. Rani Yamaya. Y tú no puedes salir en esa vaina. Hello, sigue. 4 wheel drive. Mi hermana vive en Jordan. No dio nada de nieve. Bueno, yo no solo está en la casa de ella. Bueno, puede que si en las montañas sí, sí haya. Más muerta que vivas. ¿A qué hora te llamo a las 8:00 de la mañana, pero no como si Ah anoche eso queda como por la mitad del estedo? No, no, no, eso están en medio del Estado, estado al norte o al sur de Games. It's a horse country. It's a horse country. It's really pretty actually when you, i think it's before gainesville. I think it's before gainesville. When you when you drive on 75, i think it's 75. It's real pretty. I mean you can see the big, the pieces of Property que parece que tuvieras en Virginia, where they have the horses, horse country. You see the big because has a little bit of hills, not like a lot of hills. No, no, no. Entonces. It's like horse country. So you see the big prop properties with, you know, you know the trees and you know, you see the horses. It's very pretty actually. I mean. Los 2 ya dice que lo tenían puesto, ayer también y salieron los 2 que están Perfect. Good Job, dear. Que una vez que y no el tamaño no lo pudiste escoger mejor porque no es posible. ¿Qué? Pero todo bien que a mi.se me parece a la madre. Si tú ves la la si tú ves la foto, la madre la. Very cute. No oigo el calor, tú sabes digo lo huelo, tú sabes como tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. Side so it doesn't get any sun. ¿Con el que vive aquel lado no? Cóbrale. Qué fresco. Delincuentes. A lo manda solo y él no viene. Oye que fresco una. Boy, not too bad. Right, right, right, right, right, right, right, right. Lenguaje de todo El Mundo. May be, pero. Pues sí el como.se llama Michelle, yo me paro y yo coño qué frío hace voy y veo en el y ese 61 lejano, yo lo siento mucho, hace demasiado frío. ¿Yo sé que ya para las a cuarto de la tarde no va a hacer frío, voy vamos a prender la calefacción y me dice, Allison, tú crees que tiene que no hay calefacción? Lo más probable es que no haya. Cuando centro de Michel y fue para allá y me mira neo y porque yo le dije que yo lo he aprendido y los hicimos yo me hice la que yo no doy una que me decía si un comentario yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63. Le digo bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego. A eso lo tocó él porque Alice. Ale señor, lo pusimos en 70. Sí, pero no, no se ha quitado la chaqueta, así que también tiene frío. Eso quería entrar. Por lo menos les cae el sol, pero aquí en el north side. We don't ever get Direct Sun. We get no Direct Sun we get. 61. ¿Cuándo regresa él está loco? Yeah, actually, I warm up the Los Angeles. In other words. No hace como 5 años. That's what i said about 5 years. Yo esta mañana que salí de la casa que tenía que ir al súper que o sea no meok mi carro tiene un thermasta que te dice you know, you hit it y te dice what it is 36 yo man cago en la madre fue cuando ya regresé de de Grosso decía 41 pero cuando hay drop up groceries and i let decía 39. Maybe no, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. O The only way I can turn the heat is I have to tell it auto and turn the temperature up because then the sensor will know that it's cold outside and it throws hot air like you don't you can and it takes time versus the old fashioned, you turn the knob down to the red when the core is warm, it's throws hot air. It takes time because your engine, your engine's, your engine's too cold. But the first time I went to use this heat a few years ago. Like where's where does it say heat? You know, and then right because it's auto electrical, right? Chance that you can hit on and you turn the you know you I don't like I don't like that that it's kind of weird. I don't like it. I mean, I like the old fashioned knobs, you know. I mean, it warms up, but it's weird. It's, it's, it's, it's not, it's counterintuitive because you're used to saying, OK, I want warm. Heated seats. Nunca lo he usado. Ayer que me mondé con María, fuimos a comer. Ella nos prendió y dije, ay, coño, María, tengo el fuaz calientito me dice, no, es que tienen el el hearing thin. So when i got in the car yesterday, i went, oh, mine has it too click. I turned it on. It's i didn't turn it on this morning because i was so bundled up. I didn't eat it right. And it's supposedly so i was like, cool the first time i ever use it. But you know what, I didn't use it this morning cuz I was so bundle up I didn't need it. No, I didn't even turn the Heat on because I figure I was going to like. Where's Kristen? ¿Con ustedes otra vez? La verdad que gordi es muy bueno marido. Me acuerdo que tú me dijiste. Uh-huh. ¿El enfermero no? Oh, that's not fair, right? That's not fair. That's not fair. Look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? Coño. But that's not Fair. ¿But can't he do something about it and say that they you know? De todo un poco. Right, right, right. Mm-hmm. Dad thought you can't do that. Contigo. Right, right, right. They'll give you a whole lot of unemployment. Yo no sé cómo la gente vive de un employee porque es un país. Yo me hago la única vez que. Is that coming here, Jim? Can I draw right? That's good. No, you see along with him. That's great. Right E for Kristen is good because. One of the endo Papa, which could be a little could be a little challenging. That's great. That's great. Which one?
```

### Stage 5 — Word / Phrase Boosting

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. So if you see me with this thing, I look kind of goofy. Yeah, and records it on a Chips de la Camarita, you know, the tiny little one. And then it, you know. She has some sort of program because she's doing, you know, she's got like a grant or something. Yeah. And then they transcribe it and then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, OK and everything. Or something. Y Tomé. Hello 4 wheel drive. Nieve. Mi hermana vive en Jordan. No digo nada de nieve. Bueno, yo no solo está en la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta que vivas. Pero no cara a qué hora tenemos a las 8:00 de la mañana, pero no como sí Ah noche. ¿Eso queda como por la mitad del estedo? No, no, no, eso están en medio del Estado, estado al norte o al sur de Games. Country. It's a horse country. Actually, when you, I think it's before Greensboro, I think it's before Gainesville. When you, when you drive on 75, I think it's 75. It's real pretty. I mean, you can see the big, the pieces of property in Virginia where they have the horse. It's horse country. You see the big cause has a little bit of hills, not like a lot of hills. I-75, right? No, no, no. Entonces. Like horse country. So you see the big prop purities with, you know, you know the trees and you know, you see the horses. It's very pretty actually. I mean. Los 2 ya dice que lo tenían puesto, ayer también y salieron los 2 que están Perfect. Good Job, dear. Una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible. Cute and very cute. Pero todo bien que a mi.se me parece a la madre. Si tú ves las las si tú ves la foto la madre. Very cute. Frío pero yo no, yo no oigo el calor, tú sabes, digo lo huelo, tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. The north side so it doesn't get any sun. Bueno. ¿Con el que vive aquella o no? Cóbrale. Qué fresco fresco. A lo manda solo y él no viene. Oye que fresco una. Boy, not too bad. Right, right, right, right, right, right, right, right. Porque tuvieran escrito en el lenguaje de todo El Mundo, maybe, pero. Pues sicomo.se llama Michelle yo me paro y digo coño, qué frío hace voy y veo y dice 61 lejano, yo lo siento mucho, hace demasiado frío. ¿Yo sé que ya para las a cuarto de la tarde no va a hacer frío, voy vamos a prender la calefacción y me dice, Allison, tú crees que tiene? Que no hay calección, lo más probable que no haya. Centro de Michelle. Y fue para allá y me mira a neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no doy una que ves si hacía un comentario yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63. Le digo, bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego. Eso lo tocó él porque Allison y yo. Al señor lo pusimos en 70. Sí, pero no, no se ha quitado la chaqueta, así que también tiene frío. Y eso quería entrar. Por lo menos les cae el sol. Pero aquí en el north side. We don't ever get Direct Sun. We get no Direct Sun we get. 61. ¿Cuándo regresa? Él está loco. Yeah, actually, I warm up the Los Angeles. He doesn't miss it, in other words. No. Hace como 5 años. Yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thomasta que te dice you know, you hit it y te dice what it is. 36 yo man cago en la madre fue cuando ya regresé de. 21 pero cuando groceries. And i left. Maybe no, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. Lo mueves a rojito. The only way i can turn the Heat is i have to tell it auto and turn the temperature up because in the sensor will know that it's cold outside and it throws hot air like you don't you can. It takes time versus the Old fashioned you turn the knob to the red. When the core is warm, it throws hot air. It takes time because your engine, your engines, your engines too cold better. The first time I went to use this Heat a few years ago, I'm like, where's where does it say Heat? And then right because it's auto electrical, right? Pero el aire acondicionado si te da chance. You can hit on and you turn the you know you. I don't like, I don't like that. It's kind of weird. I don't like it. I mean, I like the Old fashioned knobs. Right. I mean, it warms up, but it's weird. It's it's, it's, it's not, it's counterintuitive because you're used to saying, OK, I want warm Rojo. Déjame, cite me, carro tiene. Heated seeds. Ayer que me mondé con María, fuimos a comer. Ella nos prendió y dije, ay, coño María, tengo el fuascoalientino me dice, no, es que tienen el el hirington. So when i got in the car yesterday, i went, oh, mine has it too click. I turned it on. It's i didn't turn it on this morning 'cause i was so bundled up i didn't need it. Right. And it's supposedly para la espalda. So i was like, cool the first time i ever use it. But you know what, I didn't use it this morning cuz I was so bundle up I didn't need it. No, I didn't even turn the heat on because I figured I was going to like. Where's Kristen? Walter is stabbing in the driveway and we went Marito. Me acuerdo que tú me dijiste. ¿El enfermero no? Oh, that's not fair, right? That's not fair. That's not fair. Look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? Yeah, well, that's not fair. But can't he do something about it and say that they you know? De todo un poco. Right, right. Mm-hmm. Dad thought you can't do that. Right, right, right. Yeah. Right, right, right. That'll give you a whole lot. Yo no sé cómo la gente vive de unemployment porque es un 10. Yo no sé cómo la gente vive de un employee porque es un país. Yo me hago la única vez que. Is that coming here, Jim? Can I draw? That's good. No gets along with him. That's great. Right E for Kristen is good because. Which could be a little could be a little challenging. That's great. That's great. Which one? I want to train OH.
```

### Stage 6 — Transcript-Based Vocabulary Tuning

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. So if you see me with this thing, I look kind of goofy. Yeah, and records it on the Como Chips de la Camarita, you know, the tiny little one. And then it, you know. She has some sort of program because she's doing, you know, she's got like a grant or something. Yeah. And then they transcribe it and then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, OK, I would like and everything. Right, right. I don't know. You know. I, you know, personal or something so. Y todo mi. Hello 4 wheel drive. Nieve. Mi hermana vive en Jordan. No digo nada de nieve. Bueno, yo no solo está la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta que vivas. Pero no cara a qué hora tenemos a las 8:00 de la mañana, pero no como sí Ah noche. ¿Eso queda como por la mitad del estedo? No, no, no, eso están en medio del Estado, estado al norte o al sur de Games. Country. It's a horse country. Actually, when you, I think it's before Greensboro, I think it's before Gainesville. When you when you drive on 75, I think it's 75. It's real pretty. I mean, you can see the big, the pieces of property in Virginia where they have the horse. It's horse country. You see the big because. A little bit of hills, not like a lot of hills. I-75, right? No, no, no. Entonces. It's like horse country. So you see the big prop purities with, you know, you know the trees and you know, you see the horses. It's very pretty actually. I mean, los 2 ya dice que lo tenían puesto, ayer también y salieron los 2 que están Perfect. Good Job, dear. Una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible. Right. It's so cute. It's very cute. Pero todo bien que a mi.se me parece a la madre. Si tú ves la la si tú ves la foto, la madre la. Very cute. Frío pero yo no, yo no oigo el calor, tú sabes, digo lo huelo, tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. When does the north side so it doesn't get any sun? Bueno. ¿Cuando el que vive aquella o no? Cóbrale. Qué fresco fresco digo. I think quite a bit. Lo manda solo y él no viene. Porque fresco una. Bueno. Too bad. Right. Right, right, right, right, right. Porque tuvieran escrito en lenguaje de todo El Mundo, maybe, pero. Pues sicomo.se llama Michelle yo me paro y digo coño, qué frío hace voy y veo y dice 61 lejano, yo lo siento mucho, hace demasiado frío. ¿Yo sé que ya para las a cuarto de la tarde no hace frío, voy vamos a prender la calefacción y me dice, Allison, tú crees que tiene y? Que no hay calección, lo más probable que no haya. Centro de Michelle. Y fue para allá y me mira a neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no tengo nada que ver. Si ha sido un comentario, yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63. Le digo, bueno, por lo menos subió 2 degrees because it used to be 61 y acá decía 74. Luego eso lo tocó él porque Allison y yo. Al señor lo pusimos en 70. Sí, pero no, no se ha quitado la chaqueta, así que también tiene frío. Y eso quería entrar. Por lo menos les cae el sol. Pero aquí en el north side. We don't ever get Direct Sun. We get no Direct Sun we get. 61. Cuando regresa. Él está loco. Yeah, actually, I warm up the Los Angeles. He doesn't miss it, in other words. No hace como 5 años. That's what i said about 5 years. Yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thomasta que te dice you know, you hit it y te dice what it is. 36 yo man cago en la madre fue cuando ya regresé de. 21 pero cuando. I drop the groceries and i left they're. Maybe no, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. The only way I can turn the Heat is I have to tell it auto and turn the temperature up because then the sensor will know that it's cold outside and it throws hot air like you don't you can have and it takes time versus the Old fashioned you turn the knob to the red when the core is warm, it throws hot air. It takes time because they're your engine, your engines, your engines too cold, better. The first time I went to use this Heat a few years ago, I'm like, where's where does it say Heat? And then right because it's auto electrical, right? Pero el aire acondicionado si te da chance. You can hit on and you turn the you know you. I don't like, I don't like that. It's kind of weird. I don't like it. I mean, I like the Old fashioned knobs. Right. I mean, it warms up, but it's weird. It's it's it's, it's not, it's counterintuitive because you're used to saying, OK, I want warmuar Rojo. Déjame, cite me, carro tiene. Heated seeds. Ayer que me mondé con María, fuimos a comer. Ella nos prendió y dije, ay, coño, María, tengo el fuascoalientino me dice, no, es que tiene el el heating thing. So when i got in the car yesterday i went, oh, mine has it too click. I turned it on. It's i didn't turn it on this morning because i was so bundled up i didn't need it. Right. And it's supposedly para la espalda. So i was like, cool the first time i ever use it. But you know what? I didn't use it this morning because I was so bundled up I didn't need it. No, I didn't even turn the heat on because I figured I was going to like. Where's Kristen? Walter is something in the driveway. I love it. A Gordie and we went Marido. Me acuerdo que tú me dijiste. ¿El enfermero no? Oh, that's not fair, right? That's not fair. That's not fair. Say hi. Look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? Going yeah, well, that's not fair. But can't he do something about it and say that they you know? De todo un poco. Right, right. Mm-hmm. Dad thought you can't do that. Right, right, right. Yeah. Right, right, right. That'll give you a whole lot. Yo no sé como la gente vive de unemployment porque es un 10. Yo no sé como la gente vive de un employee porque es un país. Yo me hago la única vez que. Is that coming here, Jim? Can I draw? That's good. No gets along with him. That's great. Right E for Kristen is good because. Maybe one of the endo Papa, which could be a little could be a little challenging. That's great. That's great. Which one? I want to train.
```

### Stage 7a — Numeric: Conversation Mode (Azure native)

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. So if you see me with this thing, I look kind of goofy. Yeah, and records it on a Chips de la Camarita, you know, the tiny little one. And then it, you know. She has some sort of program because she's doing, you know, she's got like a grant or something. Yeah. And then they transcribe it and then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, OK and everything. Or something. Y Tomé. Hello 4 wheel drive. Nieve. Mi hermana vive en Jordan. No digo nada de nieve. Bueno, yo no solo está en la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta que vivas. Pero no cara a qué hora tenemos a las 8:00 de la mañana, pero no como sí Ah noche. ¿Eso queda como por la mitad del estedo? No, no, no, eso están en medio del Estado, estado al norte o al sur de Games. Country. It's a horse country. Actually, when you, I think it's before Greensboro, I think it's before Gainesville. When you, when you drive on 75, I think it's 75. It's real pretty. I mean, you can see the big, the pieces of property in Virginia where they have the horse. It's horse country. You see the big cause has a little bit of hills, not like a lot of hills. I-75, right? No, no, no. Entonces. Like horse country. So you see the big prop purities with, you know, you know the trees and you know, you see the horses. It's very pretty actually. I mean. Los 2 ya dice que lo tenían puesto, ayer también y salieron los 2 que están Perfect. Good Job, dear. Una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible. Cute and very cute. Pero todo bien que a mi.se me parece a la madre. Si tú ves las las si tú ves la foto la madre. Very cute. Frío pero yo no, yo no oigo el calor, tú sabes, digo lo huelo, tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. The north side so it doesn't get any sun. Bueno. ¿Con el que vive aquella o no? Cóbrale. Qué fresco fresco. A lo manda solo y él no viene. Oye que fresco una. Boy, not too bad. Right, right, right, right, right, right, right, right. Porque tuvieran escrito en el lenguaje de todo El Mundo, maybe, pero. Pues sicomo.se llama Michelle yo me paro y digo coño, qué frío hace voy y veo y dice 61 lejano, yo lo siento mucho, hace demasiado frío. ¿Yo sé que ya para las a cuarto de la tarde no va a hacer frío, voy vamos a prender la calefacción y me dice, Allison, tú crees que tiene? Que no hay calección, lo más probable que no haya. Centro de Michelle. Y fue para allá y me mira a neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no doy una que ves si hacía un comentario yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63. Le digo, bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego. Eso lo tocó él porque Allison y yo. Al señor lo pusimos en 70. Sí, pero no, no se ha quitado la chaqueta, así que también tiene frío. Y eso quería entrar. Por lo menos les cae el sol. Pero aquí en el north side. We don't ever get Direct Sun. We get no Direct Sun we get. 61. ¿Cuándo regresa? Él está loco. Yeah, actually, I warm up the Los Angeles. He doesn't miss it, in other words. No. Hace como 5 años. Yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thomasta que te dice you know, you hit it y te dice what it is. 36 yo man cago en la madre fue cuando ya regresé de. 21 pero cuando groceries. And i left. Maybe no, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. Lo mueves a rojito. The only way i can turn the Heat is i have to tell it auto and turn the temperature up because in the sensor will know that it's cold outside and it throws hot air like you don't you can. It takes time versus the Old fashioned you turn the knob to the red. When the core is warm, it throws hot air. It takes time because your engine, your engines, your engines too cold better. The first time I went to use this Heat a few years ago, I'm like, where's where does it say Heat? And then right because it's auto electrical, right? Pero el aire acondicionado si te da chance. You can hit on and you turn the you know you. I don't like, I don't like that. It's kind of weird. I don't like it. I mean, I like the Old fashioned knobs. Right. I mean, it warms up, but it's weird. It's it's, it's, it's not, it's counterintuitive because you're used to saying, OK, I want warm Rojo. Déjame, cite me, carro tiene. Heated seeds. Ayer que me mondé con María, fuimos a comer. Ella nos prendió y dije, ay, coño María, tengo el fuascoalientino me dice, no, es que tienen el el hirington. So when i got in the car yesterday, i went, oh, mine has it too click. I turned it on. It's i didn't turn it on this morning 'cause i was so bundled up i didn't need it. Right. And it's supposedly para la espalda. So i was like, cool the first time i ever use it. But you know what, I didn't use it this morning cuz I was so bundle up I didn't need it. No, I didn't even turn the heat on because I figured I was going to like. Where's Kristen? Walter is stabbing in the driveway and we went Marito. Me acuerdo que tú me dijiste. ¿El enfermero no? Oh, that's not fair, right? That's not fair. That's not fair. Look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? Yeah, well, that's not fair. But can't he do something about it and say that they you know? De todo un poco. Right, right. Mm-hmm. Dad thought you can't do that. Right, right, right. Yeah. Right, right, right. That'll give you a whole lot. Yo no sé cómo la gente vive de unemployment porque es un 10. Yo no sé cómo la gente vive de un employee porque es un país. Yo me hago la única vez que. Is that coming here, Jim? Can I draw? That's good. No gets along with him. That's great. Right E for Kristen is good because. Which could be a little could be a little challenging. That's great. That's great. Which one? I want to train OH.
```

### Stage 7b — Numeric: Dictation Mode

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work so if you see me with this thing I look kind of goofy yeah and records it on a chips de la camarita you know the tiny little one and then it you know she has some sort of program because she's doing you know she's got like a grant or something yeah and then they transcribe it and then I guess they figure out you know whatever she does something to do with transcribing and spanglish and you know OK and everything Or something Y tomé Hello 4 wheel drive Nieve mi hermana vive en Jordan no digo nada de nieve bueno yo no solo está en la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora tenemos a las 8:00 de la mañana pero no como sí Ah noche eso queda como por la mitad del estedo no no no eso están en medio del Estado estado al norte o al sur de Games Country it's a horse country Actually when you I think it's before Greensboro I think it's before Gainesville when you when you drive on 75 I think it's 75 it's real pretty I mean you can see the big the pieces of property in Virginia where they have the horse it's horse country you see the big cause has a little bit of hills not like a lot of hills I-75 right No no no entonces like horse country so you see the big prop purities with you know you know the trees and you know you see the horses it's very pretty actually i mean los 2 ya dice que lo tenían puesto ayer también y salieron los 2 que están Perfect good Job dear Una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible Cute and very cute Pero todo bien que a mi.se me parece a la madre si tú ves las las si tú ves la foto la madre Very cute Frío pero yo no yo no oigo el calor tú sabes digo lo huelo tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61 The north side so it doesn't get any sun Bueno con el que vive aquella o no Cóbrale Qué fresco fresco A lo manda solo y él no viene oye que fresco una Boy not too bad Right right right right right right right right Porque tuvieran escrito en el lenguaje de todo El Mundo maybe pero Pues sicomo.se llama Michelle yo me paro y digo coño qué frío hace voy y veo y dice 61 lejano yo lo siento mucho hace demasiado frío yo sé que ya para las a cuarto de la tarde no va a hacer frío voy vamos a prender la calefacción y me dice Allison tú crees que tiene que no hay calección lo más probable que no haya Centro de Michelle y fue para allá y me mira a neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no doy una que ves si hacía un comentario yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63 le digo bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego eso lo tocó él porque Allison y yo Al señor lo pusimos en 70 Sí pero no no se ha quitado la chaqueta así que también tiene frío Y eso quería entrar por lo menos les cae el sol pero aquí en el north side we don't ever get Direct Sun we get no Direct Sun we get 61 Cuándo regresa él está loco Yeah actually I warm up the Los Angeles He doesn't miss it in other words No hace como 5 años yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thomasta que te dice you know you hit it y te dice what it is 36 yo man cago en la madre fue cuando ya regresé de 21 pero cuando groceries and i left Maybe no because it has a sensor and the sensor is on on the dashboard so I guess it takes and the one thing I I I mean this car provides heat Lo mueves a rojito the only way i can turn the Heat is i have to tell it auto and turn the temperature up because in the sensor will know that it's cold outside and it throws hot air like you don't you can it takes time versus the Old fashioned you turn the knob to the red when the core is warm it throws hot air it takes time because your engine your engines your engines too cold better the first time i went to use this Heat a few years ago I'm like where's where does it say Heat and then right because it's auto electrical right pero el aire acondicionado si te da chance you can hit on and you turn the you know you i don't like i don't like that it's kind of weird i don't like it i mean i like the Old fashioned knobs Right I mean it warms up but it's weird it's it's it's it's not it's counterintuitive because you're used to saying OK I want warm Rojo Déjame cite me carro tiene heated seeds Ayer que me mondé con María fuimos a comer ella nos prendió y dije ay coño María tengo el fuascoalientino me dice no es que tienen el el hirington so when i got in the car yesterday i went oh mine has it too click i turned it on it's i didn't turn it on this morning 'cause i was so bundled up i didn't need it right and it's supposedly para la espalda so i was like cool the first time i ever use it but you know what i didn't use it this morning cuz i was so bundle up i didn't need it No I didn't even turn the heat on because I figured I was going to like Where's Kristen Walter is stabbing in the driveway and we went marito Me acuerdo que tú me dijiste El enfermero no Oh that's not fair right That's not fair that's not fair Look I have this issue they're not going to give a good reference can he just tell them they're not Yeah well that's not fair but can't he do something about it and say that they you know De todo un poco Right right Mm-hmm Dad thought you can't do that Right right right yeah Right right right that'll give you a whole lot yo no sé cómo la gente vive de unemployment porque es un 10 Yo no sé cómo la gente vive de un employee porque es un país yo me hago la única vez que Is that coming here Jim can I draw That's good no gets along with him that's great Right E for Kristen is good because Which could be a little could be a little challenging That's great that's great Which one I want to train oh
```

### Stage 7c — Numeric: Dictation + Context-Aware Post-Processor

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work so if you see me with this thing I look kind of goofy yeah and records it on a chips de la camarita you know the tiny little one and then it you know she has some sort of program because she's doing you know she's got like a grant or something yeah and then they transcribe it and then I guess they figure out you know whatever she does something to do with transcribing and spanglish and you know OK and everything Or something Y tomé Hello 4 wheel drive Nieve mi hermana vive en Jordan no digo nada de nieve bueno yo no solo está en la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora tenemos a las 8:00 de la mañana pero no como sí Ah noche eso queda como por la mitad del estedo no no no eso están en medio del Estado estado al norte o al sur de Games Country it's a horse country Actually when you I think it's before Greensboro I think it's before Gainesville when you when you drive on 75 I think it's 75 it's real pretty I mean you can see the big the pieces of property in Virginia where they have the horse it's horse country you see the big cause has a little bit of hills not like a lot of hills I-75 right No no no entonces like horse country so you see the big prop purities with you know you know the trees and you know you see the horses it's very pretty actually i mean los 2 ya dice que lo tenían puesto ayer también y salieron los 2 que están Perfect good Job dear Una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible Cute and very cute Pero todo bien que a mi.se me parece a la madre si tú ves las las si tú ves la foto la madre Very cute Frío pero yo no yo no oigo el calor tú sabes digo lo huelo tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61 The north side so it doesn't get any sun Bueno con el que vive aquella o no Cóbrale Qué fresco fresco A lo manda solo y él no viene oye que fresco una Boy not too bad Right right right right right right right right Porque tuvieran escrito en el lenguaje de todo El Mundo maybe pero Pues sicomo.se llama Michelle yo me paro y digo coño qué frío hace voy y veo y dice 61 lejano yo lo siento mucho hace demasiado frío yo sé que ya para las a cuarto de la tarde no va a hacer frío voy vamos a prender la calefacción y me dice Allison tú crees que tiene que no hay calección lo más probable que no haya Centro de Michelle y fue para allá y me mira a neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no doy una que ves si hacía un comentario yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63 le digo bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego eso lo tocó él porque Allison y yo Al señor lo pusimos en 70 Sí pero no no se ha quitado la chaqueta así que también tiene frío Y eso quería entrar por lo menos les cae el sol pero aquí en el north side we don't ever get Direct Sun we get no Direct Sun we get 61 Cuándo regresa él está loco Yeah actually I warm up the Los Angeles He doesn't miss it in other words No hace como 5 años yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thomasta que te dice you know you hit it y te dice what it is 36 yo man cago en la madre fue cuando ya regresé de 21 pero cuando groceries and i left Maybe no because it has a sensor and the sensor is on on the dashboard so I guess it takes and the one thing I I I mean this car provides heat Lo mueves a rojito the only way i can turn the Heat is i have to tell it auto and turn the temperature up because in the sensor will know that it's cold outside and it throws hot air like you don't you can it takes time versus the Old fashioned you turn the knob to the red when the core is warm it throws hot air it takes time because your engine your engines your engines too cold better the first time i went to use this Heat a few years ago I'm like where's where does it say Heat and then right because it's auto electrical right pero el aire acondicionado si te da chance you can hit on and you turn the you know you i don't like i don't like that it's kind of weird i don't like it i mean i like the Old fashioned knobs Right I mean it warms up but it's weird it's it's it's it's not it's counterintuitive because you're used to saying OK I want warm Rojo Déjame cite me carro tiene heated seeds Ayer que me mondé con María fuimos a comer ella nos prendió y dije ay coño María tengo el fuascoalientino me dice no es que tienen el el hirington so when i got in the car yesterday i went oh mine has it too click i turned it on it's i didn't turn it on this morning 'cause i was so bundled up i didn't need it right and it's supposedly para la espalda so i was like cool the first time i ever use it but you know what i didn't use it this morning cuz i was so bundle up i didn't need it No I didn't even turn the heat on because I figured I was going to like Where's Kristen Walter is stabbing in the driveway and we went marito Me acuerdo que tú me dijiste El enfermero no Oh that's not fair right That's not fair that's not fair Look I have this issue they're not going to give a good reference can he just tell them they're not Yeah well that's not fair but can't he do something about it and say that they you know De todo un poco Right right Mm-hmm Dad thought you can't do that Right right right yeah Right right right that'll give you a whole lot yo no sé cómo la gente vive de unemployment porque es un 10 Yo no sé cómo la gente vive de un employee porque es un país yo me hago la única vez que Is that coming here Jim can I draw That's good no gets along with him that's great Right E for Kristen is good because Which could be a little could be a little challenging That's great that's great Which one I want to train oh
```

### Stage 8 — Emotion / Tone Evaluation

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. So if you see me with this thing, I look kind of goofy. Yeah, and records it on a Chips de la Camarita, you know, the tiny little one. And then it, you know. She has some sort of program because she's doing, you know, she's got like a grant or something. Yeah. And then they transcribe it and then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, OK and everything. Or something. Y Tomé. Hello 4 wheel drive. Nieve. Mi hermana vive en Jordan. No digo nada de nieve. Bueno, yo no solo está en la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta que vivas. Pero no cara a qué hora tenemos a las 8:00 de la mañana, pero no como sí Ah noche. ¿Eso queda como por la mitad del estedo? No, no, no, eso están en medio del Estado, estado al norte o al sur de Games. Country. It's a horse country. Actually, when you, I think it's before Greensboro, I think it's before Gainesville. When you, when you drive on 75, I think it's 75. It's real pretty. I mean, you can see the big, the pieces of property in Virginia where they have the horse. It's horse country. You see the big cause has a little bit of hills, not like a lot of hills. I-75, right? No, no, no. Entonces. Like horse country. So you see the big prop purities with, you know, you know the trees and you know, you see the horses. It's very pretty actually. I mean. Los 2 ya dice que lo tenían puesto, ayer también y salieron los 2 que están Perfect. Good Job, dear. Una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible. Cute and very cute. Pero todo bien que a mi.se me parece a la madre. Si tú ves las las si tú ves la foto la madre. Very cute. Frío pero yo no, yo no oigo el calor, tú sabes, digo lo huelo, tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61. The north side so it doesn't get any sun. Bueno. ¿Con el que vive aquella o no? Cóbrale. Qué fresco fresco. A lo manda solo y él no viene. Oye que fresco una. Boy, not too bad. Right, right, right, right, right, right, right, right. Porque tuvieran escrito en el lenguaje de todo El Mundo, maybe, pero. Pues sicomo.se llama Michelle yo me paro y digo coño, qué frío hace voy y veo y dice 61 lejano, yo lo siento mucho, hace demasiado frío. ¿Yo sé que ya para las a cuarto de la tarde no va a hacer frío, voy vamos a prender la calefacción y me dice, Allison, tú crees que tiene? Que no hay calección, lo más probable que no haya. Centro de Michelle. Y fue para allá y me mira a neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no doy una que ves si hacía un comentario yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63. Le digo, bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego. Eso lo tocó él porque Allison y yo. Al señor lo pusimos en 70. Sí, pero no, no se ha quitado la chaqueta, así que también tiene frío. Y eso quería entrar. Por lo menos les cae el sol. Pero aquí en el north side. We don't ever get Direct Sun. We get no Direct Sun we get. 61. ¿Cuándo regresa? Él está loco. Yeah, actually, I warm up the Los Angeles. He doesn't miss it, in other words. No. Hace como 5 años. Yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thomasta que te dice you know, you hit it y te dice what it is. 36 yo man cago en la madre fue cuando ya regresé de. 21 pero cuando groceries. And i left. Maybe no, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the one thing I I I mean this car provides heat. Lo mueves a rojito. The only way i can turn the Heat is i have to tell it auto and turn the temperature up because in the sensor will know that it's cold outside and it throws hot air like you don't you can. It takes time versus the Old fashioned you turn the knob to the red. When the core is warm, it throws hot air. It takes time because your engine, your engines, your engines too cold better. The first time I went to use this Heat a few years ago, I'm like, where's where does it say Heat? And then right because it's auto electrical, right? Pero el aire acondicionado si te da chance. You can hit on and you turn the you know you. I don't like, I don't like that. It's kind of weird. I don't like it. I mean, I like the Old fashioned knobs. Right. I mean, it warms up, but it's weird. It's it's, it's, it's not, it's counterintuitive because you're used to saying, OK, I want warm Rojo. Déjame, cite me, carro tiene. Heated seeds. Ayer que me mondé con María, fuimos a comer. Ella nos prendió y dije, ay, coño María, tengo el fuascoalientino me dice, no, es que tienen el el hirington. So when i got in the car yesterday, i went, oh, mine has it too click. I turned it on. It's i didn't turn it on this morning 'cause i was so bundled up i didn't need it. Right. And it's supposedly para la espalda. So i was like, cool the first time i ever use it. But you know what, I didn't use it this morning cuz I was so bundle up I didn't need it. No, I didn't even turn the heat on because I figured I was going to like. Where's Kristen? Walter is stabbing in the driveway and we went Marito. Me acuerdo que tú me dijiste. ¿El enfermero no? Oh, that's not fair, right? That's not fair. That's not fair. Look, I have this issue. They're not going to give a good reference. Can he just tell them they're not? Yeah, well, that's not fair. But can't he do something about it and say that they you know? De todo un poco. Right, right. Mm-hmm. Dad thought you can't do that. Right, right, right. Yeah. Right, right, right. That'll give you a whole lot. Yo no sé cómo la gente vive de unemployment porque es un 10. Yo no sé cómo la gente vive de un employee porque es un país. Yo me hago la única vez que. Is that coming here, Jim? Can I draw? That's good. No gets along with him. That's great. Right E for Kristen is good because. Which could be a little could be a little challenging. That's great. That's great. Which one? I want to train OH.
```

### Stage 9 — Latency & Timeout Testing

```
(empty)
```

### Stage 10 — Load & Concurrency Testing

```
(empty)
```

### Stage 11 — Logging & Alerts Setup

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work so if you see me with this thing I look kind of goofy yeah and records it on a chips de la camarita you know the tiny little one and then it you know she has some sort of program because she's doing you know she's got like a grant or something yeah and then they transcribe it and then I guess they figure out you know whatever she does something to do with transcribing and spanglish and you know OK and everything Or something Y tomé Hello 4 wheel drive Nieve mi hermana vive en Jordan no digo nada de nieve bueno yo no solo está en la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora tenemos a las 8:00 de la mañana pero no como sí Ah noche eso queda como por la mitad del estedo no no no eso están en medio del Estado estado al norte o al sur de Games Country it's a horse country Actually when you I think it's before Greensboro I think it's before Gainesville when you when you drive on 75 I think it's 75 it's real pretty I mean you can see the big the pieces of property in Virginia where they have the horse it's horse country you see the big cause has a little bit of hills not like a lot of hills I-75 right No no no entonces like horse country so you see the big prop purities with you know you know the trees and you know you see the horses it's very pretty actually i mean los 2 ya dice que lo tenían puesto ayer también y salieron los 2 que están Perfect good Job dear Una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible Cute and very cute Pero todo bien que a mi.se me parece a la madre si tú ves las las si tú ves la foto la madre Very cute Frío pero yo no yo no oigo el calor tú sabes digo lo huelo tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61 The north side so it doesn't get any sun Bueno con el que vive aquella o no Cóbrale Qué fresco fresco A lo manda solo y él no viene oye que fresco una Boy not too bad Right right right right right right right right Porque tuvieran escrito en el lenguaje de todo El Mundo maybe pero Pues sicomo.se llama Michelle yo me paro y digo coño qué frío hace voy y veo y dice 61 lejano yo lo siento mucho hace demasiado frío yo sé que ya para las a cuarto de la tarde no va a hacer frío voy vamos a prender la calefacción y me dice Allison tú crees que tiene que no hay calección lo más probable que no haya Centro de Michelle y fue para allá y me mira a neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no doy una que ves si hacía un comentario yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63 le digo bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego eso lo tocó él porque Allison y yo Al señor lo pusimos en 70 Sí pero no no se ha quitado la chaqueta así que también tiene frío Y eso quería entrar por lo menos les cae el sol pero aquí en el north side we don't ever get Direct Sun we get no Direct Sun we get 61 Cuándo regresa él está loco Yeah actually I warm up the Los Angeles He doesn't miss it in other words No hace como 5 años yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thomasta que te dice you know you hit it y te dice what it is 36 yo man cago en la madre fue cuando ya regresé de 21 pero cuando groceries and i left Maybe no because it has a sensor and the sensor is on on the dashboard so I guess it takes and the one thing I I I mean this car provides heat Lo mueves a rojito the only way i can turn the Heat is i have to tell it auto and turn the temperature up because in the sensor will know that it's cold outside and it throws hot air like you don't you can it takes time versus the Old fashioned you turn the knob to the red when the core is warm it throws hot air it takes time because your engine your engines your engines too cold better the first time i went to use this Heat a few years ago I'm like where's where does it say Heat and then right because it's auto electrical right pero el aire acondicionado si te da chance you can hit on and you turn the you know you i don't like i don't like that it's kind of weird i don't like it i mean i like the Old fashioned knobs Right I mean it warms up but it's weird it's it's it's it's not it's counterintuitive because you're used to saying OK I want warm Rojo Déjame cite me carro tiene heated seeds Ayer que me mondé con María fuimos a comer ella nos prendió y dije ay coño María tengo el fuascoalientino me dice no es que tienen el el hirington so when i got in the car yesterday i went oh mine has it too click i turned it on it's i didn't turn it on this morning 'cause i was so bundled up i didn't need it right and it's supposedly para la espalda so i was like cool the first time i ever use it but you know what i didn't use it this morning cuz i was so bundle up i didn't need it No I didn't even turn the heat on because I figured I was going to like Where's Kristen Walter is stabbing in the driveway and we went marito Me acuerdo que tú me dijiste El enfermero no Oh that's not fair right That's not fair that's not fair Look I have this issue they're not going to give a good reference can he just tell them they're not Yeah well that's not fair but can't he do something about it and say that they you know De todo un poco Right right Mm-hmm Dad thought you can't do that Right right right yeah Right right right that'll give you a whole lot yo no sé cómo la gente vive de unemployment porque es un 10 Yo no sé cómo la gente vive de un employee porque es un país yo me hago la única vez que Is that coming here Jim can I draw That's good no gets along with him that's great Right E for Kristen is good because Which could be a little could be a little challenging That's great that's great Which one I want to train oh
```

### Stage 12 — Fallback Validation

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work so if you see me with this thing I look kind of goofy yeah and records it on a chips de la camarita you know the tiny little one and then it you know she has some sort of program because she's doing you know she's got like a grant or something yeah and then they transcribe it and then I guess they figure out you know whatever she does something to do with transcribing and spanglish and you know OK and everything Or something Y tomé Hello 4 wheel drive Nieve mi hermana vive en Jordan no digo nada de nieve bueno yo no solo está en la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora tenemos a las 8:00 de la mañana pero no como sí Ah noche eso queda como por la mitad del estedo no no no eso están en medio del Estado estado al norte o al sur de Games Country it's a horse country Actually when you I think it's before Greensboro I think it's before Gainesville when you when you drive on 75 I think it's 75 it's real pretty I mean you can see the big the pieces of property in Virginia where they have the horse it's horse country you see the big cause has a little bit of hills not like a lot of hills I-75 right No no no entonces like horse country so you see the big prop purities with you know you know the trees and you know you see the horses it's very pretty actually i mean los 2 ya dice que lo tenían puesto ayer también y salieron los 2 que están Perfect good Job dear Una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible Cute and very cute Pero todo bien que a mi.se me parece a la madre si tú ves las las si tú ves la foto la madre Very cute Frío pero yo no yo no oigo el calor tú sabes digo lo huelo tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61 The north side so it doesn't get any sun Bueno con el que vive aquella o no Cóbrale Qué fresco fresco A lo manda solo y él no viene oye que fresco una Boy not too bad Right right right right right right right right Porque tuvieran escrito en el lenguaje de todo El Mundo maybe pero Pues sicomo.se llama Michelle yo me paro y digo coño qué frío hace voy y veo y dice 61 lejano yo lo siento mucho hace demasiado frío yo sé que ya para las a cuarto de la tarde no va a hacer frío voy vamos a prender la calefacción y me dice Allison tú crees que tiene que no hay calección lo más probable que no haya Centro de Michelle y fue para allá y me mira a neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no doy una que ves si hacía un comentario yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63 le digo bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego eso lo tocó él porque Allison y yo Al señor lo pusimos en 70 Sí pero no no se ha quitado la chaqueta así que también tiene frío Y eso quería entrar por lo menos les cae el sol pero aquí en el north side we don't ever get Direct Sun we get no Direct Sun we get 61 Cuándo regresa él está loco Yeah actually I warm up the Los Angeles He doesn't miss it in other words No hace como 5 años yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thomasta que te dice you know you hit it y te dice what it is 36 yo man cago en la madre fue cuando ya regresé de 21 pero cuando groceries and i left Maybe no because it has a sensor and the sensor is on on the dashboard so I guess it takes and the one thing I I I mean this car provides heat Lo mueves a rojito the only way i can turn the Heat is i have to tell it auto and turn the temperature up because in the sensor will know that it's cold outside and it throws hot air like you don't you can it takes time versus the Old fashioned you turn the knob to the red when the core is warm it throws hot air it takes time because your engine your engines your engines too cold better the first time i went to use this Heat a few years ago I'm like where's where does it say Heat and then right because it's auto electrical right pero el aire acondicionado si te da chance you can hit on and you turn the you know you i don't like i don't like that it's kind of weird i don't like it i mean i like the Old fashioned knobs Right I mean it warms up but it's weird it's it's it's it's not it's counterintuitive because you're used to saying OK I want warm Rojo Déjame cite me carro tiene heated seeds Ayer que me mondé con María fuimos a comer ella nos prendió y dije ay coño María tengo el fuascoalientino me dice no es que tienen el el hirington so when i got in the car yesterday i went oh mine has it too click i turned it on it's i didn't turn it on this morning 'cause i was so bundled up i didn't need it right and it's supposedly para la espalda so i was like cool the first time i ever use it but you know what i didn't use it this morning cuz i was so bundle up i didn't need it No I didn't even turn the heat on because I figured I was going to like Where's Kristen Walter is stabbing in the driveway and we went marito Me acuerdo que tú me dijiste El enfermero no Oh that's not fair right That's not fair that's not fair Look I have this issue they're not going to give a good reference can he just tell them they're not Yeah well that's not fair but can't he do something about it and say that they you know De todo un poco Right right Mm-hmm Dad thought you can't do that Right right right yeah Right right right that'll give you a whole lot yo no sé cómo la gente vive de unemployment porque es un 10 Yo no sé cómo la gente vive de un employee porque es un país yo me hago la única vez que Is that coming here Jim can I draw That's good no gets along with him that's great Right E for Kristen is good because Which could be a little could be a little challenging That's great that's great Which one I want to train oh
```

### Stage C1 — Combined Best  ✅ PRODUCTION RECOMMENDATION

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work so if you see me with this thing I look kind of goofy yeah and records it on a chips de la camarita you know the tiny little one and then it you know she has some sort of program because she's doing you know she's got like a grant or something yeah and then they transcribe it and then I guess they figure out you know whatever she does something to do with transcribing and spanglish and you know OK and everything Or something Y tomé Hello 4 wheel drive Nieve mi hermana vive en Jordan no digo nada de nieve bueno yo no solo está en la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora tenemos a las 8:00 de la mañana pero no como sí Ah noche eso queda como por la mitad del estedo no no no eso están en medio del Estado estado al norte o al sur de Games Country it's a horse country Actually when you I think it's before Greensboro I think it's before Gainesville when you when you drive on 75 I think it's 75 it's real pretty I mean you can see the big the pieces of property in Virginia where they have the horse it's horse country you see the big cause has a little bit of hills not like a lot of hills I-75 right No no no entonces like horse country so you see the big prop purities with you know you know the trees and you know you see the horses it's very pretty actually i mean los 2 ya dice que lo tenían puesto ayer también y salieron los 2 que están Perfect good Job dear Una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible Cute and very cute Pero todo bien que a mi.se me parece a la madre si tú ves las las si tú ves la foto la madre Very cute Frío pero yo no yo no oigo el calor tú sabes digo lo huelo tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61 The north side so it doesn't get any sun Bueno con el que vive aquella o no Cóbrale Qué fresco fresco A lo manda solo y él no viene oye que fresco una Boy not too bad Right right right right right right right right Porque tuvieran escrito en el lenguaje de todo El Mundo maybe pero Pues sicomo.se llama Michelle yo me paro y digo coño qué frío hace voy y veo y dice 61 lejano yo lo siento mucho hace demasiado frío yo sé que ya para las a cuarto de la tarde no va a hacer frío voy vamos a prender la calefacción y me dice Allison tú crees que tiene que no hay calección lo más probable que no haya Centro de Michelle y fue para allá y me mira a neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no doy una que ves si hacía un comentario yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63 le digo bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego eso lo tocó él porque Allison y yo Al señor lo pusimos en 70 Sí pero no no se ha quitado la chaqueta así que también tiene frío Y eso quería entrar por lo menos les cae el sol pero aquí en el north side we don't ever get Direct Sun we get no Direct Sun we get 61 Cuándo regresa él está loco Yeah actually I warm up the Los Angeles He doesn't miss it in other words No hace como 5 años yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thomasta que te dice you know you hit it y te dice what it is 36 yo man cago en la madre fue cuando ya regresé de 21 pero cuando groceries and i left Maybe no because it has a sensor and the sensor is on on the dashboard so I guess it takes and the one thing I I I mean this car provides heat Lo mueves a rojito the only way i can turn the Heat is i have to tell it auto and turn the temperature up because in the sensor will know that it's cold outside and it throws hot air like you don't you can it takes time versus the Old fashioned you turn the knob to the red when the core is warm it throws hot air it takes time because your engine your engines your engines too cold better the first time i went to use this Heat a few years ago I'm like where's where does it say Heat and then right because it's auto electrical right pero el aire acondicionado si te da chance you can hit on and you turn the you know you i don't like i don't like that it's kind of weird i don't like it i mean i like the Old fashioned knobs Right I mean it warms up but it's weird it's it's it's it's not it's counterintuitive because you're used to saying OK I want warm Rojo Déjame cite me carro tiene heated seeds Ayer que me mondé con María fuimos a comer ella nos prendió y dije ay coño María tengo el fuascoalientino me dice no es que tienen el el hirington so when i got in the car yesterday i went oh mine has it too click i turned it on it's i didn't turn it on this morning 'cause i was so bundled up i didn't need it right and it's supposedly para la espalda so i was like cool the first time i ever use it but you know what i didn't use it this morning cuz i was so bundle up i didn't need it No I didn't even turn the heat on because I figured I was going to like Where's Kristen Walter is stabbing in the driveway and we went marito Me acuerdo que tú me dijiste El enfermero no Oh that's not fair right That's not fair that's not fair Look I have this issue they're not going to give a good reference can he just tell them they're not Yeah well that's not fair but can't he do something about it and say that they you know De todo un poco Right right Mm-hmm Dad thought you can't do that Right right right yeah Right right right that'll give you a whole lot yo no sé cómo la gente vive de unemployment porque es un 10 Yo no sé cómo la gente vive de un employee porque es un país yo me hago la única vez que Is that coming here Jim can I draw That's good no gets along with him that's great Right E for Kristen is good because Which could be a little could be a little challenging That's great that's great Which one I want to train oh
```

### Stage C2 — Combined All Stages

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work Pavel como El cambio Don idio malotro so if you see me with this thing I look kind of goofy yeah it records it on a como little chips de la camarita you know the tiny little one and then it you know she has some sort of program because she's doing you know she's got like a grant or something yeah and then they transcribe it and then I guess they figure out you know whatever she does something to do with transcribing and spanglish and you know okabe and everything right right I don't know I love the lot you know you know you know personal or something so Yeah Y tú no puedes salir en esa vaina hello sigue 4 wheel drive Mi hermana vive en Jordan no digo nada de nieve bueno yo no solo está en la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas A qué hora te llamo a las 8:00 de la mañana pero no como si Ah anoche eso queda como por la mitad del estedo no no no eso están en medio del Estado estado al norte o al sur de Games it's a horse country it's a horse country it's really pretty actually when you i think it's before gainesville i think it's before gainesville when you when you drive on 75 i think it's 75 it's real pretty i mean you can see the big the pieces of Property que parece que tuvieras en Virginia where they have the horse it's horse country you see the big because has a little bit of hills not like a lot of hills No no no entonces it's like horse country so you see the big prop properties with you know you know the trees and you know you see the horses it's very pretty actually i mean los 2 ella dice que lo tenían puesto ayer también y salieron los 2 que están Perfect good Job dear que como una vez porque y no el el tamaño no lo pudiste escoger mejor porque no es posible ay it's so cute it's very cute Pero todo bien que a mi.se me parece a la madre si tú ves la la si tú ves la foto la madre la Very cute a little mama no cheeky coño tengo hace frío pero yo no yo no oigo el calor tú sabes digo lo huelo tú sabes cuando tú pones la calefacción tiene un la primera vezque.se prende tiene un colorcito el lado ya hace 61 es el north side so it doesn't get any Sun Cuando el que vive aquel lado no Cóbrale Que fresco fresco Delincuentes A lo manda solo y él no viene porque fresco una Boy not too bad Right right right right right right right right Lengua de todo mundo maybe pero pues sí el como.se llama Michelle yo me paro y digo coño qué frío hace voy y veo en el y dice 61 lejano yo lo siento mucho hace demasiado frío yo sé que ya para las a cuarto de la tarde no hace frío voy vamos a prender la calefacción y me dice Allison tú crees que tiene que no hay calefacción lo más probable es que no haya Cuando centro de Michel y fue para allá y me mira neo y porque yo le dije que yo lo he aprendido y los hicimos yo me hice la que yo no doy una que me decía así un comentario yo voy a decir yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice ahora dice 63 le digo bueno por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego a eso lo tocó él porque Allison y yo ale señor lo pusimos en 70 Sí pero no no se ha quitado la chaqueta así que también tiene frío Eso quería entrar por lo menos les cae el sol pero aquí en el north side we don't ever get Direct Sun we get no Direct Sun we get 61 Cuando regresa él está loco Yeah actually I warm up the Los Angeles In other words No hace como 5 años that's what i said about 5 years yo esta mañana que salí de la casa que tenía que ir al súper que o sea no meok mi carro tiene un thermasta que te dice you know you hit it y te dice what it is 36 yo man cago en la madre fue cuando ya regresé de de Grosso decía 41 pero cuando hay Dropbox groceries and i left decía 39 coño Maybe no because it has a sensor and the sensor is on on the dashboard so I guess it takes and the one thing I I I mean this car provides heat O the only way I can turn the heat is I have to tell it auto and turn the temperature up because then the sensor will know that it's cold outside and it throws hot air like you don't you can and it takes time versus the old fashioned you turn the knob down to the red when the core is warm it's throws hot air it takes time because your engine your engine's your engine's too cold you know the first time I went to use this heat a few years ago like where's where's where does it say heat you know and then right because it's auto electrical right chance that you can hit on and you turn the you know you I don't like I don't like that that they're it's kind of weird I don't like it I mean I like the old fashioned knobs you know I mean it warms up but it's weird it's it's it's it's not it's counterintuitive because you're used to saying OK I want warm Heated seats nunca lo he usado ayer que me mondé con María fuimos a comer ella nos prendió y dije ay coño María tengo el fuaz calientito me dice no es que tiene el el heating thing so when i got in the car yesterday i went oh mine has it to click i turned it on it's i didn't turn it on this morning because i was so bundled up i didn't eat it right and it's supposedly para la espalda so i was like cool the first time i ever use it but you know what i didn't use it this morning because i was so bundled up i didn't need it no i didn't even turn the Heat on because i figure i was going to like Where's Kristen Con ustedes otra vez la verdad que gordi es muy bueno marido Me acuerdo que tú me dijiste Uh-huh el enfermero no Oh that's not fair right That's not fair that's not fair look I have this issue they're not going to give a good reference can he just tell them they're not Coño but that's not Fair but can't he do something about it and say that they you know De todo un poco Right right right Mm-hmm Dad thought you can't do that No hay no tienen como que otra tienen cogida contigo Right right right they give you a whole lot unemployment Yo no sé como la gente vive de un employee porque es un país yo me hago la única vez que Is that coming here Jim can I draw right That's good no you see along with him that's great Right EE for Kristen is good because one of the endo Papa which could be a little could be a little challenging that's great that's great Which one All right
```
