# Azure Speech-to-Text — Incremental Improvement Observations
**Generated:** 2026-04-21 13:20
**Stages run:** 13

---

## Overview

This document records observations at each stage of the Azure STT
incremental improvement process. Each stage adds one feature or configuration
on top of the previous and compares resulting metrics, transcript quality,
and stage-specific behavior.

**Important interpretation note:** baseline is treated as a reference point,
not as ground truth. Quality improvements are evaluated independently using
punctuation, numeric handling, domain phrase quality, VAD quality, short-word
risk, readability, and overall quality score.

---

## Stage 0: `baseline`

**Phase:** Baseline  &nbsp;|&nbsp;  **Task:** Original working script

**What was added:** Original working script [Baseline]

### Parameters / configuration tested

**Azure / SDK parameters**

- `SpeechConfig(subscription, region)`
- `OutputFormat = Detailed`
- `AutoDetectSourceLanguageConfig(en-US, es-US)`
- `WAV PCM 16kHz mono 16-bit input`

**Changes applied / tested**

- No optimization feature enabled
- Used as baseline reference only

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **0** |
| Stage Name | **`baseline`** |
| Phase | Baseline |
| Task | Original working script |
| Detected Language | en-US |
| TTFT Partial | 2505.8 ms |
| TTFT Final | 3656.8 ms |
| Total Time | 442.96 sec |
| Segments | 83 |
| Words | 1240 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 647 |

### Transcript

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. Cambio de un idioma al otro. So if you see me this thing, i look kind of goofy. Como los chips de la camarita. You know that tiny little One. And then you know. She has some sort of Program because she's doing, you know, she's got like a Grant or something. Yeah. And then they transcribe it. And then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, everything. Inglés. Or something so. Y tu amiga. Y tú no puedes salir en esa vaina. Hello. Sigue a 4 wheel drive. Nieve. Mi hermana vive en Jordan. No ha ido nada de nieve. La casa de ella. Puede que si en las montañas sí sí haya. M
...(truncated)
```

### Observations

- This is the baseline/reference stage.
- Future stages are compared against it, but it is not treated as ground truth.

---

## Stage 1: `asr_config`

**Phase:** Setup  &nbsp;|&nbsp;  **Task:** ASR Config Finalization

**What was added:** ASR Config Finalization [Setup]

### Parameters / configuration tested

**Azure / SDK parameters**

- `AutoDetectSourceLanguageConfig(languages=['en-US','es-US'])`
- `Fixed WAV PCM 16kHz mono 16-bit audio format`
- `SpeechConfig output_format = Detailed`

**Changes applied / tested**

- Locked candidate locales
- Standardized input audio format
- Avoided open-ended language detection
- language_locked = ['en-US', 'es-US']
- audio_format = WAV PCM 16kHz mono 16-bit (converted via FFmpeg)
- auto_detect_locked = True
- open_ended_detect = False

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **1** |
| Stage Name | **`asr_config`** |
| Phase | Setup |
| Task | ASR Config Finalization |
| Detected Language | en-US |
| TTFT Partial | 2571.5 ms |
| TTFT Final | 3652.7 ms |
| Total Time | 443.12 sec |
| Segments | 83 |
| Words | 1240 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 639 |

### Transcript

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. Cambio de un idioma al otro. So if you see me this thing, i look kind of goofy. Como los chips de la camarita. You know that tiny little One. And then you know. She has some sort of Program because she's doing, you know, she's got like a Grant or something. Yeah. And then they transcribe it. And then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, everything. Inglés. Or something so. Y tu amiga. Y tú no puedes salir en esa vaina. Hello. Sigue a 4 wheel drive. Nieve. Mi hermana vive en Jordan. No ha ido nada de nieve. La casa de ella. Puede que si en las montañas sí sí haya. M
...(truncated)
```

### Change vs Previous Stage (`baseline` → `asr_config`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 2505.8 | 2571.5 | +65.7 | ⚠️ worse |
| ttft_final_ms | 3656.8 | 3652.7 | -4.1 | ✅ improved |
| total_time_sec | 442.96 | 443.12 | +0.16 | ⚠️ worse |
| segment_count | 83 | 83 | 0 | ➡️ same |
| word_count | 1240 | 1240 | 0 | ➡️ same |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 100.0%
**Word-level changes:** 0

### Observations

- ➡️  Transcript unchanged (100.0% similar) — feature impact is in metrics, not text
- ℹ️  Expected: Stable, predictable recognition

---

## Stage 2: `vad_tuning`

**Phase:** Audio  &nbsp;|&nbsp;  **Task:** VAD Evaluation & Tuning

**What was added:** VAD Evaluation & Tuning [Audio]

### Parameters / configuration tested

**Azure / SDK parameters**

- `SpeechServiceConnection_EndSilenceTimeoutMs = 1500`
- `SpeechServiceConnection_InitialSilenceTimeoutMs = 5000`
- `Speech_SegmentationSilenceTimeoutMs = 800`

**Changes applied / tested**

- Increased end silence timeout
- Tested segmentation and initial silence thresholds
- Evaluated truncation / false cutoff behavior
- end_silence_ms = 1500
- init_silence_ms = 5000
- seg_silence_ms = 800
- note = end_silence increased 800→1500ms to reduce truncation

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **2** |
| Stage Name | **`vad_tuning`** |
| Phase | Audio |
| Task | VAD Evaluation & Tuning |
| Detected Language | es-US |
| TTFT Partial | 4748.7 ms |
| TTFT Final | 6872.0 ms |
| Total Time | 179.69 sec |
| Segments | 34 |
| Words | 460 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 251 |

### Transcript

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. So if you see me this thing, I look kind of goofy. Yeah, and records it on como los chips de la camarita. You know the tiny little One, and then you know. She has some sort of Program because she's doing, you know, she's got like a Grant or something. Yeah. And then they transcribe it. And then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, everything. Get a can I, oh, you know, personal or something so. Y tu amiga. Hello 4 wheel drive. Nieve mi hermana vive en Georgia y no dio nada de nieve. Bueno, yo no soy hasta la casa de ella. Bueno, puede que si en las montañas sí sí h
...(truncated)
```

### Change vs Previous Stage (`asr_config` → `vad_tuning`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 2571.5 | 4748.7 | +2177.2 | ⚠️ worse |
| ttft_final_ms | 3652.7 | 6872.0 | +3219.3 | ⚠️ worse |
| total_time_sec | 443.12 | 179.69 | -263.43 | ✅ improved |
| segment_count | 83 | 34 | -49 | ⚠️ worse |
| word_count | 1240 | 460 | -780 | ⚠️ worse |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 45.6%
**Word-level changes:** 27

**Sample word changes:**

- `[delete]` `cambio de un idioma al otro.` → `(nothing)`
- `[insert]` `(nothing)` → `yeah, and records it on`
- `[replace]` `that` → `the`
- `[replace]` `one.` → `one,`
- `[replace]` `inglés.` → `get a can i, oh, you know, personal`
- `[replace]` `y tú no puedes salir en esa vaina. hello. sigue a` → `hello`
- `[replace]` `nieve.` → `nieve`
- `[replace]` `jordan. no ha ido` → `georgia y no dio`
- `[insert]` `(nothing)` → `bueno, yo no soy hasta`
- `[insert]` `(nothing)` → `bueno,`

### Observations

- ⚠️  TTFT Final slower by 3219ms (may be acceptable)
- ⚠️  780 fewer words — check endpointing
- ⚠️  Transcript changed significantly (45.6%) — review word diff
- ℹ️  Expected: Reduced truncation and false cut-offs

---

## Stage 3: `phrase_boost`

**Phase:** Accuracy  &nbsp;|&nbsp;  **Task:** Word / Phrase Boosting

**What was added:** Word / Phrase Boosting [Accuracy]

### Parameters / configuration tested

**Azure / SDK parameters**

- `PhraseListGrammar.from_recognizer(...)`
- `Added numeric phrases`
- `Added finance / domain phrases`

**Changes applied / tested**

- Boosted digits and domain vocabulary
- Tested phrase hit improvement in transcript
- Phrase boosting active: total_phrases=28, hits=1

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **3** |
| Stage Name | **`phrase_boost`** |
| Phase | Accuracy |
| Task | Word / Phrase Boosting |
| Detected Language | en-US |
| TTFT Partial | 2966.2 ms |
| TTFT Final | 4104.5 ms |
| Total Time | 442.23 sec |
| Segments | 86 |
| Words | 1292 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 734 |

### Transcript

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. So if you see me this thing, I look kind of goofy. Yeah, and records it on como los chips de la camarita. You know the tiny little One, and then you know. She has some sort of Program because she's doing, you know, she's got like a Grant or something. Yeah. And then they transcribe it. And then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, everything. You know, personal or something so. Y tu amiga. Hello 4 wheel drive. Nieve mi hermana vive en Georgia y no dio nada de nieve. Bueno, yo no soy hasta la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta q
...(truncated)
```

### Change vs Previous Stage (`vad_tuning` → `phrase_boost`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 4748.7 | 2966.2 | -1782.5 | ✅ improved |
| ttft_final_ms | 6872.0 | 4104.5 | -2767.5 | ✅ improved |
| total_time_sec | 179.69 | 442.23 | +262.54 | ⚠️ worse |
| segment_count | 34 | 86 | +52 | ✅ improved |
| word_count | 460 | 1292 | +832 | ✅ improved |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 51.9%
**Word-level changes:** 2

**Sample word changes:**

- `[delete]` `get a can i, oh,` → `(nothing)`
- `[insert]` `(nothing)` → `too bad. right, right, right, right, right, right, right. y te han hecho para que nadie entienda lo que dice. porque te hubieran escrito en lenguaje de todo el mundo, maybe, pero. pues sí el como. ¿e llama michelle, yo me paro y yo coño qué frío hace voy y veo 61 lejano, yo lo siento mucho, hace demasiado frío, yo sé que ya para las a cuarto de la tarde no va a hacer frío, voy vamos a prender la calefacción y me dice, allison, tú crees que tiene? que no hay calefacción. lo más probable que no haya. central michel. y fue para allá y me mira neo y porque yo le dije que yo lo he aprendido y los yo me hice la que yo no tomaba, que si ha sido un comentario, yo voy a decir si yo hacía mucho friodice.se levanta a los 5 minutos lizeth y dice, ahora dice 63. le digo, bueno, por lo menos subió 2 degrees because it used to be 61 y acá decía 74 lego. eso lo tocó él porque allison y yo puse. al señor lo pusimos en 70. sí, pero no, no se ha quitado la chaqueta, así que también tiene frío. y eso quería entrar. por lo menos les cae el sol. pero aquí en el north side. we don't ever get direct sun. we get no direct sun we get. 61. ¿cuándo regresa? él está loco. la la sangre.se le ha puesto. he doesn't miss it, in other words. no hace como 5 años. that's what i said about 5 years. yo esta mañana que salí de la casa que tenía que ir al súper que o sea no me mi carro tiene un thermasta que te dice you know, you hit it y te dice what it is. 36 yo me cago en la madre. fue cuando ya regresé de. the owner better one. i dropped the groceries and i left. maybe no, because it has a sensor and the sensor is on on the dashboard. so i guess it takes and the one thing i i i mean this car provides heat. rojito, the only way i can turn the heat is i have to tell it auto and turn the temperature up because then the sensor will know that it's cold outside and it throws hot air like you don't, you can have and it takes time versus the old fashioned, you turn the knob to the red when the core is warm, it throws hot air. it takes time because your engine, your engine's, your engine's too cold. but the first time i went to use this heat a few years ago, i'm like, where's where does it say heat? and then right, because it's auto electrical, right? so you can hit on and you turn the, you know, you the i don't like, i don't like that that it's kind of weird. i don't like it. i mean, i like the old fashioned knobs, you know. i mean, it warms up, but it's weird. it's, it's, it's, it's not, it's counterintuitive because you're used to saying, ok, i want warm. déjame decirte mi carro tiene gire 6. nunca lo he usado. ayer que me mondé con maría, fuimos a comer. ella nos prendió y dije, ay, coño, maría, tengo el fajco alientino. me dice, no, es que tienen el el heating thing. so when i got in the car yesterday i went, oh, mine, has it too click. i turned it on. it's i didn't turn it on this morning 'cause i was so bundled up i didn't eat it. right. and it's supposedly para la espalda. so i was like, cool the first time i ever use it. but you know what, i didn't use it this morning cuz i was so bundle up i didn't need it. no, i didn't even turn the heat on because i figured i was going to like. where's kristen? walter. i love it a gordie and we went marito. me acuerdo que tú me dijiste. ¿el enfermero no? oh, that's not fair, right? that's not fair. that's not fair. say hello, look, i have this issue. they're not going to give a good reference. can he just tell them they're not? well, that's not fair. but can't he do something about it and say that they. you know. de todo un poco. right, right. dad thought you can't do that. que atrás de encogida contigo. right, right, right. that'll give you a whole lot. yo no sé cómo la gente vive de unemployment. yo no sé cómo la gente vive de un employee porque es un país. yo me hago la única vez que. is that coming here, jim? can i draw? that's good. no gets along with him. that's great. right e for kristen is good because. one of the papa, which could be a little could be a little challenging. that's great. that's great. which one? 43.`

### Observations

- ✅ TTFT Final improved by 2768ms
- ✅ 832 more words captured — less truncation
- ⚠️  Transcript changed significantly (51.9%) — review word diff
- ℹ️  Expected: Improved numeric and domain accuracy

---

## Stage 4: `vocab_tuning`

**Phase:** Accuracy  &nbsp;|&nbsp;  **Task:** Transcript-Based Vocabulary Tuning

**What was added:** Transcript-Based Vocabulary Tuning [Accuracy]

### Parameters / configuration tested

**Azure / SDK parameters**

- `PhraseListGrammar with mined transcript terms`

**Changes applied / tested**

- Extracted repeated useful words from prior transcripts
- Added mined vocabulary as boost phrases
- Mined vocabulary terms added: 50

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **4** |
| Stage Name | **`vocab_tuning`** |
| Phase | Accuracy |
| Task | Transcript-Based Vocabulary Tuning |
| Detected Language | en-US |
| TTFT Partial | 2502.8 ms |
| TTFT Final | 3748.1 ms |
| Total Time | 442.86 sec |
| Segments | 87 |
| Words | 1328 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 747 |

### Transcript

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. So if you see me this thing, I look kind of goofy. Yeah, and records it on como los chips de la camarita. You know the tiny little One, and then you know. She has some sort of Program because she's doing, you know, she's got like a Grant or something. Yeah. And then they transcribe it. And then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, everything. Right. Right. But i don't know. Si, si. Yo sé que estamos todo mundo hablando inglés o como dice mi hermano, si tú no quieres que nadie oye, no personal or something. So. Y tu amiga. Hello 4 wheel drive. Nieve mi hermana vive 
...(truncated)
```

### Change vs Previous Stage (`phrase_boost` → `vocab_tuning`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 2966.2 | 2502.8 | -463.4 | ✅ improved |
| ttft_final_ms | 4104.5 | 3748.1 | -356.4 | ✅ improved |
| total_time_sec | 442.23 | 442.86 | +0.63 | ⚠️ worse |
| segment_count | 86 | 87 | +1 | ✅ improved |
| word_count | 1292 | 1328 | +36 | ✅ improved |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 94.1%
**Word-level changes:** 38

**Sample word changes:**

- `[replace]` `you know,` → `right. right. but i don't know. si, si. yo sé que estamos todo mundo hablando inglés o como dice mi hermano, si tú no quieres que nadie oye, no`
- `[replace]` `something` → `something.`
- `[replace]` `georgia y no dio` → `jordan, no digo`
- `[replace]` `country, you see.` → `country. you see`
- `[replace]` `cause` → `because`
- `[insert]` `(nothing)` → `el`
- `[replace]` `como` → `cuando`
- `[replace]` `con` → `cuando`
- `[replace]` `oye que fresco.` → `¿porque fresco?`
- `[insert]` `(nothing)` → `right, right,`

### Observations

- ✅ TTFT Final improved by 356ms
- ✅ 36 more words captured — less truncation
- ➡️  Transcript similar (94.1%) — small word-level changes
- ℹ️  Expected: Domain alignment

---

## Stage 5: `numeric_handling`

**Phase:** Logic  &nbsp;|&nbsp;  **Task:** Numeric Handling Validation

**What was added:** Numeric Handling Validation [Logic]

### Parameters / configuration tested

**Azure / SDK parameters**

- `Detailed JSON parsing from SpeechServiceResponse_JsonResult`
- `ITN / Lexical / Display field analysis`

**Changes applied / tested**

- Evaluated number rendering quality
- Used context-aware digit analysis
- Prevented blind conversions like to -> 2

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **5** |
| Stage Name | **`numeric_handling`** |
| Phase | Logic |
| Task | Numeric Handling Validation |
| Detected Language | en-US |
| TTFT Partial | 3049.3 ms |
| TTFT Final | 4157.3 ms |
| Total Time | 442.54 sec |
| Segments | 86 |
| Words | 1292 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 745 |

### Transcript

```
My sister-in-law, you know, she, she's a linguist and so she wants to sort of track, you know, Spanglish. So she asked me if I would wear it at work. So if you see me this thing, I look kind of goofy. Yeah, and records it on como los chips de la camarita. You know the tiny little One, and then you know. She has some sort of Program because she's doing, you know, she's got like a Grant or something. Yeah. And then they transcribe it. And then I guess they figure out, you know, whatever she does something to do with transcribing and Spanglish and, you know, everything. You know, personal or something so. Y tu amiga. Hello 4 wheel drive. Nieve mi hermana vive en Georgia y no dio nada de nieve. Bueno, yo no soy hasta la casa de ella. Bueno, puede que si en las montañas sí sí haya. Más muerta q
...(truncated)
```

### Change vs Previous Stage (`vocab_tuning` → `numeric_handling`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 2502.8 | 3049.3 | +546.5 | ⚠️ worse |
| ttft_final_ms | 3748.1 | 4157.3 | +409.2 | ⚠️ worse |
| total_time_sec | 442.86 | 442.54 | -0.32 | ✅ improved |
| segment_count | 87 | 86 | -1 | ⚠️ worse |
| word_count | 1328 | 1292 | -36 | ⚠️ worse |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 94.1%
**Word-level changes:** 38

**Sample word changes:**

- `[replace]` `right. right. but i don't know. si, si. yo sé que estamos todo mundo hablando inglés o como dice mi hermano, si tú no quieres que nadie oye, no` → `you know,`
- `[replace]` `something.` → `something`
- `[replace]` `jordan, no digo` → `georgia y no dio`
- `[replace]` `country. you see` → `country, you see.`
- `[replace]` `because` → `cause`
- `[delete]` `el` → `(nothing)`
- `[replace]` `cuando` → `como`
- `[replace]` `cuando` → `con`
- `[replace]` `¿porque fresco?` → `oye que fresco.`
- `[delete]` `right, right,` → `(nothing)`

### Observations

- ⚠️  TTFT Final slower by 409ms (may be acceptable)
- ⚠️  36 fewer words — check endpointing
- ➡️  Transcript similar (94.1%) — small word-level changes
- ℹ️  Expected: Reduced verification failures

---

## Stage 6: `dictation_mode`

**Phase:** Accuracy  &nbsp;|&nbsp;  **Task:** Dictation Mode

**What was added:** Dictation Mode [Accuracy]

### Parameters / configuration tested

**Azure / SDK parameters**

- `SpeechConfig.enable_dictation()`

**Changes applied / tested**

- Enabled dictation mode
- Evaluated punctuation and readability improvement
- Dictation punctuation counts: {'commas': 0, 'periods': 5, 'questions': 0, 'total_punct': 5}

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **6** |
| Stage Name | **`dictation_mode`** |
| Phase | Accuracy |
| Task | Dictation Mode |
| Detected Language | en-US |
| TTFT Partial | 2485.1 ms |
| TTFT Final | 3557.5 ms |
| Total Time | 442.42 sec |
| Segments | 86 |
| Words | 1291 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 735 |

### Transcript

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work So if you see me this thing i look kind of goofy yeah and records it on como los chips de la camarita you know the tiny little One and then you know she has some sort of Program because she's doing you know she's got like a Grant or something yeah and then they transcribe it and then i guess they figure out you know whatever she does something to do with transcribing and spanglish and you know everything You know personal or something so Y tu amiga Hello 4 wheel drive Nieve mi hermana vive en Georgia y no dio nada de nieve bueno yo no soy hasta la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora
...(truncated)
```

### Change vs Previous Stage (`numeric_handling` → `dictation_mode`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 3049.3 | 2485.1 | -564.2 | ✅ improved |
| ttft_final_ms | 4157.3 | 3557.5 | -599.8 | ✅ improved |
| total_time_sec | 442.54 | 442.42 | -0.12 | ✅ improved |
| segment_count | 86 | 86 | 0 | ➡️ same |
| word_count | 1292 | 1291 | -1 | ⚠️ worse |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 78.3%
**Word-level changes:** 180

**Sample word changes:**

- `[replace]` `sister-in-law, you know, she,` → `sister-in-law you know she`
- `[replace]` `track, you know, spanglish.` → `track you know spanglish`
- `[replace]` `work.` → `work`
- `[replace]` `thing,` → `thing`
- `[replace]` `goofy. yeah,` → `goofy yeah`
- `[replace]` `camarita.` → `camarita`
- `[replace]` `one,` → `one`
- `[replace]` `know.` → `know`
- `[replace]` `doing, you know,` → `doing you know`
- `[replace]` `something. yeah.` → `something yeah`

### Observations

- ✅ TTFT Final improved by 600ms
- ⚠️  Transcript changed significantly (78.3%) — review word diff
- ℹ️  Expected: More readable, structured transcript

---

## Stage 7: `emotion_tone`

**Phase:** Quality  &nbsp;|&nbsp;  **Task:** Emotion / Tone Evaluation

**What was added:** Emotion / Tone Evaluation [Quality]

### Parameters / configuration tested

**Azure / SDK parameters**

- `Confidence from NBest JSON`
- `Segment word rate estimation`
- `Tone/disfluency keyword proxy analysis`

**Changes applied / tested**

- No transcript rewriting
- Quality proxy from confidence/rate/disfluency markers

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **7** |
| Stage Name | **`emotion_tone`** |
| Phase | Quality |
| Task | Emotion / Tone Evaluation |
| Detected Language | en-US |
| TTFT Partial | 2452.9 ms |
| TTFT Final | 3568.4 ms |
| Total Time | 442.58 sec |
| Segments | 86 |
| Words | 1291 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 745 |

### Transcript

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work So if you see me this thing i look kind of goofy yeah and records it on como los chips de la camarita you know the tiny little One and then you know she has some sort of Program because she's doing you know she's got like a Grant or something yeah and then they transcribe it and then i guess they figure out you know whatever she does something to do with transcribing and spanglish and you know everything You know personal or something so Y tu amiga Hello 4 wheel drive Nieve mi hermana vive en Georgia y no dio nada de nieve bueno yo no soy hasta la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora
...(truncated)
```

### Change vs Previous Stage (`dictation_mode` → `emotion_tone`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 2485.1 | 2452.9 | -32.2 | ✅ improved |
| ttft_final_ms | 3557.5 | 3568.4 | +10.9 | ⚠️ worse |
| total_time_sec | 442.42 | 442.58 | +0.16 | ⚠️ worse |
| segment_count | 86 | 86 | 0 | ➡️ same |
| word_count | 1291 | 1291 | 0 | ➡️ same |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 100.0%
**Word-level changes:** 0

### Observations

- ➡️  Transcript unchanged (100.0% similar) — feature impact is in metrics, not text
- ℹ️  Expected: Robust recognition measurement under varied speech

---

## Stage 8: `latency_testing`

**Phase:** Testing  &nbsp;|&nbsp;  **Task:** Latency & Timeout Testing

**What was added:** Latency & Timeout Testing [Testing]

### Parameters / configuration tested

**Azure / SDK parameters**

- `3 repeated recognition runs`
- `TTFT Partial`
- `TTFT Final`
- `avg / p90 / p95 latency estimation`

**Changes applied / tested**

- Measured latency stability across runs
- Checked SLA-style thresholds

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **8** |
| Stage Name | **`latency_testing`** |
| Phase | Testing |
| Task | Latency & Timeout Testing |
| Detected Language | en-US |
| TTFT Partial | 2545.4 ms |
| TTFT Final | 3652.0 ms |
| Total Time | 442.19 sec |
| Segments | 86 |
| Words | 1291 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 727 |

### Transcript

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work So if you see me this thing i look kind of goofy yeah and records it on como los chips de la camarita you know the tiny little One and then you know she has some sort of Program because she's doing you know she's got like a Grant or something yeah and then they transcribe it and then i guess they figure out you know whatever she does something to do with transcribing and spanglish and you know everything You know personal or something so Y tu amiga Hello 4 wheel drive Nieve mi hermana vive en Georgia y no dio nada de nieve bueno yo no soy hasta la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora
...(truncated)
```

### Change vs Previous Stage (`emotion_tone` → `latency_testing`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 2452.9 | 2545.4 | +92.5 | ⚠️ worse |
| ttft_final_ms | 3568.4 | 3652.0 | +83.6 | ⚠️ worse |
| total_time_sec | 442.58 | 442.19 | -0.39 | ✅ improved |
| segment_count | 86 | 86 | 0 | ➡️ same |
| word_count | 1291 | 1291 | 0 | ➡️ same |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 100.0%
**Word-level changes:** 0

### Observations

- ⚠️  TTFT Final slower by 84ms (may be acceptable)
- ➡️  Transcript unchanged (100.0% similar) — feature impact is in metrics, not text
- ℹ️  Expected: Smooth turn-taking

---

## Stage 9: `realtime_socket`

**Phase:** Integration  &nbsp;|&nbsp;  **Task:** Real-Time Socket Integration

**What was added:** Real-Time Socket Integration [Integration]

### Parameters / configuration tested

**Azure / SDK parameters**

- `PushAudioInputStream`
- `AudioStreamFormat(samples_per_second, bits_per_sample, channels)`
- `40 ms chunk streaming`

**Changes applied / tested**

- Simulated real-time streaming ingestion
- Compared with file-based recognition
- Push stream config: chunk_ms=40, chunk_count=22557, sample_rate=16000

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **9** |
| Stage Name | **`realtime_socket`** |
| Phase | Integration |
| Task | Real-Time Socket Integration |
| Detected Language | en-US |
| TTFT Partial | 5103.4 ms |
| TTFT Final | 9677.4 ms |
| Total Time | 600.94 sec |
| Segments | 64 |
| Words | 1122 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 689 |

### Transcript

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work So if you see me this thing i look kind of goofy yeah and records it on como los chips de la camarita you know the tiny little One and then you know she has some sort of Program because she's doing you know she's got like a Grant or something yeah and then they transcribe it and then i guess they figure out you know whatever she does something to do with transcribing and spanglish and you know everything You know personal or something so Y tu amiga Hello 4 wheel drive Nieve mi hermana vive en Georgia y no dio nada de nieve bueno yo no soy hasta la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora
...(truncated)
```

### Change vs Previous Stage (`latency_testing` → `realtime_socket`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 2545.4 | 5103.4 | +2558.0 | ⚠️ worse |
| ttft_final_ms | 3652.0 | 9677.4 | +6025.4 | ⚠️ worse |
| total_time_sec | 442.19 | 600.94 | +158.75 | ⚠️ worse |
| segment_count | 86 | 64 | -22 | ⚠️ worse |
| word_count | 1291 | 1122 | -169 | ⚠️ worse |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 93.0%
**Word-level changes:** 1

**Sample word changes:**

- `[delete]` `i love it a gordie and we went marito me acuerdo que tú me dijiste el enfermero no oh that's not fair right that's not fair that's not fair say hello look i have this issue they're not going to give a good reference can he just tell them they're not well that's not fair but can't he do something about it and say that they you know de todo un poco right right dad thought you can't do that que atrás de encogida contigo right right right that'll give you a whole lot yo no sé cómo la gente vive de unemployment yo no sé cómo la gente vive de un employee porque es un país yo me hago la única vez que is that coming here jim can i draw that's good no gets along with him that's great right e for kristen is good because one of the papa which could be a little could be a little challenging that's great that's great which one 43` → `(nothing)`

### Observations

- ⚠️  TTFT Final slower by 6025ms (may be acceptable)
- ⚠️  169 fewer words — check endpointing
- ➡️  Transcript similar (93.0%) — small word-level changes
- ℹ️  Expected: Low-latency real-time ASR

---

## Stage 10: `concurrency`

**Phase:** Testing  &nbsp;|&nbsp;  **Task:** Load & Concurrency Testing

**What was added:** Load & Concurrency Testing [Testing]

### Parameters / configuration tested

**Azure / SDK parameters**

- `Parallel SpeechRecognizer sessions`
- `Concurrency levels [1, 3, 5, 10]`
- `Throttle/quota detection`

**Changes applied / tested**

- Measured multi-stream stability
- Checked probable concurrency ceiling
- Concurrency tested: levels=[1, 3, 5, 10], max_safe=10, ceiling=None

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **10** |
| Stage Name | **`concurrency`** |
| Phase | Testing |
| Task | Load & Concurrency Testing |
| Detected Language | en-US |
| TTFT Partial | 2355.1 ms |
| TTFT Final | 3362.1 ms |
| Total Time | 442.47 sec |
| Segments | 86 |
| Words | 1291 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 738 |

### Transcript

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work So if you see me this thing i look kind of goofy yeah and records it on como los chips de la camarita you know the tiny little One and then you know she has some sort of Program because she's doing you know she's got like a Grant or something yeah and then they transcribe it and then i guess they figure out you know whatever she does something to do with transcribing and spanglish and you know everything You know personal or something so Y tu amiga Hello 4 wheel drive Nieve mi hermana vive en Georgia y no dio nada de nieve bueno yo no soy hasta la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora
...(truncated)
```

### Change vs Previous Stage (`realtime_socket` → `concurrency`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 5103.4 | 2355.1 | -2748.3 | ✅ improved |
| ttft_final_ms | 9677.4 | 3362.1 | -6315.3 | ✅ improved |
| total_time_sec | 600.94 | 442.47 | -158.47 | ✅ improved |
| segment_count | 64 | 86 | +22 | ✅ improved |
| word_count | 1122 | 1291 | +169 | ✅ improved |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 93.0%
**Word-level changes:** 1

**Sample word changes:**

- `[insert]` `(nothing)` → `i love it a gordie and we went marito me acuerdo que tú me dijiste el enfermero no oh that's not fair right that's not fair that's not fair say hello look i have this issue they're not going to give a good reference can he just tell them they're not well that's not fair but can't he do something about it and say that they you know de todo un poco right right dad thought you can't do that que atrás de encogida contigo right right right that'll give you a whole lot yo no sé cómo la gente vive de unemployment yo no sé cómo la gente vive de un employee porque es un país yo me hago la única vez que is that coming here jim can i draw that's good no gets along with him that's great right e for kristen is good because one of the papa which could be a little could be a little challenging that's great that's great which one 43`

### Observations

- ✅ TTFT Final improved by 6315ms
- ✅ 169 more words captured — less truncation
- ➡️  Transcript similar (93.0%) — small word-level changes
- ℹ️  Expected: Stable under load

---

## Stage 11: `logging_alerts`

**Phase:** Monitoring  &nbsp;|&nbsp;  **Task:** Logging & Alerts Setup

**What was added:** Logging & Alerts Setup [Monitoring]

### Parameters / configuration tested

**Azure / SDK parameters**

- `SPEECH_SDK_LOGFILE`
- `Structured session JSON logging`
- `Alert rules for latency/confidence/cancel/error`

**Changes applied / tested**

- Enabled diagnostic logging
- Generated alert artifacts
- Logging enabled: alerts_fired=86, sdk_log=observations\stage_11_logging_alerts\logs\azure_sdk.log

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **11** |
| Stage Name | **`logging_alerts`** |
| Phase | Monitoring |
| Task | Logging & Alerts Setup |
| Detected Language | en-US |
| TTFT Partial | 2435.1 ms |
| TTFT Final | 3552.3 ms |
| Total Time | 442.39 sec |
| Segments | 86 |
| Words | 1291 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 737 |

### Transcript

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work So if you see me this thing i look kind of goofy yeah and records it on como los chips de la camarita you know the tiny little One and then you know she has some sort of Program because she's doing you know she's got like a Grant or something yeah and then they transcribe it and then i guess they figure out you know whatever she does something to do with transcribing and spanglish and you know everything You know personal or something so Y tu amiga Hello 4 wheel drive Nieve mi hermana vive en Georgia y no dio nada de nieve bueno yo no soy hasta la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora
...(truncated)
```

### Change vs Previous Stage (`concurrency` → `logging_alerts`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 2355.1 | 2435.1 | +80.0 | ⚠️ worse |
| ttft_final_ms | 3362.1 | 3552.3 | +190.2 | ⚠️ worse |
| total_time_sec | 442.47 | 442.39 | -0.08 | ✅ improved |
| segment_count | 86 | 86 | 0 | ➡️ same |
| word_count | 1291 | 1291 | 0 | ➡️ same |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 100.0%
**Word-level changes:** 0

### Observations

- ⚠️  TTFT Final slower by 190ms (may be acceptable)
- ➡️  Transcript unchanged (100.0% similar) — feature impact is in metrics, not text
- ℹ️  Expected: Early issue detection

---

## Stage 12: `fallback`

**Phase:** Go-Live  &nbsp;|&nbsp;  **Task:** Fallback Validation

**What was added:** Fallback Validation [Go-Live]

### Parameters / configuration tested

**Azure / SDK parameters**

- `Reduced InitialSilenceTimeoutMs = 3000 for fallback test`
- `Language retry with reversed candidate order`
- `Silence file no-speech path`

**Changes applied / tested**

- Simulated re-prompt flow
- Simulated language retry
- Simulated DTMF / agent escalation fallback
- Fallback chain tested: reprompt=True, lang_retry=True, dtmf=True

### Metrics

| Metric | Value |
|--------|-------|
| Stage Number | **12** |
| Stage Name | **`fallback`** |
| Phase | Go-Live |
| Task | Fallback Validation |
| Detected Language | en-US |
| TTFT Partial | 2494.8 ms |
| TTFT Final | 3562.2 ms |
| Total Time | 442.23 sec |
| Segments | 86 |
| Words | 1291 |
| Empty Segments | 0 |
| Avg Confidence | N/A |
| Min Confidence | N/A |
| Max Confidence | N/A |
| Partial Count | 729 |

### Transcript

```
My sister-in-law you know she she's a linguist and so she wants to sort of track you know spanglish so she asked me if I would wear it at work So if you see me this thing i look kind of goofy yeah and records it on como los chips de la camarita you know the tiny little One and then you know she has some sort of Program because she's doing you know she's got like a Grant or something yeah and then they transcribe it and then i guess they figure out you know whatever she does something to do with transcribing and spanglish and you know everything You know personal or something so Y tu amiga Hello 4 wheel drive Nieve mi hermana vive en Georgia y no dio nada de nieve bueno yo no soy hasta la casa de ella bueno puede que si en las montañas sí sí haya Más muerta que vivas Pero no cara a qué hora
...(truncated)
```

### Change vs Previous Stage (`logging_alerts` → `fallback`)

| Metric | Before | After | Change | Signal |
|--------|--------|-------|--------|--------|
| ttft_partial_ms | 2435.1 | 2494.8 | +59.7 | ⚠️ worse |
| ttft_final_ms | 3552.3 | 3562.2 | +9.9 | ⚠️ worse |
| total_time_sec | 442.39 | 442.23 | -0.16 | ✅ improved |
| segment_count | 86 | 86 | 0 | ➡️ same |
| word_count | 1291 | 1291 | 0 | ➡️ same |
| empty_segments | 0 | 0 | 0 | ➡️ same |

**Transcript similarity vs previous:** 100.0%
**Word-level changes:** 0

### Observations

- ➡️  Transcript unchanged (100.0% similar) — feature impact is in metrics, not text
- ℹ️  Expected: Resilient failure handling

---

## Net Gain: Baseline → Latest Stage

Comparing **Stage 0: `baseline`** → **Stage 12: `fallback`**

| Metric | Baseline | Latest | Net Change |
|--------|----------|--------|------------|
| word_count | 1240 | 1291 | +51 |
| segment_count | 83 | 86 | +3 |
| ttft_final_ms | 3656.8 | 3562.2 | -94.6 |
| ttft_partial_ms | 2505.8 | 2494.8 | -11.0 |
| empty_segments | 0 | 0 | 0 |
| total_time_sec | 442.96 | 442.23 | -0.73 |

---

## Stage Progression Summary

| # | Stage Name | Phase | Words | Segs | Overall Quality | Avg Conf | TTFT Final (ms) | Total Time (s) |
|---|-----------|-------|-------|------|-----------------|----------|-----------------|----------------|
| 0 | `baseline` | Baseline | 1240 | 83 | N/A | N/A | 3656.8 | 442.96 |
| 1 | `asr_config` | Setup | 1240 | 83 | N/A | N/A | 3652.7 | 443.12 |
| 2 | `vad_tuning` | Audio | 460 | 34 | N/A | N/A | 6872.0 | 179.69 |
| 3 | `phrase_boost` | Accuracy | 1292 | 86 | N/A | N/A | 4104.5 | 442.23 |
| 4 | `vocab_tuning` | Accuracy | 1328 | 87 | N/A | N/A | 3748.1 | 442.86 |
| 5 | `numeric_handling` | Logic | 1292 | 86 | N/A | N/A | 4157.3 | 442.54 |
| 6 | `dictation_mode` | Accuracy | 1291 | 86 | N/A | N/A | 3557.5 | 442.42 |
| 7 | `emotion_tone` | Quality | 1291 | 86 | N/A | N/A | 3568.4 | 442.58 |
| 8 | `latency_testing` | Testing | 1291 | 86 | N/A | N/A | 3652.0 | 442.19 |
| 9 | `realtime_socket` | Integration | 1122 | 64 | N/A | N/A | 9677.4 | 600.94 |
| 10 | `concurrency` | Testing | 1291 | 86 | N/A | N/A | 3362.1 | 442.47 |
| 11 | `logging_alerts` | Monitoring | 1291 | 86 | N/A | N/A | 3552.3 | 442.39 |
| 12 | `fallback` | Go-Live | 1291 | 86 | N/A | N/A | 3562.2 | 442.23 |

---

## Metric Reference

| Metric | Description | Better direction |
|--------|-------------|-----------------|
| TTFT Partial | Time to first interim result | Lower |
| TTFT Final | Time to first committed segment | Lower |
| Avg Confidence | Azure certainty score 0–1 | Higher |
| Word Count | Total words captured | Usually higher |
| Empty Segments | Segments with no text | Lower |
| Total Time | End-to-end processing time | Lower |
| Overall Quality | Composite transcript quality score | Higher |
| Numeric Quality | Number rendering / numeric-context quality | Higher |
| VAD Quality | Segmentation / truncation quality | Higher |

*Generated by `generate_observation_doc.py` on 2026-04-21 13:20*