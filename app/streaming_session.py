import time
from app.vad import AdaptiveEnergyVAD


class StreamingSession:
    """
    Handles:
    - VAD start/end
    - partial streaming
    - long speech rollover
    - cumulative transcript
    """

    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg

        self.vad = AdaptiveEnergyVAD(
            cfg.sample_rate,
            cfg.vad_frame_ms,
            cfg.vad_start_margin,
            cfg.vad_min_noise_rms,
            cfg.pre_speech_ms,
        )

        self.session = engine.new_session(max_buffer_ms=cfg.max_utt_ms)

        self.frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
        self.raw_buf = bytearray()

        self.utt_started = False
        self.utt_audio_ms = 0

        self.t_utt_start = None
        self.t_first_partial = None
        self.silence_ms = 0

        # cumulative transcript across multiple finalized chunks
        self.full_transcript = []
        self.current_lang = "en"

        # hard rollover for long no-pause speech
        self.force_rollover_ms = 15000

    def process_chunk(self, pcm):
        events = []

        self.raw_buf.extend(pcm)

        while len(self.raw_buf) >= self.frame_bytes:
            frame = bytes(self.raw_buf[:self.frame_bytes])
            del self.raw_buf[:self.frame_bytes]

            is_speech, pre = self.vad.push_frame(frame)

            self.silence_ms = 0 if is_speech else self.silence_ms + self.cfg.vad_frame_ms

            if pre and not self.utt_started:
                self.utt_started = True
                self.utt_audio_ms = 0
                self.t_utt_start = time.time()
                self.t_first_partial = None

                self.session.accept_pcm16(pre)

            if not self.utt_started:
                continue

            self.session.accept_pcm16(frame)
            self.utt_audio_ms += self.cfg.vad_frame_ms

            if self.engine.caps.partials:
                result = self.session.step_if_ready()

                if result:
                    text, lang = result
                    self.current_lang = lang

                    if self.t_first_partial is None:
                        self.t_first_partial = time.time()

                    ttfb_ms = int((self.t_first_partial - self.t_utt_start) * 1000)

                    # partial should include previous finalized chunks + live partial
                    if self.full_transcript:
                        partial_text = " ".join(self.full_transcript + [text]).strip()
                    else:
                        partial_text = text

                    events.append(("partial", partial_text, lang, ttfb_ms))

            # forced rollover for very long continuous speech
            if self.utt_audio_ms >= self.force_rollover_ms:
                final, lang = self.session.finalize(self.engine.finalize_pad_ms)

                if final:
                    self.full_transcript.append(final)
                    self.current_lang = lang

                    ttfb_ms = (
                        int((self.t_first_partial - self.t_utt_start) * 1000)
                        if self.t_first_partial else None
                    )

                    full_text = " ".join(self.full_transcript).strip()
                    events.append(("final", full_text, lang, ttfb_ms))

                self._reset_utterance_only()
                continue

            # normal silence-based finalization
            if (
                not is_speech
                and self.utt_audio_ms >= self.engine.min_utt_ms
                and self.silence_ms >= self.engine.end_silence_ms
            ):
                final, lang = self.session.finalize(self.engine.finalize_pad_ms)

                if final:
                    self.full_transcript.append(final)
                    self.current_lang = lang

                    ttfb_ms = (
                        int((self.t_first_partial - self.t_utt_start) * 1000)
                        if self.t_first_partial else None
                    )

                    full_text = " ".join(self.full_transcript).strip()
                    events.append(("final", full_text, lang, ttfb_ms))

                self._reset_utterance_only()

        return events

    def flush(self):
        events = []

        if self.utt_started:
            final, lang = self.session.finalize(self.engine.finalize_pad_ms)

            if final:
                self.full_transcript.append(final)
                self.current_lang = lang

        if self.full_transcript:
            ttfb_ms = (
                int((self.t_first_partial - self.t_utt_start) * 1000)
                if self.t_first_partial and self.t_utt_start else None
            )

            full_text = " ".join(self.full_transcript).strip()
            events.append(("final", full_text, self.current_lang, ttfb_ms))

        self.reset_all()
        return events

    def _reset_utterance_only(self):
        self.vad.reset()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None
        self.raw_buf.clear()

    def reset_all(self):
        self._reset_utterance_only()
        self.full_transcript = []
        self.current_lang = "en"