try:
    fmt = speechsdk.audio.AudioStreamFormat(
        samples_per_second=sample_rate,
        bits_per_sample=16,
        channels=1
    )
except Exception:
    fmt = None


if fmt:
    stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
else:
    stream = speechsdk.audio.PushAudioInputStream()
