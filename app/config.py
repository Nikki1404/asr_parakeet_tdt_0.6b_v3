# app/config.py

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Global app config.
    Keep ONLY server-level / shared settings here.
    Model-specific tuning stays inside engine files.
    """

    # Backend selection
    asr_backend: str = "parakeet"
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    device: str = "cuda"

    # Audio
    sample_rate: int = 16000

    # VAD
    vad_frame_ms: int = 20
    vad_start_margin: float = 2.5
    vad_min_noise_rms: float = 0.003
    pre_speech_ms: int = 300

    # Safety
    max_utt_ms: int = 30000

    # Logging
    log_level: str = "INFO"


MODEL_MAP = {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
}


def load_config() -> Config:
    cfg = Config()

    print(
        f"DEBUG | "
        f"backend={cfg.asr_backend} | "
        f"model={cfg.model_name} | "
        f"device={cfg.device} | "
        f"sample_rate={cfg.sample_rate}"
    )

    return cfg