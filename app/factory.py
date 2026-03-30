from app.config import Config
from app.asr_engines.parakeet_asr import ParakeetASR


def build_engine(cfg: Config):
    if cfg.asr_backend == "parakeet":
        return ParakeetASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
        )

    raise ValueError(f"Unsupported ASR_BACKEND={cfg.asr_backend}")