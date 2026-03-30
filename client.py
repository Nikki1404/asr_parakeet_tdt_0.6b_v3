(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/asr_parakeet_tdt_0.6b_v3# docker run --gpus all -p 8000:8000 parakeet_asr

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

DEBUG | backend=parakeet | model=nvidia/parakeet-tdt-0.6b-v3 | device=cuda | sample_rate=16000
INFO:     Started server process [1]
INFO:     Waiting for application startup.
2026-03-30 11:30:48,587 | INFO | asr_server | Server startup initiated
2026-03-30 11:30:48,587 | INFO | asr_server | Preloading ASR engines...
2026-03-30 11:30:48,587 | INFO | asr_server | Initializing engine: parakeet
2026-03-30 11:30:50,586 | INFO | matplotlib.font_manager | generated new fontManager
2026-03-30 11:30:52,114 | INFO | numexpr.utils | NumExpr defaulting to 8 threads.
[NeMo W 2026-03-30 11:30:53 nemo_logging:405] Megatron num_microbatches_calculator not found, using Apex version.
2026-03-30 11:30:53,434 | WARNING | nv_one_logger.api.config | OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
2026-03-30 11:30:53,444 | INFO | nv_one_logger.exporter.export_config_manager | Final configuration contains 0 exporter(s)
2026-03-30 11:30:53,444 | WARNING | nv_one_logger.training_telemetry.api.training_telemetry_provider | No exporters were provided. This means that no telemetry data will be collected.

                                                                                                                                                    
                                                                                                                                                    python3.11 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3.11 -c "import nemo.collections.asr as nemo_asr; print('nemo import ok')"
python3.11 -c "import nemo.collections.asr as nemo_asr; m=nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v3'); print('model load ok')"
