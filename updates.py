 => [6/9] RUN python3.11 -m pip install --no-cache-dir -r requirements.txt                                                                                                                                                         197.1s
 => ERROR [7/9] RUN python3.11 - <<EOF                                                                                                                                                                                               9.1s
------
 > [7/9] RUN python3.11 - <<EOF:
7.598 [NeMo W 2026-03-30 11:07:23 nemo_logging:405] Megatron num_microbatches_calculator not found, using Apex version.
8.077 OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
8.088 No exporters were provided. This means that no telemetry data will be collected.
9.065 Segmentation fault (core dumped)
------
Dockerfile:23
--------------------
  22 |
  23 | >>> RUN python3.11 - <<EOF
  24 | >>> import nemo.collections.asr as nemo_asr
  25 | >>> print("Downloading Parakeet model...")
  26 | >>> _ = nemo_asr.models.ASRModel.from_pretrained(
  27 | >>>     model_name="nvidia/parakeet-tdt-0.6b-v3",
  28 | >>>     map_location="cpu"
  29 | >>> )
  30 | >>> print("Parakeet downloaded successfully.")
  31 | >>> EOF
  32 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c python3.11 - <<EOF\nimport nemo.collections.asr as nemo_asr\nprint(\"Downloading Parakeet model...\")\n_ = nemo_asr.models.ASRModel.from_pretrained(\n    model_name=\"nvidia/parakeet-tdt-0.6b-v3\",\n    map_location=\"cpu\"\n)\nprint(\"Parakeet downloaded successfully.\")\nEOF" did not complete successfully: exit code: 139
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/asr_parakeet_tdt_0.6b_v3#
