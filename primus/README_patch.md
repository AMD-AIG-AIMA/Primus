
# Primus Patch Notes & Extended Argument Documentation

This document records the modifications made to integrate and extend **Megatron** and **TorchTitan** within the Primus framework via patching. It includes new arguments introduced for configuration and highlights the affected code paths.

---

## Sections

- [Primus Patch Notes \& Extended Argument Documentation](#primus-patch-notes--extended-argument-documentation)
  - [Sections](#sections)
  - [1. Base Module Parameters](#1-base-module-parameters)
  - [2. Megatron Patch Summary](#2-megatron-patch-summary)
    - [2.1 Module-Level Parameters](#21-module-level-parameters)
    - [2.2 Model-Definition Parameters](#22-model-definition-parameters)
  - [3. TorchTitan Patch Summary](#3-torchtitan-patch-summary)

---

## 1. Base Module Parameters
The following arguments are defined in the base module configuration file:
[primus/configs/modules/module_base.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/primus/configs/modules/module_base.yaml)
This base config is inherited by all other modules in the framework, so every module supports these parameters. These options control whether a module participates in training and how its logging behaves.

| Argument Name       | Default Value | Description                                                                                |
| ------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| `trainable`         | `false`       | Whether the module is trainable.                                                           |
| `sink_level`        | `null`        | Global sink level for logging. Overrides `file_sink_level` and `stderr_sink_level` if set. |
| `file_sink_level`   | `DEBUG`       | Logging level for file sink (e.g., log file output).                                       |
| `stderr_sink_level` | `INFO`        | Logging level for standard error (console) output.                                         |

---


## 2. Megatron Patch Summary

### 2.1 Module-Level Parameters

These arguments are introduced in the Megatron module logic (e.g., training loop, logging, resume logic). They are defined via patching and can be configured to control training behavior and logging utilities.

| New Argument                         | Default Value | Version | Description                                                                                    | Patched Files                                                                                                                                                                                                                                                                                                | Notes                                            |
| ------------------------------------ | ------------- | ------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| `disable_tensorboard`                | `true`        | v0.1.0  | Whether to disable TensorBoard. Set to `false` if you want to enable profiling or torch trace. | NA                                                                                                                                                                                                                                                                                                           | Required for timeline and performance debugging. |
| `disable_wandb`                      | `true`        | v0.1.0  | Whether to disable Weights & Biases logging.                                                   | NA                                                                                                                                                                                                                                                                                                           | Useful for internal benchmarking.                |
| `disable_compile_dependencies`       | `true`        | v0.1.0  | Disables Megatron’s custom kernel compilation. Most ops are already covered by TE.             | NA                                                                                                                                                                                                                                                                                                           | Avoids redundant compilation steps.              |
| `auto_continue_train`                | `false`       | v0.1.0  | Automatically resume training from the latest checkpoint if found in the `--save` path.        | NA                                                                                                                                                                                                                                                                                                           | Simplifies job restarts.                         |
| `disable_last_saving`                | `false`       | v0.1.0  | Skip saving the final checkpoint at the last iteration.                                        | NA                                                                                                                                                                                                                                                                                                           | Useful for profiling or benchmarking runs.       |
| `no_fp8_weight_transpose_cache`      | `false`       | v0.2.0  | Disable the FP8 weight transpose cache to save memory.                                         | `megatron.core.extensions.transformer_engine.TELinear`, `megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear`, `megatron.core.extensions.transformer_engine.TEDelayedScaling`                                                                                                        | May affect performance but reduce memory use.    |
| `decoder_pipeline_manual_split_list` | `null`        | v0.2.0  | Enable manual pipeline split in (interleaved) 1F1B pipeline parallelism.                       | `megatron.core.transformer.transformer_block.get_num_layers_to_build`, `megatron.core.transformer.transformer_layer.get_transformer_layer_offset`                                                                                                                                                            | May be deprecated when megatron gets updated.    |
| `pp_warmup`                          | `false`       | v0.2.0  | Add fwd/bwd warmup to save iter1's time when pp degree is large.                             | NA                                                                                                                                                                                                                                                                                                           | Can save much time for pipeline debug.           |
| `dump_pp_data`                       | `false`       | v0.2.0  | Enable dumping pp schedule data for visualization.                                             | `megatron.core.pipeline_parallel.schedules.forward_step`, `megatron.core.pipeline_parallel.schedules.backward_step`, `megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving`, `megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving` | Useful for pipeline schedule visualization.      |
| `disable_profiler_activity_cpu`                | `false`       | v0.2.0  | Disable CPU activityt in torch profiling, .                                        | NA                                                                                                                                                                                                                                                                                                           | If you only want to trace CUDA kernels and get a smaller trace JSON file, you can enable this option. However, if you plan to run with TraceLen, please do not enable it.       |

---

### 2.2 Model-Definition Parameters

These arguments affect the internal architecture or layer implementations. They are patched into the model construction logic and used for tuning or debugging specific variants.

| New Argument                        | Default Value | Version | Description                                                               | Patched Files                                                                                                                                                                                                                                                                                                                   | Notes                                       |
| ----------------------------------- | ------------- | ------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `disable_primus_topk_router`   | `false`       | v0.1.0  | Disable PrimusTopkRouter and use TopkRouter implemented by megatron. | `megatron.core.transformer.moe.router.TopKRouter`                                                                                                                                                                                                                                                                               | Used to debug internal.         |
| `moe_router_force_load_balancing`   | `false`       | v0.1.0  | Force token redistribution in MoE to achieve load balance across experts. | `megatron.core.transformer.moe.router.TopKRouter`                                                                                                                                                                                                                                                                               | Use to debug MoE imbalance issues.          |
| `use_deprecated_20241209_moe_layer` | `false`       | v0.1.0  | Enable legacy MoE implementation for debugging/perf comparison.           | `megatron.core.transformer.moe.moe_layer.MoELayer`, `megatron.core.transformer.moe.moe_layer.MoESubmodules`, `megatron.core.transformer.moe.experts.GroupedMLP`, `megatron.core.transformer.moe.experts.SequentialMLP`, `megatron.core.transformer.moe.experts.TEGroupedMLP`, `megatron.core.transformer.moe.router.TopKRouter` | Deprecated, used for internal testing only. |
| `moe_permute_fusion`   | `false`       | v0.1.0  | Permutation and unpermutation fusion. | `megatron.core.extensions.transformer_engine`, `megatron.core.transformer.moe.moe_utils`                                                                                                                                                                                                                                                                               | Fuse permutation and unpermutation in moe layer.         |
| `fused_padded_mla_attention`   | `false`       | v0.1.0  | Pad the V head dim to match the Q head dim. | `megatron.core.transformer.multi_latent_attention.PaddedMLASelfAttention`                                                                                                                                                                                                                                                                               | To enable fused attention and reduce memory usage, this module pads the V tensor so that all Q, K, and V have a uniform head dimension of 192. After padding, AMD TE's fused attention can be invoked, resulting in more efficient memory usage and improved performance..         |
| `moe_use_fused_router_with_aux_score`   | `false`       | v0.2.0  | Fused router topk and calculation of moe aux loss score. | `megatron.core.transformer.moe.router.TopKRouter`                                                                                                                                                                                                                                                                               | Used to reduce launch overhead of the small kernels in router.         |
| `enable_primus_turbo`   | `false`       | v0.2.0  | Use Primus turbo as backend. | `megatron.core.models.gpt.gpt_layer_specs.TEDotProductAttention`, `megatron.core.models.gpt.gpt_layer_specs.PrimusTurboRowParallelLinear`, `megatron.core.models.gpt.gpt_layer_specs.TELayerNormColumnParallelLinear`, `megatron.core.models.gpt.gpt_layer_specs.TEColumnParallelLinear`, `megatron.core.models.gpt.gpt_model.tensor_parallel.ColumnParallelLinear`, `megatron.core.models.gpt.moe_module_specs.GroupedMLP`, `megatron.core.models.gpt.moe_module_specs.TEColumnParallelLinear`, `megatron.core.models.gpt.moe_module_specs.TERowParallelLinear`                                                                                                                                                                                                                                                                           | Used to accelerate training. See fine-grained control flags in primus-turbo.yaml        |


---

## 3. TorchTitan Patch Summary

| New Argument | Default Value | Version | Description | Patched Files | Notes |
| ------------ | ------------- | ------- | ----------- | ------------- | ----- |
| `ABC`        | `true`        | v0.1.0  | ABC         | `abc.py`      | ABC   |

---
