from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from typing import Any, Mapping


class _FallbackConfig:
    model_type = "config"

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


try:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as _BaseQwen3Config
except ImportError:
    _BaseQwen3Config = _FallbackConfig


@dataclass(frozen=True)
class GfstHmbConfig:
    reduced_dim: int = 64
    flag_dims: tuple[int, ...] = (8, 16, 32)
    block_len: int = 2048
    retrieval_depth: int = 8
    key_dim: int = 256
    exact_cache_blocks: int = 4
    store_exact_hidden_states: bool = False
    exact_hidden_store_mode: str = "tuple"
    pointer_value_source: str = "token_embedding"
    centroid_recall_value_source: str = "token_embedding"
    centroid_activation_enabled: bool = True
    cells_per_block: int = 16
    centroid_heads: int = 4
    centroid_key_dim: int = 32
    centroid_value_dim: int = 16
    centroid_top_cells_per_head: int = 8
    centroid_assignment_top_k: int = 2
    centroid_assignment_temperature: float = 0.35
    centroid_train_support_projection: str = "topk_sparsemax"
    centroid_train_support_k: int = 2
    centroid_write_layer: int = 0
    centroid_phase_binding_enabled: bool = True
    centroid_phase_period: int = 32
    centroid_phase_bands: int = 4
    centroid_score_normalization: str = "robust_zscore"
    centroid_background_subtract: bool = True
    centroid_count_penalty: float = 0.5
    centroid_selection_policy: str = "bh_fdr"
    centroid_fdr_q: float = 0.10
    centroid_min_selected_cells_per_head: int = 2
    centroid_selection_temperature: float = 1.0
    centroid_block_support_threshold: float = 0.10
    centroid_target_active_cells_per_block: float = 3.5
    centroid_self_field_weight: float = 0.15
    centroid_gain_entropy_low: float = 0.35
    centroid_gain_entropy_high: float = 0.75
    centroid_gain_margin_low: float = 0.25
    centroid_gain_margin_high: float = 1.0
    copy_readout_layer: int = -1
    forgetting_rates: tuple[float, ...] = (0.05, 0.10, 0.20)
    retrieval_temperature: float = 1.0
    eps: float = 1e-6
    gfst_update_rule: str = "residual_rank_one_v1"
    summary_layout_version: int = 1
    retrieval_key_version: str = "mean_top1_v1"
    copy_readout_version: str = "exact_pointer_v1"
    router_hidden_dim: int = 256
    eps_update: float = 1e-6
    eps_sign: float = 1e-8
    init_scale: float = 0.02

    def __post_init__(self) -> None:
        if self.reduced_dim <= 0:
            raise ValueError("reduced_dim must be positive")
        if self.reduced_dim % 2 != 0:
            raise ValueError("reduced_dim must be even so RoPE can be applied")
        if self.block_len <= 0:
            raise ValueError("block_len must be positive")
        if self.retrieval_depth <= 0:
            raise ValueError("retrieval_depth must be positive")
        if self.key_dim <= 0:
            raise ValueError("key_dim must be positive")
        if self.exact_cache_blocks <= 0:
            raise ValueError("exact_cache_blocks must be positive")
        if self.exact_hidden_store_mode not in {"tuple", "packed_bf16"}:
            raise ValueError("exact_hidden_store_mode must be one of {'tuple', 'packed_bf16'}")
        if self.pointer_value_source not in {"token_embedding", "exact_hidden_state"}:
            raise ValueError("pointer_value_source must be one of {'token_embedding', 'exact_hidden_state'}")
        if self.centroid_recall_value_source not in {"token_embedding", "exact_hidden_state"}:
            raise ValueError(
                "centroid_recall_value_source must be one of {'token_embedding', 'exact_hidden_state'}"
            )
        if self.cells_per_block <= 0:
            raise ValueError("cells_per_block must be positive")
        if self.centroid_heads <= 0:
            raise ValueError("centroid_heads must be positive")
        if self.centroid_key_dim <= 0:
            raise ValueError("centroid_key_dim must be positive")
        if self.centroid_value_dim <= 0:
            raise ValueError("centroid_value_dim must be positive")
        if self.centroid_top_cells_per_head <= 0:
            raise ValueError("centroid_top_cells_per_head must be positive")
        if self.centroid_assignment_top_k <= 0:
            raise ValueError("centroid_assignment_top_k must be positive")
        if self.centroid_assignment_top_k > self.cells_per_block:
            raise ValueError("centroid_assignment_top_k must be <= cells_per_block")
        if self.centroid_assignment_temperature <= 0.0:
            raise ValueError("centroid_assignment_temperature must be positive")
        if self.centroid_train_support_projection != "topk_sparsemax":
            raise ValueError("centroid_train_support_projection must be 'topk_sparsemax'")
        if self.centroid_train_support_k <= 0:
            raise ValueError("centroid_train_support_k must be positive")
        if self.centroid_train_support_k > self.cells_per_block:
            raise ValueError("centroid_train_support_k must be <= cells_per_block")
        if self.centroid_write_layer < 0:
            raise ValueError("centroid_write_layer must be nonnegative")
        if self.centroid_phase_period <= 0:
            raise ValueError("centroid_phase_period must be positive")
        if self.centroid_phase_bands <= 0:
            raise ValueError("centroid_phase_bands must be positive")
        if self.centroid_score_normalization not in {"robust_zscore"}:
            raise ValueError("centroid_score_normalization must be 'robust_zscore'")
        if self.centroid_count_penalty < 0.0:
            raise ValueError("centroid_count_penalty must be nonnegative")
        if self.centroid_selection_policy != "bh_fdr":
            raise ValueError("centroid_selection_policy must be 'bh_fdr'")
        if not (0.0 < self.centroid_fdr_q <= 1.0):
            raise ValueError("centroid_fdr_q must be in (0, 1]")
        if self.centroid_min_selected_cells_per_head <= 0:
            raise ValueError("centroid_min_selected_cells_per_head must be positive")
        if self.centroid_min_selected_cells_per_head > self.centroid_top_cells_per_head:
            raise ValueError("centroid_min_selected_cells_per_head must be <= centroid_top_cells_per_head")
        if self.centroid_selection_temperature <= 0.0:
            raise ValueError("centroid_selection_temperature must be positive")
        if not (0.0 <= self.centroid_block_support_threshold <= 1.0):
            raise ValueError("centroid_block_support_threshold must be in [0, 1]")
        if not (0.0 < self.centroid_target_active_cells_per_block <= float(self.cells_per_block)):
            raise ValueError("centroid_target_active_cells_per_block must be in (0, cells_per_block]")
        if self.centroid_self_field_weight < 0.0:
            raise ValueError("centroid_self_field_weight must be nonnegative")
        if not (0.0 <= self.centroid_gain_entropy_low < self.centroid_gain_entropy_high):
            raise ValueError("centroid gain entropy bounds must satisfy 0 <= low < high")
        if not (0.0 <= self.centroid_gain_margin_low < self.centroid_gain_margin_high):
            raise ValueError("centroid gain margin bounds must satisfy 0 <= low < high")
        if self.gfst_update_rule != "residual_rank_one_v1":
            raise ValueError("gfst_update_rule must be 'residual_rank_one_v1'")
        if int(self.summary_layout_version) <= 0:
            raise ValueError("summary_layout_version must be positive")
        if not self.retrieval_key_version:
            raise ValueError("retrieval_key_version must not be empty")
        if not self.copy_readout_version:
            raise ValueError("copy_readout_version must not be empty")
        if self.router_hidden_dim <= 0:
            raise ValueError("router_hidden_dim must be positive")
        if self.retrieval_temperature <= 0.0:
            raise ValueError("retrieval_temperature must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.eps_update <= 0.0:
            raise ValueError("eps_update must be positive")
        if self.eps_sign <= 0.0:
            raise ValueError("eps_sign must be positive")
        if not self.flag_dims:
            raise ValueError("flag_dims must not be empty")
        if tuple(sorted(self.flag_dims)) != self.flag_dims:
            raise ValueError("flag_dims must be strictly increasing")
        if len(set(self.flag_dims)) != len(self.flag_dims):
            raise ValueError("flag_dims must be unique")
        if max(self.flag_dims) > self.reduced_dim:
            raise ValueError("max(flag_dims) must be <= reduced_dim")
        if len(self.forgetting_rates) != len(self.flag_dims):
            raise ValueError("forgetting_rates must match flag_dims length")
        for rate in self.forgetting_rates:
            if not (0.0 < rate < 1.0):
                raise ValueError("each forgetting rate must be in (0, 1)")
        if tuple(sorted(self.forgetting_rates)) != self.forgetting_rates:
            raise ValueError("forgetting_rates must be nondecreasing from coarse to fine")

    @property
    def feature_dim(self) -> int:
        return 4 * len(self.flag_dims)

    @property
    def max_flag_dim(self) -> int:
        return self.flag_dims[-1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | "GfstHmbConfig" | None = None) -> "GfstHmbConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        normalised = dict(payload)
        for key in tuple(normalised.keys()):
            if str(key).startswith("span_"):
                normalised.pop(key, None)
        allowed = {spec.name for spec in dataclass_fields(cls)}
        filtered = {key: value for key, value in normalised.items() if key in allowed}
        return cls(**filtered)


@dataclass(frozen=True)
class FaissGpuRetrievalConfig:
    index_type: str = "ivfflatip"
    metric: str = "inner_product"
    top_r: int = 8
    nlist: int = 64
    nprobe: int = 8
    flat_fallback_threshold: int = 128
    min_train_points_per_centroid: int = 39
    gpu_device: int = 0
    use_float16: bool = False
    temp_memory_bytes: int = 0
    rebuild_policy: str = "on_state_change"

    def __post_init__(self) -> None:
        if self.index_type not in {"ivfflatip", "flatip"}:
            raise ValueError("index_type must be one of {'ivfflatip', 'flatip'}")
        if self.metric != "inner_product":
            raise ValueError("metric must be 'inner_product'")
        if self.top_r <= 0:
            raise ValueError("top_r must be positive")
        if self.nlist <= 0:
            raise ValueError("nlist must be positive")
        if self.nprobe <= 0:
            raise ValueError("nprobe must be positive")
        if self.flat_fallback_threshold <= 0:
            raise ValueError("flat_fallback_threshold must be positive")
        if self.min_train_points_per_centroid <= 0:
            raise ValueError("min_train_points_per_centroid must be positive")
        if self.gpu_device < 0:
            raise ValueError("gpu_device must be nonnegative")
        if self.temp_memory_bytes < 0:
            raise ValueError("temp_memory_bytes must be nonnegative")
        if self.rebuild_policy != "on_state_change":
            raise ValueError("rebuild_policy must be 'on_state_change'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "FaissGpuRetrievalConfig" | None = None,
    ) -> "FaissGpuRetrievalConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        normalised = dict(payload)
        for key in (
            "top_r",
            "nlist",
            "nprobe",
            "flat_fallback_threshold",
            "min_train_points_per_centroid",
            "gpu_device",
            "temp_memory_bytes",
        ):
            if key in normalised:
                normalised[key] = int(normalised[key])
        if "use_float16" in normalised:
            normalised["use_float16"] = bool(normalised["use_float16"])
        return cls(**normalised)


@dataclass(frozen=True)
class DatasetStreamConfig:
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    train_split: str = "train"
    validation_split: str = "validation"
    text_field: str = "text"
    streaming: bool = True
    cache_dir: str = ""
    shuffle_buffer_size: int = 10_000
    source_manifest_path: str = ""
    require_streaming_only: bool = False
    streaming_max_open_shards: int = 1
    stream_stats_path: str = ""
    bucket_candidate_buffer_size: int = 512
    min_document_tokens: int = 3072
    max_document_chars: int = 131072
    memory_positive_min_oracle_positions: int = 16
    hard_negative_max_oracle_positions: int = 4
    hard_negative_min_repeat_anchors: int = 16
    train_bucket_weights: tuple[float, float, float] = (0.5, 0.25, 0.25)
    num_workers: int = 2
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    train_sequence_length: int = 256
    block_len: int = 512
    debug_validation_sequence_lengths: tuple[int, ...] = (256, 512)
    debug_validation_num_windows: tuple[int, ...] = (16, 8)
    validation_sequence_lengths: tuple[int, ...] = (256, 512)
    validation_num_windows: tuple[int, ...] = (128, 64)
    diagnostic_fixed_windows_path: str = ""
    dev_memory_fixed_windows_path: str = ""
    dev_generic_fixed_windows_path: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty")
        if not self.dataset_config:
            raise ValueError("dataset_config must not be empty")
        if not self.train_split:
            raise ValueError("train_split must not be empty")
        if not self.validation_split:
            raise ValueError("validation_split must not be empty")
        if not self.text_field:
            raise ValueError("text_field must not be empty")
        if self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        if self.streaming_max_open_shards <= 0:
            raise ValueError("streaming_max_open_shards must be positive")
        if self.bucket_candidate_buffer_size <= 0:
            raise ValueError("bucket_candidate_buffer_size must be positive")
        if self.min_document_tokens <= 0:
            raise ValueError("min_document_tokens must be positive")
        if self.max_document_chars <= 0:
            raise ValueError("max_document_chars must be positive")
        if self.memory_positive_min_oracle_positions <= 0:
            raise ValueError("memory_positive_min_oracle_positions must be positive")
        if self.hard_negative_max_oracle_positions < 0:
            raise ValueError("hard_negative_max_oracle_positions must be nonnegative")
        if self.hard_negative_min_repeat_anchors <= 0:
            raise ValueError("hard_negative_min_repeat_anchors must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        if self.train_sequence_length <= 0:
            raise ValueError("train_sequence_length must be positive")
        if self.block_len <= 0:
            raise ValueError("block_len must be positive")
        if len(self.train_bucket_weights) != 3:
            raise ValueError("train_bucket_weights must contain exactly 3 entries for M/H/G")
        if any(float(value) <= 0.0 for value in self.train_bucket_weights):
            raise ValueError("train_bucket_weights values must be positive")
        if not self.debug_validation_sequence_lengths:
            raise ValueError("debug_validation_sequence_lengths must not be empty")
        if len(self.debug_validation_sequence_lengths) != len(self.debug_validation_num_windows):
            raise ValueError("debug_validation_sequence_lengths and debug_validation_num_windows must match")
        if not self.validation_sequence_lengths:
            raise ValueError("validation_sequence_lengths must not be empty")
        if len(self.validation_sequence_lengths) != len(self.validation_num_windows):
            raise ValueError("validation_sequence_lengths and validation_num_windows must match")
        for length in self.debug_validation_sequence_lengths:
            if length <= 0:
                raise ValueError("debug validation sequence lengths must be positive")
        for windows in self.debug_validation_num_windows:
            if windows <= 0:
                raise ValueError("debug validation windows must be positive")
        for length in self.validation_sequence_lengths:
            if length <= 0:
                raise ValueError("validation sequence lengths must be positive")
        for windows in self.validation_num_windows:
            if windows <= 0:
                raise ValueError("validation windows must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "DatasetStreamConfig" | None = None,
    ) -> "DatasetStreamConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        normalised = dict(payload)
        for key in (
            "streaming",
            "require_streaming_only",
            "pin_memory",
            "persistent_workers",
        ):
            if key in normalised:
                normalised[key] = bool(normalised[key])
        for key in (
            "shuffle_buffer_size",
            "streaming_max_open_shards",
            "bucket_candidate_buffer_size",
            "min_document_tokens",
            "max_document_chars",
            "memory_positive_min_oracle_positions",
            "hard_negative_max_oracle_positions",
            "hard_negative_min_repeat_anchors",
            "num_workers",
            "prefetch_factor",
            "train_sequence_length",
            "block_len",
        ):
            if key in normalised:
                normalised[key] = int(normalised[key])
        for key in (
            "train_bucket_weights",
            "debug_validation_sequence_lengths",
            "debug_validation_num_windows",
            "validation_sequence_lengths",
            "validation_num_windows",
        ):
            if key in normalised:
                caster = float if key == "train_bucket_weights" else int
                normalised[key] = tuple(caster(value) for value in normalised[key])
        return cls(**normalised)


@dataclass(frozen=True)
class OptimizerPolicy:
    optimizer_name: str = "soap_hybrid"
    peak_lr: float = 2e-5
    min_lr_scale: float = 0.1
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    lr_group_router_mult: float = 1.0
    lr_group_exact_readout_mult: float = 1.0
    lr_group_query_stack_mult: float = 1.0
    lr_group_gfst_mult: float = 1.5
    lr_group_centroid_mult: float = 8.0
    lr_group_core_mult: float = 1.0
    lr_group_embed_mult: float = 0.35
    lr_group_norm_mult: float = 0.2
    wd_width_coeff_matrix: float = 16.0
    wd_width_coeff_embed: float = 4.0
    plateau_reduction_factor: float = 0.8
    plateau_relative_improvement: float = 0.02
    plateau_patience_evals: int = 2
    soap_precondition_frequency: int = 10
    soap_max_preconditioner_dim: int = 4096
    soap_param_threshold: int = 4096
    soap_betas: tuple[float, float] = (0.95, 0.95)
    soap_shampoo_beta: float = -1.0
    soap_eps: float = 1e-8
    soap_correct_bias: bool = True
    soap_merge_dims: bool = False
    soap_precondition_1d: bool = False
    soap_normalize_grads: bool = False
    max_vram_bytes: int = 31_500_000_000

    def __post_init__(self) -> None:
        if self.optimizer_name not in {"soap_hybrid", "adamw_only"}:
            raise ValueError("optimizer_name must be one of {'soap_hybrid', 'adamw_only'}")
        if self.peak_lr <= 0.0:
            raise ValueError("peak_lr must be positive")
        if not (0.0 < self.min_lr_scale <= 1.0):
            raise ValueError("min_lr_scale must be in (0, 1]")
        if not (0.0 <= self.warmup_ratio < 1.0):
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        if (
            self.lr_group_router_mult <= 0.0
            or self.lr_group_exact_readout_mult <= 0.0
            or self.lr_group_query_stack_mult <= 0.0
            or self.lr_group_gfst_mult <= 0.0
            or self.lr_group_centroid_mult <= 0.0
            or self.lr_group_core_mult <= 0.0
        ):
            raise ValueError("matrix learning-rate multipliers must be positive")
        if self.lr_group_embed_mult <= 0.0 or self.lr_group_norm_mult <= 0.0:
            raise ValueError("non-matrix learning-rate multipliers must be positive")
        if self.wd_width_coeff_matrix < 0.0 or self.wd_width_coeff_embed < 0.0:
            raise ValueError("weight-decay width coefficients must be nonnegative")
        if not (0.0 < self.plateau_reduction_factor < 1.0):
            raise ValueError("plateau_reduction_factor must be in (0, 1)")
        if self.plateau_relative_improvement < 0.0:
            raise ValueError("plateau_relative_improvement must be nonnegative")
        if self.plateau_patience_evals <= 0:
            raise ValueError("plateau_patience_evals must be positive")
        if self.soap_precondition_frequency <= 0:
            raise ValueError("soap_precondition_frequency must be positive")
        if self.soap_max_preconditioner_dim <= 0:
            raise ValueError("soap_max_preconditioner_dim must be positive")
        if self.soap_param_threshold <= 0:
            raise ValueError("soap_param_threshold must be positive")
        if any(beta <= 0.0 or beta >= 1.0 for beta in self.soap_betas):
            raise ValueError("soap betas must be in (0, 1)")
        if self.soap_eps <= 0.0:
            raise ValueError("soap_eps must be positive")
        if self.max_vram_bytes <= 0:
            raise ValueError("max_vram_bytes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "OptimizerPolicy" | None = None,
    ) -> "OptimizerPolicy":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        normalised = dict(payload)
        float_keys = (
            "peak_lr",
            "min_lr_scale",
            "warmup_ratio",
            "lr_group_router_mult",
            "lr_group_exact_readout_mult",
            "lr_group_query_stack_mult",
            "lr_group_gfst_mult",
            "lr_group_centroid_mult",
            "lr_group_core_mult",
            "lr_group_embed_mult",
            "lr_group_norm_mult",
            "wd_width_coeff_matrix",
            "wd_width_coeff_embed",
            "plateau_reduction_factor",
            "plateau_relative_improvement",
            "soap_shampoo_beta",
            "soap_eps",
        )
        int_keys = (
            "warmup_steps",
            "plateau_patience_evals",
            "soap_precondition_frequency",
            "soap_max_preconditioner_dim",
            "soap_param_threshold",
            "max_vram_bytes",
        )
        for key in float_keys:
            if key in normalised:
                normalised[key] = float(normalised[key])
        for key in int_keys:
            if key in normalised:
                normalised[key] = int(normalised[key])
        if "soap_betas" in normalised:
            normalised["soap_betas"] = tuple(float(value) for value in normalised["soap_betas"])
        return cls(**normalised)


@dataclass(frozen=True)
class SoapBootstrapConfig:
    bootstrap_dir: str = ""
    sequence_length: int = 64
    num_batches: int = 1
    micro_batch_size: int = 1
    core_layers_per_pass: int = 2
    save_every_passes: int = 0

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.core_layers_per_pass <= 0:
            raise ValueError("core_layers_per_pass must be positive")
        if self.save_every_passes < 0:
            raise ValueError("save_every_passes must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "SoapBootstrapConfig" | None = None,
    ) -> "SoapBootstrapConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        normalised = dict(payload)
        for key in ("sequence_length", "num_batches", "micro_batch_size", "core_layers_per_pass", "save_every_passes"):
            if key in normalised:
                normalised[key] = int(normalised[key])
        return cls(**normalised)


@dataclass(frozen=True)
class TrainingPhaseConfig:
    phase1_end_ratio: float = 0.2
    phase2_end_ratio: float = 0.6
    force_teacher_on: bool = False
    phase_teacher_every: tuple[int, int, int] = (1, 1, 2)
    phase_distill_weight: tuple[float, float, float] = (1.0, 0.5, 0.25)
    phase_memory_active_distill_scale: tuple[float, float, float] = (0.0, 0.0, 0.0)
    teacher_gap_short_context: str = "none"
    teacher_gap_scale: float = 0.0
    teacher_gap_only_on_memory_active: bool = True
    phase_next_token_weight: tuple[float, float, float] = (1.0, 1.0, 1.0)
    phase_copy_weight: tuple[float, float, float] = (1.0, 1.0, 1.0)
    phase_copy_positive_class_weight: tuple[float, float, float] = (1.0, 1.0, 1.0)
    phase_copy_negative_class_weight: tuple[float, float, float] = (1.0, 1.0, 1.0)
    phase_copy_advantage_weight: tuple[float, float, float] = (0.0, 0.0, 0.0)
    phase_pointer_weight: tuple[float, float, float] = (1.0, 1.0, 1.0)
    phase_retrieval_weight: tuple[float, float, float] = (1.0, 1.0, 1.0)
    phase_geometry_weight: tuple[float, float, float] = (0.02, 0.02, 0.00)
    phase_boundary_weight: tuple[float, float, float] = (0.02, 0.02, 0.00)
    phase_centroid_compactness_weight: tuple[float, float, float] = (0.10, 0.05, 0.00)
    phase_centroid_sharpness_weight: tuple[float, float, float] = (0.03, 0.02, 0.00)
    phase_centroid_occupancy_balance_weight: tuple[float, float, float] = (0.00, 0.00, 0.00)
    phase_centroid_coherence_weight: tuple[float, float, float] = (0.02, 0.01, 0.00)
    phase_centroid_cardinality_weight: tuple[float, float, float] = (0.15, 0.08, 0.00)
    phase_centroid_mitigation_weight: tuple[float, float, float] = (0.00, 0.00, 0.00)
    phase_centroid_write_gate_recovery_weight: tuple[float, float, float] = (0.00, 0.00, 0.00)
    phase_centroid_write_usage_balance_weight: tuple[float, float, float] = (0.00, 0.00, 0.00)
    phase_centroid_head_overlap_weight: tuple[float, float, float] = (0.02, 0.01, 0.00)
    phase_centroid_head_diversity_weight: tuple[float, float, float] = (0.01, 0.01, 0.00)
    phase_precondition_frequency: tuple[int, int, int] = (10, 20, 40)
    phase3_disable_teacher_gap: float = 1.5

    def __post_init__(self) -> None:
        if not (0.0 < self.phase1_end_ratio < self.phase2_end_ratio < 1.0):
            raise ValueError("phase end ratios must satisfy 0 < phase1 < phase2 < 1")
        expected_lengths = (
            len(self.phase_teacher_every),
            len(self.phase_distill_weight),
            len(self.phase_memory_active_distill_scale),
            len(self.phase_next_token_weight),
            len(self.phase_copy_weight),
            len(self.phase_copy_positive_class_weight),
            len(self.phase_copy_negative_class_weight),
            len(self.phase_copy_advantage_weight),
            len(self.phase_pointer_weight),
            len(self.phase_retrieval_weight),
            len(self.phase_geometry_weight),
            len(self.phase_boundary_weight),
            len(self.phase_centroid_compactness_weight),
            len(self.phase_centroid_sharpness_weight),
            len(self.phase_centroid_occupancy_balance_weight),
            len(self.phase_centroid_coherence_weight),
            len(self.phase_centroid_cardinality_weight),
            len(self.phase_centroid_mitigation_weight),
            len(self.phase_centroid_write_gate_recovery_weight),
            len(self.phase_centroid_write_usage_balance_weight),
            len(self.phase_centroid_head_overlap_weight),
            len(self.phase_centroid_head_diversity_weight),
            len(self.phase_precondition_frequency),
        )
        if len(set(expected_lengths)) != 1 or expected_lengths[0] != 3:
            raise ValueError("all training phase tuples must contain exactly 3 entries")
        if any(value <= 0 for value in self.phase_teacher_every):
            raise ValueError("phase_teacher_every values must be positive")
        if any(value < 0.0 for value in self.phase_distill_weight):
            raise ValueError("phase_distill_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_memory_active_distill_scale):
            raise ValueError("phase_memory_active_distill_scale values must be nonnegative")
        if str(self.teacher_gap_short_context) not in {"none", "512", "1024"}:
            raise ValueError("teacher_gap_short_context must be one of {'none', '512', '1024'}")
        if float(self.teacher_gap_scale) < 0.0:
            raise ValueError("teacher_gap_scale must be nonnegative")
        if any(value < 0.0 for value in self.phase_next_token_weight):
            raise ValueError("phase_next_token_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_copy_weight):
            raise ValueError("phase_copy_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_copy_positive_class_weight):
            raise ValueError("phase_copy_positive_class_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_copy_negative_class_weight):
            raise ValueError("phase_copy_negative_class_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_copy_advantage_weight):
            raise ValueError("phase_copy_advantage_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_pointer_weight):
            raise ValueError("phase_pointer_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_retrieval_weight):
            raise ValueError("phase_retrieval_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_geometry_weight):
            raise ValueError("phase_geometry_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_boundary_weight):
            raise ValueError("phase_boundary_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_centroid_compactness_weight):
            raise ValueError("phase_centroid_compactness_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_centroid_sharpness_weight):
            raise ValueError("phase_centroid_sharpness_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_centroid_occupancy_balance_weight):
            raise ValueError("phase_centroid_occupancy_balance_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_centroid_coherence_weight):
            raise ValueError("phase_centroid_coherence_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_centroid_cardinality_weight):
            raise ValueError("phase_centroid_cardinality_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_centroid_mitigation_weight):
            raise ValueError("phase_centroid_mitigation_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_centroid_write_gate_recovery_weight):
            raise ValueError("phase_centroid_write_gate_recovery_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_centroid_write_usage_balance_weight):
            raise ValueError("phase_centroid_write_usage_balance_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_centroid_head_overlap_weight):
            raise ValueError("phase_centroid_head_overlap_weight values must be nonnegative")
        if any(value < 0.0 for value in self.phase_centroid_head_diversity_weight):
            raise ValueError("phase_centroid_head_diversity_weight values must be nonnegative")
        if any(value <= 0 for value in self.phase_precondition_frequency):
            raise ValueError("phase_precondition_frequency values must be positive")
        if self.phase3_disable_teacher_gap < 0.0:
            raise ValueError("phase3_disable_teacher_gap must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "TrainingPhaseConfig" | None = None,
    ) -> "TrainingPhaseConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        normalised = dict(payload)
        if "force_teacher_on" in normalised:
            value = normalised["force_teacher_on"]
            if isinstance(value, str):
                normalised["force_teacher_on"] = value.strip().lower() not in {"0", "false", "no", ""}
            else:
                normalised["force_teacher_on"] = bool(value)
        if "phase1_end_ratio" in normalised:
            normalised["phase1_end_ratio"] = float(normalised["phase1_end_ratio"])
        if "phase2_end_ratio" in normalised:
            normalised["phase2_end_ratio"] = float(normalised["phase2_end_ratio"])
        if "teacher_gap_short_context" in normalised:
            normalised["teacher_gap_short_context"] = str(normalised["teacher_gap_short_context"])
        if "teacher_gap_scale" in normalised:
            normalised["teacher_gap_scale"] = float(normalised["teacher_gap_scale"])
        if "teacher_gap_only_on_memory_active" in normalised:
            value = normalised["teacher_gap_only_on_memory_active"]
            if isinstance(value, str):
                normalised["teacher_gap_only_on_memory_active"] = value.strip().lower() not in {"0", "false", "no", ""}
            else:
                normalised["teacher_gap_only_on_memory_active"] = bool(value)
        for key in (
            "phase_teacher_every",
            "phase_precondition_frequency",
        ):
            if key in normalised:
                normalised[key] = tuple(int(value) for value in normalised[key])
        for key in (
            "phase_distill_weight",
            "phase_memory_active_distill_scale",
            "phase_next_token_weight",
            "phase_copy_weight",
            "phase_copy_positive_class_weight",
            "phase_copy_negative_class_weight",
            "phase_copy_advantage_weight",
            "phase_pointer_weight",
            "phase_retrieval_weight",
            "phase_geometry_weight",
            "phase_boundary_weight",
            "phase_centroid_compactness_weight",
            "phase_centroid_sharpness_weight",
            "phase_centroid_occupancy_balance_weight",
            "phase_centroid_coherence_weight",
            "phase_centroid_cardinality_weight",
            "phase_centroid_mitigation_weight",
            "phase_centroid_write_gate_recovery_weight",
            "phase_centroid_write_usage_balance_weight",
            "phase_centroid_head_overlap_weight",
            "phase_centroid_head_diversity_weight",
        ):
            if key in normalised:
                normalised[key] = tuple(float(value) for value in normalised[key])
        if "phase3_disable_teacher_gap" in normalised:
            normalised["phase3_disable_teacher_gap"] = float(normalised["phase3_disable_teacher_gap"])
        return cls(**normalised)


@dataclass(frozen=True)
class TrainingCentroidReplayConfig:
    enabled: bool = True
    steps: int = 8
    replay_only: bool = False
    pointer_weight: float = 6.0
    centroid_compactness_weight_scale: float = 1.0
    centroid_sharpness_weight_scale: float = 1.0
    centroid_occupancy_balance_weight_scale: float = 1.0
    centroid_coherence_weight_scale: float = 1.0
    centroid_cardinality_weight_scale: float = 1.25
    centroid_mitigation_weight_scale: float = 0.0
    centroid_write_usage_balance_weight_scale: float = 1.0
    centroid_head_overlap_weight_scale: float = 1.0
    centroid_head_diversity_weight_scale: float = 1.0
    centroid_replay_pointer_margin_threshold: float = 0.20
    centroid_replay_entropy_threshold: float = 0.60
    centroid_replay_confidence_threshold: float = 0.15
    write_recovery_mode: str = "off"
    write_recovery_temperature_start: float = 1.0
    write_recovery_temperature_end: float = 1.0
    write_recovery_noise_scale_start: float = 0.0
    write_recovery_noise_scale_end: float = 0.0
    write_recovery_load_bias_scale_start: float = 0.0
    write_recovery_load_bias_scale_end: float = 0.0
    write_recovery_load_bias_clip_start: float = 0.0
    write_recovery_load_bias_clip_end: float = 0.0

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("centroid replay steps must be nonnegative")
        if self.pointer_weight < 0.0:
            raise ValueError("centroid replay pointer_weight must be nonnegative")
        for name in (
            "centroid_compactness_weight_scale",
            "centroid_sharpness_weight_scale",
            "centroid_occupancy_balance_weight_scale",
            "centroid_coherence_weight_scale",
            "centroid_cardinality_weight_scale",
            "centroid_mitigation_weight_scale",
            "centroid_write_usage_balance_weight_scale",
            "centroid_head_overlap_weight_scale",
            "centroid_head_diversity_weight_scale",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.centroid_replay_pointer_margin_threshold < 0.0:
            raise ValueError("centroid_replay_pointer_margin_threshold must be nonnegative")
        if self.centroid_replay_entropy_threshold < 0.0:
            raise ValueError("centroid_replay_entropy_threshold must be nonnegative")
        if self.centroid_replay_confidence_threshold < 0.0:
            raise ValueError("centroid_replay_confidence_threshold must be nonnegative")
        if self.write_recovery_mode not in {"off", "recover", "converge"}:
            raise ValueError("write_recovery_mode must be one of {'off', 'recover', 'converge'}")
        for name in ("write_recovery_temperature_start", "write_recovery_temperature_end"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "write_recovery_noise_scale_start",
            "write_recovery_noise_scale_end",
            "write_recovery_load_bias_scale_start",
            "write_recovery_load_bias_scale_end",
            "write_recovery_load_bias_clip_start",
            "write_recovery_load_bias_clip_end",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "TrainingCentroidReplayConfig" | None = None,
    ) -> "TrainingCentroidReplayConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        normalised = dict(payload)
        if "enabled" in normalised:
            normalised["enabled"] = bool(normalised["enabled"])
        if "replay_only" in normalised:
            normalised["replay_only"] = bool(normalised["replay_only"])
        if "steps" in normalised:
            normalised["steps"] = int(normalised["steps"])
        for key in (
            "pointer_weight",
            "centroid_compactness_weight_scale",
            "centroid_sharpness_weight_scale",
            "centroid_occupancy_balance_weight_scale",
            "centroid_coherence_weight_scale",
            "centroid_cardinality_weight_scale",
            "centroid_mitigation_weight_scale",
            "centroid_write_usage_balance_weight_scale",
            "centroid_head_overlap_weight_scale",
            "centroid_head_diversity_weight_scale",
            "centroid_replay_pointer_margin_threshold",
            "centroid_replay_entropy_threshold",
            "centroid_replay_confidence_threshold",
            "write_recovery_temperature_start",
            "write_recovery_temperature_end",
            "write_recovery_noise_scale_start",
            "write_recovery_noise_scale_end",
            "write_recovery_load_bias_scale_start",
            "write_recovery_load_bias_scale_end",
            "write_recovery_load_bias_clip_start",
            "write_recovery_load_bias_clip_end",
        ):
            if key in normalised:
                normalised[key] = float(normalised[key])
        if "write_recovery_mode" in normalised:
            normalised["write_recovery_mode"] = str(normalised["write_recovery_mode"])
        return cls(**normalised)


@dataclass(frozen=True)
class TrainingRampConfig:
    stage_names: tuple[str, str, str] = ("A", "B", "C")
    stage_sequence_lengths: tuple[int, int, int] = (128, 256, 256)
    stage_gradient_accumulation_steps: tuple[int, int, int] = (2, 4, 8)
    step_wall_ceiling_ms: float = 300000.0
    timing_ema_beta: float = 0.8
    optimizer_regression_tolerance: float = 1.35
    full_eval_improvement_streak: int = 2

    def __post_init__(self) -> None:
        lengths = (
            len(self.stage_names),
            len(self.stage_sequence_lengths),
            len(self.stage_gradient_accumulation_steps),
        )
        if len(set(lengths)) != 1 or lengths[0] != 3:
            raise ValueError("training ramp stages must contain exactly 3 entries")
        if any(length <= 0 for length in self.stage_sequence_lengths):
            raise ValueError("stage_sequence_lengths values must be positive")
        if any(value <= 0 for value in self.stage_gradient_accumulation_steps):
            raise ValueError("stage_gradient_accumulation_steps values must be positive")
        if self.step_wall_ceiling_ms <= 0.0:
            raise ValueError("step_wall_ceiling_ms must be positive")
        if not (0.0 < self.timing_ema_beta < 1.0):
            raise ValueError("timing_ema_beta must be in (0, 1)")
        if self.optimizer_regression_tolerance < 1.0:
            raise ValueError("optimizer_regression_tolerance must be >= 1.0")
        if self.full_eval_improvement_streak <= 0:
            raise ValueError("full_eval_improvement_streak must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "TrainingRampConfig" | None = None,
    ) -> "TrainingRampConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        normalised = dict(payload)
        if "stage_names" in normalised:
            normalised["stage_names"] = tuple(str(value) for value in normalised["stage_names"])
        if "stage_sequence_lengths" in normalised:
            normalised["stage_sequence_lengths"] = tuple(int(value) for value in normalised["stage_sequence_lengths"])
        if "stage_gradient_accumulation_steps" in normalised:
            normalised["stage_gradient_accumulation_steps"] = tuple(
                int(value) for value in normalised["stage_gradient_accumulation_steps"]
            )
        for key in ("step_wall_ceiling_ms", "timing_ema_beta", "optimizer_regression_tolerance"):
            if key in normalised:
                normalised[key] = float(normalised[key])
        if "full_eval_improvement_streak" in normalised:
            normalised["full_eval_improvement_streak"] = int(normalised["full_eval_improvement_streak"])
        return cls(**normalised)




@dataclass
class MpucGfstHmbConfig:
    model_type: str = "mpuc_gfst_hmb"
    d_model: int = 2048
    d_intermediate: int = 0
    n_layer: int = 12
    vocab_size: int = 151936
    mpuc_cfg: dict[str, Any] = field(default_factory=lambda: {
        "branch_specs": ((3, 1), (5, 2), (9, 4)),
        "cross_rank": 128,
        "semantic_rank": 128,
    })
    attn_layer_idx: list[int] = field(default_factory=list)
    attn_cfg: dict[str, Any] = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = False
    pad_vocab_size_multiple: int = 1
    tie_embeddings: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 2
    pad_token_id: int = 2
    copy_token_id: int = -1
    teacher_model_name: str | None = None
    rope_theta: float = 10000.0
    gfst_hmb: dict[str, Any] = field(default_factory=lambda: GfstHmbConfig().to_dict())
    retrieval: dict[str, Any] = field(default_factory=lambda: FaissGpuRetrievalConfig().to_dict())

    def __post_init__(self) -> None:
        if int(self.copy_token_id) < 0:
            self.copy_token_id = int(self.vocab_size)
            self.vocab_size = int(self.vocab_size) + 1
        if int(self.copy_token_id) >= int(self.vocab_size):
            raise ValueError("copy_token_id must be within vocab_size")
        self.gfst_hmb = GfstHmbConfig.from_mapping(self.gfst_hmb).to_dict()
        self.retrieval = FaissGpuRetrievalConfig.from_mapping(self.retrieval).to_dict()

    @property
    def hidden_size(self) -> int:
        return int(self.d_model)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.n_layer)

    @property
    def gfst_hmb_config(self) -> GfstHmbConfig:
        return GfstHmbConfig.from_mapping(self.gfst_hmb)

    @property
    def retrieval_config(self) -> FaissGpuRetrievalConfig:
        return FaissGpuRetrievalConfig.from_mapping(self.retrieval)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_size"] = self.hidden_size
        payload["num_hidden_layers"] = self.num_hidden_layers
        return payload

    @classmethod
    def from_tokenizer_spec(
        cls,
        *,
        vocab_size: int,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 2,
        copy_token_id: int | None = None,
        gfst_hmb: Mapping[str, Any] | GfstHmbConfig | None = None,
        retrieval: Mapping[str, Any] | FaissGpuRetrievalConfig | None = None,
    ) -> "MpucGfstHmbConfig":
        gfst_cfg = GfstHmbConfig.from_mapping(gfst_hmb)
        retrieval_cfg = FaissGpuRetrievalConfig.from_mapping(retrieval)
        resolved_copy_token_id = int(vocab_size) if copy_token_id is None else int(copy_token_id)
        resolved_vocab_size = max(int(vocab_size), resolved_copy_token_id + 1)
        return cls(
            vocab_size=resolved_vocab_size,
            bos_token_id=int(bos_token_id),
            eos_token_id=int(eos_token_id),
            pad_token_id=int(pad_token_id),
            copy_token_id=resolved_copy_token_id,
            teacher_model_name=None,
            gfst_hmb=gfst_cfg.to_dict(),
            retrieval=retrieval_cfg.to_dict(),
        )

    @classmethod
    def from_qwen_config(
        cls,
        qwen_config: Any,
        *,
        gfst_hmb: Mapping[str, Any] | GfstHmbConfig | None = None,
        retrieval: Mapping[str, Any] | FaissGpuRetrievalConfig | None = None,
        teacher_model_name: str | None = "Qwen/Qwen3-1.7B",
        copy_token_id: int | None = None,
    ) -> "MpucGfstHmbConfig":
        gfst_cfg = GfstHmbConfig.from_mapping(gfst_hmb)
        retrieval_cfg = FaissGpuRetrievalConfig.from_mapping(retrieval)
        d_model = int(getattr(qwen_config, "hidden_size"))
        n_layer = 12
        vocab_size = int(getattr(qwen_config, "vocab_size"))
        resolved_copy_token_id = vocab_size if copy_token_id is None else int(copy_token_id)
        bos_token_id = int(getattr(qwen_config, "bos_token_id", 0))
        eos_token_id = int(getattr(qwen_config, "eos_token_id", 2))
        pad_token = getattr(qwen_config, "pad_token_id", None)
        pad_token_id = eos_token_id if pad_token is None else int(pad_token)
        base_rank = max(64, d_model // 16)
        mpuc_cfg = {
            "branch_specs": ((3, 1), (5, 2), (9, 4)),
            "cross_rank": base_rank,
            "semantic_rank": base_rank,
        }
        return cls(
            d_model=d_model,
            d_intermediate=0,
            n_layer=n_layer,
            vocab_size=max(vocab_size, resolved_copy_token_id + 1),
            mpuc_cfg=mpuc_cfg,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=False,
            pad_vocab_size_multiple=1,
            tie_embeddings=False,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            copy_token_id=resolved_copy_token_id,
            teacher_model_name=teacher_model_name,
            rope_theta=float(getattr(qwen_config, "rope_theta", 10000.0) or 10000.0),
            gfst_hmb=gfst_cfg.to_dict(),
            retrieval=retrieval_cfg.to_dict(),
        )
