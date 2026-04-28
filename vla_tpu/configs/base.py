from dataclasses import dataclass, field


@dataclass
class BackboneConfig:
    name: str = "dummy_qwen_adapter"
    impl: str = "dummy_qwen_adapter"
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    hidden_size: int = 1024
    n_visual_tokens: int = 64
    text_vocab_size: int = 4096
    image_channels: int = 3
    patch_size: int = 16
    num_layers: int = 4
    num_heads: int = 8
    num_key_value_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    max_text_tokens: int = 32
    rope_max_wavelength: int = 10000
    rope_scaling_factor: float = 1.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    layer_norm_epsilon: float = 1e-6
    vision_hidden_size: int = 768
    vision_intermediate_size: int = 3072
    vision_num_heads: int = 12
    vision_depth: int = 8
    vision_patch_size: int = 14
    vision_spatial_merge_size: int = 2
    vision_num_position_embeddings: int = 256
    vision_rope_theta: float = 10000.0
    deepstack_visual_indexes: tuple[int, ...] = (1, 3, 5)
    fixed_image_grid_thw: tuple[tuple[int, int, int], ...] = ()


@dataclass
class ActionHeadConfig:
    impl: str = "cross_attn_regressor"
    hidden_size: int = 512
    action_dim: int = 32
    action_horizon: int = 16
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    flow_noise_s: float = 0.999
    num_inference_steps: int = 4
    num_timestep_buckets: int = 1000
    add_action_pos_embed: bool = True


@dataclass
class TrainConfig:
    seed: int = 0
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    batch_size: int = 8
    num_steps: int = 20
    log_every: int = 5
    save_every: int = 500
    checkpoint_dir: str = "outputs/checkpoints"
    freeze_backbone: bool = False
    use_pmap: bool = False
    use_wandb: bool = False
    wandb_project: str = "vla-tpu"
    wandb_entity: str = ""
    wandb_mode: str = "online"


@dataclass
class DataConfig:
    dataset_type: str = "dummy"
    dataset_root: str = ""
    image_height: int = 224
    image_width: int = 224
    seed: int = 0
    num_cameras: int = 3
    camera_keys: tuple[str, ...] = (
        "left_view",
        "right_view",
        "wrist_view",
    )
    state_keys: tuple[str, ...] = ()
    action_keys: tuple[str, ...] = ()
    language_key: str = "annotation.human.action.task_description"
    observation_indices: tuple[int, ...] = (0,)
    action_indices: tuple[int, ...] = tuple(range(16))
    state_normalization_modes: dict[str, str] = field(default_factory=dict)
    action_normalization_modes: dict[str, str] = field(default_factory=dict)
    state_rotation_targets: dict[str, str] = field(default_factory=dict)
    video_crop_scale: float = 0.95
    use_color_jitter: bool = False
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.5
    color_jitter_hue: float = 0.08
    state_dim: int = 16
    instruction_length: int = 32
    max_episodes: int = 8
    max_samples: int = 256
    episode_stride: int = 4
    fixed_episode_indices: tuple[int, ...] = ()
    num_workers: int = 0
    prefetch_size: int = 2


@dataclass
class ExperimentConfig:
    name: str = "small_debug"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def get_experiment_config(name: str) -> ExperimentConfig:
    if name == "small_debug":
        return ExperimentConfig()
    if name == "small_debug_jax":
        cfg = ExperimentConfig()
        cfg.name = "small_debug_jax"
        cfg.backbone.name = "jax_qwen_experimental"
        cfg.backbone.impl = "jax_qwen_experimental"
        cfg.backbone.hidden_size = 768
        cfg.backbone.n_visual_tokens = 196
        cfg.backbone.patch_size = 16
        cfg.backbone.num_layers = 6
        cfg.backbone.num_heads = 12
        cfg.action_head.hidden_size = 768
        return cfg
    if name == "small_debug_jax_qwen3":
        cfg = ExperimentConfig()
        cfg.name = "small_debug_jax_qwen3"
        cfg.backbone.name = "jax_qwen3_adapter"
        cfg.backbone.impl = "jax_qwen3_adapter"
        cfg.backbone.hidden_size = 768
        cfg.backbone.n_visual_tokens = 128
        cfg.backbone.patch_size = 16
        cfg.backbone.num_layers = 6
        cfg.backbone.num_heads = 12
        cfg.backbone.num_key_value_heads = 4
        cfg.action_head.hidden_size = 768
        return cfg
    if name == "small_debug_jax_qwen3_vl":
        cfg = ExperimentConfig()
        cfg.name = "small_debug_jax_qwen3_vl"
        cfg.backbone.name = "jax_qwen3_vl_full"
        cfg.backbone.impl = "jax_qwen3_vl_full"
        cfg.backbone.hidden_size = 768
        cfg.backbone.vision_hidden_size = 768
        cfg.backbone.vision_intermediate_size = 3072
        cfg.backbone.n_visual_tokens = 128
        cfg.backbone.num_layers = 6
        cfg.backbone.num_heads = 12
        cfg.backbone.num_key_value_heads = 4
        cfg.backbone.vision_num_heads = 12
        cfg.backbone.vision_depth = 6
        cfg.backbone.vision_patch_size = 14
        cfg.backbone.vision_spatial_merge_size = 2
        cfg.backbone.vision_num_position_embeddings = 256
        cfg.backbone.deepstack_visual_indexes = (1, 3, 5)
        cfg.action_head.hidden_size = 768
        return cfg
    if name == "robocasa_debug":
        cfg = ExperimentConfig()
        cfg.name = "robocasa_debug"
        cfg.action_head.action_dim = 12
        cfg.action_head.hidden_size = 512
        cfg.backbone.hidden_size = 1024
        cfg.data.dataset_type = "robocasa_lerobot"
        cfg.data.dataset_root = "debug_data/robocasa_debug_subset"
        cfg.data.state_dim = 53
        cfg.data.num_cameras = 3
        cfg.data.max_episodes = 2
        cfg.data.max_samples = 16
        cfg.train.batch_size = 2
        cfg.train.num_steps = 5
        cfg.train.log_every = 1
        cfg.train.checkpoint_dir = "outputs/checkpoints/robocasa_debug"
        return cfg
    if name == "libero_debug":
        cfg = ExperimentConfig()
        cfg.name = "libero_debug"
        cfg.action_head.impl = "flow_matching_cross_attn_dit"
        cfg.data.dataset_type = "lerobot"
        cfg.data.dataset_root = "/mnt/disks/vla-data/libero_gr00t_delta_hf"
        cfg.data.seed = 0
        cfg.data.num_cameras = 2
        cfg.data.camera_keys = ("front_view", "left_wrist_view")
        cfg.data.state_keys = (
            "state.eef_pos_absolute",
            "state.eef_rot_absolute",
            "state.gripper_close",
        )
        cfg.data.action_keys = (
            "action.eef_pos_delta",
            "action.eef_rot_delta",
            "action.gripper_close",
        )
        cfg.data.language_key = "annotation.human.action.task_description"
        cfg.data.observation_indices = (0,)
        cfg.data.action_indices = tuple(range(16))
        cfg.data.state_normalization_modes = {
            "state.eef_pos_absolute": "min_max",
            "state.gripper_close": "min_max",
        }
        cfg.data.action_normalization_modes = {
            "action.eef_pos_delta": "min_max",
            "action.eef_rot_delta": "min_max",
            "action.gripper_close": "min_max",
        }
        cfg.data.state_rotation_targets = {
            "state.eef_rot_absolute": "rotation_6d",
        }
        cfg.data.state_dim = 11
        cfg.data.max_episodes = 8
        cfg.data.max_samples = 64
        cfg.data.episode_stride = 1
        cfg.data.num_workers = 4
        cfg.data.prefetch_size = 8
        cfg.data.use_color_jitter = True
        cfg.action_head.action_dim = 7
        cfg.train.batch_size = 2
        cfg.train.num_steps = 5
        cfg.train.log_every = 1
        cfg.train.checkpoint_dir = "outputs/checkpoints/libero_debug"
        return cfg
    if name == "libero_debug_jax":
        cfg = get_experiment_config("libero_debug")
        cfg.name = "libero_debug_jax"
        cfg.backbone.name = "jax_qwen_experimental"
        cfg.backbone.impl = "jax_qwen_experimental"
        cfg.backbone.hidden_size = 768
        cfg.backbone.n_visual_tokens = 392
        cfg.backbone.patch_size = 16
        cfg.backbone.num_layers = 6
        cfg.backbone.num_heads = 12
        cfg.action_head.hidden_size = 768
        cfg.train.checkpoint_dir = "outputs/checkpoints/libero_debug_jax"
        return cfg
    if name == "libero_debug_jax_qwen3":
        cfg = get_experiment_config("libero_debug")
        cfg.name = "libero_debug_jax_qwen3"
        cfg.backbone.name = "jax_qwen3_adapter"
        cfg.backbone.impl = "jax_qwen3_adapter"
        cfg.backbone.hidden_size = 768
        cfg.backbone.n_visual_tokens = 256
        cfg.backbone.patch_size = 16
        cfg.backbone.num_layers = 8
        cfg.backbone.num_heads = 12
        cfg.backbone.num_key_value_heads = 4
        cfg.action_head.hidden_size = 768
        cfg.train.checkpoint_dir = "outputs/checkpoints/libero_debug_jax_qwen3"
        return cfg
    if name == "libero_debug_jax_qwen3_vl":
        cfg = get_experiment_config("libero_debug")
        cfg.name = "libero_debug_jax_qwen3_vl"
        from vla_tpu.models.qwen3_vl_weight_loader import backbone_config_from_hf_qwen3_vl

        cfg.backbone = backbone_config_from_hf_qwen3_vl(
            "Qwen/Qwen3-VL-2B-Instruct",
            n_visual_tokens=192,
            max_text_tokens=512,
        )
        cfg.backbone.name = "jax_qwen3_vl_pure"
        cfg.backbone.impl = "jax_qwen3_vl_pure"
        cfg.backbone.fixed_image_grid_thw = ((1, 14, 14), (1, 14, 14))
        cfg.action_head.hidden_size = 768
        cfg.action_head.impl = "flow_matching_cross_attn_dit"
        cfg.train.checkpoint_dir = "outputs/checkpoints/libero_debug_jax_qwen3_vl"
        cfg.train.freeze_backbone = True
        cfg.train.use_wandb = True
        return cfg
    if name == "libero_debug_single_episode":
        cfg = get_experiment_config("libero_debug_jax_qwen3_vl")
        cfg.name = "libero_debug_single_episode"
        cfg.data.fixed_episode_indices = (0,)
        cfg.data.max_episodes = 1
        cfg.data.max_samples = 16
        cfg.data.episode_stride = 1
        cfg.data.num_workers = 0
        cfg.data.prefetch_size = 2
        cfg.train.use_wandb = False
        cfg.train.checkpoint_dir = "outputs/checkpoints/libero_debug_single_episode"
        return cfg
    if name == "libero_full":
        cfg = get_experiment_config("libero_debug")
        cfg.name = "libero_full"
        cfg.data.max_episodes = 1693
        cfg.data.max_samples = 273465
        cfg.data.num_workers = 8
        cfg.data.prefetch_size = 16
        cfg.train.batch_size = 8
        cfg.train.num_steps = 1000
        cfg.train.log_every = 10
        cfg.train.checkpoint_dir = "outputs/checkpoints/libero_full"
        cfg.train.use_wandb = True
        return cfg
    if name == "libero_full_jax":
        cfg = get_experiment_config("libero_full")
        cfg.name = "libero_full_jax"
        cfg.backbone.name = "jax_qwen_experimental"
        cfg.backbone.impl = "jax_qwen_experimental"
        cfg.backbone.hidden_size = 768
        cfg.backbone.n_visual_tokens = 392
        cfg.backbone.patch_size = 16
        cfg.backbone.num_layers = 6
        cfg.backbone.num_heads = 12
        cfg.action_head.hidden_size = 768
        cfg.train.checkpoint_dir = "outputs/checkpoints/libero_full_jax"
        cfg.train.use_wandb = True
        return cfg
    if name == "libero_full_jax_qwen3":
        cfg = get_experiment_config("libero_full")
        cfg.name = "libero_full_jax_qwen3"
        cfg.backbone.name = "jax_qwen3_adapter"
        cfg.backbone.impl = "jax_qwen3_adapter"
        cfg.backbone.hidden_size = 768
        cfg.backbone.n_visual_tokens = 256
        cfg.backbone.patch_size = 16
        cfg.backbone.num_layers = 8
        cfg.backbone.num_heads = 12
        cfg.backbone.num_key_value_heads = 4
        cfg.action_head.hidden_size = 768
        cfg.train.checkpoint_dir = "outputs/checkpoints/libero_full_jax_qwen3"
        cfg.train.use_wandb = True
        return cfg
    if name == "libero_full_jax_qwen3_vl":
        cfg = get_experiment_config("libero_full")
        cfg.name = "libero_full_jax_qwen3_vl"
        from vla_tpu.models.qwen3_vl_weight_loader import backbone_config_from_hf_qwen3_vl

        cfg.backbone = backbone_config_from_hf_qwen3_vl(
            "Qwen/Qwen3-VL-2B-Instruct",
            n_visual_tokens=192,
            max_text_tokens=512,
        )
        cfg.backbone.name = "jax_qwen3_vl_pure"
        cfg.backbone.impl = "jax_qwen3_vl_pure"
        cfg.backbone.fixed_image_grid_thw = ((1, 14, 14), (1, 14, 14))
        cfg.action_head.hidden_size = 768
        cfg.action_head.impl = "flow_matching_cross_attn_dit"
        cfg.train.batch_size = 64
        cfg.train.num_steps = 30000
        cfg.train.freeze_backbone = True
        cfg.train.use_pmap = True
        cfg.train.checkpoint_dir = "outputs/checkpoints/libero_full_jax_qwen3_vl"
        cfg.train.use_wandb = True
        return cfg
    if name == "libero_full_jax_qwen3_vl_legacy_eval":
        cfg = get_experiment_config("libero_full_jax_qwen3_vl")
        cfg.name = "libero_full_jax_qwen3_vl_legacy_eval"
        cfg.action_head.impl = "legacy_flow_matching_cross_attn_dit"
        cfg.data.num_workers = 0
        cfg.data.prefetch_size = 2
        cfg.train.use_wandb = False
        return cfg
    if name == "libero_full_jax_qwen3_vl_legacy":
        cfg = get_experiment_config("libero_full_jax_qwen3_vl")
        cfg.name = "libero_full_jax_qwen3_vl_legacy"
        cfg.action_head.impl = "legacy_flow_matching_cross_attn_dit"
        return cfg
    if name == "libero_debug_single_episode_legacy":
        cfg = get_experiment_config("libero_debug_single_episode")
        cfg.name = "libero_debug_single_episode_legacy"
        cfg.action_head.impl = "legacy_flow_matching_cross_attn_dit"
        cfg.train.checkpoint_dir = "outputs/checkpoints/libero_debug_single_episode_legacy"
        return cfg
    raise ValueError(
        "Unknown config "
        f"'{name}'. Currently available: ['small_debug', 'small_debug_jax', 'robocasa_debug', "
        "'small_debug_jax_qwen3', 'small_debug_jax_qwen3_vl', 'libero_debug', 'libero_debug_jax', "
        "'libero_debug_jax_qwen3', 'libero_debug_jax_qwen3_vl', 'libero_full', 'libero_full_jax', "
        "'libero_full_jax_qwen3', 'libero_full_jax_qwen3_vl', 'libero_full_jax_qwen3_vl_legacy_eval', "
        "'libero_full_jax_qwen3_vl_legacy', 'libero_debug_single_episode_legacy']"
    )
