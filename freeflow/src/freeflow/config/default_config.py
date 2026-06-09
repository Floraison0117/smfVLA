"""FreeFlow default configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "freeflow_base"
    description: str = "Basic FreeFlow with error correction"
    action_dim: int = 32
    action_horizon: int = 1
    teacher_nfe: int = 10
    lambda_correction: float = 0.1
    use_curriculum: bool = False
    adaptive_correction: bool = False
    correction_schedule: str = "linear"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 4
    learning_rate: float = 2.5e-5
    warmup_steps: int = 500
    total_steps: int = 30000
    eval_every: int = 1000
    save_every: int = 5000
    log_every: int = 100
    gradient_clip: float = 1.0
    precision: str = "bfloat16"


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "adamw"
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = "cosine_decay"
    warmup_steps: int = 500


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_path: str = "../data/libero"
    norm_stats: str = "../data/libero/norm_stats.json"
    num_workers: int = 4
    prefetch_size: int = 2


@dataclass
class CheckpointingConfig:
    """Checkpointing configuration."""
    base_checkpoint: str = "../checkpoints/base/pi05_libero"
    save_dir: str = "../checkpoints/finetuned/freeflow"
    resume: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""
    project: str = "freeflow"
    entity: Optional[str] = None
    mode: str = "online"
    tags: List[str] = field(default_factory=lambda: ["libero-plus", "1nfe", "data-free"])


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    nfe: int = 1
    eval_during_training: bool = True
    eval_tasks: List[str] = field(default_factory=lambda: ["libero_spatial", "libero_object", "libero_goal"])
    eval_episodes: int = 10


@dataclass
class FreeFlowConfig:
    """Complete FreeFlow configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    freeze: List[str] = field(default_factory=list)
    trainable: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "FreeFlowConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert nested dict to config objects
        config = cls()

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "optimizer" in data:
            config.optimizer = OptimizerConfig(**data["optimizer"])
        if "scheduler" in data:
            config.scheduler = SchedulerConfig(**data["scheduler"])
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "checkpointing" in data:
            config.checkpointing = CheckpointingConfig(**data["checkpointing"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "evaluation" in data:
            config.evaluation = EvaluationConfig(**data["evaluation"])

        config.freeze = data.get("freeze", [])
        config.trainable = data.get("trainable", [])

        return config
