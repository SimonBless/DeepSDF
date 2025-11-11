"""Configuration management for DeepSDF."""

from typing import Any, Dict, Optional
import yaml
import json
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    """Configuration for the DeepSDF model."""
    latent_size: int = 256
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 512, 512, 512, 512])
    dropout_prob: float = 0.2
    norm_layers: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    latent_in: list[int] = field(default_factory=lambda: [4])
    weight_norm: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    num_epochs: int = 2000
    learning_rate: float = 1e-4
    latent_lr: float = 1e-3
    latent_reg_lambda: float = 1e-4
    clamp_dist: float = 0.1
    num_samples_per_shape: int = 10000
    checkpoint_every: int = 100
    log_every: int = 10
    num_workers: int = 4
    device: str = "cuda"


@dataclass
class DataConfig:
    """Configuration for data."""
    data_dir: str = "./data"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


@dataclass
class Config:
    """Complete configuration for DeepSDF."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment_name: str = "deepsdf_experiment"
    output_dir: str = "./output"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            experiment_name=config_dict.get('experiment_name', 'deepsdf_experiment'),
            output_dir=config_dict.get('output_dir', './output'),
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
        }
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
