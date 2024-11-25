from dataclasses import dataclass
from typing import Optional
import os
import yaml

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    num_workers: int = 4
    model_name: str = 'resnet50'
    checkpoint_bucket: Optional[str] = None
    data_bucket: Optional[str] = None
    device: str = 'cuda'
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    warmup_epochs: int = 1
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    mixed_precision: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save(self, yaml_path: str) -> None:
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f)