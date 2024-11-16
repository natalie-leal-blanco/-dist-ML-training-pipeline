import pytest
import torch
from src.pipeline.trainer import DistributedTrainer

@pytest.fixture
def trainer_config():
    return {
        'checkpoint_bucket': 'test-bucket',
        'data_bucket': 'test-data-bucket',
        'batch_size': 32,
        'learning_rate': 0.001
    }

def test_trainer_initialization(trainer_config):
    trainer = DistributedTrainer(trainer_config)
    assert trainer.config == trainer_config

def test_model_loading(trainer_config):
    trainer = DistributedTrainer(trainer_config)
    model = torch.nn.Linear(10, 2)  # Simple test model
    wrapped_model = trainer.load_model(model)
    assert isinstance(wrapped_model, torch.nn.parallel.DistributedDataParallel) or \
           isinstance(wrapped_model, torch.nn.Module)

# Add more tests as needed