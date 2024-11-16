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
    """Test trainer initialization without distributed setup"""
    trainer = DistributedTrainer(trainer_config, distributed=False)
    assert trainer.config == trainer_config
    assert trainer.distributed == False

def test_model_loading(trainer_config):
    """Test model loading without distributed setup"""
    trainer = DistributedTrainer(trainer_config, distributed=False)
    model = torch.nn.Linear(10, 2)  # Simple test model
    wrapped_model = trainer.load_model(model)
    assert isinstance(wrapped_model, torch.nn.Module)

def test_save_checkpoint(trainer_config, tmp_path):
    """Test checkpoint saving functionality"""
    trainer = DistributedTrainer(trainer_config, distributed=False)
    model = torch.nn.Linear(10, 2)
    
    # Mock s3_client upload_file to avoid actual S3 upload
    trainer.s3_client.upload_file = lambda *args, **kwargs: None
    
    # This should not raise any errors
    trainer.save_checkpoint(model, epoch=1)