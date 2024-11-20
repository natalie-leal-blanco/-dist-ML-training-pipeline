import pytest
import torch
import torch.nn as nn
from src.pipeline.trainer import DistributedTrainer
from src.pipeline.data_loader import create_dataloaders
import tempfile
import os
import boto3
from unittest.mock import MagicMock, patch

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 53 * 53, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 6 * 53 * 53)
        x = self.fc1(x)
        return x

@pytest.fixture
def mock_s3():
    with patch('boto3.client') as mock_client:
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        yield mock_s3

@pytest.fixture
def training_config():
    return {
        'checkpoint_bucket': 'test-checkpoints',
        'data_bucket': 'test-data',
        'batch_size': 4,
        'learning_rate': 0.001,
        'epochs': 2,
        'num_workers': 0
    }

def test_full_training_pipeline(training_config, mock_s3):
    """Test complete training pipeline including model training and checkpointing"""
    # Setup trainer
    trainer = DistributedTrainer(training_config, distributed=False)
    
    # Create mock data
    batch_size = training_config['batch_size']
    mock_data = torch.randn(batch_size, 3, 224, 224)
    mock_labels = torch.randint(0, 10, (batch_size,))
    
    # Setup model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=training_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Mock training step
    loss = trainer.train_step(model, (mock_data, mock_labels), optimizer, criterion)
    
    # Verify training produces expected outputs
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    
    # Test checkpoint saving
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        trainer.save_checkpoint(model, epoch=1)
        
        # Verify S3 upload was called
        mock_s3.upload_file.assert_called_once()

def test_model_validation(training_config, mock_s3):
    """Test model validation functionality"""
    trainer = DistributedTrainer(training_config, distributed=False)
    model = SimpleModel()
    
    # Create mock validation data
    batch_size = training_config['batch_size']
    mock_data = torch.randn(batch_size, 3, 224, 224)
    mock_labels = torch.randint(0, 10, (batch_size,))
    
    # Perform validation
    val_loss, accuracy = trainer.validate(model, [(mock_data, mock_labels)])
    
    # Verify validation metrics
    assert isinstance(val_loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100