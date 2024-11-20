import pytest
import torch
import torch.nn as nn
from src.pipeline.trainer import DistributedTrainer
import tempfile
import os
from unittest.mock import MagicMock, patch

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Calculate the correct input features for the fully connected layer
        # For 224x224 input: After conv1 and pool: 110x110
        # After conv2 and pool: 53x53
        # Therefore: 16 channels * 53 * 53
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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
    trainer = DistributedTrainer(training_config, distributed=False)
    
    # Create mock data
    batch_size = training_config['batch_size']
    mock_data = torch.randn(batch_size, 3, 224, 224)
    mock_labels = torch.randint(0, 10, (batch_size,))
    
    # Setup model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=training_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Test training step
    loss = trainer.train_step(model, (mock_data, mock_labels), optimizer, criterion)
    
    # Verify training produces expected outputs
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    
    # Test checkpoint saving
    trainer.save_checkpoint(model, epoch=1)
    mock_s3.upload_file.assert_called_once()

def test_model_validation(training_config, mock_s3):
    """Test model validation functionality"""
    trainer = DistributedTrainer(training_config, distributed=False)
    model = SimpleModel()
    
    # Create mock validation data
    batch_size = training_config['batch_size']
    mock_data = torch.randn(batch_size, 3, 224, 224)
    mock_labels = torch.randint(0, 10, (batch_size,))
    
    # Create validation loader with single batch
    val_loader = [(mock_data, mock_labels)]
    
    # Perform validation
    val_loss, accuracy = trainer.validate(model, val_loader)
    
    # Verify validation metrics
    assert isinstance(val_loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100
    assert not torch.isnan(torch.tensor(val_loss))