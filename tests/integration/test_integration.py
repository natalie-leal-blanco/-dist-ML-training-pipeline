import pytest
import boto3
import torch
from src.pipeline.data_loader import create_dataloaders
from src.pipeline.trainer import DistributedTrainer
import os

@pytest.mark.integration
def test_s3_connectivity():
    """Test actual S3 connectivity and operations"""
    if not os.getenv('INTEGRATION_TESTS'):
        pytest.skip("Skipping integration tests")
        
    s3 = boto3.client('s3')
    response = s3.list_buckets()
    assert 'Buckets' in response

@pytest.mark.integration
def test_data_loading():
    """Test actual data loading from S3"""
    if not os.getenv('INTEGRATION_TESTS'):
        pytest.skip("Skipping integration tests")
    
    # Create multiple dummy files to match batch size
    s3 = boto3.client('s3')
    bucket = os.getenv('TEST_DATA_BUCKET')
    
    # Upload multiple dummy files
    dummy_data = b"dummy data"
    for i in range(4):  # Match batch size
        s3.put_object(
            Bucket=bucket,
            Key=f'train/class_0/dummy{i}.jpg',
            Body=dummy_data
        )
        s3.put_object(
            Bucket=bucket,
            Key=f'val/class_0/dummy{i}.jpg',
            Body=dummy_data
        )
    
    config = {
        'data_bucket': bucket,
        'batch_size': 4,
        'num_workers': 0
    }
    
    train_loader, val_loader = create_dataloaders(config)
    
    # Test data loader functionality
    batch = next(iter(train_loader))
    data, labels = batch
    
    assert data.shape[0] == config['batch_size']
    assert len(labels) == config['batch_size']

@pytest.mark.integration
def test_model_checkpointing():
    """Test full model checkpointing to S3"""
    if not os.getenv('INTEGRATION_TESTS'):
        pytest.skip("Skipping integration tests")
        
    config = {
        'checkpoint_bucket': os.getenv('TEST_CHECKPOINT_BUCKET'),
        'data_bucket': os.getenv('TEST_DATA_BUCKET'),
        'batch_size': 4,
        'learning_rate': 0.001
    }
    
    trainer = DistributedTrainer(config, distributed=False)
    model = torch.nn.Linear(10, 2)
    
    # Test actual S3 upload
    trainer.save_checkpoint(model, epoch=1)
    
    # Verify checkpoint exists in S3
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(
        Bucket=config['checkpoint_bucket'],
        Prefix='checkpoints/epoch_1'
    )
    assert 'Contents' in response