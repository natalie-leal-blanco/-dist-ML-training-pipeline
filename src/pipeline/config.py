import os
from typing import Dict, Any

def get_training_config() -> Dict[str, Any]:
    """Get training configuration from environment variables."""
    return {
        'checkpoint_bucket': os.getenv('CHECKPOINT_BUCKET'),
        'data_bucket': os.getenv('DATA_BUCKET'),
        'batch_size': int(os.getenv('BATCH_SIZE', '32')),
        'learning_rate': float(os.getenv('LEARNING_RATE', '0.001')),
        'epochs': int(os.getenv('EPOCHS', '10')),
        'num_workers': int(os.getenv('NUM_WORKERS', '4')),
        'model_name': os.getenv('MODEL_NAME', 'resnet50'),
    }