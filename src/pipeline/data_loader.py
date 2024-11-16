import boto3
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple, List
import io
import os
from PIL import Image

class S3Dataset(Dataset):
    def __init__(self, bucket_name: str, prefix: str):
        self.s3_client = boto3.client('s3')
        self.bucket = bucket_name
        self.prefix = prefix
        self.image_list = self._get_image_list()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_image_list(self) -> List[Tuple[str, int]]:
        """Get list of images from S3 bucket."""
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.prefix
        )
        return [(obj['Key'], self._get_label(obj['Key'])) 
                for obj in response.get('Contents', [])]
    
    def _get_label(self, key: str) -> int:
        """Extract label from file path."""
        # Implement your labeling logic here
        # Example: extract from path structure
        return int(key.split('/')[-2])
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_key, label = self.image_list[idx]
        
        # Download image from S3
        response = self.s3_client.get_object(
            Bucket=self.bucket,
            Key=image_key
        )
        image_data = response['Body'].read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply transformations
        image = self.transform(image)
        return image, label

def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    train_dataset = S3Dataset(
        config['data_bucket'],
        prefix='train/'
    )
    
    val_dataset = S3Dataset(
        config['data_bucket'],
        prefix='val/'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    return train_loader, val_loader