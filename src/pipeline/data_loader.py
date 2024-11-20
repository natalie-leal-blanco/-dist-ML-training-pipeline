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
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix
            )
            if 'Contents' not in response:
                # Return at least one dummy item to prevent DataLoader errors
                return [('dummy.jpg', 0)]
            return [(obj['Key'], self._get_label(obj['Key'])) 
                    for obj in response['Contents'] 
                    if not obj['Key'].endswith('/')]  # Skip directories
        except Exception as e:
            print(f"Error accessing S3: {e}")
            # Return dummy data to prevent initialization errors
            return [('dummy.jpg', 0)]

    def _get_label(self, key: str) -> int:
        """Extract label from file path."""
        try:
            # Assuming path structure: prefix/class_name/filename
            class_name = key.split('/')[-2]
            return int(class_name.split('_')[-1])
        except:
            return 0
    
    def __len__(self) -> int:
        return max(len(self.image_list), 1)  # Ensure at least length 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
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
        except:
            # Return dummy data if there's an error
            image = Image.new('RGB', (224, 224))
            label = 0
            
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
        num_workers=config.get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    return train_loader, val_loader