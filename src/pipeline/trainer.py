import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import boto3
from typing import Dict, Any

class DistributedTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_client = boto3.client('s3')
        self.setup_distributed()
        
    def setup_distributed(self) -> None:
        """Initialize distributed training setup."""
        if torch.cuda.is_available():
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(dist.get_rank())
        else:
            dist.init_process_group(backend='gloo')
    
    def load_model(self, model: torch.nn.Module) -> DistributedDataParallel:
        """Wrap model for distributed training."""
        if torch.cuda.is_available():
            model = model.cuda()
        return DistributedDataParallel(model)
    
    def save_checkpoint(self, model: torch.nn.Module, epoch: int) -> None:
        """Save model checkpoint to S3."""
        if dist.get_rank() == 0:  # Only save on main process
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }
            path = f'/tmp/checkpoint_{epoch}.pt'
            torch.save(checkpoint, path)
            
            # Upload to S3
            self.s3_client.upload_file(
                path,
                self.config['checkpoint_bucket'],
                f'checkpoints/epoch_{epoch}.pt'
            )
            os.remove(path)  # Clean up local file
    
    def train_epoch(self, model: torch.nn.Module, 
                    dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.Module) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
                
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)