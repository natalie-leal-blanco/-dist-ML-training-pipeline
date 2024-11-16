import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import boto3
from typing import Dict, Any

class DistributedTrainer:
    def __init__(self, config: Dict[str, Any], distributed: bool = False):
        self.config = config
        self.s3_client = boto3.client('s3')
        self.distributed = distributed
        if distributed:
            self.setup_distributed()
    
    def setup_distributed(self) -> None:
        """Initialize distributed training setup."""
        if torch.cuda.is_available():
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(dist.get_rank())
        else:
            dist.init_process_group(backend='gloo')
    
    def load_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        if torch.cuda.is_available():
            model = model.cuda()
        if self.distributed:
            return DistributedDataParallel(model)
        return model
    
    def save_checkpoint(self, model: torch.nn.Module, epoch: int) -> None:
        """Save model checkpoint to S3."""
        if not self.distributed or (self.distributed and dist.get_rank() == 0):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }
            path = f'/tmp/checkpoint_{epoch}.pt'
            torch.save(checkpoint, path)
            
            self.s3_client.upload_file(
                path,
                self.config['checkpoint_bucket'],
                f'checkpoints/epoch_{epoch}.pt'
            )
            if os.path.exists(path):
                os.remove(path)