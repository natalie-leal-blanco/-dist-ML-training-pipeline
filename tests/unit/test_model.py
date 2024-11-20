import torch
from tests.unit.test_pipeline import SimpleModel

def test_model_forward():
    """Test model forward pass with correct dimensions"""
    model = SimpleModel()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)
    
    # Check output dimensions
    assert output.shape == (batch_size, 10)
    
    # Check that output is valid (no NaN values)
    assert not torch.isnan(output).any()