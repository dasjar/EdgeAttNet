import torch

def count_parameters(model):
    """
    Function to count the total number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
    
    Returns:
        int: The total number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
