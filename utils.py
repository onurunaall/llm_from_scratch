import torch

def create_padding_mask(seq, pad_id):
    # Returns mask shape: (batch, 1, 1, seq_len)
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size, device):
    mask = torch.triu(torch.ones((size, size), device=device), diagonal=1).type(torch.bool)
    return ~mask  # Allowed positions are True