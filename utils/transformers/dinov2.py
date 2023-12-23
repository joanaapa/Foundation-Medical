import torch

def dinov2_vits14():
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

def dinov2_vitb14():
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

def dinov2_vitl14():
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

def dinov2_vitg14():
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')