import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, num_features, eps=1e-7):
        """
        Args:
            num_features (int): The feature dimension of the content vector (base_img).
            eps (float): A small constant to prevent division by zero.
        """
        super(AdaIN, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, noise, base_img):
        """
        Args:
            noise (Tensor): Content vector to be normalized, shape (B, num_features).
            base_img (Tensor): Style vector with the same shape as the content vector,
                               shape (B, num_features).
        Returns:
            out (Tensor): The result after applying AdaIN, shape (B, num_features).
        """
        
        # Compute the per-instance mean and standard deviation for the content vector.
        style_mean = base_img.mean(dim=1, keepdim=True)  # Shape: (B, 1)
        style_std  = base_img.std(dim=1, keepdim=True)   # Shape: (B, 1)
        
        # Normalize the content vector using instance normalization.
        normalized = style_std*(noise - noise.mean(dim=1, keepdim=True)) / (noise.std(dim=1, keepdim=True) + self.eps) + style_mean
        
        return normalized


def AdaIN_fcn(noise, base_img, eps=1e-7):
    """
    Args:
        noise (Tensor): Content vector to be normalized, shape (B, num_features).
        base_img (Tensor): Style vector with the same shape as the content vector,
                           shape (B, num_features).
        eps (float): A small constant to prevent division by zero.
    Returns:
        out (Tensor): The result after applying AdaIN, shape (B, num_features).
    """
    # Compute the per-instance mean and standard deviation for the style vector.
    style_mean = base_img.mean(dim=1, keepdim=True)  # Shape: (B, 1)
    style_std  = base_img.std(dim=1, keepdim=True)   # Shape: (B, 1)
    
    # Normalize the content vector using instance normalization.
    normalized = style_std * (noise - noise.mean(dim=1, keepdim=True)) / (noise.std(dim=1, keepdim=True) + eps) + style_mean

    return normalized