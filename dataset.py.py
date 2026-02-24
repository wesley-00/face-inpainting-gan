import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np


class SmartResize:
    """Custom transform: Center Crop -> Resize to maintain aspect ratio"""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((self.size, self.size), Image.BICUBIC)
        return img


class ImageRestorationDataset(Dataset):
    """
    Dataset for image inpainting/restoration tasks.
    Supports train/val split from single directory.
    """
    def __init__(self, root_dir, img_size=256, mode='train', 
                 corruption_type='inpainting', split_ratio=0.9):
        """
        Args:
            root_dir: Directory containing images
            img_size: Target image size
            mode: 'train' or 'val'
            corruption_type: 'inpainting' or 'denoising'
            split_ratio: Train/val split ratio (0.9 = 90% train)
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        self.corruption_type = corruption_type
        
        # Load and split files
        all_files = sorted([
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Deterministic split to avoid data leakage
        split_idx = int(len(all_files) * split_ratio)
        
        if mode == 'train':
            self.image_files = all_files[:split_idx]
        else:
            self.image_files = all_files[split_idx:]
        
        print(f"[{mode.upper()}] Loaded {len(self.image_files)} images from {root_dir}")
        
        # Base transforms
        self.base_transform = transforms.Compose([
            SmartResize(img_size),
            transforms.ToTensor(),
        ])
        
        # Augmentation (only for training)
        if mode == 'train':
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.1
                ),
            ])
        else:
            self.augment = None
        
        # Normalization to [-1, 1] for Tanh output
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        )
    
    def __len__(self):
        return len(self.image_files)
    
    def add_corruption(self, img_tensor):
        """Apply corruption based on type"""
        if self.corruption_type == 'inpainting':
            return self.add_mask(img_tensor)
        elif self.corruption_type == 'denoising':
            return self.add_noise(img_tensor)
        else:
            return img_tensor
    
    def add_mask(self, img_tensor):
        """Add rectangular mask for inpainting"""
        c, h, w = img_tensor.shape
        masked_img = img_tensor.clone()
        
        # Random mask size (reasonable for face inpainting)
        mask_h = random.randint(h // 6, h // 3)
        mask_w = random.randint(w // 6, w // 3)
        
        # Random position
        top = random.randint(0, h - mask_h)
        left = random.randint(0, w - mask_w)
        
        # Apply mask (set to -1.0 after normalization)
        masked_img[:, top:top+mask_h, left:left+mask_w] = -1.0
        
        return masked_img
    
    def add_noise(self, img_tensor):
        """Add Gaussian noise for denoising"""
        noise_level = random.uniform(0.05, 0.2)
        noise = torch.randn_like(img_tensor) * noise_level
        noisy_img = img_tensor + noise
        noisy_img = torch.clamp(noisy_img, -1.0, 1.0)
        return noisy_img
    
    def __getitem__(self, idx):
        """Get corrupted and clean image pair"""
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        # Apply augmentation if training
        if self.augment is not None:
            img = self.augment(img)
        
        # Convert to tensor and normalize
        img_tensor = self.base_transform(img)
        clean_img = self.normalize(img_tensor)
        
        # Add corruption
        corrupted_img = self.add_corruption(clean_img)
        
        return corrupted_img, clean_img
