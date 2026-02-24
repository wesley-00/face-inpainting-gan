import torch
import torch.nn as nn
import torchvision.models as models


# ========================
# VGG PERCEPTUAL LOSS
# ========================
class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    Captures high-level semantic similarity between images.
    """
    def __init__(self):
        super().__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True)
        
        # Extract features from relu3_3 layer (16th layer)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Loss criterion
        self.criterion = nn.L1Loss()
        
        # CRITICAL FIX: Add ImageNet normalization
        # VGG was trained on ImageNet with these stats
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def normalize_imagenet(self, x):
        """Convert from [-1, 1] to ImageNet normalized range"""
        # First convert to [0, 1]
        x = (x + 1) / 2
        # Then apply ImageNet normalization
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, x, y):
        """
        Args:
            x, y: Images in range [-1, 1]
        Returns:
            Perceptual loss (L1 distance in feature space)
        """
        # Normalize for VGG
        x_norm = self.normalize_imagenet(x)
        y_norm = self.normalize_imagenet(y)
        
        # Extract features
        x_features = self.feature_extractor(x_norm)
        y_features = self.feature_extractor(y_norm)
        
        # Compute L1 loss in feature space
        return self.criterion(x_features, y_features)


# ========================
# GAN ADVERSARIAL LOSS
# ========================
class GANLoss(nn.Module):
    """
    Adversarial loss with label smoothing support.
    Uses BCEWithLogitsLoss (more stable than BCE).
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target_is_real, smoothing=0.0):
        """
        Args:
            pred: Discriminator predictions (logits)
            target_is_real: True if target is real images, False otherwise
            smoothing: Label smoothing factor (0.0 - 0.2 recommended)
                      Real labels: 1.0 -> (1.0 - smoothing)
                      Fake labels: 0.0 -> (0.0 + smoothing)
        Returns:
            BCE loss with optional label smoothing
        """
        if target_is_real:
            # Smooth real labels: 1.0 -> 0.9 (if smoothing=0.1)
            target = torch.ones_like(pred) * (1.0 - smoothing)
        else:
            # Optionally smooth fake labels: 0.0 -> 0.1 (rarely used)
            target = torch.zeros_like(pred) + smoothing
        
        return self.criterion(pred, target)


# ========================
# COMBINED GENERATOR LOSS
# ========================
class TotalLoss:
    """
    Combined loss for generator training.
    Includes: Pixel loss + Perceptual loss + Adversarial loss
    """
    def __init__(self, lambda_pixel=100.0, lambda_perceptual=10.0, lambda_adv=1.0):
        """
        Args:
            lambda_pixel: Weight for L1 pixel loss
            lambda_perceptual: Weight for VGG perceptual loss
            lambda_adv: Weight for adversarial loss
        """
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.adversarial_loss = GANLoss()
        
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adv = lambda_adv
    
    def to(self, device):
        """Move perceptual loss to device"""
        self.perceptual_loss.to(device)
        return self
    
    def compute_generator_loss(self, fake_img, real_img, disc_fake):
        """
        Compute combined generator loss.
        
        Args:
            fake_img: Generated image [-1, 1]
            real_img: Ground truth image [-1, 1]
            disc_fake: Discriminator output for fake image
        
        Returns:
            total_loss: Weighted combination of all losses
            loss_dict: Dictionary with individual loss components
        """
        # 1. Pixel-wise L1 loss (low-frequency reconstruction)
        pixel_loss = self.pixel_loss(fake_img, real_img)
        
        # 2. Perceptual loss (high-level semantic similarity)
        perceptual_loss = self.perceptual_loss(fake_img, real_img)
        
        # 3. Adversarial loss (fool the discriminator)
        adv_loss = self.adversarial_loss(disc_fake, target_is_real=True)
        
        # 4. Weighted combination
        total_loss = (
            self.lambda_pixel * pixel_loss + 
            self.lambda_perceptual * perceptual_loss + 
            self.lambda_adv * adv_loss
        )
        
        # Return both total loss and individual components for logging
        return total_loss, {
            'pixel': pixel_loss.item(),
            'perceptual': perceptual_loss.item(),
            'adversarial': adv_loss.item(),
            'total': total_loss.item()
        }
