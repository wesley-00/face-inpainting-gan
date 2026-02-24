import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# ========================
# CHECKPOINT MANAGEMENT
# ========================
def save_checkpoint(state, filename):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model states, optimizer states, etc.
        filename: Path to save checkpoint
    """
    torch.save(state, filename)
    print(f"✅ Checkpoint saved: {filename}")


def load_checkpoint(checkpoint_path, generator, discriminator, opt_g, opt_d, device='cuda'):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        generator: Generator model
        discriminator: Discriminator model
        opt_g: Generator optimizer
        opt_d: Discriminator optimizer
        device: Device to load models on
    
    Returns:
        epoch: Training epoch number
        best_psnr: Best validation PSNR achieved
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 0, 0.0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel state dicts (remove 'module.' prefix if present)
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        return new_state_dict
    
    generator.load_state_dict(remove_module_prefix(checkpoint['generator']))
    discriminator.load_state_dict(remove_module_prefix(checkpoint['discriminator']))
    opt_g.load_state_dict(checkpoint['opt_g'])
    opt_d.load_state_dict(checkpoint['opt_d'])
    
    epoch = checkpoint['epoch']
    best_psnr = checkpoint.get('best_psnr', 0.0)
    
    print(f"✅ Checkpoint loaded from epoch {epoch} (Best PSNR: {best_psnr:.2f})")
    return epoch, best_psnr


# ========================
# IMAGE PROCESSING
# ========================
def denormalize(tensor):
    """
    Convert from [-1, 1] to [0, 1] range.
    
    Args:
        tensor: Image tensor in range [-1, 1]
    Returns:
        Tensor in range [0, 1]
    """
    return (tensor + 1) / 2


# ========================
# METRICS
# ========================
def calculate_psnr(img1, img2, max_val=2.0):
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    Args:
        img1, img2: Image tensors in range [-1, 1]
        max_val: Maximum possible pixel value (2.0 for [-1, 1] range)
    Returns:
        PSNR in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11):
    """
    Calculate Structural Similarity Index (simplified version).
    For production, use torchmetrics.StructuralSimilarityIndexMeasure
    
    Args:
        img1, img2: Image tensors in range [-1, 1]
        window_size: Size of Gaussian window
    Returns:
        SSIM value (0-1, higher is better)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Local mean
    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, 1, window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, 1, window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Local variance and covariance
    sigma1_sq = torch.nn.functional.avg_pool2d(img1 ** 2, window_size, 1, window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 ** 2, window_size, 1, window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, 1, window_size//2) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


# ========================
# VISUALIZATION
# ========================
def visualize_results(corrupted, fake, real, epoch, save_path=None, num_images=4):
    """
    Visualize and save comparison of corrupted, generated, and real images.
    
    Args:
        corrupted: Corrupted input images
        fake: Generated images
        real: Ground truth images
        epoch: Current epoch number
        save_path: Path to save visualization
        num_images: Number of images to display
    """
    num_images = min(num_images, corrupted.size(0))
    
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Denormalize and convert to numpy
        c_img = denormalize(corrupted[i]).cpu().permute(1, 2, 0).numpy()
        f_img = denormalize(fake[i]).cpu().permute(1, 2, 0).numpy()
        r_img = denormalize(real[i]).cpu().permute(1, 2, 0).numpy()
        
        # Clip to valid range
        c_img = np.clip(c_img, 0, 1)
        f_img = np.clip(f_img, 0, 1)
        r_img = np.clip(r_img, 0, 1)
        
        # Display
        axes[i, 0].imshow(c_img)
        axes[i, 0].set_title('Input (Corrupted)' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(f_img)
        axes[i, 1].set_title('Generated (Restored)' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(r_img)
        axes[i, 2].set_title('Ground Truth' if i == 0 else '')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Epoch {epoch} - Results', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"📊 Visualization saved: {save_path}")
    
#   plt.show()
    plt.close()


def create_comparison_grid(images_dict, save_path=None):
    """
    Create a grid comparing multiple image versions.
    
    Args:
        images_dict: Dictionary of {label: tensor} pairs
        save_path: Path to save grid
    """
    num_types = len(images_dict)
    num_images = list(images_dict.values())[0].size(0)
    
    fig, axes = plt.subplots(num_images, num_types, figsize=(5 * num_types, 5 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for col, (label, images) in enumerate(images_dict.items()):
        for row in range(num_images):
            img = denormalize(images[row]).cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            axes[row, col].imshow(img)
            if row == 0:
                axes[row, col].set_title(label, fontsize=14)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
#   plt.show()
    plt.close()
