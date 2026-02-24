import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Chuyển sang thư viện mới torch.amp để hỗ trợ device_type đa nền tảng
from torch.amp import autocast, GradScaler 
import numpy as np
from tqdm import tqdm
import os
import wandb

import config
from dataset import ImageRestorationDataset
from models import Generator, Discriminator
from loss import TotalLoss, GANLoss
from utils import (save_checkpoint, load_checkpoint, calculate_psnr, 
                   calculate_ssim, visualize_results, denormalize)


# ========================
# INSTANCE NOISE INJECTION
# ========================
def add_instance_noise(images, std=0.05):
    """
    Add Gaussian noise to images to prevent discriminator overpowering.
    """
    if std <= 0:
        return images
    noise = torch.randn_like(images) * std
    return images + noise


def get_noise_std(epoch, initial_std=0.1, decay_epochs=50):
    """
    Calculate decaying noise standard deviation.
    """
    return max(0.0, initial_std * (1.0 - epoch / decay_epochs))


# ========================
# TRAINING EPOCH
# ========================
def train_epoch(generator, discriminator, train_loader, opt_g, opt_d, 
                criterion_g, criterion_d, scaler, device, epoch):
    """
    Train for one epoch.
    """
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    total_psnr = 0
    
    # Calculate current noise level
    current_noise_std = get_noise_std(
        epoch, 
        config.INSTANCE_NOISE_INITIAL, 
        config.INSTANCE_NOISE_DECAY_EPOCHS
    )
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.EPOCHS}')
    
    for batch_idx, (corrupted, real) in enumerate(pbar):
        corrupted = corrupted.to(device)
        real = real.to(device)
        
        # ========================
        # TRAIN DISCRIMINATOR
        # ========================
        opt_d.zero_grad(set_to_none=True)
        
        # [FIX] Sử dụng config.device.type để linh hoạt chạy trên CPU/GPU
        with autocast(device_type=config.device.type):
            # Generate fake images
            fake = generator(corrupted)
            
            # Apply instance noise
            real_noisy = add_instance_noise(real, std=current_noise_std)
            fake_noisy = add_instance_noise(fake.detach(), std=current_noise_std)
            corrupted_noisy = add_instance_noise(corrupted, std=current_noise_std)
            
            # Discriminator predictions
            pred_real = discriminator(corrupted_noisy, real_noisy)
            pred_fake = discriminator(corrupted_noisy, fake_noisy)
            
            # Calculate loss
            loss_real = criterion_d(pred_real, target_is_real=True, 
                                   smoothing=config.LABEL_SMOOTHING)
            loss_fake = criterion_d(pred_fake, target_is_real=False, 
                                   smoothing=0.0)
            
            loss_d = (loss_real + loss_fake) * 0.5
        
        # Backward pass
        scaler.scale(loss_d).backward()
        
        # Gradient clipping
        scaler.unscale_(opt_d)
        torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(), 
            max_norm=config.GRADIENT_CLIP_MAX_NORM
        )
        
        scaler.step(opt_d)
        
        # ========================
        # TRAIN GENERATOR
        # ========================
        opt_g.zero_grad(set_to_none=True)
        
        # [FIX] Cập nhật autocast ở Generator
        with autocast(device_type=config.device.type):
            # Generate images again
            fake_g = generator(corrupted)
            
            # Apply noise to G's inputs for D check
            corrupted_noisy_g = add_instance_noise(corrupted, std=current_noise_std)
            fake_noisy_g = add_instance_noise(fake_g, std=current_noise_std)
            
            # Discriminator prediction
            pred_fake_g = discriminator(corrupted_noisy_g, fake_noisy_g)
            
            # Combined generator loss
            loss_g, loss_dict = criterion_g.compute_generator_loss(
                fake_g, real, pred_fake_g
            )
        
        scaler.scale(loss_g).backward()
        
        scaler.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(
            generator.parameters(), 
            max_norm=config.GRADIENT_CLIP_MAX_NORM
        )
        
        scaler.step(opt_g)
        scaler.update()
        
        # ========================
        # METRICS & LOGGING
        # ========================
        with torch.no_grad():
            psnr = calculate_psnr(fake_g, real)
        
        total_g_loss += loss_g.item()
        total_d_loss += loss_d.item()
        total_psnr += psnr
        
        if batch_idx % config.LOG_BATCH_INTERVAL == 0:
            global_step = epoch * len(train_loader) + batch_idx
            wandb.log({
                "Batch/G_Loss": loss_g.item(),
                "Batch/D_Loss": loss_d.item(),
                "Batch/D_Loss_Real": loss_real.item(),
                "Batch/D_Loss_Fake": loss_fake.item(),
                "Batch/Pixel_Loss": loss_dict['pixel'],
                "Batch/Perceptual_Loss": loss_dict['perceptual'],
                "Batch/Adv_Loss": loss_dict['adversarial'],
                "Batch/PSNR": psnr,
                "Batch/Noise_Std": current_noise_std,
                "global_step": global_step
            }, commit=True)
        
        pbar.set_postfix({
            'G_loss': f'{loss_g.item():.4f}',
            'D_loss': f'{loss_d.item():.4f}',
            'PSNR': f'{psnr:.2f}',
            'noise': f'{current_noise_std:.3f}'
        })
    
    return (
        total_g_loss / len(train_loader), 
        total_d_loss / len(train_loader), 
        total_psnr / len(train_loader)
    )


# ========================
# VALIDATION
# ========================
@torch.no_grad()
def validate(generator, val_loader, device):
    generator.eval()
    total_psnr = 0
    total_ssim = 0
    
    for corrupted, real in tqdm(val_loader, desc='Validating'):
        corrupted = corrupted.to(device)
        real = real.to(device)
        
        fake = generator(corrupted)
        
        psnr = calculate_psnr(fake, real)
        ssim = calculate_ssim(fake, real)
        
        total_psnr += psnr
        total_ssim += ssim
    
    return total_psnr / len(val_loader), total_ssim / len(val_loader)


# ========================
# WANDB IMAGE LOGGING
# ========================
def log_images_to_wandb(generator, val_loader, device, epoch):
    generator.eval()
    with torch.no_grad():
        corrupted, real = next(iter(val_loader))
        corrupted = corrupted.to(device)
        real = real.to(device)
        
        fake = generator(corrupted)
        num_images = min(config.NUM_VISUALIZATION_IMAGES, corrupted.size(0))
        images = []
        
        for i in range(num_images):
            c_img = denormalize(corrupted[i]).cpu().permute(1, 2, 0).numpy()
            f_img = denormalize(fake[i]).cpu().permute(1, 2, 0).numpy()
            r_img = denormalize(real[i]).cpu().permute(1, 2, 0).numpy()
            
            c_img = np.clip(c_img, 0, 1)
            f_img = np.clip(f_img, 0, 1)
            r_img = np.clip(r_img, 0, 1)
            
            combined = np.hstack((c_img, f_img, r_img))
            images.append(
                wandb.Image(
                    combined, 
                    caption=f"Epoch {epoch} | Image {i+1}"
                )
            )
        
        wandb.log({"Validation/Samples": images}, commit=False)


# ========================
# MAIN
# ========================
def main():
    wandb.init(
        project="Face-Inpainting-CelebA-HQ",
        name=f"UNet-Attention-SpectralNorm-{config.EPOCHS}ep",
        config={
            "architecture": "U-Net + Self-Attention",
            "discriminator": "PatchGAN + Spectral Norm",
            "dataset": "CelebA-HQ 256x256",
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LR,
            "img_size": config.IMG_SIZE,
            "lambda_pixel": config.LAMBDA_PIXEL,
            "lambda_perceptual": config.LAMBDA_PERCEPTUAL,
            "lambda_adv": config.LAMBDA_ADV,
            "label_smoothing": config.LABEL_SMOOTHING,
            "instance_noise": config.INSTANCE_NOISE_INITIAL,
            "gradient_clip": config.GRADIENT_CLIP_MAX_NORM,
        }
    )
    
    print("=" * 60)
    print("🚀 TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Image Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning Rate: {config.LR}")
    print(f"Loss Weights - Pixel: {config.LAMBDA_PIXEL}, "
          f"Perceptual: {config.LAMBDA_PERCEPTUAL}, Adv: {config.LAMBDA_ADV}")
    print("=" * 60)
    
    train_dataset = ImageRestorationDataset(
        config.TRAIN_DIR, 
        config.IMG_SIZE, 
        mode='train',
        split_ratio=config.TRAIN_VAL_SPLIT_RATIO
    )
    
    val_dataset = ImageRestorationDataset(
        config.VAL_DIR, 
        config.IMG_SIZE, 
        mode='val',
        split_ratio=config.TRAIN_VAL_SPLIT_RATIO
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    generator = Generator().to(config.device)
    discriminator = Discriminator().to(config.device)
    
    if torch.cuda.device_count() > 1:
        print(f"📊 Using {torch.cuda.device_count()} GPUs!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    
    opt_g = torch.optim.Adam(
        generator.parameters(), 
        lr=config.LR, 
        betas=(config.BETA1, config.BETA2)
    )
    
    opt_d = torch.optim.Adam(
        discriminator.parameters(), 
        lr=config.LR, 
        betas=(config.BETA1, config.BETA2)
    )
    
    criterion_g = TotalLoss(
        config.LAMBDA_PIXEL, 
        config.LAMBDA_PERCEPTUAL, 
        config.LAMBDA_ADV
    ).to(config.device)
    
    criterion_d = GANLoss().to(config.device)
    
    # [FIX] Khởi tạo GradScaler với thiết bị lấy từ config
    scaler = GradScaler(device=config.device.type)
    
    start_epoch = 1
    best_psnr = 0.0
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'last_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Found existing checkpoint: {checkpoint_path}")
        user_input = 'y' # Tự động resume khi chạy ngầm
        if user_input.lower() == 'y':
            start_epoch, best_psnr = load_checkpoint(
                checkpoint_path, generator, discriminator, 
                opt_g, opt_d, config.device
            )
            start_epoch += 1
    
    print(f"\n🏋️ Starting training from epoch {start_epoch}...\n")
    
    for epoch in range(start_epoch, config.EPOCHS + 1):
        train_g_loss, train_d_loss, train_psnr = train_epoch(
            generator, discriminator, train_loader, 
            opt_g, opt_d, criterion_g, criterion_d, 
            scaler, config.device, epoch
        )
        
        val_psnr, val_ssim = validate(generator, val_loader, config.device)
        
        wandb.log({
            "Epoch": epoch,
            "Train/Avg_G_Loss": train_g_loss,
            "Train/Avg_D_Loss": train_d_loss,
            "Train/Avg_PSNR": train_psnr,
            "Val/PSNR": val_psnr,
            "Val/SSIM": val_ssim,
            "Learning_Rate/G": opt_g.param_groups[0]['lr'],
            "Learning_Rate/D": opt_d.param_groups[0]['lr']
        }, commit=False)
        
        if epoch % config.LOG_IMAGES_EVERY_N_EPOCHS == 0:
            log_images_to_wandb(generator, val_loader, config.device, epoch)
        
        wandb.log({}, commit=True)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.EPOCHS} Summary")
        print(f"{'='*60}")
        print(f"Train - G Loss: {train_g_loss:.4f} | D Loss: {train_d_loss:.4f} | PSNR: {train_psnr:.2f} dB")
        print(f"Val   - PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")
        print(f"{'='*60}\n")
        
        save_checkpoint(
            {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
                'best_psnr': best_psnr,
                'train_g_loss': train_g_loss,
                'train_d_loss': train_d_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim
            },
            os.path.join(config.CHECKPOINT_DIR, 'last_checkpoint.pth')
        )
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(
                {
                    'epoch': epoch,
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'opt_g': opt_g.state_dict(),
                    'opt_d': opt_d.state_dict(),
                    'best_psnr': best_psnr,
                    'val_psnr': val_psnr,
                    'val_ssim': val_ssim
                },
                os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            )
            print(f"🏆 New best model saved! PSNR: {best_psnr:.2f} dB\n")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    print(f"Best Validation PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved in: {config.CHECKPOINT_DIR}")
    print("="*60)
    
    wandb.finish()

if __name__ == '__main__':
    main()