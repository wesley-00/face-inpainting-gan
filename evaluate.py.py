import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# Import metrics from torchmetrics
from torchmetrics.image import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure, 
    LearnedPerceptualImagePatchSimilarity
)
from torchmetrics.image.fid import FrechetInceptionDistance

import config
from dataset import ImageRestorationDataset
from models import Generator
from utils import denormalize


def evaluate(model_path, device='cuda'):
    """
    Comprehensive evaluation of trained model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to run evaluation on
    """
    print("=" * 70)
    print("🔍 MODEL EVALUATION - CelebA-HQ Image Inpainting")
    print("=" * 70)
    print(f"📂 Model Path: {model_path}")
    print(f"🖥️  Device: {device}\n")
    
    # ========================
    # LOAD MODEL
    # ========================
    print("🔄 Loading generator model...")
    generator = Generator().to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle DataParallel state dict
    state_dict = checkpoint['generator']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    
    generator.load_state_dict(new_state_dict)
    generator.eval()
    
    epoch = checkpoint.get('epoch', 'Unknown')
    print(f"✅ Model loaded successfully (Epoch: {epoch})\n")
    
    # ========================
    # SETUP DATALOADER
    # ========================
    print("📊 Preparing validation dataset...")
    val_dataset = ImageRestorationDataset(
        config.VAL_DIR, 
        config.IMG_SIZE, 
        mode='val', 
        corruption_type='inpainting',
        split_ratio=config.TRAIN_VAL_SPLIT_RATIO
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,  # Larger batch for faster evaluation
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Validation set: {len(val_dataset)} images\n")
    
    # ========================
    # INITIALIZE METRICS
    # ========================
    print("⚙️  Initializing metrics (downloading pretrained models if needed)...")
    print("    This may take a few minutes on first run...\n")
    
    # PSNR & SSIM (input range [0, 1])
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # LPIPS - VGG-based perceptual similarity (lower is better)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type='vgg',  # Can also use 'alex' or 'squeeze'
        normalize=True   # Expects input in [0, 1]
    ).to(device)
    
    # FID - Fréchet Inception Distance (lower is better)
    # Note: Requires uint8 images in [0, 255]
    fid_metric = FrechetInceptionDistance(
        feature=64,  # Inception feature dimension
        normalize=True
    ).to(device)
    
    print("✅ Metrics initialized successfully\n")
    
    # ========================
    # EVALUATION LOOP
    # ========================
    print(f"🧪 Running evaluation on {len(val_dataset)} images...")
    print(f"{'='*70}\n")
    
    with torch.no_grad():
        for batch_idx, (corrupted, real) in enumerate(tqdm(val_loader, desc="Evaluating")):
            corrupted = corrupted.to(device)
            real = real.to(device)
            
            # Generate restored images
            fake = generator(corrupted)
            
            # Denormalize from [-1, 1] to [0, 1]
            fake_01 = denormalize(fake).clamp(0, 1)
            real_01 = denormalize(real).clamp(0, 1)
            
            # Update PSNR, SSIM, LPIPS (expect [0, 1] range)
            psnr_metric.update(fake_01, real_01)
            ssim_metric.update(fake_01, real_01)
            lpips_metric.update(fake_01, real_01)
            
            # Update FID (expects uint8 [0, 255])
            fake_uint8 = (fake_01 * 255).to(torch.uint8)
            real_uint8 = (real_01 * 255).to(torch.uint8)
            
            fid_metric.update(real_uint8, real=True)
            fid_metric.update(fake_uint8, real=False)
    
    # ========================
    # COMPUTE FINAL METRICS
    # ========================
    print("\n" + "="*70)
    print("📊 Computing final metrics...")
    print("="*70 + "\n")
    
    final_psnr = psnr_metric.compute().item()
    final_ssim = ssim_metric.compute().item()
    final_lpips = lpips_metric.compute().item()
    final_fid = fid_metric.compute().item()
    
    # ========================
    # DISPLAY RESULTS
    # ========================
    print("="*70)
    print(f"📝 EVALUATION RESULTS - Epoch {epoch}")
    print("="*70)
    print()
    
    # PSNR
    print("1️⃣  PSNR (Peak Signal-to-Noise Ratio)")
    print("   " + "-"*66)
    print(f"   Score: {final_psnr:.2f} dB")
    if final_psnr >= 28:
        status = "🟢 Excellent"
    elif final_psnr >= 25:
        status = "🟡 Good"
    elif final_psnr >= 22:
        status = "🟠 Fair"
    else:
        status = "🔴 Needs Improvement"
    print(f"   Status: {status}")
    print(f"   Interpretation: Measures pixel-level accuracy")
    print(f"   Benchmark: >25 dB is acceptable, >28 dB is excellent")
    print()
    
    # SSIM
    print("2️⃣  SSIM (Structural Similarity Index)")
    print("   " + "-"*66)
    print(f"   Score: {final_ssim:.4f}")
    if final_ssim >= 0.90:
        status = "🟢 Excellent"
    elif final_ssim >= 0.85:
        status = "🟡 Good"
    elif final_ssim >= 0.80:
        status = "🟠 Fair"
    else:
        status = "🔴 Needs Improvement"
    print(f"   Status: {status}")
    print(f"   Interpretation: Measures structural similarity (1.0 is perfect)")
    print(f"   Benchmark: >0.85 is good, >0.90 is excellent")
    print()
    
    # LPIPS
    print("3️⃣  LPIPS (Learned Perceptual Image Patch Similarity)")
    print("   " + "-"*66)
    print(f"   Score: {final_lpips:.4f}")
    if final_lpips <= 0.10:
        status = "🟢 Excellent"
    elif final_lpips <= 0.15:
        status = "🟡 Good"
    elif final_lpips <= 0.25:
        status = "🟠 Fair"
    else:
        status = "🔴 Needs Improvement"
    print(f"   Status: {status}")
    print(f"   Interpretation: Perceptual similarity (lower is better)")
    print(f"   Benchmark: <0.15 is good for GANs, <0.10 is excellent")
    print(f"   Note: Most important metric for perceptual quality")
    print()
    
    # FID
    print("4️⃣  FID (Fréchet Inception Distance)")
    print("   " + "-"*66)
    print(f"   Score: {final_fid:.2f}")
    if final_fid <= 20:
        status = "🟢 Excellent"
    elif final_fid <= 30:
        status = "🟡 Good"
    elif final_fid <= 50:
        status = "🟠 Fair"
    else:
        status = "🔴 Needs Improvement"
    print(f"   Status: {status}")
    print(f"   Interpretation: Distribution distance (lower is better)")
    print(f"   Benchmark: <30 is good, <20 is excellent")
    print(f"   Note: Measures how 'realistic' the entire generated set is")
    
    if len(val_dataset) < 2048:
        print(f"   ⚠️  Warning: FID calculated on {len(val_dataset)} images")
        print(f"      (recommended: 2048+). Results may be less stable.")
    
    print()
    print("="*70)
    
    # ========================
    # OVERALL ASSESSMENT
    # ========================
    # Calculate overall score (weighted average)
    overall_score = (
        (min(final_psnr, 30) / 30) * 0.20 +  # PSNR (20% weight)
        final_ssim * 0.20 +                    # SSIM (20% weight)
        (1 - min(final_lpips, 0.5) / 0.5) * 0.30 +  # LPIPS (30% weight - most important)
        (1 - min(final_fid, 100) / 100) * 0.30   # FID (30% weight)
    )
    
    print("\n📈 OVERALL ASSESSMENT")
    print("="*70)
    print(f"Overall Score: {overall_score*100:.1f}/100")
    
    if overall_score >= 0.85:
        assessment = "🟢 Excellent - Production ready!"
    elif overall_score >= 0.75:
        assessment = "🟡 Good - Minor improvements recommended"
    elif overall_score >= 0.65:
        assessment = "🟠 Fair - Significant improvements needed"
    else:
        assessment = "🔴 Poor - Requires substantial training/tuning"
    
    print(f"Assessment: {assessment}")
    print("="*70)
    
    # ========================
    # SAVE RESULTS
    # ========================
    results_path = os.path.join(config.OUTPUT_DIR, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Results - Epoch {epoch}\n")
        f.write("="*50 + "\n\n")
        f.write(f"PSNR: {final_psnr:.2f} dB\n")
        f.write(f"SSIM: {final_ssim:.4f}\n")
        f.write(f"LPIPS: {final_lpips:.4f}\n")
        f.write(f"FID: {final_fid:.2f}\n")
        f.write(f"\nOverall Score: {overall_score*100:.1f}/100\n")
    
    print(f"\n💾 Results saved to: {results_path}\n")


if __name__ == "__main__":
    # Path to best model checkpoint
    MODEL_PATH = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    
    # Alternative: Evaluate last checkpoint
    # MODEL_PATH = os.path.join(config.CHECKPOINT_DIR, 'last_checkpoint.pth')
    
    evaluate(MODEL_PATH, device=config.device)
