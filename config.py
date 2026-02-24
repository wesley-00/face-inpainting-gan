import torch
import os

# ========================
# HYPERPARAMETERS
# ========================
IMG_SIZE = 256
BATCH_SIZE = 32  # Optimized for dual T4 GPUs
EPOCHS = 70
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999

# ========================
# LOSS WEIGHTS
# ========================
LAMBDA_PIXEL = 100.0
LAMBDA_PERCEPTUAL = 10.0
LAMBDA_ADV = 1.0

# ========================
# TRAINING STABILITY
# ========================
LABEL_SMOOTHING = 0.1  # For real labels (0.9 instead of 1.0)
GRADIENT_CLIP_MAX_NORM = 1.0
INSTANCE_NOISE_INITIAL = 0.1  # Initial noise std
INSTANCE_NOISE_DECAY_EPOCHS = 50  # Decay to 0 over this many epochs

# ========================
# LOGGING
# ========================
LOG_BATCH_INTERVAL = 50  # Log to WandB every N batches
LOG_IMAGES_EVERY_N_EPOCHS = 1  # Visualize results every N epochs
NUM_VISUALIZATION_IMAGES = 4  # Number of images to log to WandB

# ========================
# DATA PATHS
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'celeba_hq_256')
TRAIN_DIR = DATA_PATH 
VAL_DIR = DATA_PATH
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Tự động tạo các thư mục này nếu chúng chưa tồn tại trên máy để tránh lỗi crash
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# DATALOADER SETTINGS
# ========================
NUM_WORKERS = 4  # Optimized for dual GPU setup
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# ========================
# DATASET SPLIT
# ========================
TRAIN_VAL_SPLIT_RATIO = 0.9  # 90% train, 10% validation

# ========================
# DEVICE CONFIGURATION
# ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# CREATE DIRECTORIES
# ========================
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# PRINT CONFIGURATION
# ========================
if __name__ == '__main__':
    print("=" * 50)
    print("CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LR}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
    print("=" * 50)