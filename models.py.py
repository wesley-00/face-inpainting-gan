import torch
import torch.nn as nn
import torch.nn.utils as utils


def init_weights(m):
    """Initialize network weights using standard GAN initialization"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ========================
# SELF-ATTENTION MODULE
# ========================
class SelfAttention(nn.Module):
    """
    Self-Attention mechanism for capturing long-range dependencies.
    Based on SAGAN (Self-Attention GAN) and Pluralistic Inpainting.
    Helps maintain global consistency (e.g., facial symmetry).
    """
    def __init__(self, in_dim):
        super().__init__()
        # Reduce channels for attention computation (memory efficiency)
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        # Learnable scaling parameter (starts at 0 for stable training)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Compute attention maps
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        
        # Attention scores
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention to values
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Residual connection with learnable weight
        return self.gamma * out + x


# ========================
# U-NET BUILDING BLOCKS
# ========================
class DownBlock(nn.Module):
    """Encoder block: Conv2d -> (BatchNorm) -> LeakyReLU -> (Dropout)"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block: Upsample -> Conv2d -> BatchNorm -> ReLU -> (Dropout)"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.dropout = nn.Dropout(dropout) if dropout else None
    
    def forward(self, x, skip=None):
        """
        Args:
            x: Input tensor
            skip: Skip connection from encoder (concatenated after upsampling)
        """
        x = self.up(x)
        if self.dropout:
            x = self.dropout(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return x


# ========================
# GENERATOR (U-NET + ATTENTION)
# ========================
class Generator(nn.Module):
    """
    U-Net based generator with Self-Attention.
    Architecture: 8 encoder blocks -> Bottleneck -> 8 decoder blocks
    Attention applied at 32x32 feature map resolution.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # ========================
        # ENCODER (Downsampling path)
        # ========================
        self.down1 = DownBlock(in_channels, 64, normalize=False)  # 256 -> 128
        self.down2 = DownBlock(64, 128)                            # 128 -> 64
        self.down3 = DownBlock(128, 256)                           # 64 -> 32
        self.down4 = DownBlock(256, 512, dropout=0.5)              # 32 -> 16
        self.down5 = DownBlock(512, 512, dropout=0.5)              # 16 -> 8
        self.down6 = DownBlock(512, 512, dropout=0.5)              # 8 -> 4
        self.down7 = DownBlock(512, 512, dropout=0.5)              # 4 -> 2
        self.down8 = DownBlock(512, 512, normalize=False, dropout=0.5)  # 2 -> 1 (bottleneck)
        
        # ========================
        # DECODER (Upsampling path)
        # ========================
        self.up1 = UpBlock(512, 512, dropout=0.5)       # 1 -> 2
        self.up2 = UpBlock(1024, 512, dropout=0.5)      # 2 -> 4 (concat: 512+512=1024)
        self.up3 = UpBlock(1024, 512, dropout=0.5)      # 4 -> 8 (concat: 512+512=1024)
        
        # Self-Attention at 32x32 resolution (after up3, before skip concat)
        self.attention = SelfAttention(512)
        
        self.up4 = UpBlock(1024, 512, dropout=0.5)      # 8 -> 16 (concat: 512+512=1024)
        self.up5 = UpBlock(1024, 256)                   # 16 -> 32 (concat: 512+512=1024)
        self.up6 = UpBlock(512, 128)                    # 32 -> 64 (concat: 256+256=512)
        self.up7 = UpBlock(256, 64)                     # 64 -> 128 (concat: 128+128=256)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, out_channels, 3, 1, 1),  # 128 -> 128 (concat: 64+64=128)
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, x):
        # ========================
        # ENCODER
        # ========================
        d1 = self.down1(x)      # [B, 64, 128, 128]
        d2 = self.down2(d1)     # [B, 128, 64, 64]
        d3 = self.down3(d2)     # [B, 256, 32, 32]
        d4 = self.down4(d3)     # [B, 512, 16, 16]
        d5 = self.down5(d4)     # [B, 512, 8, 8]
        d6 = self.down6(d5)     # [B, 512, 4, 4]
        d7 = self.down7(d6)     # [B, 512, 2, 2]
        d8 = self.down8(d7)     # [B, 512, 1, 1] - Bottleneck
        
        # ========================
        # DECODER
        # ========================
        u1 = self.up1(d8, d7)   # [B, 1024, 2, 2] after concat
        u2 = self.up2(u1, d6)   # [B, 1024, 4, 4] after concat
        u3 = self.up3(u2, d5)   # [B, 1024, 8, 8] after concat
        
        # CRITICAL FIX: Apply attention BEFORE concatenation with d4
        # Extract the upsampled part (512 channels) before skip connection
        # We need to split u3 or apply attention differently
        
        # Solution: Apply attention on the 512-channel upsampled features
        # Since u3 already has skip concat (1024 channels), we split it
        u3_features = u3[:, :512, :, :]  # First 512 channels (upsampled part)
        u3_skip = u3[:, 512:, :, :]      # Last 512 channels (skip from d5)
        
        # Apply attention on upsampled features only
        u3_attended = self.attention(u3_features)
        
        # Recombine with skip connection
        u3 = torch.cat([u3_attended, u3_skip], dim=1)  # [B, 1024, 8, 8]
        
        u4 = self.up4(u3, d4)   # [B, 1024, 16, 16] after concat
        u5 = self.up5(u4, d3)   # [B, 512, 32, 32] after concat
        u6 = self.up6(u5, d2)   # [B, 256, 64, 64] after concat
        u7 = self.up7(u6, d1)   # [B, 128, 128, 128] after concat
        
        return self.final(u7)   # [B, 3, 256, 256]


# ========================
# DISCRIMINATOR (PATCHGAN + SPECTRAL NORM)
# ========================
class Discriminator(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization.
    - Outputs a patch map instead of single scalar
    - Spectral Norm applied to all conv layers for training stability
    - No BatchNorm (Spectral Norm replaces it)
    """
    def __init__(self, in_channels=6):  # 6 = 3 (corrupted) + 3 (real/fake)
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=False):
            """
            Discriminator block with Spectral Normalization.
            Note: We don't use BatchNorm with Spectral Norm (common practice)
            """
            layers = [
                utils.spectral_norm(
                    nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False)
                )
            ]
            # Optionally use InstanceNorm instead of BatchNorm
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            # Input: [B, 6, 256, 256]
            *discriminator_block(in_channels, 64, normalize=False),  # -> [B, 64, 128, 128]
            *discriminator_block(64, 128, normalize=False),          # -> [B, 128, 64, 64]
            *discriminator_block(128, 256, normalize=False),         # -> [B, 256, 32, 32]
            *discriminator_block(256, 512, normalize=False),         # -> [B, 512, 16, 16]
            
            # Output layer (PatchGAN - outputs patch map)
            utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))          # -> [B, 1, 15, 15]
        )
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, x, y):
        """
        Args:
            x: Corrupted/masked input image [B, 3, H, W]
            y: Real or fake image [B, 3, H, W]
        Returns:
            Patch-level predictions [B, 1, 15, 15]
        """
        return self.model(torch.cat([x, y], dim=1))
