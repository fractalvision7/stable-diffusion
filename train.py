"""
Enhanced Diffusion Model Training with U-Net 2.5D Architecture
Optimized for 150K dataset on 32GB GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
from pathlib import Path
import wandb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = True  # FP16 training
    gradient_checkpointing = True  # Save memory
    compile_model = True  # Torch 2.0 speedup
    
    # Paths
    data_root = "./dataset"  # CHANGE: Your dataset path
    metadata_file = "./dataset/metadata.json"  # Optional: JSON with captions
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"
    
    # Dataset
    image_size = 512  # Optimal for 32GB GPU
    image_channels = 3
    caption_max_length = 77
    
    # U-Net Architecture (Modern Configuration)
    unet_channels = 128
    unet_channel_mult = [1, 2, 4, 8]  # Progressive downsampling
    unet_num_res_blocks = 2
    attention_resolutions = [16, 8]  # Apply attention at these resolutions
    dropout = 0.1
    num_heads = 8  # Multi-head attention
    use_spatial_transformer = True
    transformer_depth = 1
    context_dim = 768  # Text embedding dimension
    
    # Training
    batch_size = 8  # Adjust based on memory
    micro_batch_size = 2  # Gradient accumulation
    gradient_accumulation_steps = batch_size // micro_batch_size
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    adam_betas = (0.9, 0.999)
    adam_weight_decay = 1e-2
    clip_grad_norm = 1.0
    
    # Diffusion
    timesteps = 1000
    beta_schedule = "cosine"  # Better than linear
    variance_type = "fixed_small"
    loss_type = "huber"  # More stable than MSE
    
    # Optimization
    epochs = 100
    save_every = 1000  # Save checkpoint every N steps
    sample_every = 500  # Generate samples during training
    validation_every = 2000
    log_every = 50
    
    # Text Encoder
    text_model = "google/flan-t5-base"  # Better than CLIP for understanding
    
    # Distributed Training
    ddp = False
    local_rank = 0

config = Config()

# ==================== ENHANCED DATASET ====================
class DiffusionDataset(Dataset):
    """Robust dataset with multiple caption sources"""
    
    def __init__(self, data_root, image_size=512, transform=None, is_train=True):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.is_train = is_train
        
        # Collect all valid images
        self.image_paths = []
        self.captions = []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        # Option 1: Check for metadata.json
        metadata_path = Path(data_root) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            for img_name, caption in metadata.items():
                img_path = Path(data_root) / "images" / img_name
                if img_path.exists():
                    self.image_paths.append(img_path)
                    self.captions.append(caption)
        else:
            # Option 2: Look for images with .txt captions
            images_dir = Path(data_root) / "images"
            captions_dir = Path(data_root) / "captions"
            
            for img_path in images_dir.glob("*"):
                if img_path.suffix.lower() in valid_extensions:
                    # Try to find caption
                    caption_path = captions_dir / f"{img_path.stem}.txt"
                    if caption_path.exists():
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                    else:
                        # Use filename as fallback caption
                        caption = img_path.stem.replace('_', ' ')
                    
                    self.image_paths.append(img_path)
                    self.captions.append(caption)
        
        print(f"Loaded {len(self.image_paths)} images with captions")
        
        # Enhanced transforms with augmentation
        if transform is None:
            if is_train:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size + 32),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])
        else:
            self.transform = transform
        
        # Text tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            
            # Apply transform
            image_tensor = self.transform(image)
            
            # Process caption
            caption = self.captions[idx]
            if len(caption) == 0:
                caption = "a high quality image"
            
            # Tokenize with proper padding/truncation
            tokens = self.tokenizer(
                caption,
                max_length=config.caption_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": image_tensor,
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "caption": caption
            }
            
        except Exception as e:
            # Return a different sample if error occurs
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

# ==================== ENHANCED U-NET ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.nin_shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhdn,bhem->bhnm', q * scale, k)
        attn = F.softmax(attn, dim=-1)
        h = torch.einsum('bhnm,bhem->bhdn', attn, v)
        h = h.reshape(B, C, H, W)
        return x + self.proj_out(h)

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, context_dim, num_heads=8, depth=1):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.proj_in = nn.Conv2d(in_channels, in_channels, 1)
        
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn1': nn.MultiheadAttention(in_channels, num_heads, batch_first=True),
                'ff': nn.Sequential(
                    nn.Linear(in_channels, in_channels * 4),
                    nn.GELU(),
                    nn.Linear(in_channels * 4, in_channels)
                ),
                'norm1': nn.LayerNorm(in_channels),
                'norm2': nn.LayerNorm(in_channels),
                'attn2': nn.MultiheadAttention(in_channels, num_heads, batch_first=True, kdim=context_dim, vdim=context_dim)
            }) for _ in range(depth)
        ])
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = self.proj_in(h)
        
        # Reshape for attention
        h = h.reshape(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        
        for block in self.transformer_blocks:
            # Self-attention
            residual = h
            h = block['norm1'](h)
            h_attn, _ = block['attn1'](h, h, h)
            h = residual + h_attn
            
            # Cross-attention
            residual = h
            h = block['norm2'](h)
            h_attn, _ = block['attn2'](h, context, context)
            h = residual + h_attn
            
            # Feed-forward
            residual = h
            h = block['ff'](h)
            h = residual + h
        
        # Reshape back
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj_out(h)

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class EnhancedUNet(nn.Module):
    """Modern U-Net 2.5D with attention and transformer blocks"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )
        
        # Text embedding projection
        self.context_proj = nn.Linear(config.context_dim, 512)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(config.image_channels, config.unet_channels, 3, padding=1)
        
        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        ch = config.unet_channels
        chs = [ch]
        
        for i, mult in enumerate(config.unet_channel_mult):
            out_ch = ch * mult
            for _ in range(config.unet_num_res_blocks):
                block = nn.ModuleList([
                    ResidualBlock(ch, out_ch, config.dropout),
                    AttentionBlock(out_ch, config.num_heads) if (i in config.attention_resolutions) else nn.Identity(),
                    SpatialTransformer(out_ch, config.context_dim, config.num_heads, config.transformer_depth) 
                    if config.use_spatial_transformer and (i in config.attention_resolutions) else nn.Identity()
                ])
                self.down_blocks.append(block)
                ch = out_ch
                chs.append(ch)
            
            if i != len(config.unet_channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))
                chs.append(ch)
        
        # Middle blocks
        self.middle_block = nn.ModuleList([
            ResidualBlock(ch, ch, config.dropout),
            AttentionBlock(ch, config.num_heads),
            SpatialTransformer(ch, config.context_dim, config.num_heads, config.transformer_depth),
            ResidualBlock(ch, ch, config.dropout)
        ])
        
        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(config.unet_channel_mult))):
            out_ch = config.unet_channels * mult
            for j in range(config.unet_num_res_blocks + 1):
                block = nn.ModuleList([
                    ResidualBlock(ch + chs.pop(), out_ch, config.dropout),
                    AttentionBlock(out_ch, config.num_heads) if (i in config.attention_resolutions) else nn.Identity(),
                    SpatialTransformer(out_ch, config.context_dim, config.num_heads, config.transformer_depth)
                    if config.use_spatial_transformer and (i in config.attention_resolutions) else nn.Identity()
                ])
                self.up_blocks.append(block)
                ch = out_ch
            
            if i != 0:
                self.up_blocks.append(Upsample(ch))
        
        # Final
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, config.image_channels, 3, padding=1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, timesteps, context):
        # Timestep embedding
        t_emb = get_timestep_embedding(timesteps, 256)
        t_emb = self.time_embed(t_emb)
        
        # Context embedding
        if context is not None:
            context = self.context_proj(context)
            t_emb = t_emb + context.mean(dim=1, keepdim=True)
        
        # Initial convolution
        h = self.conv_in(x)
        hs = [h]
        
        # Downsample
        for block in self.down_blocks:
            if isinstance(block, Downsample):
                h = block(h)
                hs.append(h)
            else:
                for layer in block:
                    if isinstance(layer, SpatialTransformer):
                        h = layer(h, context)
                    else:
                        h = layer(h)
                hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, SpatialTransformer):
                h = layer(h, context)
            else:
                h = layer(h)
        
        # Upsample
        for block in self.up_blocks:
            if isinstance(block, Upsample):
                h = block(h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                for layer in block:
                    if isinstance(layer, SpatialTransformer):
                        h = layer(h, context)
                    else:
                        h = layer(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h

def get_timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiffusionModel:
    """Main training class with EMA, gradient checkpointing, and mixed precision"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Models
        self.unet = EnhancedUNet(config).to(self.device)
        
        # EMA for stable inference
        self.ema_unet = EnhancedUNet(config).to(self.device)
        self.ema_unet.load_state_dict(self.unet.state_dict())
        self.ema_decay = 0.9999
        
        # Text encoder
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model)
        self.text_encoder = AutoModel.from_pretrained(config.text_model).to(self.device)
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Diffusion scheduler
        self.betas = get_beta_schedule(config.timesteps, config.beta_schedule)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=config.learning_rate,
            betas=config.adam_betas,
            weight_decay=config.adam_weight_decay
        )
        
        # Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs * (150000 // config.batch_size),
            eta_min=1e-6
        )
        
        # Mixed precision
        self.scaler = GradScaler(enabled=config.mixed_precision)
        
        # Gradient checkpointing
        if config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        
        # Compile for speed (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            self.unet = torch.compile(self.unet)
        
        print(f"Model initialized on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.unet.parameters()):,}")
    
    def encode_text(self, text):
        """Encode text to embeddings"""
        with torch.no_grad():
            tokens = self.tokenizer(
                text,
                max_length=config.caption_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings from last hidden state
            outputs = self.text_encoder(**tokens)
            return outputs.last_hidden_state
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(1. - self.alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def compute_loss(self, x_start, text_embeddings, t=None):
        """Compute diffusion loss"""
        if t is None:
            t = torch.randint(0, self.config.timesteps, (x_start.shape[0],), device=self.device).long()
        
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        noise_pred = self.unet(x_noisy, t, text_embeddings)
        
        # Loss
        if self.config.loss_type == "huber":
            loss = F.smooth_l1_loss(noise_pred, noise)
        else:
            loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def update_ema(self):
        """Update EMA model"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_unet.parameters(), self.unet.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def train_step(self, batch, global_step):
        """Single training step"""
        images = batch["pixel_values"].to(self.device)
        text = batch["caption"]
        
        # Encode text
        text_embeddings = self.encode_text(text)
        
        # Compute loss
        with autocast(enabled=self.config.mixed_precision):
            loss = self.compute_loss(images, text_embeddings)
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.config.clip_grad_norm)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Update EMA
            self.update_ema()
            
            # LR scheduler
            self.lr_scheduler.step()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def save_checkpoint(self, path, global_step, epoch, best=False):
        """Save model checkpoint"""
        checkpoint = {
            'global_step': global_step,
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'ema_unet_state_dict': self.ema_unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.config.__dict__,
        }
        
        torch.save(checkpoint, path)
        
        if best:
            best_path = Path(path).parent / "model_best.pth"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.ema_unet.load_state_dict(checkpoint['ema_unet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        return checkpoint['global_step'], checkpoint['epoch']

# ==================== UTILITY FUNCTIONS ====================
def extract(a, t, x_shape):
    """Extract values from a at indices t and reshape to match x_shape"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def get_beta_schedule(num_timesteps, schedule="cosine"):
    """Get beta schedule for diffusion"""
    if schedule == "linear":
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    elif schedule == "cosine":
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

# ==================== MAIN TRAINING ====================
def main():
    # Initialize
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(project="diffusion-model", config=config.__dict__)
    
    # Create model
    model = DiffusionModel(config)
    
    # Dataset and dataloader
    dataset = DiffusionDataset(config.data_root, config.image_size, is_train=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        progress_bar = tqdm(dataloader, desc=f"Training")
        
        for batch in progress_bar:
            # Training step
            loss = model.train_step(batch, global_step)
            
            # Logging
            if global_step % config.log_every == 0:
                wandb.log({
                    "train/loss": loss,
                    "train/lr": model.lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/step": global_step
                })
                progress_bar.set_postfix({"loss": f"{loss:.4f}", "step": global_step})
            
            # Validation
            if global_step % config.validation_every == 0:
                val_loss = validate(model, config)
                wandb.log({"val/loss": val_loss})
                
                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    model.save_checkpoint(
                        f"{config.checkpoint_dir}/model_best.pth",
                        global_step,
                        epoch,
                        best=True
                    )
            
            # Save checkpoint
            if global_step % config.save_every == 0:
                model.save_checkpoint(
                    f"{config.checkpoint_dir}/model_step_{global_step}.pth",
                    global_step,
                    epoch
                )
            
            # Sample generation
            if global_step % config.sample_every == 0:
                generate_samples(model, global_step, config)
            
            global_step += 1
        
        # Save epoch checkpoint
        model.save_checkpoint(
            f"{config.checkpoint_dir}/model_epoch_{epoch+1}.pth",
            global_step,
            epoch
        )
    
    # Save final model
    model.save_checkpoint(f"{config.checkpoint_dir}/model_final.pth", global_step, config.epochs)
    print("Training completed!")

def validate(model, config):
    """Validation loop"""
    model.unet.eval()
    
    val_dataset = DiffusionDataset(config.data_root, config.image_size, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=config.micro_batch_size, shuffle=False)
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["pixel_values"].to(model.device)
            text = batch["caption"]
            
            text_embeddings = model.encode_text(text)
            loss = model.compute_loss(images, text_embeddings)
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 10:  # Limit validation batches
                break
    
    model.unet.train()
    return total_loss / num_batches

def generate_samples(model, step, config):
    """Generate sample images during training"""
    model.ema_unet.eval()
    
    with torch.no_grad():
        # Sample noise
        z = torch.randn(1, 3, config.image_size, config.image_size, device=model.device)
        
        # Sample prompt
        prompt = "a high quality photo of a realistic scene"
        text_emb = model.encode_text([prompt])
        
        # DDIM sampling
        images = ddim_sample(model.ema_unet, z, text_emb, config)
        
        # Save
        save_image(images, f"{config.log_dir}/samples/sample_step_{step}.png")
    
    model.ema_unet.train()

if __name__ == "__main__":
    import math
    main()
