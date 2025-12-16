"""
Optimized Training Script - Fixed image saving & uses full 32GB GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import os
import zipfile
import tempfile
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import gc
import sys
import re

# Install missing packages
def install_package(package):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from diffusers import DDPMScheduler, UNet2DConditionModel
    print("✓ diffusers imported")
except ImportError:
    print("Installing diffusers...")
    install_package("diffusers")
    from diffusers import DDPMScheduler, UNet2DConditionModel

try:
    from transformers import CLIPTokenizer, CLIPTextModel
    print("✓ transformers imported")
except ImportError:
    print("Installing transformers...")
    install_package("transformers")
    from transformers import CLIPTokenizer, CLIPTextModel

# ==================== CONFIGURATION OPTIMIZED FOR 32GB GPU ====================
class Config:
    # Paths
    data_source = "./training_dataset.zip"
    
    # ========== OPTIMIZED FOR 32GB GPU ==========
    image_size = 512  # Keep at 512 for quality
    latent_size = image_size // 8
    
    # U-Net - LARGER for 32GB
    unet_channels = 128  # Increased from 128
    channel_multipliers = [1, 2, 4, 8, 8]  # Added extra layer
    num_res_blocks = 3  # Increased from 2
    
    # ========== TRAINING OPTIMIZED FOR 32GB ==========
    batch_size = 6  # DOUBLED from 6 (fits in 32GB)
    micro_batch = 4  # Increased
    gradient_accumulation_steps = batch_size // micro_batch
    
    learning_rate = 1e-4
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    weight_decay = 1e-2
    
    epochs = 30  # More epochs for better quality
    warmup_steps = 1000
    
    # Diffusion
    timesteps = 1000
    beta_schedule = "linear"
    
    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = True  # FP16 for memory
    seed = 42
    
    # Checkpoints
    checkpoint_dir = "./checkpoints"
    sample_dir = "./training_samples"
    save_every = 1000  # Save more often
    log_every = 50
    
    # Validation
    val_split = 0.01  # Reduced for more training data
    sample_prompt = "anime artwork, a person"

config = Config()

# Set seeds
torch.manual_seed(config.seed)
np.random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)
    # Enable TF32 for faster computation (A100/RTX 30xx/40xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ==================== MEMORY OPTIMIZATION ====================
def optimize_memory():
    """Optimize memory usage for 32GB GPU"""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set memory fraction if needed
        # torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU
        
        # Enable memory optimizations
        torch.backends.cudnn.benchmark = True  # Faster convolutions
        torch.backends.cudnn.deterministic = False  # Faster but non-deterministic
        
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

optimize_memory()

# ==================== DATASET HANDLER ====================
class DatasetHandler:
    """Handles zip with duplicate name captions"""
    
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.temp_dir = None
        self.extracted_path = None
        
        print(f"Loading dataset from {zip_path}")
        self._extract_zip()
    
    def _extract_zip(self):
        """Extract zip to temp directory"""
        print("Extracting zip file...")
        
        self.temp_dir = tempfile.mkdtemp(prefix="anime_train_")
        self.extracted_path = os.path.join(self.temp_dir, "dataset")
        
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                files = zip_ref.namelist()
                print(f"Total files in zip: {len(files)}")
                
                # Extract with progress
                total_size = sum(zinfo.file_size for zinfo in zip_ref.infolist())
                
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                    for member in zip_ref.infolist():
                        zip_ref.extract(member, self.extracted_path)
                        pbar.update(member.file_size)
                
            print(f"✓ Extracted to {self.extracted_path}")
            
        except Exception as e:
            print(f"Failed to extract zip: {e}")
            raise
    
    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

# ==================== DATASET CLASS ====================
class AnimeDataset(Dataset):
    """Optimized dataset loading"""
    
    def __init__(self, handler, image_size=512, is_train=True):
        self.handler = handler
        self.image_size = image_size
        self.is_train = is_train
        
        # Find images and captions
        self._find_files()
        self._match_files()
        
        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            local_files_only=False
        )
        
        # Image transforms
        self.transform = self._get_transforms()
    
    def _find_files(self):
        """Find all files"""
        # Look for training_dataset folder
        training_folder = os.path.join(self.handler.extracted_path, "training_dataset")
        if os.path.exists(training_folder):
            self.base_path = training_folder
        else:
            self.base_path = self.handler.extracted_path
        
        # Images folder
        self.images_folder = os.path.join(self.base_path, "images")
        if not os.path.exists(self.images_folder):
            # Find any folder with images
            for root, dirs, files in os.walk(self.base_path):
                if any(f.lower().endswith(('.jpg', '.png')) for f in files[:10]):
                    self.images_folder = root
                    break
        
        # Get all images
        self.image_files = []
        for root, dirs, files in os.walk(self.images_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.image_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_files)} images")
    
    def _match_files(self):
        """Match images with captions"""
        self.image_paths = []
        self.caption_paths = []
        
        # Get all txt files in base path
        txt_files = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.lower().endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        
        print(f"Found {len(txt_files)} caption files")
        
        # Create caption dict for faster lookup
        caption_dict = {}
        for txt_path in txt_files:
            base = os.path.splitext(os.path.basename(txt_path))[0]
            base = base.replace('.jpg', '').replace('.png', '')
            caption_dict[base] = txt_path
        
        # Match images
        matched = 0
        for img_path in tqdm(self.image_files[:100000], desc="Matching"):  # Limit to 100k for speed
            img_name = os.path.basename(img_path)
            img_base = os.path.splitext(img_name)[0]
            
            # Try multiple matching strategies
            caption_path = None
            
            # 1. Exact match
            if img_base in caption_dict:
                caption_path = caption_dict[img_base]
            
            # 2. Remove duplicate show name (A_Lull_in_the_sea_A_Lull_in_the_sea_001 -> A_Lull_in_the_sea_001)
            if not caption_path and '_' in img_base:
                parts = img_base.split('_')
                # Check for duplicate pattern
                for i in range(1, len(parts)//2 + 1):
                    if parts[:i] == parts[i:2*i]:
                        # Remove duplicate
                        simple = '_'.join(parts[i:])
                        if simple in caption_dict:
                            caption_path = caption_dict[simple]
                            break
            
            # 3. Match by numbers
            if not caption_path:
                img_numbers = re.findall(r'\d+', img_base)
                if img_numbers:
                    for cap_base, cap_path in caption_dict.items():
                        cap_numbers = re.findall(r'\d+', cap_base)
                        if cap_numbers and img_numbers[-1] == cap_numbers[-1]:
                            caption_path = cap_path
                            break
            
            if caption_path:
                self.image_paths.append(img_path)
                self.caption_paths.append(caption_path)
                matched += 1
        
        print(f"✓ Matched {matched} image-caption pairs")
        
        if matched == 0:
            raise ValueError("No matches found!")
    
    def _get_transforms(self):
        """Get image transforms"""
        if self.is_train:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img = Image.open(self.image_paths[idx]).convert('RGB')
            pixel_values = self.transform(img)
            
            # Load caption
            with open(self.caption_paths[idx], 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            # Tokenize
            tokens = self.tokenizer(
                caption,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": pixel_values,
                "input_ids": tokens.input_ids.squeeze(0),
                "caption": caption
            }
        except:
            # Return dummy on error
            pixel_values = torch.zeros(3, config.image_size, config.image_size)
            caption = "anime artwork"
            tokens = self.tokenizer(
                caption,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                "pixel_values": pixel_values,
                "input_ids": tokens.input_ids.squeeze(0),
                "caption": caption
            }

# ==================== LARGER AUTOENCODER FOR 32GB ====================
class LargeAutoencoder(nn.Module):
    """Larger autoencoder for better quality"""
    
    def __init__(self, scale_factor=0.18215):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Encoder - larger
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),  # Increased channels
            nn.GroupNorm(32, 96),
            nn.SiLU(),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 192),
            nn.SiLU(),
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 384),
            nn.SiLU(),
            nn.Conv2d(384, 4, kernel_size=3, padding=1),
        )
        
        # Decoder - larger
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 384, kernel_size=3, padding=1),
            nn.GroupNorm(32, 384),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.GroupNorm(32, 192),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.GroupNorm(32, 96),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
    
    def encode(self, images):
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        latents = self.encoder(images)
        return latents * self.scale_factor
    
    def decode(self, latents):
        latents = latents / self.scale_factor
        images = self.decoder(latents)
        return images

# ==================== MODEL INITIALIZATION ====================
def initialize_models():
    """Initialize larger models for 32GB GPU"""
    print("\nInitializing LARGER models for 32GB GPU...")
    
    # Text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).to(config.device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    # Autoencoder
    vae = LargeAutoencoder().to(config.device)
    vae.requires_grad_(False)
    vae.eval()
    
    # LARGER U-Net for 32GB
    print("Creating LARGER U-Net...")
    unet = UNet2DConditionModel(
        sample_size=config.latent_size,
        in_channels=4,
        out_channels=4,
        layers_per_block=config.num_res_blocks,
        block_out_channels=[config.unet_channels * m for m in config.channel_multipliers],
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=768,
        attention_head_dim=8,  # Increased attention
    ).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"✓ U-Net parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.timesteps,
        beta_schedule=config.beta_schedule,
        prediction_type="epsilon"
    )
    
    # Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    
    return {
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
        "tokenizer": tokenizer
    }

# ==================== FIXED IMAGE SAVING ====================
def save_image_fixed(image_tensor, path):
    """
    FIXED: Save image tensor properly
    image_tensor: (3, H, W) in range [-1, 1] or [0, 1]
    """
    # Ensure it's on CPU and detached
    image = image_tensor.cpu().detach()
    
    # Clamp to valid range
    image = torch.clamp(image, -1, 1)
    
    # Convert from [-1, 1] to [0, 1] if needed
    if image.min() < 0:
        image = (image + 1) / 2
    
    # Ensure it's in [0, 1] range
    image = torch.clamp(image, 0, 1)
    
    # Convert to numpy
    img_np = image.permute(1, 2, 0).numpy()
    
    # Save with matplotlib (FIXED)
    plt.imsave(path, img_np)
    return True

def generate_sample_fixed(models, prompt, epoch, save_path):
    """FIXED: Generate and save sample image"""
    print(f"Generating sample for epoch {epoch+1}")
    
    unet = models["unet"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    
    unet.eval()
    
    with torch.no_grad():
        # Tokenize
        text_input = tokenizer(
            [prompt],
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(config.device)
        
        text_embeddings = text_encoder(text_input.input_ids)[0]
        
        # Create noise
        latents = torch.randn(
            (1, 4, config.latent_size, config.latent_size),
            device=config.device
        )
        
        # Simple sampling (30 steps)
        num_steps = 30
        alphas = torch.linspace(0.99, 0.01, num_steps).to(config.device)
        
        for i in range(num_steps):
            alpha = alphas[i]
            
            # Predict noise
            noise_pred = unet(
                latents,
                torch.tensor([i * 33], device=config.device),  # Scaled to 0-1000
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Update latents (simplified DDPM)
            latents = (latents - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
            
            if i < num_steps - 1:
                next_alpha = alphas[i + 1]
                noise = torch.randn_like(latents)
                latents = latents + (1 - next_alpha).sqrt() * noise
        
        # Decode
        image = vae.decode(latents)
        
        # FIXED: Proper image saving
        image = image[0].cpu()  # (3, H, W)
        
        # Save using fixed function
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image_fixed(image, save_path)
        print(f"✓ Sample saved: {save_path}")
    
    unet.train()

# ==================== OPTIMIZED TRAINING ====================
def train_epoch_optimized(models, dataloader, optimizer, scaler, epoch, global_step):
    """Optimized training for 32GB GPU"""
    unet = models["unet"]
    noise_scheduler = models["noise_scheduler"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    
    unet.train()
    total_loss = 0
    num_batches = 0
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move to device
            images = batch["pixel_values"].to(config.device, non_blocking=True)
            input_ids = batch["input_ids"].to(config.device, non_blocking=True)
            
            # Get text embeddings
            with torch.no_grad():
                text_embeddings = text_encoder(input_ids)[0]
            
            # Encode images
            with torch.no_grad():
                latents = vae.encode(images)
            
            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=config.device
            ).long()
            
            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise with mixed precision
            with autocast(enabled=config.mixed_precision):
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # More memory efficient
                global_step += 1
            
            # Update metrics
            total_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1
            
            # Logging
            if global_step % config.log_every == 0:
                current_loss = total_loss / num_batches
                # Memory usage
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    progress_bar.set_postfix({
                        "loss": f"{current_loss:.4f}",
                        "step": global_step,
                        "GPU": f"{mem_alloc:.1f}/{mem_reserved:.1f}GB"
                    })
                else:
                    progress_bar.set_postfix({
                        "loss": f"{current_loss:.4f}",
                        "step": global_step
                    })
            
            # Save checkpoint
            if global_step % config.save_every == 0 and global_step > 0:
                save_checkpoint(models, optimizer, scaler, epoch, global_step, total_loss/num_batches)
            
            # Clear cache occasionally
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n⚠ Out of memory at batch {batch_idx}. Reducing batch size...")
            torch.cuda.empty_cache()
            # Reduce batch size for next batch
            if config.micro_batch > 1:
                config.micro_batch //= 2
                config.gradient_accumulation_steps = config.batch_size // config.micro_batch
                print(f"Reduced micro_batch to {config.micro_batch}")
            continue
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            optimizer.zero_grad(set_to_none=True)
            continue
    
    epoch_loss = total_loss / num_batches if num_batches > 0 else 0
    return epoch_loss, global_step

# ==================== CHECKPOINT ====================
def save_checkpoint(models, optimizer, scaler, epoch, step, loss):
    """Save checkpoint"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'unet_state_dict': models['unet'].state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'config': config.__dict__
    }
    
    path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}_step_{step}.pth")
    torch.save(checkpoint, path)
    
    # Best model
    best_path = os.path.join(config.checkpoint_dir, "model_best.pth")
    if not hasattr(save_checkpoint, 'best_loss') or loss < save_checkpoint.best_loss:
        torch.save(models['unet'].state_dict(), best_path)
        save_checkpoint.best_loss = loss
        print(f"  ✓ New best model (loss: {loss:.4f})")
    
    print(f"  Checkpoint saved: {path}")

# ==================== MAIN OPTIMIZED TRAINING ====================
def main():
    print("\n" + "="*70)
    print("ANIME DIFFUSION TRAINING - OPTIMIZED FOR 32GB GPU")
    print("="*70)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size} (micro: {config.micro_batch})")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Image size: {config.image_size}")
    print(f"U-Net channels: {config.unet_channels}")
    print("="*70 + "\n")
    
    # Memory optimization
    optimize_memory()
    
    # Initialize handler
    try:
        handler = DatasetHandler(config.data_source)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    try:
        # Initialize LARGER models
        models = initialize_models()
        
        # Create dataset (use first 100K for speed)
        print("\nCreating dataset...")
        dataset = AnimeDataset(handler, config.image_size, is_train=True)
        
        # Use subset if too large
        if len(dataset) > 100000:
            from torch.utils.data import Subset
            indices = list(range(min(100000, len(dataset))))
            dataset = Subset(dataset, indices)
            print(f"Using subset of {len(indices)} images for faster training")
        
        # Split
        from torch.utils.data import random_split
        val_size = int(config.val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, _ = random_split(dataset, [train_size, val_size])
        
        print(f"\nDataset Info:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Training: {train_size:,}")
        print(f"  Validation: {val_size:,}")
        
        # OPTIMIZED DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.micro_batch,
            shuffle=True,
            num_workers=4,  # Increased for faster loading
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,  # Prefetch batches
            persistent_workers=True  # Keep workers alive
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            models["unet"].parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay
        )
        
        # Scaler for mixed precision
        scaler = GradScaler(enabled=config.mixed_precision)
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
        
        # Training loop
        print("\n" + "="*50)
        print("STARTING OPTIMIZED TRAINING")
        print("="*50)
        
        global_step = 0
        
        for epoch in range(config.epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{config.epochs}")
            print(f"{'='*70}")
            
            # Train with optimization
            train_loss, global_step = train_epoch_optimized(
                models, train_loader, optimizer, scaler, epoch, global_step
            )
            
            print(f"\n✓ Epoch {epoch+1} completed")
            print(f"  Average loss: {train_loss:.4f}")
            print(f"  Global step: {global_step}")
            
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                print(f"  GPU memory used: {mem_alloc:.1f} GB")
            
            # Generate sample (FIXED)
            sample_path = os.path.join(config.sample_dir, f"epoch_{epoch+1:03d}.png")
            generate_sample_fixed(models, config.sample_prompt, epoch, sample_path)
            
            # Save checkpoint
            save_checkpoint(models, optimizer, scaler, epoch, global_step, train_loss)
            
            # Clear cache between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Save final model
        final_path = os.path.join(config.checkpoint_dir, "model_final.pth")
        torch.save(models["unet"].state_dict(), final_path)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Final model: {final_path}")
        print(f"Best model: {os.path.join(config.checkpoint_dir, 'model_best.pth')}")
        print(f"Total steps: {global_step}")
        print(f"Total epochs: {config.epochs}")
        
        # Cleanup
        handler.cleanup()
        
        print("\n🎉 Training finished successfully!")
        print("\nTo generate images:")
        print(f"  python generate_optimized.py --prompt 'anime character' --checkpoint {final_path}")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        handler.cleanup()

if __name__ == "__main__":
    # Environment optimizations
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Memory optimization
    
    # Run training
    main()
