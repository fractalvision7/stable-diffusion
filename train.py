"""
Complete Diffusion Model Training for Anime
Optimized for 32GB GPU with 150K images
Dimension-safe architecture with automatic sample generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, AutoencoderKL
from PIL import Image
import os
import gc
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ==================== CONFIGURATION ====================
class Config:
    # ========== Paths ==========
    data_root = "./training_dataset"  # Main folder
    images_dir = "images"             # Images subfolder
    # Captions are in data_root folder with same names as images
    
    # ========== Model Architecture ==========
    # Optimized for 32GB GPU with dimension safety
    image_size = 512                  # Final image size
    latent_size = image_size // 8     # U-Net input size (64x64)
    
    # U-Net dimensions (carefully tuned for memory)
    unet_channels = 128               # Base channels
    channel_multipliers = [1, 2, 4, 8]  # Progressive downsampling
    num_res_blocks = 2                # Residual blocks per level
    attention_resolutions = [16, 8]   # Apply attention at these resolutions
    num_heads = 8                     # Attention heads
    dropout = 0.0                     # No dropout for anime clarity
    
    # Text encoder
    text_encoder_dim = 768            # CLIP text embedding dimension
    max_text_length = 77              # CLIP token limit
    
    # ========== Training Parameters ==========
    batch_size = 8                    # Fits in 32GB
    micro_batch = 2                   # For gradient accumulation
    gradient_accumulation_steps = batch_size // micro_batch
    
    learning_rate = 1e-4
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    weight_decay = 1e-2
    
    epochs = 30                       # Good starting point
    warmup_steps = 500                # LR warmup
    
    # ========== Diffusion Process ==========
    timesteps = 1000
    beta_schedule = "linear"          # Stable choice
    
    # ========== System ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = True            # Use FP16 for memory
    seed = 42
    
    # ========== Checkpoints & Logging ==========
    checkpoint_dir = "./checkpoints"
    sample_dir = "./training_samples"
    save_every = 2000                 # Save checkpoint every N steps
    log_every = 100                   # Log loss every N steps
    
    # ========== Validation ==========
    val_split = 0.02                  # 2% for validation
    sample_prompt = "anime artwork, a person"  # Prompt for epoch samples

config = Config()

# Set all seeds for reproducibility
torch.manual_seed(config.seed)
np.random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== DIMENSION-SAFE DATASET ====================
class AnimeDataset(Dataset):
    """Dataset with dimension validation and error handling"""
    
    def __init__(self, data_root, image_size=512, is_train=True):
        self.data_root = data_root
        self.image_size = image_size
        self.is_train = is_train
        
        # Collect valid image-caption pairs
        self.image_paths = []
        self.caption_paths = []
        
        images_folder = os.path.join(data_root, config.images_dir)
        
        # Validate folder structure
        if not os.path.exists(images_folder):
            raise ValueError(f"Images folder not found: {images_folder}")
        
        print(f"Scanning dataset at {data_root}...")
        
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_files = [f for f in os.listdir(images_folder) 
                      if any(f.lower().endswith(ext) for ext in valid_extensions)]
        
        print(f"Found {len(image_files)} image files")
        
        # Match with captions
        for img_file in tqdm(image_files, desc="Matching captions"):
            img_path = os.path.join(images_folder, img_file)
            
            # Try multiple possible caption file names
            base_name = os.path.splitext(img_file)[0]
            caption_candidates = [
                os.path.join(data_root, f"{base_name}.txt"),
                os.path.join(data_root, f"{base_name}.jpg.txt"),
                os.path.join(data_root, f"{base_name}.png.txt"),
            ]
            
            caption_path = None
            for candidate in caption_candidates:
                if os.path.exists(candidate):
                    caption_path = candidate
                    break
            
            if caption_path:
                self.image_paths.append(img_path)
                self.caption_paths.append(caption_path)
            else:
                print(f"Warning: No caption found for {img_file}")
        
        print(f"Loaded {len(self.image_paths)} valid image-caption pairs")
        
        if len(self.image_paths) == 0:
            raise ValueError("No valid image-caption pairs found!")
        
        # Initialize tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="tokenizer"
        )
        
        # Image transforms with dimension safety
        self.transform = self._get_transforms()
        
        # Validate first few samples
        self._validate_samples()
    
    def _get_transforms(self):
        """Get image transforms with dimension checking"""
        if self.is_train:
            return transforms.Compose([
                transforms.Resize(config.image_size + 32),
                transforms.RandomCrop(config.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # To [-1, 1]
            ])
        else:
            return transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
    
    def _validate_samples(self):
        """Validate first few samples for dimension consistency"""
        print("Validating dataset dimensions...")
        valid_samples = 0
        
        for i in range(min(10, len(self.image_paths))):
            try:
                img = Image.open(self.image_paths[i]).convert('RGB')
                original_size = img.size
                
                # Test transform
                img_tensor = self.transform(img)
                
                # Check dimensions
                assert img_tensor.shape == (3, config.image_size, config.image_size), \
                    f"Sample {i}: Wrong tensor shape {img_tensor.shape}"
                
                # Check caption
                with open(self.caption_paths[i], 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                tokens = self.tokenizer(
                    caption,
                    padding="max_length",
                    max_length=config.max_text_length,
                    truncation=True,
                    return_tensors="pt"
                )
                assert tokens.input_ids.shape == (1, config.max_text_length), \
                    f"Sample {i}: Wrong token shape {tokens.input_ids.shape}"
                
                valid_samples += 1
                if i == 0:
                    print(f"  Sample 0: Image {original_size} -> {img_tensor.shape}, "
                          f"Caption: {caption[:50]}...")
                    
            except Exception as e:
                print(f"  Sample {i} validation failed: {e}")
        
        print(f"Validation: {valid_samples}/10 samples passed")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get item with robust error handling"""
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Load image
                img_path = self.image_paths[idx]
                img = Image.open(img_path).convert('RGB')
                
                # Apply transform
                pixel_values = self.transform(img)
                
                # Load and process caption
                with open(self.caption_paths[idx], 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                
                # Enhance simple captions for anime
                if not caption.startswith("anime"):
                    caption = f"anime style, {caption}"
                
                # Tokenize
                tokens = self.tokenizer(
                    caption,
                    padding="max_length",
                    max_length=config.max_text_length,
                    truncation=True,
                    return_tensors="pt"
                )
                
                return {
                    "pixel_values": pixel_values,
                    "input_ids": tokens.input_ids.squeeze(0),
                    "caption": caption
                }
                
            except Exception as e:
                if retry == max_retries - 1:
                    print(f"Failed to load sample {idx} after {max_retries} retries: {e}")
                    # Return a fallback sample
                    return self._get_fallback_sample()
                
                # Try next sample on error
                idx = (idx + 1) % len(self)
                continue
    
    def _get_fallback_sample(self):
        """Create a fallback sample if loading fails"""
        # Create black image
        pixel_values = torch.zeros(3, config.image_size, config.image_size)
        
        # Create simple caption
        caption = "anime artwork"
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=config.max_text_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
            "caption": caption
        }

# ==================== MODEL INITIALIZATION ====================
def initialize_models():
    """Initialize all models with dimension validation"""
    print("\n" + "="*50)
    print("INITIALIZING MODELS")
    print("="*50)
    
    # Load pretrained models from Stable Diffusion
    print("Loading pretrained components...")
    
    # 1. Text encoder (frozen)
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="text_encoder"
    ).to(config.device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    print(f"  Text encoder: {sum(p.numel() for p in text_encoder.parameters()):,} params")
    
    # 2. VAE for latent space (frozen)
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae"
    ).to(config.device)
    vae.requires_grad_(False)
    vae.eval()
    print(f"  VAE: {sum(p.numel() for p in vae.parameters()):,} params")
    
    # 3. Create U-Net from scratch with validated dimensions
    print(f"\nCreating U-Net with:")
    print(f"  Input shape: (batch, 4, {config.latent_size}, {config.latent_size})")
    print(f"  Text embedding dim: {config.text_encoder_dim}")
    print(f"  Channels: {config.unet_channels}")
    print(f"  Channel multipliers: {config.channel_multipliers}")
    
    unet = UNet2DConditionModel(
        sample_size=config.latent_size,  # 64 for 512px images
        in_channels=4,
        out_channels=4,
        layers_per_block=config.num_res_blocks,
        block_out_channels=[config.unet_channels * m for m in config.channel_multipliers],
        down_block_types=(
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
        ),
        cross_attention_dim=config.text_encoder_dim,
    ).to(config.device)
    
    print(f"  U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # 4. Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.timesteps,
        beta_schedule=config.beta_schedule,
        prediction_type="epsilon"
    )
    
    # 5. Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer"
    )
    
    return {
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
        "tokenizer": tokenizer
    }

# ==================== TRAINING UTILITIES ====================
def encode_images(vae, images):
    """Encode images to latents with dimension validation"""
    # images: (B, 3, H, W) in [-1, 1]
    with torch.no_grad():
        # Convert to [0, 1] for VAE
        images = (images + 1) / 2
        
        # Validate input dimensions
        assert images.min() >= 0 and images.max() <= 1, "Images not in [0, 1] range"
        assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
        
        # Encode
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # Scaling factor from Stable Diffusion
        
        # Validate output dimensions
        expected_shape = (images.shape[0], 4, images.shape[2] // 8, images.shape[3] // 8)
        assert latents.shape == expected_shape, \
            f"Latent shape mismatch: {latents.shape} vs {expected_shape}"
    
    return latents

def decode_latents(vae, latents):
    """Decode latents to images"""
    with torch.no_grad():
        latents = latents / 0.18215
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)  # [0, 1]
    return images

def validate_batch_dimensions(batch, models):
    """Validate all dimensions in a training batch"""
    images = batch["pixel_values"].to(config.device)
    input_ids = batch["input_ids"].to(config.device)
    
    # Validate image dimensions
    assert images.shape[1:] == (3, config.image_size, config.image_size), \
        f"Image shape mismatch: {images.shape[1:]}"
    
    # Validate text input dimensions
    assert input_ids.shape == (images.shape[0], config.max_text_length), \
        f"Input IDs shape mismatch: {input_ids.shape}"
    
    # Encode to latents and validate
    latents = encode_images(models["vae"], images)
    
    # Validate text embeddings
    with torch.no_grad():
        text_embeddings = models["text_encoder"](input_ids)[0]
        assert text_embeddings.shape == (images.shape[0], config.max_text_length, config.text_encoder_dim), \
            f"Text embeddings shape mismatch: {text_embeddings.shape}"
    
    return images, latents, text_embeddings

# ==================== TRAINING LOOP ====================
def train_epoch(models, dataloader, optimizer, scaler, epoch, global_step):
    """Train for one epoch"""
    unet = models["unet"]
    noise_scheduler = models["noise_scheduler"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    
    unet.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Validate dimensions
            images, latents, text_embeddings = validate_batch_dimensions(batch, models)
            
            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=config.device
            ).long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise with mixed precision
            with autocast(enabled=config.mixed_precision):
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update global step
                global_step += 1
            
            # Update metrics
            total_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1
            
            # Logging
            if global_step % config.log_every == 0:
                current_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "step": global_step,
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
            
            # Save checkpoint
            if global_step % config.save_every == 0 and global_step > 0:
                save_checkpoint(models, optimizer, scaler, epoch, global_step, total_loss/num_batches)
            
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            print("Skipping batch...")
            optimizer.zero_grad()
            continue
    
    epoch_loss = total_loss / num_batches if num_batches > 0 else 0
    return epoch_loss, global_step

# ==================== SAMPLING & CHECKPOINTING ====================
def generate_sample(models, prompt, step, save_path):
    """Generate sample image during training"""
    print(f"\nGenerating sample: '{prompt}'")
    
    unet = models["unet"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    
    unet.eval()
    
    with torch.no_grad():
        # Tokenize prompt
        text_input = tokenizer(
            [prompt],
            padding="max_length",
            max_length=config.max_text_length,
            truncation=True,
            return_tensors="pt"
        ).to(config.device)
        
        # Get text embeddings
        text_embeddings = text_encoder(text_input.input_ids)[0]
        
        # Create noise
        latents = torch.randn(
            (1, 4, config.latent_size, config.latent_size),
            device=config.device
        )
        
        # Use DDIM for faster sampling
        ddim_scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        ddim_scheduler.set_timesteps(30, device=config.device)
        
        # Denoising loop
        for t in ddim_scheduler.timesteps:
            # Expand for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = ddim_scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = unet(
                latent_model_input,
                t.expand(2),
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Apply guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            # Update latents
            latents = ddim_scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode to image
        image = decode_latents(vae, latents)
        image = image[0].cpu()  # (3, H, W) in [0, 1]
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path, image.permute(1, 2, 0).numpy())
        print(f"Saved sample to {save_path}")
    
    unet.train()
    return image

def save_checkpoint(models, optimizer, scaler, epoch, step, loss):
    """Save training checkpoint"""
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
    
    # Also save best model
    best_path = os.path.join(config.checkpoint_dir, "model_best.pth")
    if not os.path.exists(best_path) or loss < getattr(save_checkpoint, 'best_loss', float('inf')):
        torch.save(models['unet'].state_dict(), best_path)
        save_checkpoint.best_loss = loss
        print(f"  New best model saved with loss {loss:.4f}")
    
    print(f"  Checkpoint saved to {path}")

# ==================== MAIN TRAINING FUNCTION ====================
def main():
    print("\n" + "="*50)
    print("ANIME DIFFUSION MODEL TRAINING")
    print("="*50)
    print(f"Device: {config.device}")
    print(f"Image size: {config.image_size}")
    print(f"Batch size: {config.batch_size} (micro batch: {config.micro_batch})")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Latent size: {config.latent_size}")
    print(f"Epochs: {config.epochs}")
    print("="*50 + "\n")
    
    # Initialize models
    models = initialize_models()
    
    # Create dataset
    print("\n" + "="*50)
    print("LOADING DATASET")
    print("="*50)
    
    try:
        dataset = AnimeDataset(config.data_root, config.image_size, is_train=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please check your dataset structure:")
        print(f"  Data root: {config.data_root}")
        print(f"  Images folder: {os.path.join(config.data_root, config.images_dir)}")
        print("  Captions should be in data_root folder with same names as images")
        return
    
    # Split dataset
    from torch.utils.data import random_split
    val_size = int(config.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {train_size:,}")
    print(f"Validation samples: {val_size:,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        models["unet"].parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_dacay=config.weight_decay
    )
    
    # Initialize mixed precision scaler
    scaler = GradScaler(enabled=config.mixed_precision)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{config.epochs}")
        print(f"{'='*60}")
        
        # Train for one epoch
        train_loss, global_step = train_epoch(
            models, train_loader, optimizer, scaler, epoch, global_step
        )
        
        print(f"\nEpoch {epoch+1} completed")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Global Step: {global_step}")
        
        # Generate sample after each epoch
        sample_path = os.path.join(config.sample_dir, f"epoch_{epoch+1:03d}.png")
        generate_sample(models, config.sample_prompt, epoch+1, sample_path)
        
        # Save epoch checkpoint
        save_checkpoint(models, optimizer, scaler, epoch, global_step, train_loss)
        
        # Clear cache to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save final model
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    
    final_path = os.path.join(config.checkpoint_dir, "model_final.pth")
    torch.save(models["unet"].state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {config.epochs}")
    print(f"  Total steps: {global_step}")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Best checkpoint: {os.path.join(config.checkpoint_dir, 'model_best.pth')}")
    print(f"  Training samples: {len(train_loader) * config.micro_batch * config.epochs:,}")
    print(f"\nTo generate images:")
    print(f"  python generate_complete.py --prompt 'your prompt' --checkpoint {final_path}")

if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
