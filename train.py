import os
import zipfile
import io
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import warnings
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, PretrainedConfig

# Suppress warnings
warnings.filterwarnings("ignore")

class SimpleAnimeDataset(Dataset):
    """Simple dataset for anime fine-tuning"""
    
    def __init__(self, zip_path, tokenizer, image_size=512, max_samples=None):
        self.zip_path = zip_path
        self.image_size = image_size
        self.tokenizer = tokenizer
        
        print(f"📦 Loading anime dataset from {zip_path}")
        
        # Open ZIP
        self.zip_file = zipfile.ZipFile(zip_path, 'r')
        all_files = self.zip_file.namelist()
        
        # Find image files
        self.image_files = []
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        for file in all_files:
            if any(file.lower().endswith(ext) for ext in image_exts):
                self.image_files.append(file)
        
        print(f"Found {len(self.image_files)} anime image files")
        
        # Get captions
        self.captions = {}
        for img_file in self.image_files[:max_samples] if max_samples else self.image_files:
            img_name = os.path.basename(img_file)
            base_name = os.path.splitext(img_name)[0]
            caption_file = f"{base_name}.txt"
            
            if caption_file in all_files:
                try:
                    with self.zip_file.open(caption_file) as f:
                        caption = f.read().decode('utf-8', errors='ignore').strip()
                except:
                    caption = f"anime style {base_name}"
            else:
                caption = f"anime style {base_name}"
            
            self.captions[img_file] = caption
        
        self.valid_files = list(self.captions.keys())[:max_samples] if max_samples else list(self.captions.keys())
        print(f"✅ Loaded {len(self.valid_files)} anime image-caption pairs")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        img_file = self.valid_files[idx]
        
        try:
            # Load image
            with self.zip_file.open(img_file) as f:
                img_data = f.read()
            
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            image = self.transform(image)
            
            # Get caption
            caption = self.captions[img_file]
            
            # Tokenize
            text_inputs = self.tokenizer(
                caption,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = text_inputs.input_ids[0]
            
            return {
                'pixel_values': image,
                'input_ids': input_ids,
                'caption': caption
            }
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def close(self):
        self.zip_file.close()

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def main():
    parser = argparse.ArgumentParser(description="Simple Anime Stable Diffusion Fine-tuning")
    
    parser.add_argument("--zip_path", type=str, default="training_dataset.zip")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="./anime_model")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--max_samples", type=int, default=8000)
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🔧 Loading model: {args.model_name}")
    
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_name,
        subfolder="tokenizer",
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_name,
        subfolder="text_encoder",
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.model_name,
        subfolder="vae",
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.model_name,
        subfolder="unet",
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name,
        subfolder="scheduler",
    )
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Enable gradient checkpointing to save memory
    unet.enable_gradient_checkpointing()
    
    print(f"✅ Models loaded successfully")
    print(f"   Trainable parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    
    # Load dataset
    print("📦 Loading anime dataset...")
    train_dataset = SimpleAnimeDataset(
        zip_path=args.zip_path,
        tokenizer=tokenizer,
        image_size=args.resolution,
        max_samples=args.max_samples
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    
    # Learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Training
    print(f"\n🔥 Starting anime fine-tuning")
    print(f"   Dataset: {len(train_dataset)} anime images")
    print(f"   Batch size: {args.train_batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"   Total steps: {args.max_train_steps}")
    print(f"   Learning rate: {args.learning_rate}")
    print()
    
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), desc="Training")
    
    for epoch in range(100):
        unet.train()
        
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Convert images to latents
                latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
                latents = latents * 0.18215
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                
                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Check if gradient accumulation is done
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        
                        # Also save the unet
                        unet_to_save = accelerator.unwrap_model(unet)
                        torch.save(unet_to_save.state_dict(), os.path.join(save_path, "unet.pth"))
                        print(f"💾 Saved checkpoint to {save_path}")
            
            # Break if max steps reached
            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break
    
    # Save final model
    print("💾 Saving final model...")
    
    if accelerator.is_main_process:
        # Save the pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_name,
            unet=accelerator.unwrap_model(unet),
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            scheduler=noise_scheduler,
            safety_checker=None,
        )
        
        pipeline.save_pretrained(os.path.join(args.output_dir, "anime_pipeline"))
        
        # Save just the unet weights
        torch.save(
            accelerator.unwrap_model(unet).state_dict(),
            os.path.join(args.output_dir, "unet_weights.pth")
        )
        
        print(f"✅ Model saved to {args.output_dir}")
        print(f"   - Full pipeline: {args.output_dir}/anime_pipeline/")
        print(f"   - UNet weights: {args.output_dir}/unet_weights.pth")
    
    train_dataset.close()
    print("🎉 Training complete!")

if __name__ == "__main__":
    main()