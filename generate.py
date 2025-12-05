import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler
import argparse
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Generate anime images with fine-tuned model")
    
    parser.add_argument("--model_path", type=str, default="./anime_model/anime_pipeline", help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model if loading weights separately")
    parser.add_argument("--prompt", type=str, default="anime artwork of a beautiful girl", help="Generation prompt")
    parser.add_argument("--negative_prompt", type=str, default="blurry, bad quality, realistic, photo", help="Negative prompt")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Output directory")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--no_half", action="store_true", help="Don't use half precision (use if getting NaN errors)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🔧 Loading model...")
    
    # Determine torch dtype
    if torch.cuda.is_available() and not args.no_half:
        torch_dtype = torch.float16
        print("✅ Using half precision (fp16)")
    else:
        torch_dtype = torch.float32
        print("⚠️  Using full precision (fp32)")
    
    try:
        # Try to load the full pipeline
        if os.path.exists(args.model_path):
            print(f"Loading from pipeline directory: {args.model_path}")
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.model_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
        else:
            # Try to load base model with custom unet
            print(f"Loading base model: {args.base_model}")
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.base_model,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Try to load fine-tuned unet weights
            unet_paths = [
                os.path.join("./anime_model", "unet_weights.pth"),
                os.path.join(args.model_path, "unet.pth"),
                args.model_path + ".pth"
            ]
            
            unet_loaded = False
            for unet_path in unet_paths:
                if os.path.exists(unet_path):
                    print(f"Loading fine-tuned UNet from: {unet_path}")
                    unet_weights = torch.load(unet_path, map_location="cpu")
                    pipeline.unet.load_state_dict(unet_weights)
                    unet_loaded = True
                    break
            
            if not unet_loaded:
                print("⚠️  No fine-tuned UNet found, using base model")
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Trying alternative loading method...")
        
        # Alternative: Load with low_cpu_mem_usage
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True
        )
    
    # Use a compatible scheduler
    try:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    except:
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
        
        # Disable xformers to avoid compatibility issues
        try:
            pipeline.disable_xformers_memory_efficient_attention()
            print("✅ Disabled xformers (using standard attention)")
        except:
            pass
        
        # Enable memory efficient attention if available
        try:
            pipeline.enable_attention_slicing()
            print("✅ Enabled attention slicing for memory efficiency")
        except:
            pass
    
    print(f"\n🎨 Generating anime images...")
    print(f"   Prompt: {args.prompt}")
    print(f"   Negative: {args.negative_prompt}")
    print(f"   Steps: {args.steps}")
    print(f"   Guidance: {args.guidance}")
    print(f"   Seed: {args.seed}")
    print(f"   Size: {args.height}x{args.width}")
    print()
    
    # Test generation first
    print("🧪 Testing generation with a simple prompt...")
    try:
        test_image = pipeline(
            prompt="anime",
            num_inference_steps=5,
            guidance_scale=1.0,
            height=64,
            width=64,
            generator=torch.Generator(device=pipeline.device).manual_seed(42)
        ).images[0]
        print("✅ Test generation successful!")
    except Exception as e:
        print(f"⚠️  Test generation warning: {e}")
        print("Continuing anyway...")
    
    # Generate images
    successful = 0
    for i in range(args.num_images):
        try:
            current_seed = args.seed + i if args.seed else None
            generator = torch.Generator(device=pipeline.device).manual_seed(current_seed) if current_seed else None
            
            print(f"   Generating image {i+1}/{args.num_images} (seed: {current_seed})...")
            
            image = pipeline(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
                height=args.height,
                width=args.width
            ).images[0]
            
            # Save image
            output_path = os.path.join(args.output_dir, f"anime_{i+1:03d}.png")
            image.save(output_path)
            print(f"✅ Saved: {output_path}")
            successful += 1
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ Out of memory for image {i+1}, reducing size...")
            try:
                # Try smaller size
                image = pipeline(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                    height=384,
                    width=384
                ).images[0]
                output_path = os.path.join(args.output_dir, f"anime_{i+1:03d}_small.png")
                image.save(output_path)
                print(f"✅ Saved smaller version: {output_path}")
                successful += 1
            except Exception as e:
                print(f"❌ Failed to generate image {i+1}: {e}")
                
        except Exception as e:
            print(f"❌ Error generating image {i+1}: {e}")
    
    print(f"\n🎉 Generated {successful}/{args.num_images} anime images in {args.output_dir}")
    
    # Save generation settings
    settings_path = os.path.join(args.output_dir, "generation_settings.txt")
    with open(settings_path, "w") as f:
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Negative Prompt: {args.negative_prompt}\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Guidance Scale: {args.guidance}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Size: {args.height}x{args.width}\n")
        f.write(f"Model: {args.model_path}\n")
    
    print(f"📝 Settings saved to {settings_path}")

if __name__ == "__main__":
    main()