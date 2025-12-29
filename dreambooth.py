
# 1. Create the virtual environment named 'testing_env'
python3 -m venv testing_env

# 2. Activate the environment
source testing_env/bin/activate

# 3. Upgrade pip (good practice)
pip install --upgrade pip

# 4. Install PyTorch (with CUDA support) and Diffusers
# Note: This installs the standard CUDA version. If you are on a specific server, check your cuda version.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors scipy omegaconf

import torch
from diffusers import StableDiffusionPipeline
import os
import random

# --- CONFIGURATION ---
# Path to your .safetensors file
model_path = "/workspace/kohya_ss/outputs/final.safetensors"

# Directory to save images
output_dir = "generated_images_15epch"
os.makedirs(output_dir, exist_ok=True)

# --- LOAD MODEL ---
print(f"Loading model from {model_path}...")

try:
    # We use from_single_file because it is a standalone .safetensors file
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.to("cuda")
    # Enable memory efficient attention if you have low VRAM
    # pipe.enable_xformers_memory_efficient_attention() 
    print("Model loaded successfully!")
    
except Exception as e:
    print("\nERROR: Could not load the model.")
    print("If your file size is small (~144MB), you trained a LoRA, not a full model.")
    print("This script is for Full Dreambooth models (2GB+).")
    print(f"Error details: {e}")
    exit()

# --- PROMPT LIST ---
# We mix your trigger word into various scenarios
base_prompts = [
    "sfa a girl",
    "sfa a boy in a suit",
    "sfa a cat",
    "sfa a dog",
    "sfa a car in front of a building",
    "sfa a man standing on a street",
    "sfa a woman smiling",
    "sfa a child playing",
    "sfa a bicycle on the road",
    "sfa a house with trees",
    "sfa a person sitting on a bench",
    "sfa a couple walking together",
    "sfa a bird flying in the sky",
    "sfa a flower in a garden",
    "sfa a laptop on a desk",
    "sfa a cup of coffee",
    "sfa a city street at night",
    "sfa a mountain landscape",
    "sfa a beach with waves",
    "sfa a train at a station"
]



# --- GENERATION LOOP ---
print(f"\nStarting generation of {len(base_prompts)} images...")

for prompt in base_prompts:
    print(f"Generating: {prompt}")
    
    # Generate the image
    # guidance_scale=7.5 is standard. num_inference_steps=30 is a good balance.
    image = pipe(
        prompt, 
        num_inference_steps=30, 
        guidance_scale=7.5
    ).images[0]
    
    # Create a safe filename from the prompt
    # Remove spaces and special chars, limit length
    safe_name = "".join([c if c.isalnum() else "_" for c in prompt])
    safe_name = safe_name[:50] # Limit to 50 chars
    
    save_path = os.path.join(output_dir, f"{safe_name}.png")
    image.save(save_path)
    print(f"Saved: {save_path}")

print(f"\nDone! Check the '{output_dir}' folder.")
