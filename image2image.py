import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils import load_image
import os
import sys

# --- CONFIGURATION ---
model_path = "/workspace/kohya_ss/outputs/last.safetensors"
input_image_path = "image.png"
output_dir = "anime_controlnet_conversions"
trigger_word = "ble " 

# ControlNet Parameters
# How strictly to follow the lines (0.0 = ignore, 1.0 = strict)
controlnet_scale = 0.6 

os.makedirs(output_dir, exist_ok=True)

# --- CHECK INPUT ---
if not os.path.exists(input_image_path):
    print(f"ERROR: File '{input_image_path}' not found.")
    sys.exit(1)

# --- LOAD MODELS ---
print("1. Loading ControlNet (Canny Edge)...")
try:
    # We download the standard SDXL Canny ControlNet
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
except Exception as e:
    print(f"Error loading ControlNet. Check internet connection.\n{e}")
    sys.exit(1)

print(f"2. Loading Fine-tuned Model from {model_path}...")
try:
    # We wrap your model with the ControlNet we just loaded
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to("cuda")
    print("Pipeline ready!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- PREPARE IMAGES ---
print("Preparing images...")
# 1. Load original image
original_image = load_image(input_image_path).convert("RGB")
original_image = original_image.resize((1024, 1024))

# 2. Create Canny Edge Map (The "Skeleton")
# We use OpenCV to detect edges
image_array = np.array(original_image)
# Detect edges (100, 200 are standard thresholds)
canny_edges = cv2.Canny(image_array, 100, 200)
canny_edges = canny_edges[:, :, None]
canny_edges = np.concatenate([canny_edges, canny_edges, canny_edges], axis=2)
control_image = Image.fromarray(canny_edges)

# Save the edge map so you can see what the AI "sees"
control_image.save(os.path.join(output_dir, "debug_edges.png"))
print(f"Debug: Edge map saved to {output_dir}/debug_edges.png")

# --- PROMPT ---
prompt = f"{trigger_word}, highly detailed, vibrant colors, 2d shading"
negative_prompt = "rendering, bad anatomy, blurry, photograph"

# --- GENERATION LOOP ---
strengths = [0.9,1.0] 

print(f"\nStarting ControlNet Generation...")

for strength in strengths:
    print(f"Generating with Img2Img Strength {strength}...")
    
    # We pass BOTH the original image and the control image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=original_image,      # Reference for colors/general shape
        control_image=control_image, # Reference for strict lines
        strength=strength,         # How much to change the original pixels
        controlnet_conditioning_scale=controlnet_scale, # How strictly to follow edges
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    save_path = os.path.join(output_dir, f"anime_cn_str{strength}.png")
    image.save(save_path)
    print(f"Saved: {save_path}")

print(f"\nDone! Check '{output_dir}'.")
