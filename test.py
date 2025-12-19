import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# Model ID for Dreamlike Anime
model_id = "dreamlike-art/dreamlike-anime-1.0"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
).to("cuda")

# Safety optimization
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe.enable_model_cpu_offload()

# Prepare image (Dreamlike prefers 768px for best detail)
init_image = Image.open("image.png").convert("RGB").resize((768, 768))

prompt = "anime, masterpiece, high quality, digital art, stylized, vibrant colors"
negative_prompt = "photorealistic, 3d render, low quality, bad anatomy"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=0.25,  # Lower strength = more like the original photo
    guidance_scale=8.5
).images[0]

image.save("dreamlike_result.png")
