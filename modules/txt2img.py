# txt2img.py
import torch
import random
import sys
from PIL import Image
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from modules import models
# Set an initial value for the pipe variable
pipe = models.loadmodel("digiplay/Realisian_v5")
def txt2img(prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, seed):
    global pipe  # Use the global variable pipe
    # Seed randomizer
    if seed == "":
        seed = random.randint(0, sys.maxsize)
    else:
        try:
            seed = int(seed)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            seed = None

    # Set seed for torch generator
    generator = torch.Generator("cuda").manual_seed(seed)

    # Assuming your_model_or_function is the function responsible for text-to-image generation
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, height=height, width=width, guidance_scale=10, generator=generator).images[0]

    # Convert image to NumPy array
    image_np = np.array(image)

    # Convert NumPy array to PIL image
    image_pil = Image.fromarray(image_np)

    return [image_pil], seed
