# /content/functions/text2image.py
import torch
import random
import sys
from PIL import Image
import numpy as np
from functions.load import CustomDiffusionLoader
from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler
)


class TextToImageGenerator:
    def __init__(self, model_name="digiplay/Realisian_v5"):
        self.model_loader = CustomDiffusionLoader(model_name=model_name)

    def txt2img(self, prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, num_images_per_prompt, seed, model_name, scheduler):
        # Seed randomizer
        if seed == "":
            seed = random.randint(0, sys.maxsize)
        else:
            try:
                seed = int(seed)
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                seed = None

        generator = torch.Generator("cuda").manual_seed(seed)

        # Load or get the pipeline
        self.model_loader.model_name = model_name
        pipe = self.model_loader.load_pipeline()
        pipe.safety_checker = None

        # Set the scheduler
        pipe.scheduler = eval(scheduler)

        num_inference_steps = int(num_inference_steps)
        pipe.enable_vae_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
                     num_inference_steps=num_inference_steps, 
                     guidance_scale=guidance_scale,
                     num_images_per_prompt=num_images_per_prompt,
                     generator=generator).images

        # Convert image to NumPy array
        images_np = [np.array(img) for img in image]
        images_pil = [Image.fromarray(img_np) for img_np in images_np]

        return images_pil, seed
