# /content/functions/text2image.py
# text2image.py
import torch
import random
import sys
from PIL import Image
import numpy as np
from functions.load import CustomDiffusionLoader

class TextToImageGenerator:
    def __init__(self, model_name="digiplay/Realisian_v5"):
        self.model_loader = CustomDiffusionLoader(model_name=model_name)

    def txt2img(self, prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, seed, model_name):
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
      num_inference_steps = int(num_inference_steps)
      
      image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
                  num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                  generator=generator).images[0]

      # Convert image to NumPy array
      image_np = np.array(image)

      # Convert NumPy array to PIL image
      image_pil = Image.fromarray(image_np)

      return [image_pil], seed
