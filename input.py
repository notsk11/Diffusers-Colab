import torch
from IPython.display import clear_output
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline
import random, sys
import mediapy as media
prompt = "scarlett johansson, girl sitting on kitchen bar, thighs separated, high res, 4k, cinematic, black hair, face fix" #@param {type: "string"}
negative_prompt = "bad anatomy, bad eye, bad body parts, bad face" #@param {type: "string"}

from diffusers import StableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

repo_id = "runwayml/stable-diffusion-v1-5" #@param {type: "string"}

# Add a parameter for selecting the scheduler
scheduler_type = "ddpm" #@param ["ddpm", "ddim", "pndm", "lms", "euler_anc", "euler"]
if scheduler_type == "ddpm":
    scheduler = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")
elif scheduler_type == "ddim":
    scheduler = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler")
elif scheduler_type == "pndm":
    scheduler = PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler")
elif scheduler_type == "lms":
    scheduler = LMSDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
elif scheduler_type == "euler_anc":
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
elif scheduler_type == "euler":
    scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
else:
    scheduler = DPMSolverMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

# Provide a placeholder value for pretrained_model_name_or_path

# Use the selected scheduler in the pipeline
pipeline = StableDiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler, repo_id=repo_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipeline.to("cuda")



seed_input = "" #@param {type: "string"}
num_inference_steps = 30 # @param {type:"slider", min:1, max:100, step:1}
guidance_scale = 7 # @param {type:"slider", min:0, max:10, step:1}
pipeline.safety_checker = None
selected_resolution = 400, 700 # @param ["1600, 900","1280, 720","700, 400","500, 200","200, 500","400, 700","720, 1280","900, 1600","512, 512","768, 768","1024, 1024" ] {type:"raw"}
num_images_per_prompt = 1 # @param {type:"slider", min:1, max:10, step:1}

if seed_input == "":
    seed = random.randint(0, sys.maxsize)
else:
    try:
        seed = int(seed_input)
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        seed = None

print("Seed:", seed)
pipeline.enable_vae_tiling()
pipeline.enable_xformers_memory_efficient_attention()

images = pipeline(
    prompt = prompt,
    negative_prompt = negative_prompt,
    guidance_scale = guidance_scale,
    num_inference_steps = num_inference_steps,
    width = selected_resolution[0] - selected_resolution[0] % 8,
    height = selected_resolution[1] - selected_resolution[1] % 8,
    num_images_per_prompt = num_images_per_prompt,
    generator = torch.Generator("cuda").manual_seed(seed),
    ).images

media.show_images(images)
images[0].save("output.jpg")
