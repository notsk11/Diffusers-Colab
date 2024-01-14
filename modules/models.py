# models.py
from diffusers import DiffusionPipeline, DDIMScheduler
import torch

# Set default values for width and height
number_choices = [1600, 1280, 704, 504, 408]
number_choicess = [1600, 1280, 704, 504, 408]

modelnames = [
    ("digiplay/Realisian_v5", "digiplay/Realisian_v5"),
    ("SG161222/Realistic_Vision_V6.0_B1_noVAE", "SG161222/Realistic_Vision_V6.0_B1_noVAE"),
    ("segmind/SSD-1B", "segmind/SSD-1B"),
    ("digiplay/RealEpicMajicRevolution_v1", "digiplay/RealEpicMajicRevolution_v1"),
    ("imagepipeline/Realities-Edge-XL", "imagepipeline/Realities-Edge-XL"),
    ("stablediffusionapi/realisian111", "stablediffusionapi/realisian111")
]

def loadmodel(modeloptions):
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1
    )
    global pipe
    pipe = DiffusionPipeline.from_pretrained(
        modeloptions,
        torch_dtype=torch.float16,
        requires_safety_checker=None,
        safety_checker=None,
        scheduler=noise_scheduler,
    )
    pipe = pipe.to('cuda')
    pipe.safety_checker = None
    return pipe
