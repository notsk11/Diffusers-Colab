# /content/main.py
import functions
from functions.text2image import TextToImageGenerator
from functions.style import css
from IPython.display import display
import gradio as gr
import torch
import random
import sys
from PIL import Image
import numpy as np

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

# Instantiate the TextToImageGenerator with a default model_name
generator = TextToImageGenerator()

def load_pipeline(model_name):
    generator.model_loader.model_name = model_name
    generator.model_loader.pipe = None

# Create a function to be triggered when the button is clicked
def on_button_click():
    model_name = model_name_gr.value
    load_pipeline(model_name)


scheduler_choices = [
    ("Euler", "EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=False)"),
    ("Euler a", "EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=False)"),
    ("LMS", "LMSDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=False)"),
    ("LMS Karras", "LMSDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)"),
    ("DPM++ SDE", "DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=False)"),
    ("DPM++ SDE Karras", "DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)"),
    ("DPM2 Karras", "KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)"),
    ("DPM++ 2M", "DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=False)"),
    ("DPM++ 2M Karras", "DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)"),
    ("DPM++ 2M SDE", "DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=False, algorithm_type='sde-dpmsolver++')"),
    ("DPM++ 2M SDE Karras", "DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type='sde-dpmsolver++')"),
]


with gr.Blocks(css=functions.style.css) as demo:
    gr.Markdown("Stable Diffusion")
    model_name_gr = gr.Dropdown(container=False, show_label=False, choices=modelnames, elem_classes='ckpt-box')
    load_pipeline_gr = gr.Button("Load Model", elem_classes='load')
    with gr.Tab("Txt2Img"):
        prompt_gr = gr.Textbox(label="Prompt", lines=4, show_label=False, elem_classes='prompt')
        negative_gr = gr.Textbox(elem_classes='neg-prompt', lines=3, container=True, show_label=False)
        height = gr.Dropdown(elem_classes='height', choices=number_choicess, value=704, label="Height")
        width = gr.Dropdown(elem_classes='width', choices=number_choices, value=408, label="Width")
        steps_gr = gr.Slider(elem_classes='steps', minimum=1, maximum=100, step=1, value=10, label="Sampling steps")
        guidance_scale_gr = gr.Slider(elem_classes='guidance', minimum=1, maximum=10, step=0.1, value=5, label="CFG Scale")
        output_image_gr = gr.Gallery(preview=True, elem_classes='img-output')
        seed = gr.Textbox(elem_classes='inp-seed', label="Seed (Leave empty for random seed)")
        output_seed = gr.Textbox(elem_classes='out-seed', label="Current Seed", show_label=False)
        scheduler_choice_gr = gr.Dropdown(label="Sampling Methods", elem_classes='samp-meth', choices=scheduler_choices)
        generate_gr = gr.Button("Generate", elem_classes='generate')
        generate_gr.click(fn=generator.txt2img, inputs=[prompt_gr, negative_gr, height, width, steps_gr, guidance_scale_gr, seed, model_name_gr, scheduler_choice_gr], outputs=[output_image_gr, output_seed])
        load_pipeline_gr.click(on_button_click)

demo.launch(share=True, debug=True)
