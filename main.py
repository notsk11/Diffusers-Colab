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
    model_name_gr = gr.Dropdown(choices=modelnames, container=False, show_label=False, elem_classes='ckpt-box')
    load_pipeline_gr = gr.Button("Load", elem_classes='load')
    with gr.Tab("Txt2Img"):
      with gr.Row(elem_classes='row_a'):
        with gr.Column(elem_classes='colm_a1', variant='default'):
          prompt_gr = gr.Textbox(label="Prompt", lines=3, show_label=False, elem_classes='prompt')
          negative_gr = gr.Textbox(elem_classes='neg-prompt', lines=3, container=True, show_label=False)
        with gr.Column(elem_classes='colm_a2', variant='default'):
          generate_gr = gr.Button("Generate", elem_classes='generate')
      with gr.Row(elem_classes='row_b'):
        with gr.Column(elem_classes='colm_b1', variant='default'):
          scheduler_choice_gr = gr.Dropdown(choices=scheduler_choices, label="Sampling Methods", elem_classes='samp-meth')
          hiresfix = gr.Checkbox(label="Hires. fix", elem_classes='hiresfix')       
          height = gr.Slider(elem_classes='height', minimum=400, maximum=1600, step=8, value=704, label="Height")
          width = gr.Slider(elem_classes='width', minimum=400, maximum=1600, step=8, value=408, label="Width")
          guidance_scale_gr = gr.Slider(elem_classes='guidance', minimum=1, maximum=10, step=0.1, value=5, label="CFG Scale")
          seed = gr.Textbox(elem_classes='inp-seed', label="Seed (Leave empty for random seed)")
          output_image_gr = gr.Gallery(allow_preview=True, preview=True, elem_classes='img-output')
        with gr.Column(elem_classes='colm_b2', variant='default'):
          steps_gr = gr.Slider(elem_classes='steps', minimum=1, maximum=100, step=5, value=10, label="Sampling steps")
          num_images_per_prompt = gr.Slider(elem_classes='b-count', minimum=1, maximum=10, step=1, value=1, label="Batch count")
          b_size = gr.Slider(elem_classes='b-size', minimum=1, maximum=10, step=1, value=1, label="Batch size")
          output_seed = gr.Textbox(elem_classes='out-seed', label="Current Seed", show_label=False)
    generate_gr.click(fn=generator.txt2img, inputs=[prompt_gr, negative_gr, height, width, steps_gr, guidance_scale_gr, num_images_per_prompt, seed, model_name_gr, scheduler_choice_gr], outputs=[output_image_gr, output_seed])
    load_pipeline_gr.click(on_button_click)

demo.launch(share=True, debug=True)
