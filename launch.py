# Import the modules module
import modules
from modules import txt2img
from modules import models
from modules import pipelines
import gradio as gr
from gradio.themes.base import Base
import time
with gr.Blocks(css="/content/style.css") as demo:
    gr.Markdown("Stable Diffusion")
    modeloptions = gr.Dropdown(container=False, show_label=False, choices=models.modelnames, elem_classes='ckpt-box')
    loadbutton = gr.Button("Load", elem_classes='load')
    with gr.Tab("Txt2Img"):
      prompt = gr.Textbox(elem_classes='prompt', lines=4, show_label=False)
      output_image = gr.Gallery(elem_classes='img-output', preview=True)
      negative_prompt = gr.Textbox(elem_classes='neg-prompt', lines=3, container=True, show_label=False)
      generate = gr.Button("Generate", elem_classes='generate')
      seed = gr.Textbox(elem_classes='inp-seed', label="Seed (Leave empty for random seed)")
      output_seed = gr.Textbox(elem_classes='out-seed', label="Current Seed", show_label=False)
      num_inference_steps = gr.Slider(elem_classes='steps', minimum=1, maximum=100, step=1, value=10, label="Sampling steps")
      sampling_methods = gr.Dropdown(elem_classes='samp-meth', label="Sampling method")
      guidance_scale = gr.Slider(elem_classes='guidance', minimum=1, maximum=10, step=0.1, value=5, label="CFG Scale")
      width = gr.Dropdown(elem_classes='width', choices=models.number_choices, value=408, label="Width")
      height = gr.Dropdown(elem_classes='height', choices=models.number_choicess, value=704, label="Height")
      batch_count = gr.Slider(elem_classes='b-count', minimum=1, maximum=10, step=1, value=1, label="Batch Count")
      batch_size = gr.Slider(elem_classes='b-size', minimum=1, maximum=10, value=1, step=1, label="Batch Size")
      generate.click(fn=modules.txt2img.txt2img, inputs=[prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, seed], outputs=[output_image, output_seed])
      loadbutton.click(fn=models.loadmodel, inputs=modeloptions)
# Launch the Gradio UI
demo.launch(share=True, debug=True)
