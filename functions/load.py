# /content/functions/load.py
from diffusers import DiffusionPipeline

class CustomDiffusionLoader:
    def __init__(self, model_name="digiplay/Realisian_v5"):
        self.model_name = model_name
        self.pipe = None

    def load_pipeline(self):
        if self.pipe is None:
            self.pipe = DiffusionPipeline.from_pretrained(self.model_name).to('cuda')
        return self.pipe
