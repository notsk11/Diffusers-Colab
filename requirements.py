# Requirements
!pip install -q diffusers xformers transformers accelerate mediapy numba --upgrade
import torch
from IPython.display import clear_output
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline
import random, sys
import mediapy as media

gpu = torch.cuda.is_available()
clear_output()
