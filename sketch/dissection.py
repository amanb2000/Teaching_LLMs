# %%  load gpt-2 
import os
import sys
import torch
import transformers

# import gpt2 
from transformers import AutoModel, AutoTokenizer

# load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# %% float16 
model.half()

# %%
