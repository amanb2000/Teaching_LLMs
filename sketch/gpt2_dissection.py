# %%
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pdb

# %% GPT2 loading 
device = torch.device("mps")
print("Device: ", device)

# %% Get the model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = model.to(device)

# %% 
model

# %%
model.transformer.h[0].mlp

# %%
type(model.transformer.h[0].mlp.act)

# %%
type(model)
# %% Run a forward pass with some tokens 
input_str = "To the assembled audience, I proclaim my will for personal freedom. Let me free, let me free, let me free,"
