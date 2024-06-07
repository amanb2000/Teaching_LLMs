"""
This script is a minimal demo of adding special tokens to a HuggingFace model 
tokenizer. 
"""
# %%
# We start by importing the necessary modules.
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pdb


# %% Get the model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = model.to(device)
# %% 
special_tokens = ["<<special>>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# %%
tokenizer.special_tokens_map
# %%
input_string = "Something that contains <<special>> token."
input_ids = tokenizer(input_string)["input_ids"]
input_ids


# %%
tokenizer.batch_decode(input_ids)


# %%
