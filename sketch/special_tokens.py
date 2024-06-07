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

# model = model.to(device)
# %% 
special_tokens = ["<<special>>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# %%
tokenizer.special_tokens_map
# %%
input_string = "Something that contains <<special>> token."
print("Input string: ", input_string)
input_ids = tokenizer(input_string)["input_ids"]
print("Input ids: ", input_ids)


# %%
decoded_ids = tokenizer.batch_decode(input_ids)
print("Decoded sequence: ", decoded_ids)


# %% 
# Now let's add the new token to the
# embedding layer of the model 
print("model.transformer.wte shape before: ", model.transformer.wte)

# now we add the new token to the embedding layer
num_added_toks = len(special_tokens)
new_embeddings = model.resize_token_embeddings(tokenizer.vocab_size + num_added_toks)

model.set_input_embeddings(new_embeddings)

print("model.transformer.wte shape after: ", model.transformer.wte)

# %%
# test to see if the new token is added to the model
input_ids = tokenizer(input_string, return_tensors="pt")["input_ids"]
output = model(input_ids)
print("Output shape: ", output.logits.shape)

