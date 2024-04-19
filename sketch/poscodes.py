# %%  load gpt-2 
import os
import sys
import torch
import transformers

# import gpt2 
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# load the model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# %%
model
# %%
pos_codes = model.transformer.wpe.weight.cpu().detach().numpy()

print("Positional codes shape (num_pos=1024, dim=768): ", pos_codes.shape)
# %%
import matplotlib.pyplot as plt
import numpy as np

# plot the positional codes
plt.plot(model.transformer.wpe.weight.cpu().detach().numpy(), alpha=0.1)
plt.xlabel("Position")
plt.ylabel("Value")

# %%
plt.imshow(pos_codes)
# %%
