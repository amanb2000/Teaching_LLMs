# %%
"""
# Key-Value Bitmaps

This script plots the key and value representaitons for a given input sentence 
to an LLM. 

This should work for any HuggingFace LLM. 
"""

# %%
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pdb

# %% Set up the GPU
device = torch.device("mps")
print("Device: ", device)

# % Get the model 
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = model.to(device)
model.config.output_attentions = True

# %%
input_text = "The quick brown fox jumped over the lazy dog."
# Tokenize the input
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

print("Number of input tokens: ", len(input_ids[0]))

tokenizer.batch_decode(input_ids)

# %% Run the model
# Ge the logits, attention values, and key-value cached values 
outputs = model(input_ids)

# %% What we get for the outputs
outputs.keys()

# %% Let's visualize the past_key_values
past_kv = outputs['past_key_values']
print("Length of past key values: ", len(past_kv))
for i, kv in enumerate(past_kv): 
    print("\n\tPast key values for layer ", i)
    print("\tShape of key: ", kv[0].shape)
    print("\tShape of value: ", kv[1].shape)
    break


# The past key values have shape [batch, num_heads, seq_len, hidden_size // num_heads]
# In this case, batch = 1, num_heads = 12, seq_len = 16, hidden_size = 768

# %% Let's visualize the values in layer 0. 

def plot_layer_values(layer_num, past_kv): 
    layer_values = past_kv[layer_num][1]
    layer_keys = past_kv[layer_num][0]

    num_heads = layer_values.shape[1]

    # 10 side-by-side plots 
    # Plotting the values for each head in the layer
    fig, axs = plt.subplots(1, num_heads, figsize=(20, 10))

    for i, head in enumerate(range(num_heads)):
        head_i_values = layer_values[0, head, :, :] 
        axs[i].imshow(head_i_values.cpu().detach().numpy().T)
        axs[i].set_title("Head {}".format(i))
        axs[i].set_xlabel("Sequence position")
        if i == 0: 
            axs[i].set_ylabel("Value vector dimension dimension")

    plt.suptitle(f"Values for Transformer layer {layer_num}")
    plt.show()

    # Plotting the keys for each head in the layer
    fig, axs = plt.subplots(1, num_heads, figsize=(20, 10))

    for i, head in enumerate(range(num_heads)):
        head_i_keys = layer_keys[0, head, :, :] 
        axs[i].imshow(head_i_keys.cpu().detach().numpy().T)
        axs[i].set_title("Head {}".format(i))
        axs[i].set_xlabel("Sequence position")
        if i == 0: 
            axs[i].set_ylabel("Key vector dimension")
        # set colormap 
    
    plt.suptitle(f"Keys for Transformer layer {layer_num}")
    plt.show()







# %%
for i in range(12): 
    plot_layer_values(i, past_kv)

# %%
