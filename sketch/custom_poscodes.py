# %%
"""
Positional encoding is super important. E.g., transformers tend to learn awfully 
local attention patterns. So, if you're like me, and want to make your transformer 
"think" using reasoning tokens, sometimes you gotta mess with the positional 
encodings so that the reasoning doesn't get in the way of this super-strong 
local attention bias.

The goal here is to replace the positional encoding logic in an LLM with some 
custom logic. In this case, we're going to make sure that the positional codes 
for register tokens don't interfere with the positional codes for the surrounding 
"real" tokens. Something like this: 

TOKENS: ['I', 'am', 'a', 'chicken', '<r>', '<r>', '<r>', '<r>', '<r>', 'I',
        'crossed', 'the', 'road']
POSCODE: [0, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7]

I think this might fix the fact that the small GPT-2 LLMs have such a hard time 
"learning to use" the register tokens. 
"""

# %% Import box
import torch 
import transformers 
import torch
import numpy as np

DEVICE = 'mps' # cuda, cpu, or mps

# %% Making the model 
###################
### MODEL SETUP ###
###################
config = transformers.GPT2Config(
    vocab_size=50257,  # Vocabulary size of the GPT-2 model
    n_embd=69,  # Hidden size of the transformer embeddings
    n_layer=3,  # Number of transformer layers
    n_head=3,  # Number of attention heads
    n_positions=2048,  # Maximum sequence length
)
# creating the model, tokenizer
print("Creating model, tokenizer...")
model = transformers.GPT2LMHeadModel(config)
tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
print("Done!")

# %% set device 
if DEVICE:
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("\nMoving model to device: ", device)
model = model.to(device)
print("Done!\n")

# %% Add register token
TOKENS_TO_ADD = ['<r>', '</r>']

special_tokens_dict = {'additional_special_tokens': TOKENS_TO_ADD + tokenizer.all_special_tokens }
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
new_embeddings = model.resize_token_embeddings(tokenizer.vocab_size + num_added_toks)
model.set_input_embeddings(new_embeddings)

try:
    tokenizer.start_reasoning_token_id = tokenizer.encode("<r>", add_special_tokens=False)[0]
    tokenizer.end_reasoning_token_id = tokenizer.encode("</r>", add_special_tokens=False)[0]
except Exception:
    print("Error: something went wrong with adding the reasoning tokens " + " ".join(TOKENS_TO_ADD))

# %%
input_ids = tokenizer.encode('Hello, world!', return_tensors='pt').to(model.device)
input_ids

# %%
output = model(input_ids)

# %% Replacing positional coding logic 
"""
GPT2LMHead models have a `transformer` object (GPT2LMHeadModel.transformer:GPT2Model). 

In any case, seems like there's a `position_ids` kwarg you can pass to the 
`forward()` function of GPTLMHeadModel which denotes the index of each token. 
We can just pass in a new `position_ids` object like this 
"""
sentence_with_reg_toks = 'Hello, <r><r><r>world!'
input_ids = tokenizer.encode(sentence_with_reg_toks, return_tensors='pt')
input_ids

# %%
position_ids = torch.range(0, input_ids.shape[1]-1, dtype=torch.int).unsqueeze(0)
position_ids = position_ids.to(torch.int)
position_ids

# %%
reg_mask = (input_ids == tokenizer.encode('<r>')[0]).to(torch.int)
reg_mask

# %%
sub_me = torch.cumsum(reg_mask, 1) 
sub_me
# %%
input_ids
# %%
position_ids -= sub_me
position_ids 
# %%
