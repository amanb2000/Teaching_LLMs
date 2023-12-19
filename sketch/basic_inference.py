# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

# %% Load the tokenizer and model
model_name = "tiiuae/falcon-7b"
print("Loading the tokenizer and model weights...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Done!")

# %% Take a look at the model:
print("The type of `model` is: ", type(model))
print("The type of `tokenizer` is: ", type(tokenizer))
print(f"\n`model` is currently on device `{model.device}`")
print(f"Number of parameters: ", model.num_parameters())

# %% Print the model object
print(model)

# %% Move to the GPU
# check if cuda is available: 
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
# check if mps is available
elif torch.backends.mps.is_available(): 
    device = torch.device("mps")
    print("We will use the MPS GPU:", device)

model = model.to(device)
model.eval()

# %% Define the input text, convert it into tokens.
input_text = "I love France. The capital of France is \""
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
print("input_text: ", input_text)
print("input_ids: ", input_ids)
print("input_ids.shape: ", input_ids.shape)

# sanity check: let's decode the input_ids back into text
print("input_ids decoded: ", tokenizer.decode(input_ids[0]))

# %% Run inference on the tokenized input text.
output = model(input_ids)
print("Output object keys: ", output.keys())
print("Output logits shape: ", output.logits.shape)

# %% Softmax the logits to get probabilities
# index the 0th logit batch (we have batch=1)
probs = torch.nn.functional.softmax(output['logits'][0], dim=-1)
probs = probs.cpu().detach().numpy()  # move to the cpu, convert to numpy array
probs.shape # [sequence_len, vocab_size]

# get the probabilities of the next token
next_token_probs = probs[-1,:] 

# %% Plot the probability distribution over the final token
import matplotlib.pyplot as plt
plt.plot(next_token_probs)
plt.title("Probability distribution over Final Token")
plt.savefig('../figures/01_falcon_probs.png')
# %% Now let's see what the highest probability tokens are. 
# First we decode range(vocab_size) to get the string representation 
# of each token in the vocabulary.
vocab_size = tokenizer.vocab_size
vocab = [tokenizer.decode([i]) for i in range(vocab_size)]

# sorted_idx will contain the indices that yield the sorted probabilities
# in descending order. 
sorted_idx = np.argsort(next_token_probs)[::-1]

# Print out the top 10 tokens and their probabilities
for i in range(10):
    print(vocab[sorted_idx[i]], "\t\t",probs[-1,sorted_idx[i]], "\t\t", sorted_idx[i])
