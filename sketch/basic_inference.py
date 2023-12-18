# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

# %%
model_name = "tiiuae/falcon-7b"
print("Loading the tokenizer and model weights...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Done!")

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

# %% 
input_text = "I love France. The capital of France is "
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
print("input_text: ", input_text)
print("input_ids: ", input_ids)
print("input_ids.shape: ", input_ids.shape)

# sanity check: let's decode the input_ids back into text
print("input_ids decoded: ", tokenizer.decode(input_ids[0]))

# %% Run inference on some text.
output = model(input_ids)
print("Output object keys: ", output.keys())


# %%
output.logits.shape
# %% Softmax the logits to get probabilities
probs = torch.nn.functional.softmax(output['logits'][0], dim=-1)
probs = probs.cpu().detach().numpy()  # move to the cpu, convert to numpy array
probs.shape # [sequence_len, vocab_size]

# %%
import matplotlib.pyplot as plt
plt.plot(probs[-1,:])
plt.title("Probability distribution over Final Token")

# %% Now let's see what the highest probability tokens are. 
# First we decode range(vocab_size) 
vocab_size = tokenizer.vocab_size
vocab = [tokenizer.decode([i]) for i in range(vocab_size)]
print("vocab: ", vocab[:10])

# %%
sorted_idx = np.argsort(probs[-1,:])[::-1]
plt.plot(probs[-1,sorted_idx])

# %% Print out the top 10 tokens and their probabilities
for i in range(10):
    print(vocab[sorted_idx[i]], "\t\t",probs[-1,sorted_idx[i]], "\t\t", sorted_idx[i])


# %%
