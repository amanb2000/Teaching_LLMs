import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model, GPT2Config
import pdb

class EvidenceAggregationGPT2(nn.Module):
    def __init__(self, d_input, d_model, n_tasks, use_mlp_observation=False):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_tasks = n_tasks
        self.use_mlp_observation = use_mlp_observation

        # Input projection
        self.input_projection = nn.Linear(d_input, d_model)

        # Observation MLP (optional)
        if use_mlp_observation:
            self.observation_mlp = nn.Sequential(
                nn.Linear(d_input, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_input)
            )

        # Modified GPT-2 model
        config = GPT2Config(n_embd=d_model, n_layer=6, n_head=8)
        self.gpt2 = GPT2Model(config)

        # Output projection
        self.output_projection = nn.Linear(d_model, n_tasks)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_input)
        
        if self.use_mlp_observation:
            x = self.observation_mlp(x)

        # Project input to model dimension
        x = self.input_projection(x)

        # Pass through GPT-2
        outputs = self.gpt2(inputs_embeds=x).last_hidden_state

        # Project output to n_tasks dimensions
        logits = self.output_projection(outputs)

        # Apply sigmoid
        probs = torch.sigmoid(logits)

        return probs

def generate_data(batch_size, sequence_length, d_input, n_tasks, sigma = 0.1):
    # Generate ground truth x*
    x_star = torch.rand(1, d_input) * 2 - 1
    # stack batch_size times to get batch_size x d_input
    x_star = x_star.repeat(batch_size, 1)

    # Generate noisy observations
    x_noisy = x_star.unsqueeze(1).repeat(1, sequence_length, 1) + sigma * torch.randn(batch_size, sequence_length, d_input)

    # Generate random decision boundaries
    decision_boundaries = torch.randn(n_tasks, d_input)
    decision_boundaries = decision_boundaries / decision_boundaries.norm(dim=1, keepdim=True)
    
    # Generate random thresholds
    thresholds = torch.randn(n_tasks)

    # Compute ground truth labels
    y_star = torch.where(
        (x_star @ decision_boundaries.T) + thresholds > 0,
        torch.ones(batch_size, n_tasks),
        torch.zeros(batch_size, n_tasks)
    )

    return x_noisy, y_star

def train_model(model, num_epochs, batch_size, sequence_length, d_input, n_tasks, learning_rate=1e-4, sigma=0.1):
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    for epoch in range(num_epochs):
        x_noisy, y_star = generate_data(batch_size, sequence_length, d_input, n_tasks, sigma=sigma)

        x_noisy = x_noisy.to(model.gpt2.device)
        y_star = y_star.to(model.gpt2.device)
        
        optimizer.zero_grad()
        y_pred = model(x_noisy)
        loss = loss_fn(y_pred[:, -1, :], y_star)  # Use only the last token's prediction
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Usage example
d_input = 3
d_model = 256
n_tasks = 20
batch_size = 32
sequence_length = 256
num_epochs = 1000
lr=1e-4
sigma=0.001

# move model to mps

model = EvidenceAggregationGPT2(d_input, d_model, n_tasks, use_mlp_observation=True)

model.gpt2 = model.gpt2.to("mps")
model = model.to("mps")

train_model(model, num_epochs, batch_size, sequence_length, d_input, n_tasks, learning_rate=lr, sigma=sigma)