"""
Minimal example of grabbing the computational graph of a PyTorch model and
examining the data (e.g., two sub_loss values)

E.g., x represents the params of a model. 
"""

# %%
import torch

# Define a simple computational graph
x = torch.tensor([2.0, 3.0, 4.0] , requires_grad=True)
loss_1 = x ** 2
loss_2 = x ** 3
z = loss_1 + loss_2

# Compute the final loss
loss = z.mean()
print("Loss: ", loss)

# %%
# Recursive function to traverse the computational graph and print gradients
def traverse_graph_with_gradients(node, level=0):
    print(f"{'  ' * level}{node}")
    if hasattr(node, 'next_functions'):
        for child in node.next_functions:
            if child[0] is not None:
                traverse_graph_with_gradients(child[0], level + 1)
    
    # Print gradients for PowBackward nodes (corresponding to sub-losses)
    if 'PowBackward' in str(node):
        input_tensor = node.next_functions[0][0].variable
        grad = input_tensor.grad
        if grad is not None:
            print(f"{'  ' * (level + 1)}Gradient of x w.r.t. {str(node)}: {grad}")

# Perform backward pass to compute gradients
loss.backward()

# Traverse the computational graph and print gradients
print("Computational Graph with Gradients:")
traverse_graph_with_gradients(loss.grad_fn)

# Print the gradient of x w.r.t. the final loss
print("Gradient of x w.r.t. the final loss:", x.grad)
# %%
