# %%
import numpy as np
import matplotlib.pyplot as plt
# Set random seed for reproducibility
np.random.seed(42)

# %%
 
N_task = 10000
D = 5

# generate random C matrix where each row is normalized -- shape = [N_task, D]
C_unif = np.random.rand(N_task, D)
# sample from normal distributoin 
C = np.random.randn(N_task, D)
C = C / np.linalg.norm(C, axis=1)[:, np.newaxis]

# test that the rows of C are normalized
assert np.allclose(np.linalg.norm(C, axis=1), 1)

print(np.linalg.norm(C, axis=1).shape)

# 
# compute singular value decomposition 
U, S, V = np.linalg.svd(C, full_matrices=False)

# Compute AAT = V S^{-2} V^T
AAT = np.dot(V, np.dot(np.diag(1/S**2), V.T))

A = np.linalg.inv(C.T @ C) @ C.T
AAT_val = A @ A.T

# assert allclose between AAT_val and AAT
# assert np.allclose(AAT, AAT_val, atol=1e-3)

# Print AAT to the screen 
print("AAT = V S^{-2} V.T: ", AAT)
print("Sigma: ", S)

# heat map of AAT 
plt.imshow(AAT)
plt.colorbar()
# title
plt.title(f"Heatmap of AAT -- N_task = {N_task}, D = {D}")

# 


# %%

AAT_val

# %%
AAT
# %%
