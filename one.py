# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data, x_data.dtype)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_data, x_data.dtype)

x_ones = torch.ones_like(x_data)
print(f"Ones tensor:\n{x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor:\n{x_rand}\n")


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random tensor:\n{rand_tensor}\n")
print(f"Ones tensor:\n{ones_tensor}\n")
print(f"Zeros tensor:\n{zeros_tensor}\n")


tensor = torch.rand(3, 4)

print(f"Shape of tensorː {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Tensor is stored on: {tensor.device}")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[...,-1]}")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1)

# three ways of matrix product
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# three ways of hadamard product
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(z1, z2, out=z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor}\n")
tensor.add_(5)
print(f"{tensor}")

t = torch.ones(5)
print(f"tː {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"tː {t}")
print(f"n: {n}")