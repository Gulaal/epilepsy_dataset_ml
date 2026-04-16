import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_np, dtype=torch.double)


shape = (2, 3, )

rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(rand_tensor.device)

if torch.cuda.is_available():
    rand_tensor = rand_tensor.to('cuda')

print(rand_tensor)

rand_tensor.t_()
print(rand_tensor)


x = np.ones(5)
t = torch.from_numpy(x)

print(x, t)