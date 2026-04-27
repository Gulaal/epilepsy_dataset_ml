import torch
import torch.nn as nn

m = nn.MaxPool1d(3, stride=3)
input = torch.randn(20, 16, 50)
output = m(input)

print(input.shape)
print(output.shape)
