import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

data = torch.rand(1, 3, 64, 64)
print(data.shape)
lables = torch.rand(1, 1000)
print(lables.shape)

predictions = model(data)

loss = (predictions - lables).sum()
loss.backward()

print(type(model.parameters()))

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

ext_grad = torch.tensor([1., 1.])
Q.backward(gradient=ext_grad)

print(a.grad)
print(9*a**2 == a.grad)
