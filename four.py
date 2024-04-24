

# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html


import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# still don't have a gpu but again just in case I decide to run on colab

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device {device}")


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork().to(device)
print(model)


X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred.item()}")


input_image = torch.rand(3, 28, 28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

class SmoothReLU(nn.Module):

    def __init__(self, smoothing_factor):
        super(SmoothReLU, self).__init__()
        self.smoothing_factor = smoothing_factor

    def forward(self, x):
        return 0.5 * (x + torch.sqrt(x * x + self.smoothing_factor))

print(f"Before smoothed ReLU: {hidden1}")
# hidden1 = nn.ReLU()(hidden1)
# make up my own approximation function (smoothed relu)
# see https://www.desmos.com/calculator/oexu9djmyg
hidden1 = SmoothReLU(smoothing_factor=1)(hidden1)
print(f"After smoothed ReLU: {hidden1}")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    SmoothReLU(smoothing_factor=1),
    nn.Linear(20, 10),
)

input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print(f"Model structure: {model}")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")