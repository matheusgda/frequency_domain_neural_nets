# -*- coding: utf-8 -*-
NUM_TRAIN = 128

import torch
import torchvision

from torch.utils.data import sampler
from torch.utils.data import DataLoader

import torch.fft as fft
import torchvision.transforms as T

import fdnn
import matplotlib.pyplot as plt

dtype = torch.cfloat
# device = torch.device("cuda:0")
device = torch.device('cpu')

N, C, H, W, K = 64, 3, 32, 32, 10

transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# Create random input and output data
cifar10_train = torchvision.datasets.CIFAR10(
    '/home/revz/Development/neural_nets/assignment2/cs682/datasets',
    train=True, download=True, transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))


def initializer(x, d):
    A = 0.01 * torch.randn(x, device=d)
    B = 0.01 * torch.randn(x, device=d)
    return (A, B)

def hadamard_initializer(x, d):
    return (0.01 * torch.randn(x, device=d), 0.01 * torch.randn(x, device=d))

dims = (N, H, W, C, 1)
num_layers = 1
num_filters = (1, 4, 5, 6)
preserved_dim = 3

bias_initializer = None

module = fdnn.FrequencyFilteringBlock(
    dims, num_layers, num_filters, preserved_dim=None,
    initializer=initializer, hadamard_initializer=hadamard_initializer,
    bias_initializer=bias_initializer, device=device)
layers = module.layers
layers.append(torch.nn.Flatten())
layers.append(fdnn.Absolute())
# layers.append(
#     fdnn.ComplexLinear(2 * H * W * C * 1, K, initializer=initializer, layer_ind=8))
model = torch.nn.Sequential(*layers)

epochs = 100

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)
losses = list()
time = list(range(epochs))
for e in time:
    print("epoch {}".format(e))
    for t, (x, y) in enumerate(loader_train):
        x = x.view(N, H, W, C, 1).to(device=device, dtype=dtype)  # move to device, e.g. GPU
        x = fft.fftn(x, dim=(-1, -2))
        real_x = torch.cat((x.real, x.imag))
        print(real_x.shape, "shape")
        y = y.to(device=device, dtype=torch.long)
        y_pred = model(real_x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())
        # print(loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
 
plt.plot(time, losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig("prototype.png")
plt.show()