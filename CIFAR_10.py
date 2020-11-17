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
import gradflow

dtype = torch.cfloat
device = torch.device("cuda:0")
# device = torch.device('cpu')

N, C, H, W, K = 64, 3, 32, 32, 10

transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# Create random input and output data
cifar10_train = torchvision.datasets.CIFAR10(
    '/home/revz/Development/neural_nets/assignment2/cs682/datasets',
    train=True, download=True, transform=transform)

cifar10_validation = torchvision.datasets.CIFAR10(
    '/home/revz/Development/neural_nets/assignment2/cs682/datasets',
    train=False, download=True, transform=transform)

loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

loader_val = DataLoader(cifar10_validation, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(
                              range(NUM_TRAIN, NUM_TRAIN + NUM_TRAIN)))


dims = (N, H, W, C, 1)
num_layers = 3
num_filters = (1, 4, 5, 6)
preserved_dim = 3
initializer = fdnn.random_complex_weight_initializer
hadamard_initializer = fdnn.random_hadamard_filter_initializer
bias_initializer = fdnn.naive_bias_initializer
output_features = H * W * C * num_filters[-1]

pfilter = fdnn.FrequencyFilteringBlock(
    dims, num_layers, num_filters, preserved_dim=3,
    initializer=initializer, hadamard_initializer=hadamard_initializer,
    bias_initializer=bias_initializer, device=device)

head = fdnn.ComplexClassificationHead(
    pfilter.num_output_features(), K, device=device)

# layers = module.layers
# layers.append(torch.nn.Flatten())
# layers.append(fdnn.ComplexLinear(output_features, K))
# layers.append(fdnn.Absolute())
model = torch.nn.Sequential(pfilter, head)

for name, param in model.named_parameters():
    print(name, param.requires_grad)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.99))

num_epochs = 50
losses = list()
accuracy = list()
epochs = list()

plot = True
for e in range(num_epochs):
    print("Epoch {}:".format(e))
    best_loss = 10

    for t, (x, y) in enumerate(loader_train):

        # preprocess x with fftn and needed reshaping
        x = x.view(N, H, W, C, 1).to(device=device, dtype=dtype)
        x = fft.fftn(x, dim=(-1, -2))
        real_x = torch.cat((x.real, x.imag))

        y = y.to(device=device, dtype=torch.long)
        y_pred = model(real_x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        # if t % 100 == 99:
            # print(t, loss.item(), "try it here")
        if best_loss > loss.item():
            best_loss = loss.item()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        # if plot:
        #     gradflow.plot_grad_flow(model.named_parameters())
        plot = False
        optimizer.step()
    print("Best loss {}.".format(best_loss))

    losses.append(best_loss)

    for t, (x, y) in enumerate(loader_val):
        
        # preprocess x with fftn and needed reshaping
        x = x.view(N, H, W, C, 1).to(device=device, dtype=dtype)
        x = fft.fftn(x, dim=(-1, -2))
        real_x = torch.cat((x.real, x.imag))

        y = y.to(device=device, dtype=torch.long)
        print(y)
        y_pred = model(real_x)
        val = (1.0 * (y_pred[y] > 0.5)).mean()
        accuracy.append(val.item())
        print(val.item())

plt.plot(list(range(num_epochs)), losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig("prototype.png")
plt.show()
plt.clf()

plt.plot(list(range(2 * num_epochs)), accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.savefig("accuracy.png")
plt.show()