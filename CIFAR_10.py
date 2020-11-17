# -*- coding: utf-8 -*-
NUM_TRAIN = 30000
NUM_VAL = 200

import torch
import torchvision

from torch.utils.data import sampler
from torch.utils.data import DataLoader

import torch.fft as fft
import torchvision.transforms as T

import fdnn
import matplotlib.pyplot as plt
import gradflow

device = torch.device("cuda:0")
dtype = torch.cfloat

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
                          sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))


# dims = (N, H, W, C, 1)
# num_filters = (1, 4, 5, 6)
# preserved_dim = 3
# initializer = fdnn.random_complex_weight_initializer
# hadamard_initializer = fdnn.random_hadamard_filter_initializer
# bias_initializer = fdnn.naive_bias_initializer

# pfilter = fdnn.FrequencyFilteringBlock(
#     dims, num_filters, preserved_dim=3,
#     initializer=initializer, hadamard_initializer=hadamard_initializer,
#     bias_initializer=bias_initializer, device=device)

# head = fdnn.ComplexClassificationHead(
#     pfilter.num_output_features(), K, device=device)

# model = torch.nn.Sequential(pfilter, head)


dims = (N, H, W, C, 1)
p_num_filters = (1, 4, 5, 6)
m_num_filters = (4, 5, 6)
preserved_dim = 3
initializer = fdnn.random_complex_weight_initializer
hadamard_initializer = fdnn.random_hadamard_filter_initializer
bias_initializer = fdnn.naive_bias_initializer

model = fdnn.FrequencyDomainNeuralNet(
    dims, p_num_filters, m_num_filters, K, preserved_dim=3,
    p_initializer=initializer, p_hadamard_initializer=hadamard_initializer,
    p_bias_initializer=bias_initializer,
    m_initializer=initializer, m_hadamard_initializer=hadamard_initializer,
    m_bias_initializer=bias_initializer,
    device=device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.99))

num_epochs = 500
losses = list()
accuracy = list()
epochs = list()

plot = True
for e in range(num_epochs):
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

    if e % 100 == 0:
        print("Best loss {}.".format(best_loss))
        print("Epoch: {} / {}".format(e, num_epochs))

    losses.append(best_loss)

    for t, (x, y) in enumerate(loader_val):
        
        # preprocess x with fftn and needed reshaping
        x = x.view(N, H, W, C, 1).to(device=device, dtype=dtype)
        x = fft.fftn(x, dim=(-1, -2))
        real_x = torch.cat((x.real, x.imag))
        scores = model(real_x)

        y = y.to(device=device, dtype=torch.long)

        val = (1.0 * (torch.argmax(scores, 1) == y)).mean()
        accuracy.append(val.item())
        # print("Batch {0} accuracy: {1}".format(t, val.item()))

torch.save(model.state_dict(), "FDNN_CIFAR10.model")

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