# -*- coding: utf-8 -*-
NUM_TRAIN = 640
NUM_VAL = 128
BATCH_SIZE = 64

import torch
import torchvision

from torch.utils.data import sampler
from torch.utils.data import DataLoader

import torch.fft as fft
import torchvision.transforms as T

import fdnn
import utils
import matplotlib.pyplot as plt



import sys

CIFAR10_PATH = '/home/revz/Development/neural_nets/assignment2/cs682/datasets'
if len(sys.argv) > 1:
    CIFAR10_PATH = sys.argv[1]



device = torch.device("cuda:0")
dtype = torch.cfloat

N, C, H, W, K = 64, 3, 32, 32, 10

transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# Create random input and output data
cifar10_train = torchvision.datasets.CIFAR10(
    CIFAR10_PATH,
    train=True, download=True, transform=transform)

cifar10_validation = torchvision.datasets.CIFAR10(
    CIFAR10_PATH,
    train=False, download=True, transform=transform)

train_loader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

val_loader = DataLoader(cifar10_validation, batch_size=BATCH_SIZE, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))


dims = (N, H, W, C, 1)
p_num_filters = (1, 32, 32, 32)
m_num_filters = (16, 16, 16)
preserved_dim = 3
initializer = fdnn.random_complex_weight_initializer
hadamard_initializer = fdnn.random_hadamard_filter_initializer
bias_initializer = fdnn.naive_bias_initializer
dropout = None

model = fdnn.FrequencyDomainNeuralNet(
    dims, p_num_filters, m_num_filters, K, preserved_dim=3,
    p_initializer=initializer, p_hadamard_initializer=hadamard_initializer,
    p_bias_initializer=bias_initializer,
    m_initializer=initializer, m_hadamard_initializer=hadamard_initializer,
    m_bias_initializer=bias_initializer,
    device=device, dropout=dropout)

utils.print_param_counter(model)

preprocess = fdnn.FourierPreprocess()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.99))

num_epochs = 50

loss, accuracy = utils.trainer(
    preprocess,
    model, num_epochs, K, train_loader, val_loader,criterion, optimizer, device)

plot = True


torch.save(model.state_dict(), "FDNN_CIFAR10_1M_Overffited.model")

plt.plot(range(len(loss)), loss)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig("prototype.png")
plt.show()
plt.clf()

plt.plot(range(len(accuracy)), accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.savefig("accuracy.png")
plt.show()