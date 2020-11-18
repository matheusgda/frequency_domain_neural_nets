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

NUM_TRAIN = 256
NUM_VAL = 128
BATCH_SIZE = 128
NUM_TEST = 128

model_name = "FDNN_CIFAR10.model"

CIFAR10_PATH = '/home/revz/Development/neural_nets/assignment2/cs682/datasets'
if len(sys.argv) > 1:
    CIFAR10_PATH = sys.argv[1]

N, C, H, W, K = BATCH_SIZE, 3, 32, 32, 10

transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])


cifar10_train = torchvision.datasets.CIFAR10(
    CIFAR10_PATH,
    train=True, download=True, transform=transform)

cifar10_test = torchvision.datasets.CIFAR10(
    CIFAR10_PATH,
    train=False, download=True, transform=transform)

train_loader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, pin_memory=True,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

val_loader = DataLoader(
    cifar10_train, batch_size=BATCH_SIZE, pin_memory=True,
    sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL)))

test_loader = DataLoader(
    cifar10_test, batch_size=BATCH_SIZE, pin_memory=True,
    sampler=sampler.SubsetRandomSampler(range(NUM_TEST)))

device = torch.device("cuda:0")

dims = (N, H, W, C, 1)
p_num_filters = (1, 3, 3, 3)
m_num_filters = (3, 3, 3)
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
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4, betas=(0.9,0.99), weight_decay=0)

num_epochs = 2

loss, accuracy = utils.trainer(
    preprocess,
    model, num_epochs, K, train_loader, val_loader, NUM_VAL, 
    criterion, optimizer, device)

test_accuracy = utils.evaluate(preprocess, model, test_loader, NUM_TEST, device)

torch.save(model.state_dict(), model_name)

plt.plot(range(len(loss)), loss)
plt.ylabel("Loss")
plt.xlabel("Step")
plt.savefig("{}_loss.png".format(model_name))
plt.grid(True)
plt.title("Training Loss")
plt.show()
plt.clf()

plt.plot(range(len(accuracy)), accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.grid(True)
plt.title("Validation Accuracy")
plt.savefig("{}_accuracy.png".format(model_name))
plt.show()

print("Accuracy on test set: {}.".format(test_accuracy))
