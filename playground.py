import pickle

with open('model1_attemp5/FDNN_CIFAR10_1.model_data.pck', 'rb') as f:
    Dict = pickle.load(f)
    print(Dict)