import pickle
import matplotlib.pyplot as plt

model_name = 'FDNN_CIFAR10'

Attempt5 = None
Attempt6 = None
Attempt7 = None

with open('model1_attemp5/FDNN_CIFAR10_1.model_data.pck', 'rb') as f:
    Attempt5 = pickle.load(f)
    # print(Dict)

with open('model1_attemp6/FDNN_CIFAR10_1.model_data.pck', 'rb') as f:
    Attempt6 = pickle.load(f)

with open('FDNN_CIFAR10_3_data.pck', 'rb') as f:
    Attempt7 = pickle.load(f)

loss5 = Attempt5['train_loss']
loss6 = Attempt6['train_loss']
loss7 = Attempt7['train_loss']

plt.plot(range(len(loss5)), loss5)
plt.plot(range(len(loss6)), loss6)
plt.plot(range(len(loss7)), loss7)
plt.ylabel("Loss")
plt.xlabel("Step")
plt.grid(True)
plt.title("Training Loss")
plt.savefig("{}_loss.png".format(model_name))
# plt.show()
plt.clf()

accuracy5 = Attempt5['val_accuracy']
accuracy6 = Attempt6['val_accuracy']
accuracy7 = Attempt7['val_accuracy']

plt.plot(range(len(accuracy5)), accuracy5)
plt.plot(range(len(accuracy6)), accuracy6)
plt.plot(range(len(accuracy7)), accuracy7)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.grid(True)
plt.title("Validation Accuracy")
plt.savefig("{}_accuracy.png".format(model_name))
# plt.show()