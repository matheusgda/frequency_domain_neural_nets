import numpy as np
import torch

def trainer(preprocess,
    model, num_epochs, initial_loss, train_loader, val_loader,
    criterion, optimizer, device, show_every=10, dtype=torch.float):

    loss_list = list()
    accuracy_list = list()
    best_loss = initial_loss

    for e in range(num_epochs):

        for _, (x, target) in enumerate(train_loader):

            x = x.to(device=device, dtype=dtype)
            x = preprocess(x)
            scores = model(x)

            target = target.to(device=device, dtype=torch.long)
            loss = criterion(scores, target)

            optimizer.zero_grad() # clear gradients
            loss.backward()
            optimizer.step()

            best_loss = loss.item() * (best_loss > loss.item()) + \
                (best_loss < loss.item()) * loss.item()
            loss_list.append(loss.item())

        if e % show_every == 0:
            print("Epoch: {} / {}".format(e, num_epochs))
            print("Best loss {}.".format(best_loss))

        acc = 0
        num_samples = 0
        for _, (x, target) in enumerate(val_loader):

            x = x.to(device=device, dtype=dtype)
            x = preprocess(x)

            scores = model(x)
            target = target.to(device=device, dtype=torch.long)

            acc += (1.0 * (torch.argmax(scores, 1) == target)).sum().item()
            num_samples += target.shape[0]
        accuracy_list.append(acc / num_samples)


    return loss_list, accuracy_list


def print_param_counter(model):
    print("The model has {} parameters.".format(parameter_counter(model)))


def parameter_counter(model):

    count = 0
    for param in model.parameters():
        count += np.prod(param.size())
    return count