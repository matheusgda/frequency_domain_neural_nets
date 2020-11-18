import numpy as np
import torch

def trainer(preprocess,
    model, num_epochs, K, train_loader, val_loader,
    criterion, optimizer, device, show_every=10):

    loss_list = list()
    accuracy_list = list()
    for e in range(num_epochs):
        best_loss = 10 * np.log(K)

        for _, (x, target) in enumerate(train_loader):

            x = x.to(device=device, dtype=dtype)
            x = preprocess(x)
            y = model(x)        

            target = target.to(device=device, dtype=torch.long)
            loss = criterion(y, target)

            optimizer.zero_grad() # clear gradients
            loss.backward()
            optimizer.step()

            best_loss = loss.item() * (best_loss > loss.item()) + \
                (best_loss < loss.item()) * loss.item()
            loss_list.append(loss.item())

        if (e + 1) % show_every == 0:
            print("Epoch: {} / {}".format(e, num_epochs))
            print("Best loss {}.".format(best_loss))

        acc = 0
        num_samples = 0
        for _, (x, target) in enumerate(val_loader):

            x = x.to(device=device, dtype=dtype)
            x = preprocess(x)

            scores = model(scores)
            target = target.to(device=device, dtype=torch.long)

            acc += (1.0 * (torch.argmax(scores, 1) == y)).sum().item()
            num_samples += y.shape[0]
        accuracy_list.append(acc / num_samples)


    return model, loss_list, acc_list
