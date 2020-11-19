import numpy as np
import torch
import torch.autograd.profiler as profiler

def trainer(preprocess,
    model, num_epochs, initial_loss, train_loader, val_loader, num_val,
    criterion, optimizer, device, show_every=10, dtype=torch.float):

    loss_list = list()
    accuracy_list = list()
    best_loss = initial_loss

    for e in range(num_epochs):

        for t, (x, target) in enumerate(train_loader):

            x = x.to(device=device, dtype=dtype)
            x = preprocess(x)
            scores = model(x)

            target = target.to(device=device, dtype=torch.long)
            loss = criterion(scores, target)

            optimizer.zero_grad() # clear gradients
            loss.backward()
            optimizer.step()

            l = loss.item()

            best_loss = l * (best_loss > l) + \
                (best_loss < l) * l
            loss_list.append(l)

        if e % show_every == 0:
            print("Epoch: {} / {}".format(e, num_epochs))
            print("Best loss {}.".format(best_loss))

        accuracy_list.append(
            evaluate(
                preprocess, model, val_loader, num_val, device, dtype=dtype))

    return loss_list, accuracy_list


def evaluate(preprocess, model, loader, num_samp, device, dtype=torch.cfloat):

    with torch.no_grad():
        acc = torch.zeros(1, device=device)
        for _, (x, target) in enumerate(loader):

            x = x.to(device=device, dtype=dtype)
            x = preprocess(x)

            scores = model(x)
            target = target.to(device=device, dtype=torch.long)

            acc += (1.0 * (torch.argmax(scores, 1) == target)).sum()
    return acc.item() / num_samp


def print_param_counter(model):
    print("The model has {} parameters.".format(parameter_counter(model)))


def parameter_counter(model):

    count = 0
    for param in model.parameters():
        count += np.prod(param.size())
    return count


def profile(x, model, batch_size, device, func, sort_by="cuda_time_total"):

    with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        with profiler.record_function(func):
            model(x)
    print(prof.key_averages().table(sort_by=sort_by, row_limit=10))
