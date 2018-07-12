import torch
import time

def train(net, dataloader, dataset_size, criterion, optimizer, max_epochs):
    # start timer
    start = time.time()
    # cpu/gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # store network to cpu/gpu
    net = net.to(device)
    # for each epoch
    for epoch in range(max_epochs):
        print()
        print('Epoch', epoch)
        print('-' * 8)
        # set network to training mode
        net.train()
        # used for losses + accuracies
        running_loss = 0
        running_correct = 0
        # iterate over data
        for i, data in enumerate(dataloader):
            # get inputs and labels
            left, right, y = data
            left = left.to(device)
            right = right.to(device)
            y = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # track history in training phase
            with torch.set_grad_enabled(True):
                # forward pass
                out = net.forward(left, right)
                loss = criterion(out, y)
                _, y_pred = torch.max(out, 1)
                correct = (y_pred == y).sum().item()

                # backward + optimize
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * left.shape[0]  # batch size
            running_correct += correct

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_correct / dataset_size
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # print elapsed time
    time_elapsed = time.time() - start
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

