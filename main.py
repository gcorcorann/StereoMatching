#!/usr/bin/env python3
import torch
import torch.nn as nn
from network import SiameseNetwork
from dataloader import get_loader
from trainer import train

def main():
    """ Main Function. """
    # dataloader parameters
    data_path = 'data/kitti2012/training'
    receptive_size = 9
    max_disp = 128
    batch_size = 5
    num_workers = 0
    # training parameters
    learning_rate = 1e-2
    max_epochs = 20
    criterion = nn.CrossEntropyLoss()

    # create network
    net = SiameseNetwork()
    print(net)

    # create dataloader
    dataloader, dataset_size = get_loader(data_path, receptive_size, max_disp,
            batch_size, num_workers)
    # create optimizer
    p = net.parameters()
    optimizer = torch.optim.Adagrad(p, learning_rate)

    # train the network
    train(net, dataloader, dataset_size, criterion, optimizer, max_epochs)

if __name__ == '__main__':
    main()
