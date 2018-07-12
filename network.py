#!/usr/bin/env python3
import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # 32 feature maps
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3))

    def forward_branch(self, x):
        out = self.cnn(x)
        out = out.squeeze(2)
        return out

    def forward(self, inp1, inp2):
        out1 = self.forward_branch(inp1)
        out2 = self.forward_branch(inp2)
        out1 = out1.transpose(1,2)
        out = torch.bmm(out1, out2)
        out = out.view(out.shape[0], -1)
        return out

def test():
    """ Test Function. """
    # create inputs
    left = torch.randn((2, 1, 9, 9))
    right = torch.randn((2, 1, 9, 136))
    y = torch.LongTensor(1).random_(0, 128)
    print('left:', left.shape)
    print('right:', right.shape)
    print('y:', y.item())

    # create network
    net = SiameseNetwork()
    print(net)

    # forward pass
    out = net.forward(left, right)
    print('out:', out.shape)

if __name__ == '__main__':
    test()

