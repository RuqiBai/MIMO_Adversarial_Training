import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def net(ensembles=1):
        return nn.Sequential(nn.Conv2d(1*ensembles, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10*ensembles))

