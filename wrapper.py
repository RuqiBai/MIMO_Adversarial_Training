from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class ModelWrapper(nn.Module):
    """
    Wrapping the model to fit the requirement of the ART toolbox.
    """
    def __init__(self, model, dataset, ensembles, criterion):
        super().__init__()
        self.model = model
        self.dataset = dataset
        if dataset == 'CIFAR10':
            self.sub_in_channels = 3
            self.num_classes = 10
            self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465)*ensembles, (0.2471, 0.2435, 0.2616)*ensembles)
        elif dataset == 'MNIST':
            self.sub_in_channels = 1
            self.num_classes = 10
        else:
            raise NotImplementedError("{} dataset not implemented for wrapping".format(dataset))
        self.ensembles = ensembles
        self.criterion = criterion
        assert self.model.module.conv1.in_channels == self.sub_in_channels * self.ensembles

    def forward(self, x, *args):
        if self.dataset == 'MNIST':
            return self.model(x)
        elif self.dataset == 'CIFAR10':
            out = self.normalize(x)
            return self.model(out)

    def calc_loss(self, outputs, ground_truth):
        loss = torch.zeros(self.ensembles, dtype=torch.float)
        for i in range(self.ensembles):
            loss[i] = self.criterion(outputs[:, i * self.num_classes:(i + 1) * self.num_classes], ground_truth[:, i])
        return loss


class TestWrapper(ModelWrapper):
    def forward(self, x, softmax=True):
        out = x.repeat(1, self.ensembles, 1, 1)
        if self.dataset == 'CIFAR10':
            out = self.normalize(out)
        outputs = self.model(out).reshape(-1, self.ensembles, 10)
        if softmax:
            outputs = F.softmax(outputs, dim=2)
        outputs = torch.mean(outputs, dim=1)
        # outputs = outputs[:,2,:]
        return outputs

    def calc_loss(self, outputs, ground_truth):
        return self.criterion(outputs, ground_truth)

    @staticmethod
    def evaluate(outputs, targets):
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        return correct   
