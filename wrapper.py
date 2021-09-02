from torch import nn
import torch
import torch.nn.functional as F


class ModelWrapper(nn.Module):
    """
    Wrapping the model to fit the requirement of the ART toolbox.
    """
    def __init__(self, model, submodel_channels, num_classes, ensembles, criterion):
        super().__init__()
        self.model = model
        self.submodel_channels = submodel_channels
        self.num_classes = num_classes
        self.ensembles = ensembles
        self.criterion = criterion
        assert self.model.conv1.in_channels == self.submodel_channels * self.ensembles

    def forward(self, x, *args):
        return self.model(x)

    def calc_loss(self, outputs, ground_truth):
        loss = torch.zeros(self.ensembles, dtype=torch.float)
        for i in range(self.ensembles):
            loss[i] = self.criterion(outputs[:, i * self.num_classes:(i + 1) * self.num_classes], ground_truth[:, i])
        return loss


class TestWrapper(ModelWrapper):
    def forward(self, x, softmax=True):
        x = x.repeat(1, self.ensembles, 1, 1)
        outputs = self.model(x).reshape(-1, self.ensembles, 10)
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
