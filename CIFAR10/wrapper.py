from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import torch
import torch.nn.functional as F


class ModelWrapper(nn.Module):
    """
    Wrapping the model to fit the requirement of the ART toolbox.
    """
    def __init__(self, model, num_classes, ensembles, criterion):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.ensembles = ensembles
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, x, ground_truth, mode):
        loss = torch.zeros(self.ensembles, dtype=torch.float)
        y = self(x)
        for i in range(self.ensembles):
            loss[i] = self.criterion(y[:, i * self.num_classes:(i + 1) * self.num_classes], ground_truth[:, i])
        if mode == 'sum':
            return torch.sum(loss)
        elif mode == 'max':
            return torch.max(loss)
        elif mode == 'all':
            return loss
        elif isinstance(mode, int):
            return loss[mode]

class TestWrapper(nn.Module):
    def __init__(self, model, ensemble):
        super().__init__()
        self.model = model
        self.ensemble = ensemble

    def forward(self, x, softmax=True):
        shape = x.shape
        x = x.repeat(1, self.ensemble, 1, 1)
        outputs = self.model(x).reshape(-1,self.ensemble,10)
        if softmax:
            outputs = F.softmax(outputs, dim=2)
        outputs = torch.mean(outputs, dim=1)
        # outputs = outputs[:,2,:]
        return outputs

