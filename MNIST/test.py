import foolbox
import foolbox.attacks as fa
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from time import time
import argparse 
from fast_adv.attacks import DDN


parser.add_argument('--mode', choices=('max', 'avg','sum'))
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
# Data
print('==> Preparing data..')
batch_size = 128
num_classes = 10
transform_train = transforms.Compose([
    transforms.ToTensor()
])

transform_test = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train)

trainloader = []
for i in range(args.ensembles):
    trainloader.append(torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1))

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=1)

# MIMO Model
print('==> Building model..')

net = ResNet18(args.ensembles, args.ensembles*num_classes).to(device)
# net = ResNet50(args.ensembles, args.ensembles*10).to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
