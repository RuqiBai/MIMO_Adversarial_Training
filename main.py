import os
import argparse

import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models import *
# from utils import progress_bar
from attack import PGDAttack
from wrapper import ModelWrapper, TestWrapper


parser = argparse.ArgumentParser(description='PyTorch MAT Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ensembles', default=3, type=int, help='ensemble number')
parser.add_argument('--adv_training', action='store_true')
parser.add_argument('--msd', action='store_true')
parser.add_argument('--epsilon', nargs='+', type=float)
parser.add_argument('--alpha', nargs='+', type=float)
parser.add_argument('--epochs', default=15, type=int, help='number of epochs to train [default: 200]')
parser.add_argument('--norm', nargs='+', type=float)
parser.add_argument('--max_iter', default=100, type=int)
parser.add_argument('--max_lr', default=1e-03, type=float)
parser.add_argument('--mode', choices=('max', 'avg','sum'))
parser.add_argument('--dataset', choices=('MNIST', 'CIFAR10'))
parser.add_argument('--model', choices=('dnn', 'resnet18', 'preact_resnet18', 'resnet50'))
args = parser.parse_args()

data_file = "/local/scratch/a/bai116/data/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 1
criterion = nn.CrossEntropyLoss()
if args.dataset == "MNIST":
    in_channels = 1
    num_classes = 10
elif args.dataset == "CIFAR10":
    in_channels = 3
    num_classes = 10
else:
    raise NotImplementedError("Only MNIST and CIFAR10 are supported now")

# MIMO Model
print('==> Building model..')
if args.model == 'nn':
    net = net(args.ensembles)
elif args.model == 'resnet18':
    net = ResNet18(args.ensembles, args.ensembles*num_classes).to(device)
elif args.model == 'resnet50':
    net = ResNet50(args.ensembles, args.ensembles*10).to(device)
elif args.model == 'preact_resnet18':
    net = PreActResNet18(args.ensembles, args.ensembles*10).to(device)
else:
    raise ModuleNotFoundError("Unknown model")
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Data
print('==> Preparing data..')
batch_size = 128
num_classes = 10

transform_train = transforms.Compose([
    transforms.ToTensor()
])

transform_test = transforms.Compose([transforms.ToTensor()])
if args.dataset == "MNIST":
    # data load
    trainset = torchvision.datasets.MNIST(
        root=data_file, train=True, download=True, transform=transform_train)

    trainloader = []
    for i in range(args.ensembles):
        trainloader.append(torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4))

    testset = torchvision.datasets.MNIST(
        root=data_file, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)

    # hyperparameter load
    optimizer = optim.Adam(net.parameters(), lr=1)
    lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.max_lr, 0])[0]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

elif args.dataset == "CIFAR10":
    # data load
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root=data_file, train=True, download=True, transform=transform_train)

    trainloader = []
    for i in range(args.ensembles):
        trainloader.append(torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4))

    testset = torchvision.datasets.CIFAR10(
        root=data_file, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)
    # hyperparameter load
    optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9, weight_decay=5e-4)
    lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs], [0, 0.1, 0.005, 0])[0]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_mimo_mnist_49.pth')
    # checkpoint = torch.load('./checkpoint/ckpt_0411.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    optimizer = checkpoint['optimizer']
    scheduler.load_state_dict(checkpoint['scheduler'])

def train(model, epoch):
    model.train()
    if args.adv_training and epoch > 1:
        attack = PGDAttack(model, alpha=args.alpha, epsilon=args.epsilon, norm=args.norm, max_iteration=args.max_iter, msd=True, random_start=True)

    correct = torch.zeros(args.ensembles)
    total = 0
    for batch_idx, data in enumerate(zip(*trainloader)):
        # initialize inputs, targets and forward pass
        inputs, targets = [], []
        for ensemble_data in data:
            inputs.append(ensemble_data[0])
            targets.append(ensemble_data[1])
        inputs = torch.cat(inputs, dim=1).to(device)
        targets = torch.stack(targets, dim=1).to(device)
        if args.adv_training and epoch > 1:
            adv_inputs = attack.generate(inputs, targets)
            outputs = model(adv_inputs)
        else:
            outputs = model(inputs)
        loss_list = model.calc_loss(outputs, targets)
        if args.mode == "max":
            loss = torch.max(loss_list)
        elif args.mode == "avg":
            loss = torch.sum(loss_list)
        else:
            raise NotImplementedError("Only support max or sum of subnetwork's loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(args.ensembles):
            _, predicted = outputs[:, i * 10:(i + 1) * 10].max(1)
            correct[i] += predicted.eq(targets[:, i]).sum().item()
        total += targets.size(0)
        print(batch_idx, len(trainloader[0]),
              'lr: {} | Loss: {} | Acc: {}'.format(str(optimizer.param_groups[0]["lr"]),
                                                       str([round(loss.item(), 3) for loss in loss_list]),
                                                       str([round(100. * cor.item() / total, 3) for cor in correct])), end='\r')
    print("")


def test(epoch, net):

        # attack = LinfPGDAttack(net, eps=8 / 255, nb_iter=40,eps_iter=2 / 255, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

        # attack = PGDAttack(net, alpha=0.05, epsilon=0.3, norm=[float('inf')], max_iteration=60, random_start=True)
        # attack = SparseL1DescentAttack(net, eps=12., nb_iter=40, eps_iter=3., l1_sparsity=0.975, rand_init=True)
    global best_acc
    net.eval()



    correct = 0
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # adv_inputs = attack(inputs, targets)
        total += targets.size(0)
        with torch.no_grad():
            outputs = net(inputs)
            correct_batch = net.evaluate(outputs, targets)
        # adv_correct_batch = evaluate(adv_inputs, targets)
        correct += correct_batch
        # adv_correct += adv_correct_batch
        print(batch_idx, len(testloader), 'Acc: {},{}'.format(str(round(100. * correct / total, 3)),
                                                              str(round(100. * adv_correct / total, 3))), end="\r")
    print("")
    # Save checkpoint.
    acc = 100. * correct / total
    # if acc > best_acc:
    if epoch > 0:
        print('Saving..')
        state = {
            'net': net.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer,
            'scheduler': scheduler.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        torch.save(state, './checkpoint/ckpt_{}_{}.pth'.format(epoch))
        best_acc = acc


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(epoch)
        train(ModelWrapper(net, sub_in_channels=in_channels, num_classes=num_classes, ensembles=args.ensembles, criterion=criterion), epoch)
        scheduler.step()
        if epoch % 5 == 0:
            test(epoch, TestWrapper(net, sub_in_channels=in_channels, num_classes=num_classes, ensembles=args.ensembles, criterion=criterion))
        
