'''Train CIFAR10 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import math

from models.resnet import *
from models.dnn import net
# from utils import progress_bar
from attack import PGDAttack
from wrapper import ModelWrapper, TestWrapper
from advertorch.attacks import LinfPGDAttack, SparseL1DescentAttack
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
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


def main(args):
    # def train(net, epoch, lr_schedule):
    def train(net):
        net.train()
        if args.adv_training:
            train_attack = PGDAttack(net, alpha=args.alpha, epsilon=args.epsilon, norm=args.norm, max_iteration=args.max_iter,
                                     ensembles=args.ensembles, msd=True, random_start=True)
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
            if args.adv_training:
                adv_inputs = train_attack.generate(inputs, targets)
                outputs = net(adv_inputs)
            else:
                adv_inputs = torch.clone(inputs)
                outputs = net(inputs)
            total_loss = torch.zeros(args.ensembles, dtype=torch.float)
            for i in range(args.ensembles):
                total_loss[i] = criterion(outputs[:, i * 10:(i + 1) * 10], targets[:, i])
            if args.mode == 'max':
                loss = torch.max(total_loss)
            elif args.mode == 'sum' or args.mode == 'avg':
                loss = torch.sum(total_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for i in range(args.ensembles):
                _, predicted = outputs[:, i * 10:(i + 1) * 10].max(1)
                correct[i] += predicted.eq(targets[:, i]).sum().item()
            total += targets.size(0)
            print(batch_idx, len(trainloader[0]),
                  'lr: {} | Loss: {} | Acc: {}'.format(str(optimizer.param_groups[0]["lr"]),
                                                       str([round(loss.item(), 3) for loss in
                                                            total_loss]),
                                                       str([round(100. * cor.item() / total, 3) for cor in correct])))

    def test(epoch, net):

        # attack = LinfPGDAttack(net, eps=8 / 255, nb_iter=40,eps_iter=2 / 255, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

        # attack = PGDAttack(net, alpha=0.05, epsilon=0.3, norm=[float('inf')], max_iteration=60, random_start=True)
        # attack = SparseL1DescentAttack(net, eps=12., nb_iter=40, eps_iter=3., l1_sparsity=0.975, rand_init=True)
        global best_acc
        net.eval()

        def evaluate(inputs, targets):
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            return correct

        correct = 0
        adv_correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # adv_inputs = attack(inputs, targets)
            total += targets.size(0)
            with torch.no_grad():
                correct_batch = evaluate(inputs, targets)
                # adv_correct_batch = evaluate(adv_inputs, targets)
            correct += correct_batch
            # adv_correct += adv_correct_batch
            print(batch_idx, len(testloader), 'Acc: {},{}'.format(str(round(100. * correct / total, 3)),
                                                                  str(round(100. * adv_correct / total, 3))))

        # Save checkpoint.
        acc = 100. * correct / total
        # if acc > best_acc:
        if epoch > 0:
            print('Saving..')
            state = {
                'net': net.model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer
                # 'scheduler': scheduler
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_msd_{}.pth'.format(epoch))
            best_acc = acc

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
        # scheduler = checkpoint['scheduler']
    else:
        start_epoch = 1
        best_acc = 0
        optimizer = optim.Adam(net.parameters(), lr=1)
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.max_lr, 0])[0]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=1, epochs=50, pct_start=0.4, anneal_strategy='linear')

    criterion = nn.CrossEntropyLoss()
    # Training
    for epoch in range(start_epoch, start_epoch + args.epochs):
        scheduler.step()
        print(epoch)
        train(ModelWrapper(net, num_classes=10, ensembles=args.ensembles, criterion=criterion))
        if epoch % 5 == 0:
            test(epoch, TestWrapper(net, ensemble=args.ensembles))

if __name__ == '__main__':
    main(args)

