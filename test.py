import foolbox
import foolbox.attacks as fa
import argparse
import torch
from models import *
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from wrapper import TestWrapper
import eagerpy as ep
import torch.nn as nn
import os
import random

# from fast_adv.attacks import DDN

parser = argparse.ArgumentParser(description='MAT, MSD Evaluating')
parser.add_argument('--ensembles', default=3, type=int, help='ensemble number')
parser.add_argument('--file_name', help='the model parameters file')
parser.add_argument('--restarts', default=10, type=int)
parser.add_argument('--attack', choices=('PGD', 'FGSM', 'BBA', 'Gaussian', 'Boundary',
                                         'DeepFool', 'DDN', 'CW', 'SP', 'EAD', 'AUTO'))
parser.add_argument('--dataset', choices=('MNIST', 'CIFAR10'))
parser.add_argument('--model', choices=('dnn', 'resnet18', 'preact_resnet18', 'resnet50'))
parser.add_argument('--batch_size', type=int, default=100)

# command parameter setting
parser.add_argument('--norm', type=float)
args = parser.parse_args()
ensembles = args.ensembles
file_name = args.file_name
norm = args.norm
restarts = args.restarts
attack_name = args.attack
dataset = args.dataset
model = args.model
batch_size = args.batch_size
# init envs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data_file = "/scratch/gilbreth/bai116/data/"
data_file = "../../../data"
best_acc = 0
criterion = nn.CrossEntropyLoss()

# dataset setting
transform_test = transforms.Compose([transforms.ToTensor()])

if dataset == "MNIST":
    in_channels = 1
    num_classes = 10
    epsilon={1:10,2:2,float('inf'):0.3}
    testset = torchvision.datasets.MNIST(root=data_file, train=False, download=True, transform=transform_test)
elif dataset == "CIFAR10":
    in_channels = 3
    num_classes = 10
    epsilon={1:12,2:0.5,float('inf'):0.03}
    testset = torchvision.datasets.CIFAR10(root=data_file, train=False, download=True, transform=transform_test)
else:
    raise NotImplementedError("Only MNIST and CIFAR10 are supported now")
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model Setting
print('==> Building model..')
if args.model == 'dnn':
    net = net(args.ensembles)
elif args.model == 'resnet18':
    net = ResNet18(in_channels*args.ensembles, args.ensembles*num_classes).to(device)
elif args.model == 'resnet50':
    net = ResNet50(in_channels*args.ensembles, args.ensembles*num_classes).to(device)
elif args.model == 'preact_resnet18':
    net = PreActResNet18(in_channels*args.ensembles, args.ensembles*num_classes).to(device)
else:
    raise ModuleNotFoundError("Unknown model")
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('checkpoint/{}'.format(args.file_name))
net.load_state_dict(checkpoint['net'])
net = TestWrapper(net, dataset=dataset, ensembles=ensembles, criterion=criterion)

if attack_name == 'PGD':
    if args.norm == 1:
        k = (random.randint(5,20))
        if dataset == 'MNIST':
            q = 1 - k/(28*28)
            attack = fa.SparseL1DescentAttack(quantile=q, abs_stepsize=0.8, steps=200, random_start=True)
        elif dataset == 'CIFAR10':
            q = 1 - k/(32*32)
            attack = fa.SparseL1DescentAttack(quantile=q, abs_stepsize=1.0, steps=100, random_start=True)
    elif args.norm == 2:
        if dataset == 'MNIST':
             attack = fa.L2ProjectedGradientDescentAttack(abs_stepsize=0.1, steps=200, random_start=True)
        elif dataset == 'CIFAR10':
             attack = fa.L2ProjectedGradientDescentAttack(abs_stepsize=0.05, steps=200, random_start=True)
    elif args.norm == float('inf'):
        if dataset == 'MNIST':
             attack = fa.LinfProjectedGradientDescentAttack(abs_stepsize=0.01, steps=200, random_start=True)
        elif dataset == 'CIFAR10':
             attack = fa.LinfProjectedGradientDescentAttack(abs_stepsize=0.003, steps=200, random_start=True)
    else:
        raise ValueError('norm should be either 1 2 or inf')


# elif attack_name == 'BBA':
#     if args.norm == float('inf'):
#         attack = fa.LinfinityBrendelBethgeAttack()
#     elif args.norm == 2:
#         attack = fa.L2BrendelBethgeAttack()
#     elif args.norm == 1:
#         attack = fa.L1BrendelBethgeAttack()
#     else:
#         raise ValueError('norm should be either 1 2 or inf')

elif attack_name == 'Gaussian':
    if args.norm == 2:
        attack = fa.L2AdditiveGaussianNoiseAttack()
    else:
        raise ValueError('norm should be 2')

elif attack_name == 'Boundary':
    if args.norm == 2:
        attack = fa.BoundaryAttack()
        restarts = 1
    else:
        raise ValueError('norm should be 2')

elif attack_name == 'CW':
    if args.norm == 2:
        attack = fa.L2CarliniWagnerAttack()
    else:
        raise ValueError('norm should be 2')

elif attack_name == 'SP':
    if args.norm == 1:
        attack = fa.SaltAndPepperNoiseAttack()
    else:
        raise ValueError('norm should be 1')

# elif attack_name == 'EAD':
    
net.eval()
save_path = "./attack/{}/".format(os.path.splitext(file_name)[0])

os.makedirs(save_path, exist_ok=True)
total = 0
res = None
fmodel = foolbox.models.PyTorchModel(net,bounds=(0., 1.), device=device)
for images, labels in testloader:
    total += images.shape[0]
    print(total)
    images = ep.astensor(images.to(device))
    labels = ep.astensor(labels.to(device))
    clean_acc = foolbox.accuracy(fmodel, images, labels)
    print(clean_acc)
    worst_case_succ = ep.astensor(torch.zeros(batch_size, dtype=torch.bool).to(device)) 
    for r in range (restarts):
        print(restarts)
        if attack_name == 'Boundary':
            _, clip_adv, success  = attack(fmodel, images, criterion=labels)
        else:
            _, clip_adv, success  = attack(fmodel, images, criterion=labels, epsilons=epsilon[norm])
        dist = (clip_adv-images).reshape((batch_size,-1)).raw.norm(p=norm,dim=1)
        worst_case_succ = worst_case_succ.logical_or(success.logical_and(dist.le(epsilon[norm])))
    if res is not None:
        res = torch.cat((res,worst_case_succ.raw))
    else:
        res = worst_case_succ.raw
    if total >= 1000:
        break
torch.save(res[0:1000], open(save_path + attack_name + '_' + str(norm), 'wb'))
print(1-res[0:1000].float().mean())
