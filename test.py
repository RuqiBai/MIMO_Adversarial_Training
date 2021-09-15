import foolbox
import foolbox.attacks as fa
import argparse
import torch
from models.resnet import ResNet18
from models.dnn import net
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from wrapper import TestWrapper
import eagerpy as ep
import torch.nn as nn

# from fast_adv.attacks import DDN

parser = argparse.ArgumentParser(description='PyTorch MNIST Evaluating')
parser.add_argument('--ensembles', default=3, type=int, help='ensemble number')
parser.add_argument('--file_name')
parser.add_argument('--norm', type=float)
parser.add_argument('--max_iter', default=100, type=int)
parser.add_argument('--restarts', default=10, type=int)
parser.add_argument('--attack', choices=('PGD', 'FGSM', 'BBA', 'Gaussian Noise', 'Boundary Attack',
                                         'DeepFool', 'DDN', 'CW', 'SP'))
                                         
parser.add_argument('--dataset', choices=('MNIST', 'CIFAR10'))
parser.add_argument('--model', choices=('dnn', 'resnet18', 'preact_resnet18', 'resnet50'))

# command parameter setting
args = parser.parse_args()
ensembles = args.ensembles
file_name = args.file_name
norm = args.norm
max_iter = args.max_iter
restarts = args.restarts
attack = args.attack
dataset = args.dataset
model = args.model

# init envs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_file = "/scratch/gilbreth/bai116/data/"
best_acc = 0
criterion = nn.CrossEntropyLoss()

# dataset setting
transform_test = transforms.Compose([transforms.ToTensor()])

if dataset == "MNIST":
    in_channels = 1
    num_classes = 10
    testset = torchvision.datasets.MNIST(root=data_file, train=False, download=True, transform=transform_test)
elif dataset == "CIFAR10":
    in_channels = 3
    num_classes = 10
    testset = torchvision.datasets.CIFAR10(root=data_file, train=False, download=True, transform=transform_test)
else:
    raise NotImplementedError("Only MNIST and CIFAR10 are supported now")
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

# Model Setting
print('==> Building model..')
if args.model == 'nn':
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

net = net(args.ensembles).to(device)
# net = ResNet18(args.ensembles, args.ensembles*num_classes).to(device)
# net = ResNet50(args.ensembles, args.ensembles*10).to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
checkpoint = torch.load('checkpoint/{}'.format(args.file_name))
net.load_state_dict(checkpoint['net'])
net = TestWrapper(net, dataset=dataset, ensembles=ensembles)

if args.attack == 'PGD':
    if args.norm == 1:
        attack = fa.L1ProjectedGradientDescentAttack()
    elif args.norm == 2:
        attack = fa.L2ProjectedGradientDescentAttack()
    elif args.norm == float('inf'):
        attack = fa.LinfProjectedGradientDescentAttack()
    else:
        raise ValueError('norm should be either 1 2 or inf')

elif args.attack == 'FGSM':
    if args.norm == float('inf'):
        attack = fa.LinfFastGradientAttack()
    elif args.norm == 2:
        attack = fa.L2FastGradientAttack()
    elif args.norm == 1:
        attack = fa.L1FastGradientAttack()
    else:
        raise ValueError('norm should be either 1 2 or inf')

elif args.attack == 'BBA':
    if args.norm == float('inf'):
        attack = fa.LinfinityBrendelBethgeAttack()
    elif args.norm == 2:
        attack = fa.L2BrendelBethgeAttack()
    elif args.norm == 1:
        attack = fa.L1BrendelBethgeAttack()
    else:
        raise ValueError('norm should be either 1 2 or inf')

elif args.attack == 'Gaussian Noise':
    if args.norm == 2:
        attack = fa.L2AdditiveGaussianNoiseAttack()
    else:
        raise ValueError('norm should be 2')

elif args.attack == 'Boundary Attack':
    if args.norm == 2:
        attack = fa.BoundaryAttack()
        restarts = 1
    else:
        raise ValueError('norm should be 2')

elif args.attack == 'DeepFool':
    if args.norm == 2:
        attack = fa.L2DeepFoolAttack()
    else:
        raise ValueError('norm should be 2')

elif args.attack == 'CW':
    if args.norm == 2:
        attack = fa.L2CarliniWagnerAttack()
    else:
        raise ValueError('norm should be 2')

elif args.attack == 'SP':
    if args.norm == 1:
        attack = fa.SaltAndPepperNoiseAttack()
    else:
        raise ValueError('norm should be 1')
fmodel = foolbox.models.PyTorchModel(net,bounds=(0., 1.), device=device)
for images, labels in testloader:
    images = ep.astensor(images.to(device))
    labels = ep.astensor(labels.to(device))
    print(images.shape)
    print(labels.shape)
    clean_acc = foolbox.accuracy(fmodel, images, labels)
    print(clean_acc)
    for r in range (restarts):
        if args.attack == 'Boundary Attack':
            raw_adv, clip_adv, success  = attack(images, labels=labels, threaded_rnd=False, threaded_gen=False)
        elif args.attack == 'CW':
            raw_adv, clip_adv, success  = attack(images, labels=labels, threaded_rnd=False, threaded_gen=False)
        else:
            raw_adv, clip_adv, success  = attack(fmodel, images, labels, epsilons=args.epsilon)
        robust_accuracy = 1 - success.float32().mean(axis=-1)
        print(robust_accuracy)