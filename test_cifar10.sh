#!/bin/bash

norm_group="1 2 inf"

for norm in $norm_group
do
    echo "mm"
    python test.py --ensembles 3 --file_name ckpt_mm_CIFAR10_50.pth --model preact_resnet18 --dataset CIFAR10 --attack PGD --norm $norm --batch_size 1000 
    # echo "msd"
    # python test.py --ensembles 1 --file_name ckpt_msd_CIFAR10_50.pth --model preact_resnet18 --dataset CIFAR10 --attack PGD --norm $norm --batch_size 1000
    # echo "mat"
    # python test.py --ensembles 3 --file_name ckpt_mat_CIFAR10_50.pth --model preact_resnet18 --dataset CIFAR10 --attack PGD --norm $norm --batch_size 1000
    # echo "l1"
    # python test.py --ensembles 1 --file_name ckpt_l1_CIFAR10_50.pth --model preact_resnet18 --dataset CIFAR10 --attack PGD --norm $norm --batch_size 1000
    # echo "mat l1"
    # python test.py --ensembles 3 --file_name ckpt_mat_l1_CIFAR10_50.pth --model preact_resnet18 --dataset CIFAR10 --attack PGD --norm $norm --batch_size 1000
done
