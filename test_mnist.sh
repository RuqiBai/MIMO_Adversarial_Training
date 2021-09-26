#!/bin/bash

attack_group="CW"
for attack in $attack_group
do
    echo "mm 25"
    python test.py --ensembles 3 --file_name ckpt_mm_MNIST_25.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000 
    echo "mm 15"
    python test.py --ensembles 3 --file_name ckpt_mm_MNIST_15.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000
    echo "msd resnet"
    python test.py --ensembles 1 --file_name ckpt_msd_MNIST_resnet_15.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000
    echo "msd dnn"
    python test.py --ensembles 1 --file_name ckpt_msd_MNIST_15.pth --model dnn --dataset MNIST --attack $attack --norm 2 --batch_size 1000
    echo "mat"
    python test.py --ensembles 3 --file_name ckpt_mat_MNIST_15.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000
    echo "l1"
    python test.py --ensembles 1 --file_name ckpt_l1_MNIST_15.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000
    echo "mat l1"
    python test.py --ensembles 3 --file_name ckpt_mat_l1_MNIST_15.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000
    echo "l2"
    python test.py --ensembles 1 --file_name ckpt_l2_MNIST_15.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000
    echo "mat l2"
    python test.py --ensembles 3 --file_name ckpt_mat_l2_MNIST_15.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000
    echo "linf"
    python test.py --ensembles 1 --file_name ckpt_linf_MNIST_15.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000
    echo "mat linf"
    python test.py --ensembles 3 --file_name ckpt_mat_linf_MNIST_15.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000
    echo "standard"
    python test.py --ensembles 1 --file_name ckpt_standard_MNIST_15.pth --model resnet18 --dataset MNIST --attack $attack --norm 2 --batch_size 1000

done
