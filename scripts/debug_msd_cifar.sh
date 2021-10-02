#!/bin/bash

python main.py --dataset CIFAR10 --model preact_resnet18 --adv_training --msd --alpha 1.0 0.02 0.003 --epsilon 12.0 0.5 0.03 --norm 1 2 inf --ensembles 1 --epochs 50 --max_iter 50 --mode avg --seed 107
