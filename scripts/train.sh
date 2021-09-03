#!/bin/bash



python main.py --dataset MNIST --model resnet18 --adv_training --msd --alpha 0.8 0.1 0.01 --epsilon 10.0 2.0 0.3 --norm 1 2 inf --mode avg
