#!/bin/bash


module load anaconda
module load use.own
module load conda-env/pytorch-py3.6.4

python main.py --adv_training --msd --alpha 0.8 0.1 0.01 --epsilon 10.0 2.0 0.3 --norm 1 2 inf --mode max
