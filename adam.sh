#!/bin/bash

# Define the array of learning rates for Adam
adam_learning_rates=(0.0003 0.0001 0.00003 0.00001  0.000003 0.000001)  # From 1e-4 to 1e-6

# Set the subsample value
subsample=0.1

# Loop over learning rates for Adam optimizer
for lr in "${adam_learning_rates[@]}"; do
    echo "Running training with Adam, learning rate: $lr"
    python3.10 gpt2_multigpu.py --optimiser Adam --lr $lr --subsample $subsample --batch_size 4
done
