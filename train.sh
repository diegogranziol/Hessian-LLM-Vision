#!/bin/bash

# Define the array of learning rates for SGD and Adam
sgd_learning_rates=(0.001 0.0001 0.00001)  # From 1e-3 to 1e-5
adam_learning_rates=(0.0001 0.00001 0.000001)  # From 1e-4 to 1e-6

# Set the subsample value
subsample=0.01

# Loop over optimizers with their specific learning rates
for optimiser in "SGD" "Adam"; do
    if [ "$optimiser" = "SGD" ]; then
        for lr in "${sgd_learning_rates[@]}"; do
            # Assuming typical momentum for SGD
            momentum=0.9
            echo "Running training with $optimiser, learning rate: $lr, momentum: $momentum"
            python3.10 gpt2_multigpu.py --optimiser $optimiser --lr $lr --subsample $subsample --momentum $momentum --batch_size 60
        done
    elif [ "$optimiser" = "Adam" ]; then
        for lr in "${adam_learning_rates[@]}"; do
            echo "Running training with $optimiser, learning rate: $lr"
            python3.10 gpt2_multigpu.py --optimiser $optimiser --lr $lr --subsample $subsample --batch_size 60
        done
    fi
done
