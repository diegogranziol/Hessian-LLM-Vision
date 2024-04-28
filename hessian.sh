#!/bin/bash

# Define the constant parameters
batch_size=4
subsample=0.1

# Array of learning rates
learning_rates=(0.001 0.0001 0.00001)

# Array of k values
k_values=(10 5)

# Loop over each learning rate
for lr in "${learning_rates[@]}"
do
    # Set delta to lr and lr/10
    delta1=$lr
    delta2=$(echo "$lr / 10" | bc -l)

    # Loop over each k value
    for k in "${k_values[@]}"
    do
        # Run the script with delta = lr
        echo "Running with lr=$lr, delta=$delta1, k=$k"
        python gpt2_hessian_cpu.py --batch_size $batch_size --lr $lr --delta $delta1 --k $k --subsample $subsample

        # Run the script with delta = lr/10
        echo "Running with lr=$lr, delta=$delta2, k=$k"
        python gpt2_hessian_cpu.py --batch_size $batch_size --lr $lr --delta $delta2 --k $k --subsample $subsample
    done
done
