#!/bin/bash

# Loop over data_seed from 997 to 999
for data_seed in {997..999}
do
    # Loop over vector_seed from 997 to 999
    for vector_seed in {997..999}
    do
        # Execute the Python script with the current seeds and batch size
        python diego_pythia_tiny.py --data_seed $data_seed --vector_seed $vector_seed --batch_size 32
    done
done
