#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import time
import gpytorch
import cola
from functools import partial
import torch.func as tf
from time import time
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Process batch_size, vector_seed, and data_seed.')
parser.add_argument('--batch_size', type=int, required=True, help='Batch size for DataLoader')
parser.add_argument('--data_seed', type=int, required=True, help='Seed number for data shuffling')
parser.add_argument('--vector_seed', type=int, required=True, help='Seed number for vector initialization')
args = parser.parse_args()

# Load dataset
#70m model
tokenizer = AutoTokenizer.from_pretrained("./offline/model/", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("./offline/model", local_files_only=True)
ds = load_dataset("./offline/dataset/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

subsample_size = int(0.1 * len(ds['train']))
seed_number = 123
torch.manual_seed(seed_number)
torch.cuda.manual_seed_all(seed_number)


def collate_fn(batch):
    try:
        input_ids = torch.stack([torch.tensor(item['Tokens'], dtype=torch.long) for item in batch])
        attention_mask = torch.ones_like(input_ids)  # Create attention_mask as all ones

        assert len(input_ids.shape) == 2, f"input_ids should be of shape [batch_size, seq_length], got {input_ids.shape}"
        assert len(attention_mask.shape) == 2, f"attention_mask should be of shape [batch_size, seq_length], got {attention_mask.shape}"

    except KeyError as e:
        print("KeyError: Make sure the dataset contains 'Tokens'.")
        raise e
    except TypeError as e:
        print("TypeError: Ensure the batch items are correctly formatted.")
        raise e
    except AssertionError as e:
        print(f"AssertionError: {e}")
        raise e

    labels = input_ids.clone()  # For causal LM, labels are the same as input_ids

    return (input_ids, labels)


# Create a random subsample of the dataset
batch_size = args.batch_size

# Define the loss criterion
from torch.utils._pytree import tree_flatten, tree_unflatten

criterion = torch.nn.CrossEntropyLoss()

# make stateless model
def flatten_params(params):
    shapes = [p.shape for p in params]
    flat_params = torch.cat([p.flatten() for p in params])
    return flat_params, shapes

def unflatten_params(flat_params, shapes):
    params = []
    i = 0
    for shape in shapes:
        size = torch.prod(torch.tensor(shape)).item()
        params.append(flat_params[i:i + size].view(shape))
        i += size
    return params

flat_p, shape = flatten_params(list(model.parameters()))
flat_p = flat_p.detach().requires_grad_(True)

def stateless_model(flatparams, x):
    params = unflatten_params(flatparams, shape)
    names = list(n for n, _ in model.named_parameters())
    nps = {n: p for n, p in zip(names, params)}
    return tf.functional_call(model, nps, x)

def flat_loss(X, y, params):
    outputs = stateless_model(params, X)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = y[..., 1:].contiguous()
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

class BatchedHessian(cola.ops.LinearOperator):
    def __init__(self, loss, params, dataloader):
        self.loss = loss
        self.params = params
        self.dataloader = dataloader
        super().__init__(dtype=params.dtype, shape=(params.numel(), params.numel()), annotations={cola.SelfAdjoint}) # mark it as self-adjoint

    def _matmat(self, V):
        HV = torch.zeros_like(V)
        with torch.no_grad():
            n = 0
            for X, y in self.dataloader:
                with torch.enable_grad():
                    H = cola.ops.Hessian(partial(self.loss, X.to(self.device), y.to(self.device)), self.params)
                    out = H @ V
                    n += 1
                HV += out
        return HV / n

# Save checkpoint
def save_checkpoint(matrix, data_seed, vector_seed, checkpoint_dir="70mpythia"):
    os.makedirs(f"{checkpoint_dir}/data_seed={data_seed}_vector_seed={vector_seed}", exist_ok=True)
    filename = f"{checkpoint_dir}/data_seed={data_seed}_vector_seed={vector_seed}/ckpt.pt"
    torch.save(matrix, filename)

device = "cuda"
P = sum(p.numel() for p in model.parameters())

# Set data_seed from args
data_seed = args.data_seed

# Shuffle and select subsample
subsample = ds['train'].shuffle(seed=data_seed).select(range(subsample_size))

# Create DataLoader
dataloader = DataLoader(subsample, batch_size=batch_size, collate_fn=collate_fn)

t0 = time()
H = BatchedHessian(flat_loss, flat_p, dataloader)

# Set vector_seed from args
vector_seed = args.vector_seed
torch.manual_seed(vector_seed)
random_vec = torch.randn(P, device=device)
random_vec = random_vec / torch.norm(random_vec, 2)

print("starting iterations")

# Perform Lanczos decomposition
with torch.no_grad():
    Q1, T, info2 = cola.Lanczos(start_vector=random_vec.clone(), pbar=True, tol=1e-7, max_iters=15)(H)

# Convert T to dense and save checkpoint
T_dense = T.to_dense()
save_checkpoint(T_dense, data_seed, vector_seed)
