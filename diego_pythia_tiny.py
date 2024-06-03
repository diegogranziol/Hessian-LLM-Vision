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
tokenizer = AutoTokenizer.from_pretrained("./offline/model/", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("./offline/model", local_files_only=True)
ds = load_dataset("./offline/dataset/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

subsample_size = int(0.01 * len(ds['train']))
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

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }



# Create a random subsample of the dataset
batch_size = args.batch_size

# Define the loss criterion
class CurvVecProduct:
    def __init__(self, loader, model, init_vec=None):
        self.loader = loader
        self.model = model
        self.init_vec = init_vec
        self.iters = 0
        self.timestamp = time.time()

    def __call__(self, vector):
        if self.iters == 0 and self.init_vec is not None:
            vector = self.init_vec
        start_time = time.time()
        output = hess_vec(
            vector,
            self.loader,
            self.model,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        time_diff = time.time() - start_time
        self.iters += 1
        print('Iter %d. Time: %.2f' % (self.iters, time_diff))
        return output.unsqueeze(1)

def hess_vec(vector, dataloader, model, device):
    param_list = list(model.parameters())
    vector_list = []

    offset = 0
    for param in param_list:
        vector_list.append(vector[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    model.eval()
    model.zero_grad()
    N = len(dataloader.dataset)
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if torch.cuda.device_count() > 1:
            loss = loss.mean()

        loss *= len(batch['input_ids']) / N

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)
        dL_dvec = torch.zeros(1, device=device)
        for v, g in zip(vector_list, grad_list):
            dL_dvec += torch.sum(v * g)
        dL_dvec.backward()

    model.eval()
    return torch.cat([param.grad.view(-1) for param in param_list]).view(-1).to(device)


# Save checkpoint
def save_checkpoint(matrix, data_seed, vector_seed, checkpoint_dir="70mpythia"):
    os.makedirs(f"{checkpoint_dir}/diegotiny_data_seed={data_seed}_vector_seed={vector_seed}", exist_ok=True)
    filename = f"{checkpoint_dir}/diegotiny_data_seed={data_seed}_vector_seed={vector_seed}/ckpt.pt"
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
# Set vector_seed from args
vector_seed = args.vector_seed
torch.manual_seed(vector_seed)
random_vec = torch.randn(P, device=device)
random_vec = random_vec / torch.norm(random_vec, 2)

lanczos_iters = 15

dimension = random_vec.shape[0]

start_start = time()

T = torch.zeros([lanczos_iters + 1, lanczos_iters + 1], device=device)
q_old = torch.zeros_like(random_vec, device=device)  # Ensure q_old is a vector of the same size as random_vec

# v = torch.randn([dimension], device=device)
# b = torch.norm(v, p=2)
# v /= b
v = random_vec.clone()
q_old = q_old.to(device)

# w = Hess_Vec(M, v)
w = hess_vec(v, dataloader, model, device)
w = w.to(device)  # Ensure w is on the correct device

alpha = torch.dot(w, v)
T[0, 0] = alpha
w -= alpha * v
v_old = v

for i in range(lanczos_iters):
    start_time = time()
    b = torch.norm(w, 2)
    T[i + 1, i] = b
    T[i, i + 1] = b
    v = w / b
    # w = Hess_Vec(M, v)
    w = hess_vec(v, dataloader, model, device)
    w = w.to(device)  # Ensure w is on the correct device

    alpha = torch.dot(w, v)
    T[i + 1, i + 1] = alpha
    w -= (alpha * v + b * v_old)
    v_old = v

    print("step", i + 1)
    print("Time:", time() - start_time)
    save_checkpoint(T, data_seed, vector_seed)

total_time = time()-start_start
print("total time {}".format(total_time))

save_checkpoint(T, data_seed, vector_seed)
