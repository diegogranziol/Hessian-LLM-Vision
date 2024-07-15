from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import time

parser = argparse.ArgumentParser(description='Example script to demonstrate argparse usage.')
parser.add_argument('--batch_size', type=int, help='A list of numbers', default=8)
parser.add_argument('--subsample', type=float, help='A list of numbers', default=1)
parser.add_argument('--lr', type=float, help='A list of numbers', default=1e-4)
parser.add_argument('--beta1', type=float, help='Beta1 for Adam', default=0.9)
parser.add_argument('--beta2', type=float, help='Beta2 for Adam', default=0.999)
parser.add_argument('--delta', type=float, help='Delta for Adam', default=1e-8)
parser.add_argument('--accumulation_steps', type=int, help='A list of numbers', default=8)
args = parser.parse_args()

# Import necessary for Data Parallel
from torch.nn import DataParallel

# Load the dataset
ds = load_dataset("wikipedia", "20220301.simple")
model_name = 'gpt2'
subsample_size = int(args.subsample * len(ds['train']))

# Create a random subsample of the dataset
subsample = ds['train'].shuffle(seed=42).select(range(subsample_size))

# Load tokenizer for your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_docs = subsample.map(tokenize_function, batched=True)

def select_model_inputs(batch):
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"]
    }

model_inputs = tokenized_docs.map(select_model_inputs, batched=True)

def manual_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    }

dataloader = DataLoader(model_inputs, batch_size=args.batch_size, collate_fn=manual_collate_fn)

config = GPT2Config(vocab_size=len(tokenizer), n_positions=512)
model = GPT2LMHeadModel(config)

if torch.cuda.device_count() >= 1:
    print("running on {} GPUS".format(torch.cuda.device_count()))
    model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
model.to("cuda")

optimiser = "adam_raw"

log_dir = "training/{}/{}/gpu={}_lr={}_batchsize={}_beta2={}_delta={}/tensorboard_logs".format(
    optimiser, args.subsample, torch.cuda.device_count(), args.lr, args.batch_size, args.beta2, args.delta)

checkpoint_dir = "training/{}/{}/gpu={}_lr={}_batchsize={}_beta2={}_delta={}/model_checkpoints".format(
    optimiser, args.subsample, torch.cuda.device_count(), args.lr, args.batch_size, args.beta2, args.delta)

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

num_epochs = 1
model.train()
total_loader_len = len(dataloader)

gradient_accumulation_steps = args.accumulation_steps
ema_loss = None

# Initialize variables for manual Adam
m = {param: torch.zeros_like(param) for param in model.parameters()}
v = {param: torch.zeros_like(param) for param in model.parameters()}
t = 0  # timestep
started = False

import pickle

time_list = []
loss_list = []
ema_loss_list = []

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        if not started:
            start_time = time.time()
            started = True
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss.mean()

        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            t += 1  # Increment timestep

            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is None:
                        continue

                    grad = param.grad.data

                    # Update biased first moment estimate
                    m[param] = args.beta1 * m[param] + (1 - args.beta1) * grad
                    # Update biased second raw moment estimate
                    v[param] = args.beta2 * v[param] + (1 - args.beta2) * grad ** 2

                    # Compute bias-corrected first moment estimate
                    m_hat = m[param] / (1 - args.beta1 ** t)
                    # Compute bias-corrected second raw moment estimate
                    v_hat = v[param] / (1 - args.beta2 ** t)

                    # Update parameters
                    param.data -= args.lr * m_hat / (torch.sqrt(v_hat) + args.delta)

                # Zero the gradients after updating
                model.zero_grad()

            end_time = time.time()
            elapsed_time = end_time - start_time

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = 0.99 * ema_loss + 0.01 * loss.item()

            writer.add_scalar('Time/train', elapsed_time, epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Ema_loss/train', ema_loss, epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
            started = False

            time_list.append(elapsed_time)
            loss_list.append(loss.item())
            ema_loss_list.append(ema_loss)

            if batch_idx % 10 == 0:
                print(f"{(10 * batch_idx) / total_loader_len} complete")
                print(f"{ema_loss}")
                with open(os.path.join(checkpoint_dir, 'training_stats.pkl'), 'ab') as f:
                    pickle.dump({'time': time_list, 'loss': loss_list, 'ema_loss': ema_loss_list}, f)

torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_trained.pt"))
if time_list or loss_list or ema_loss_list:
    with open(os.path.join(checkpoint_dir, 'training_stats.pkl'), 'ab') as f:
        pickle.dump({'time': time_list, 'loss': loss_list, 'ema_loss': ema_loss_list}, f)
