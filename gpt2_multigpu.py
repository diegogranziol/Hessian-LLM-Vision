from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
from torch.optim import SGD
from torch.optim import Adam
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
parser = argparse.ArgumentParser(description='Example script to demonstrate argparse usage.')
# parser.add_argument('--gpus', type=int,  help='A list of numbers', default=1)
parser.add_argument('--batch_size', type=int, help='A list of numbers', default=128)
parser.add_argument('--subsample', type=float, help='A list of numbers', default=1)
parser.add_argument('--lr', type=float, help='A list of numbers', default=1)
parser.add_argument('--momentum', type=float, help='A list of numbers', default=0.9)
parser.add_argument('--optimiser', type=str, help='A list of numbers', default='sgd')
args = parser.parse_args()

# Import necessary for Data Parallel
from torch.nn import DataParallel

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the dataset

ds = load_dataset("wikipedia", "20220301.simple")
model_name = 'gpt2'
# Calculate the size of the subsample (1% of the 'train' split)
subsample_size = int(args.subsample * len(ds['train']))

# Create a random subsample of the dataset
subsample = ds['train'].shuffle(seed=42).select(range(subsample_size))

# Load tokenizer for your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Tokenize the subsample
def tokenize_function(examples):
    # Truncation and padding are typically handled here if necessary
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_docs = subsample.map(tokenize_function, batched=True)

from torch.utils.data import DataLoader

def select_model_inputs(batch):
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"]
    }

# Apply the function to filter out only the necessary fields
model_inputs = tokenized_docs.map(select_model_inputs, batched=True)

# Manually collate a batch
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

# Wrap the model for Data Parallel on 4 GPUs
if torch.cuda.device_count() >= 1:
    print("running on {} GPUS".format(torch.cuda.device_count()))
    model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
model.to("cuda")

optimiser = args.optimiser.lower()
print("optimiser chosen = {}".format(optimiser))
if optimiser == 'sgd':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
elif optimiser == 'adam':
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999), eps=1e-8)
else:
    print("failed optimiser")


# Specify the directory to save TensorBoard logs and model checkpoints
log_dir = "training/{}/{}/gpu={}_lr={}_batchsize={}/tensorboard_logs".format(
    optimiser, args.subsample, torch.cuda.device_count(), args.lr, args.batch_size)

checkpoint_dir = "training/{}/{}/gpu={}_lr={}_batchsize={}/model_checkpoints".format(
    optimiser, args.subsample, torch.cuda.device_count(), args.lr, args.batch_size)

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_untrained.pt"))

num_epochs = 1
model.train()
total_loader_len = len(dataloader)

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss.mean()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
        if batch_idx % 100 == 0:
            print(f"{(100*batch_idx)/total_loader_len} complete")
            print(f"Loss: {loss.item()}")

torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_trained.pt"))
