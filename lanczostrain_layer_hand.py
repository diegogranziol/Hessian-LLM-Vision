import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.optim import SGD, Adam
import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
import os
import gc

# Argument parser setup
parser = argparse.ArgumentParser(description='Script for training GPT-2 with custom arguments.')
parser.add_argument('--gpus', type=str, help='GPUs to use', default="0,1,2,3")
parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
parser.add_argument('--subsample', type=float, help='Subsample fraction', default=0.1)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)
parser.add_argument('--momentum', type=float, help='SGD momentum', default=0.9)
parser.add_argument('--optimiser', type=str, help='Optimizer to use', default='sgd')
parser.add_argument('--num_epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--log_dir', type=str, help='Directory for TensorBoard logs', default='./tensorboard_lanczos_logs')
parser.add_argument('--checkpoint_dir', type=str, help='Directory for model checkpoints', default='./model_lanczos_checkpoints')
parser.add_argument('--delta', type=float, help='Delta value for Lanczos adjustments', default=0.001)
parser.add_argument('--max_length', type=int, help='Max token length for tokenizer', default=32)
args = parser.parse_args()


# Set CUDA_VISIBLE_DEVICES to all available GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

from torch.nn.parallel import DataParallel

# Load the dataset
ds = load_dataset("wikipedia", "20220301.simple")
model_name = 'gpt2'
# Calculate the size of the subsample
subsample_size = int(args.subsample * len(ds['train']))

# Create a random subsample of the dataset
subsample = ds['train'].shuffle(seed=42).select(range(subsample_size))

# Load tokenizer for your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the subsample
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)

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

def _bn_train_mode(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.train()

def hess_vec(vector, input_ids, model, layer_idx, device, cuda=True, bn_train_mode=False):
    param = list(model.parameters())[layer_idx]
    vector = vector.detach().view_as(param).to(param.device)

    model.eval()
    if bn_train_mode:
        model.apply(_bn_train_mode)

    model.zero_grad()
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss *= len(input_ids)

    grad = torch.autograd.grad(loss, param, create_graph=True)[0]
    dL_dvec = torch.sum(vector * grad)
    dL_dvec.backward()

    return param.grad.view(-1)


dataloader = DataLoader(model_inputs, batch_size=args.batch_size, collate_fn=manual_collate_fn)

config = GPT2Config(vocab_size=len(tokenizer), n_positions=args.max_length)
model = GPT2LMHeadModel(config)
model.to(torch.device("cuda"))
model = DataParallel(model)

# Ensure all model parameters require gradients
for param in model.parameters():
    param.requires_grad = True

# Initialize TensorBoard SummaryWriter
os.makedirs(args.log_dir, exist_ok=True)
checkpoint_dir = f"./checkpoints/layerwise_bs={args.batch_size}_subsample={args.subsample}_lr={args.lr}_momentum={args.momentum}_optim={args.optimiser}_epochs={args.num_epochs}"
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(args.log_dir)

# Initialize momentum buffers for each parameter
momentum_buffers = {param: torch.zeros_like(param.data) for param in model.parameters()}
model.train()
total_loader_len = len(dataloader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(args.num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda")

        # Forward pass
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        with torch.no_grad():
            for layer_idx, grad in enumerate(gradients):
                param = list(model.parameters())[layer_idx]
                P = param.numel()
                grad_vector = grad.view(-1)

                adjusted_grad_vector = grad_vector.clone()  # Initialize adjusted gradient vector

                random_vec = torch.randn(P, device=device)
                random_vec = random_vec / torch.norm(random_vec, 2)

                lanczos_iters = 10

                start_start = time()

                T = torch.zeros([lanczos_iters + 1, lanczos_iters + 1], device=device)
                q_old = torch.zeros_like(random_vec, device=device)  # Ensure q_old is a vector of the same size as random_vec

                Q = torch.zeros((lanczos_iters + 1, P), device=device)  # Pre-allocate Q
                v = random_vec.clone()
                q_old = q_old.to(device)
                Q[0] = v

                # w = Hess_Vec(M, v)
                w = hess_vec(v, input_ids, model, layer_idx, device)
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
                    Q[i + 1] = v  # Store the new vector in Q

                    # w = Hess_Vec(M, v)
                    w = hess_vec(v, input_ids, model, layer_idx, device)
                    w = w.to(device)  # Ensure w is on the correct device

                    alpha = torch.dot(w, v)
                    T[i + 1, i + 1] = alpha
                    w -= (alpha * v + b * v_old)
                    v_old = v

                    # print("step", i + 1)
                    # print("Time:", time() - start_time)

                eigvals, eigvects = torch.linalg.eigh(T)
                gammas = eigvects[0, :] ** 2
                V = eigvects.t() @ Q
                delta = args.delta
                maxeig = torch.max(eigvals)
                mineig = torch.min(torch.abs(eigvals))
                print(f"layer {layer_idx} max {maxeig} min {mineig}")

                # Compute adjustments based on eigenvalues and eigenvectors
                for i, eigval in enumerate(eigvals):
                    dot_product = torch.dot(grad_vector, V[i])
                    adjustment = (1 / eigval - 1 / (eigval + delta)) * dot_product * V[i]
                    adjusted_grad_vector += adjustment

                adjusted_gradient = adjusted_grad_vector.view(param.size())

                # Perform the manual SGD update with momentum, using adjusted gradients
                weight_decay_term = 0.0005 * param.data if 0.0005 != 0 else 0
                adjusted_grad_with_weight_decay = adjusted_gradient + weight_decay_term

                if param in momentum_buffers:
                    momentum_buffers[param] = momentum_buffers[param] * args.momentum + adjusted_grad_with_weight_decay
                else:
                    momentum_buffers[param] = adjusted_grad_with_weight_decay

                param.data -= args.lr * momentum_buffers[param]

        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)

        print(f"Loss: {loss.item()}")

writer.close()
torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_trained.pt"))
