import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.optim import SGD, Adam
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import gpytorch
import gc
import os

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

def hess_vec(vector, input_ids, model, cuda=True, bn_train_mode=False):
    param_list = list(model.parameters())
    vector_list = []

    offset = 0
    for param in param_list:
        vector_list.append(vector[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    model.eval()
    if bn_train_mode:
        model.apply(_bn_train_mode)

    model.zero_grad()
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss *= len(input_ids)

    grad_list = torch.autograd.grad(loss, param_list, create_graph=True)
    dL_dvec = torch.zeros(1, device='cuda' if cuda else 'cpu')
    for v, g in zip(vector_list, grad_list):
        dL_dvec += torch.sum(v * g)
    dL_dvec.backward()

    model.eval()
    return torch.cat([param.grad.view(-1) for param in param_list]).view(-1)

class CurvVecProduct(object):
    def __init__(self, input_ids, model, init_vec=None):
        self.input_ids = input_ids
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
            self.input_ids,
            self.model,
            cuda=True,
            bn_train_mode=True
        )
        time_diff = time.time() - start_time
        self.iters += 1
        print('Iter %d. Time: %.2f' % (self.iters, time_diff))
        return output.unsqueeze(1)

dataloader = DataLoader(model_inputs, batch_size=args.batch_size, collate_fn=manual_collate_fn)

config = GPT2Config(vocab_size=len(tokenizer), n_positions=args.max_length)
model = GPT2LMHeadModel(config)
model.to(torch.device("cuda"))
model = DataParallel(model)

# Initialize TensorBoard SummaryWriter
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.checkpoint_dir, exist_ok=True)
writer = SummaryWriter(args.log_dir)

# Initialize momentum buffers for each parameter
momentum_buffers = {param: torch.zeros_like(param.data) for param in model.parameters()}
model.train()
total_loader_len = len(dataloader)

for epoch in range(args.num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda")

        # Forward pass
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        P = sum(p.numel() for p in model.parameters())
        grad_vector = torch.cat([grad.view(-1) for grad in gradients])

        adjusted_grad_vector = grad_vector.clone()  # Initialize adjusted gradient vector

        productor = CurvVecProduct(input_ids, model, init_vec=grad_vector)

        # Run the Lanczos algorithm
        lanczos_iters = 10
        Q, T = gpytorch.utils.lanczos.lanczos_tridiag(
            productor,
            max_iter=lanczos_iters,
            dtype=torch.float32,
            device='cuda',
            matrix_shape=(P, P)
        )

        eigvals, eigvects = torch.linalg.eigh(T)
        gammas = eigvects[0, :] ** 2
        V = eigvects.t() @ Q.t()
        delta = args.delta

        # Compute adjustments based on eigenvalues and eigenvectors
        for i, eigval in enumerate(eigvals):
            dot_product = torch.dot(grad_vector, V[i])
            adjustment = (1 / eigval - 1 / (eigval + delta)) * dot_product * V[i]
            adjusted_grad_vector += adjustment

        split_sizes = [p.numel() for p in model.parameters()]
        split_gradients = torch.split(adjusted_grad_vector, split_sizes)

        adjusted_gradients = [g.view(p.size()) for g, p in zip(split_gradients, model.parameters())]

        # Perform the manual SGD update with momentum, using adjusted gradients
        with torch.no_grad():
            for param, adj_grad in zip(model.parameters(), adjusted_gradients):
                weight_decay_term = 0.0005 * param.data if 0.0005 != 0 else 0
                adjusted_grad_with_weight_decay = adj_grad + weight_decay_term

                if param in momentum_buffers:
                    momentum_buffers[param] = momentum_buffers[param] * args.momentum + adjusted_grad_with_weight_decay
                else:
                    momentum_buffers[param] = adjusted_grad_with_weight_decay

                param.data -= args.lr * momentum_buffers[param]

        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)

        print(f"Loss: {loss.item()}")

writer.close()
torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"model_trained.pt"))
