from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.optim import SGD
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import gpytorch
import gc
import os


# Set CUDA_VISIBLE_DEVICES to all available GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from torch.nn.parallel import DataParallel


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the dataset
ds = load_dataset("wikipedia", "20220301.simple")
model_name = 'gpt2'
# Calculate the size of the subsample (1% of the 'train' split)
subsample_size = int(0.1 * len(ds['train']))

# Create a random subsample of the dataset
subsample = ds['train'].shuffle(seed=42).select(range(subsample_size))

# Load tokenizer for your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Tokenize the subsample
def tokenize_function(examples):
    # Truncation and padding are typically handled here if necessary
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=32)

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

def _bn_train_mode(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.train()

import subprocess

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
    # N = len(loader.dataset)
    # for batch_idx, batch in enumerate(loader):
    #     if cuda:
    #         input_ids = batch["input_ids"].to("cuda")
        #output = model(input)
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    #loss = criterion(output, target)
    loss *= len(input_ids)
    # loss *= input.size()[0] / N

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
        # self.criterion = criterion
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
            # self.criterion,
            cuda=True,
            bn_train_mode=True
        )
        time_diff = time.time() - start_time
        self.iters += 1
        # print('Iter %d. Time: %.2f' % (self.iters, time_diff))
        # return output.cpu().unsqueeze(1)
        return output.unsqueeze(1)

dataloader = DataLoader(model_inputs, batch_size=1, collate_fn=manual_collate_fn)

config = GPT2Config(vocab_size=len(tokenizer), n_positions=32)
model = GPT2LMHeadModel(config)
model.to(torch.device("cuda"))
model = DataParallel(model)
# Initialize TensorBoard SummaryWriter



num_epochs = 1
lr = 0.001
momentum = 0.9
weight_decay = 0
delta = 1

log_dir = "./tensorboard_lanczos_logs"
checkpoint_dir = "./model_lanczos_checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# Initialize momentum buffers for each parameter
momentum_buffers = {}
for param in model.parameters():
    momentum_buffers[param] = torch.zeros_like(param.data)
model.train()
total_loader_len = len(dataloader)

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()
        input_ids = batch["input_ids"].to("cuda")

        # Forward pass
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        P = sum(p.numel() for p in model.parameters())
        grad_vector = torch.cat([grad.view(-1) for grad in gradients])
        # grad_vector.cuda()
        # grad_vector = grad_vector.cuda()
        """kill this extra memory cost"""

        adjusted_grad_vector = grad_vector.clone()  # Initialize adjusted gradient vector

        # grad_vector = grad_vector / torch.norm(grad_vector)

        # Pass the random vector as the initial vector to the CurvVecProduct
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
        #delta = args.lanczos_beta


        # Compute adjustments based on eigenvalues and eigenvectors
        for i, eigval in enumerate(eigvals):
            dot_product = torch.dot(grad_vector, V[i])
            adjustment = (1 / eigval - 1 / (eigval + delta)) * dot_product * V[i]
            adjusted_grad_vector += adjustment
        # Convert the split tensors to the correct shape
        # adjusted_gradients = [g.view(p.size()) for g, p in zip(adjusted_grad_vector, model.parameters())]

        # Calculate the correct splits for 'adjusted_grad_vector' based on model parameters
        split_sizes = [p.numel() for p in model.parameters()]
        # Split 'adjusted_grad_vector' accordingly
        split_gradients = torch.split(adjusted_grad_vector, split_sizes)

        # Now, correctly reshape each split gradient to match the corresponding parameter's shape
        adjusted_gradients = [g.view(p.size()) for g, p in zip(split_gradients, model.parameters())]

        # print('adjust gradient vector {}'.format(adjusted_grad_vector.shape))
        # print('gradient {}'.format(grad_vector.shape))

        # Perform the manual SGD update with momentum, using adjusted gradients
        with torch.no_grad():
            for param, adj_grad in zip(model.parameters(), adjusted_gradients):
                # Compute the 'flattened' version of the weight decay term + adjusted gradient
                weight_decay_term = weight_decay * param.data if weight_decay != 0 else 0
                adjusted_grad_with_weight_decay = adj_grad + weight_decay_term

                # Update the momentum buffer with the adjusted gradient
                if param in momentum_buffers:
                    momentum_buffers[param] = momentum_buffers[param] * momentum + adjusted_grad_with_weight_decay
                else:
                    momentum_buffers[param] = adjusted_grad_with_weight_decay

                # Update the parameter
                param.data -= lr * momentum_buffers[param]

        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time

        # Log the loss
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
        writer.add_scalar('Time/train', elapsed_time, epoch * len(dataloader) + batch_idx)

        if batch_idx % 10 == 0:
            print(f"{(10 * batch_idx) / total_loader_len} complete")
            print(f"Loss: {loss.item()}")

# Close the TensorBoard writer
writer.close()
torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_trained.pt"))