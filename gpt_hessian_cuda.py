import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import torch

# Load the compiled CUDA module
mod = SourceModule("""
extern "C" __global__ void vector_adjust(const float* grad_vector, const float* V, const float* eigvals, float* adjusted_grad_vector, int num_eigenvalues, int vec_len, float delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_len) {
        float adjustment = 0.0f;
        for (int i = 0; i < num_eigenvalues; i++) {
            float dot_product = 0.0f;
            for (int j = 0; j < vec_len; j++) {
                dot_product += grad_vector[j] * V[i * vec_len + j];
            }
            adjustment += (1.0f / eigvals[i] - 1.0f / (eigvals[i] + delta)) * dot_product * V[i * vec_len + idx];
        }
        adjusted_grad_vector[idx] += adjustment;
    }
}
""")

# Get the kernel function from the compiled module
vector_adjust = mod.get_function("vector_adjust")

def cuda_vector_adjust(grad_vector, V, eigvals, adjusted_grad_vector, delta):
    num_eigenvalues = eigvals.size(0)
    vec_len = grad_vector.size(0)

    # Allocate memory on the device
    grad_vector_gpu = grad_vector.contiguous()
    V_gpu = V.contiguous()
    eigvals_gpu = eigvals.contiguous()
    adjusted_grad_vector_gpu = adjusted_grad_vector.contiguous()

    # Define the grid and block dimensions
    block_size = 256
    grid_size = (vec_len + block_size - 1) // block_size

    # Call the kernel
    vector_adjust(
        grad_vector_gpu,
        V_gpu,
        eigvals_gpu,
        adjusted_grad_vector_gpu,
        np.int32(num_eigenvalues),
        np.int32(vec_len),
        np.float32(delta),
        block=(block_size, 1, 1),
        grid=(grid_size, 1)
    )

    return adjusted_grad_vector_gpu
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
        print('Iter %d. Time: %.2f' % (self.iters, time_diff))
        # return output.cpu().unsqueeze(1)
        return output.unsqueeze(1)

dataloader = DataLoader(model_inputs, batch_size=1, collate_fn=manual_collate_fn)

config = GPT2Config(vocab_size=len(tokenizer), n_positions=32)
model = GPT2LMHeadModel(config)
model.to(torch.device("cuda"))
model = DataParallel(model)
# Initialize TensorBoard SummaryWriter



num_epochs = 1
lr = 0.0001
momentum = 0.9
weight_decay = 0.0005
delta = 0.001

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
        input_ids = batch["input_ids"].to("cuda")

        # Forward pass
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        P = sum(p.numel() for p in model.parameters())
        grad_vector = torch.cat([grad.view(-1) for grad in gradients]).cuda()
        adjusted_grad_vector = grad_vector.clone()

        productor = CurvVecProduct(input_ids, model, init_vec=grad_vector)
        lanczos_iters = 10
        Q, T = gpytorch.utils.lanczos.lanczos_tridiag(
            productor,
            max_iter=lanczos_iters,
            dtype=torch.float32,
            device='cuda',
            matrix_shape=(P, P)
        )

        eigvals, eigvects = torch.linalg.eigh(T)
        eigvals = eigvals.cuda()
        V = (eigvects.t() @ Q.t()).cuda()

        # Use CUDA for parallel adjustment computation
        adjusted_grad_vector = cuda_vector_adjust(grad_vector, V, eigvals, adjusted_grad_vector, delta)

        split_sizes = [p.numel() for p in model.parameters()]
        split_gradients = torch.split(adjusted_grad_vector, split_sizes)
        adjusted_gradients = [g.view(p.size()) for g, p in zip(split_gradients, model.parameters())]

        with torch.no_grad():
            for param, adj_grad in zip(model.parameters(), adjusted_gradients):
                weight_decay_term = weight_decay * param.data if weight_decay != 0 else 0
                adjusted_grad_with_weight_decay = adj_grad + weight_decay_term

                if param in momentum_buffers:
                    momentum_buffers[param] = momentum_buffers[param] * momentum + adjusted_grad_with_weight_decay
                else:
                    momentum_buffers[param] = adjusted_grad_with_weight_decay

                param.data -= lr * momentum_buffers[param]

        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
        if batch_idx % 100 == 0:
            print(f"{(100*batch_idx)/total_loader_len} complete")
            print(f"Loss: {loss.item()}")
        del V
        del gradients
        del grad_vector
        del adjustment
        gc.collect()
        print(f"Loss: {loss.item()}")

writer.close()
torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_trained.pt"))
