import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import gpytorch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

# Load the dataset and model
ds = load_dataset("wikipedia", "20220301.simple")
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config(vocab_size=len(tokenizer), n_positions=512)
model = GPT2LMHeadModel(config)

# Load the state dictionary from your checkpoint file
model_state_dict = torch.load("model_checkpoints/model_trained.pt", map_location=torch.device('cuda'))
model.load_state_dict(model_state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Function to prepare data
def prepare_data_loader(dataset, tokenizer, batch_size=16):
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_docs = dataset.map(tokenize_function, batched=True)
    model_inputs = tokenized_docs.map(lambda x: {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']}, batched=True)
    return DataLoader(model_inputs, batch_size=batch_size, collate_fn=lambda x: {'input_ids': torch.tensor([y['input_ids'] for y in x], device=device), 'attention_mask': torch.tensor([y['attention_mask'] for y in x], device=device)})
# Define the Hessian-vector product function
def hess_vec_product(model, dataset, vector, batch_size=1):
    dataloader = prepare_data_loader(dataset, tokenizer, batch_size)
    accumulated_hvp = torch.zeros_like(vector)

    for batch in dataloader:
        model.zero_grad()
        outputs = model(**batch, labels=batch['input_ids'].to(model.device))
        loss = outputs.loss
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_flat = torch.cat([g.contiguous().view(-1) for g in grads])
        grad_vector_product = torch.dot(grad_flat, vector)
        hvp = torch.autograd.grad(grad_vector_product, model.parameters(), retain_graph=True)
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        accumulated_hvp += hvp_flat.detach()
    print('Hessian vector product calculated')
    return accumulated_hvp / len(dataloader)

# Prepare the random vector for the power iteration
P = sum(p.numel() for p in model.parameters())
h_vector = torch.randn(P, device=model.device)
h_vector /= torch.norm(h_vector)

# Compute the Hessian-vector product
hessian_vector_product = hess_vec_product(model, ds['train'].select(range(int(0.001 * len(ds['train'])))), h_vector)

# Lanczos algorithm to compute eigenvalues
lanczos_iters = 30
Q, T = gpytorch.utils.lanczos.lanczos_tridiag(
    lambda v: hess_vec_product(model, ds['train'].select(range(int(0.001 * len(ds['train'])))), v),
    max_iter=lanczos_iters,
    dtype=torch.float32,
    device=model.device,
    matrix_shape=(P, P)
)

# Calculate eigenvalues and eigenvectors
eigvals, eigvects = torch.linalg.eigh(T)
gammas = eigvects[0, :] ** 2
#V = eigvects.t() @ Q.t()

# Output results
print("Eigenvalues:", eigvals.cpu().numpy())

# Optionally save to disk, for example using numpy
import numpy as np
np.savez("hessian_analysis.npz", eigvals=result['eigvals'], gammas=result['gammas'], V=result['V'])
