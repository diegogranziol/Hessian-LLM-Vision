from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.optim import SGD
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import gpytorch
import argparse
parser = argparse.ArgumentParser(description='Example script to demonstrate argparse usage.')
parser.add_argument('--subsample', type=float, help='A list of numbers', default=1)
parser.add_argument('--basis', type=bool, help='A list of numbers', default=False)
parser.add_argument('--lanczos_iters', type=int, help='A list of numbers', default=30)
parser.add_argument('--batch_size', type=int, help='A list of numbers', default=16)
parser.add_argument('--max_length', type=int, help='A list of numbers', default=512)
parser.add_argument('--checkpoint', type=str, help='A list of numbers', default="model_checkpoints/model_trained.pt")
args = parser.parse_args()



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the dataset
class RandomNoiseDataset(Dataset):
    def __init__(self, num_samples, max_length, vocab_size):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            max_length (int): The maximum length of the input tensors.
            vocab_size (int): The size of the vocabulary (used to generate input_ids).
        """
        self.num_samples = num_samples
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random input_ids
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.max_length,), dtype=torch.long)

        # Generate a random attention mask (binary mask)
        attention_mask = torch.randint(low=0, high=2, size=(self.max_length,), dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
# Load tokenizer for your model
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Tokenize the subsample
def tokenize_function(examples):
    # Truncation and padding are typically handled here if necessary
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)

#tokenized_docs = subsample.map(tokenize_function, batched=True)

from torch.utils.data import DataLoader


class RandomNoiseDataset(Dataset):
    def __init__(self, num_samples, max_length, vocab_size):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            max_length (int): The maximum length of the input tensors.
            vocab_size (int): The size of the vocabulary (used to generate input_ids).
        """
        self.num_samples = num_samples
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random input_ids
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.max_length,), dtype=torch.long)

        # Generate a random attention mask (binary mask)
        attention_mask = torch.randint(low=0, high=2, size=(self.max_length,), dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
# Example parameters
num_samples = 1000  # Number of random samples you want to generate
max_length = 512  # Maximum length of the input (should match your model's expected input size)
vocab_size = 50257  # For GPT-2's vocabulary size

# Initialize the dataset
random_noise_dataset = RandomNoiseDataset(num_samples=int(100000*args.subsample), max_length=max_length, vocab_size=vocab_size)

# Initialize the DataLoader
dataloader = DataLoader(random_noise_dataset, batch_size=args.batch_size, shuffle=True)
config = GPT2Config(vocab_size=len(tokenizer), n_positions=512)
#model = GPT2LMHeadModel(config)

# Load the model from the checkpoint
# Load the checkpoint
checkpoint = torch.load(args.checkpoint, map_location="cpu")

# Assuming the architecture is GPT-2 and you want to load it directly
# Note: This step initializes a model with the default GPT-2 configuration.
# If your checkpoint was trained with a different configuration, you might need to adjust this step.
model = GPT2LMHeadModel.from_pretrained("gpt2", state_dict=checkpoint)

# Move the model to GPU if available
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Now the model is ready to be used






num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    print("running on multiple (i.e. {}) GPUs".format(num_gpus))
    from torch.nn import DataParallel
    model = DataParallel(model, device_ids=list(range(num_gpus)))
else:
    print("running on a single GPU")
model.to("cuda")

save_folder = args.checkpoint.split("/")[:-1]
save_name = args.checkpoint.split("/")[-1]
model_state_dict = torch.load(args.checkpoint, map_location=torch.device('cuda'))
model.load_state_dict(model_state_dict)
def _bn_train_mode(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.train()

def hess_vec(vector, dataloader, model, cuda=True, bn_train_mode=False):
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
    N = len(dataloader.dataset)
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda")

        # Forward pass
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        if num_gpus > 1:
            loss = loss.mean()
        """What is the correct scaling here?"""
        loss *= len(batch) / N

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)
        dL_dvec = torch.zeros(1, device='cuda' if cuda else 'cpu')
        for v, g in zip(vector_list, grad_list):
            dL_dvec += torch.sum(v * g)
        dL_dvec.backward()

    model.eval()
    return torch.cat([param.grad.view(-1) for param in param_list]).view(-1)



class CurvVecProduct(object):
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
            cuda=True,
            bn_train_mode=True
        )
        time_diff = time.time() - start_time
        self.iters += 1
        print('Iter %d. Time: %.2f' % (self.iters, time_diff))
        return output.cpu().unsqueeze(1)

print('running Lanczos on original training set model')

P = sum(p.numel() for p in model.parameters())
random_vec = torch.randn(P, device='cuda')
random_vec = random_vec / torch.norm(random_vec)

# Pass the random vector as the initial vector to the CurvVecProduct
productor = CurvVecProduct(dataloader, model, init_vec=random_vec)

# Run the Lanczos algorithm
lanczos_iters = args.lanczos_iters
Q, T = gpytorch.utils.lanczos.lanczos_tridiag(
    productor,
    max_iter=lanczos_iters,
    dtype=torch.float32,
    device='cpu',
    matrix_shape=(P, P)
)

eigvals, eigvects = torch.linalg.eigh(T)
eigvals = eigvals

gammas = eigvects[0, :] ** 2
V = eigvects.t() @ Q.t()

# Save or return the results as needed
result = {
    # 'w': w,
    'eigvals': eigvals,
    'gammas': gammas,
}
if args.basis:
    result['V']: V
print(eigvals)
relevant_folder = "subsample={}_iters={}_basis={}_noise".format(str(args.subsample),str(args.lanczos_iters),str(args.basis))
if os.path.exists('{}/{}'.format(save_folder,relevant_folder)):
    pass
else:
    os.makedirs('{}/{}'.format(save_folder,relevant_folder))

# Save the result dictionary
torch.save(result, '{}/{}/{}.ckpt'.format(save_folder,relevant_folder,save_name))