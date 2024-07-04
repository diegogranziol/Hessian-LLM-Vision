from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
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
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.max_length)

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

# config = GPT2Config(vocab_size=len(tokenizer), n_positions=512)
# model = GPT2LMHeadModel(config)

save_folder = args.checkpoint.split("/")[:-1]
save_name = args.checkpoint.split("/")[-1]
model_state_dict = torch.load(args.checkpoint, map_location=torch.device('cuda'))
# model.load_state_dict(model_state_dict)
#
# Load the configuration from the checkpoint or from the model name
config = GPT2Config.from_pretrained(model_name, n_positions=512)  # Ensure this matches the training configuration


# Create the model with the loaded configuration
model = GPT2LMHeadModel(config)
num_gpus = torch.cuda.device_count()

# Adapt the state dictionary keys if necessary
new_state_dict = {}
for k, v in model_state_dict.items():
    if k.startswith('module.'):
        k = k[len('module.'):]
    new_state_dict[k] = v
model_state_dict = new_state_dict

# Load the state dictionary into the model
model.load_state_dict(model_state_dict)

# # Check if the model state dict needs to be adapted for DataParallel
# if num_gpus > 1:
#     model = torch.nn.DataParallel(model)
#     # Adapt the state dictionary keys for DataParallel
#     new_state_dict = {}
#     for k, v in model_state_dict.items():
#         if not k.startswith('module.'):
#             k = 'module.' + k
#         new_state_dict[k] = v
#     model_state_dict = new_state_dict
# else:
#     # Remove 'module.' prefix if present in the state dict
#     new_state_dict = {}
#     for k, v in model_state_dict.items():
#         if k.startswith('module.'):
#             k = k[len('module.'):]
#         new_state_dict[k] = v
#     model_state_dict = new_state_dict

# Load the state dictionary into the model



num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    print("running on multiple (i.e. {}) GPUs".format(num_gpus))
    from torch.nn import DataParallel
    model = DataParallel(model, device_ids=list(range(num_gpus)))
else:
    print("running on a single GPU")
model.to("cuda")


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
        return output.unsqueeze(1)

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
    device='cuda',
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
save_folder = "/".join(save_folder)
print(f"save {save_folder}")
relevant_folder = "subsample={}_iters={}_basis={}".format(str(args.subsample),str(args.lanczos_iters),str(args.basis))
if os.path.exists('{}/{}'.format(save_folder,relevant_folder)):
    pass
else:
    os.makedirs('{}/{}'.format(save_folder,relevant_folder))

# Save the result dictionary

torch.save(result, '{}/{}/{}.ckpt'.format(save_folder,relevant_folder,save_name))
print('{}/{}/{}.ckpt'.format(save_folder,relevant_folder,save_name))