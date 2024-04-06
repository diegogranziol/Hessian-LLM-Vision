from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.optim import SGD
import torch
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter instance (logs will be saved to "./runs" by default)
writer = SummaryWriter()


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

def _bn_train_mode(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.train()

def hess_vec(vector, loader, model, criterion, cuda=True, bn_train_mode=False):
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
    N = len(loader.dataset)
    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)
        loss *= input.size()[0] / N

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)
        dL_dvec = torch.zeros(1, device='cuda' if cuda else 'cpu')
        for v, g in zip(vector_list, grad_list):
            dL_dvec += torch.sum(v * g)
        dL_dvec.backward()

    model.eval()
    return torch.cat([param.grad.view(-1) for param in param_list]).view(-1)



class CurvVecProduct(object):
    def __init__(self, loader, model, criterion, init_vec=None):
        self.loader = loader
        self.model = model
        self.criterion = criterion
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
            self.criterion,
            cuda=True,
            bn_train_mode=True
        )
        time_diff = time.time() - start_time
        self.iters += 1
        # print('Iter %d. Time: %.2f' % (self.iters, time_diff))
        # return output.cpu().unsqueeze(1)
        return output.unsqueeze(1)

dataloader = DataLoader(model_inputs, batch_size=16, collate_fn=manual_collate_fn)

config = GPT2Config(vocab_size=len(tokenizer), n_positions=512)
model = GPT2LMHeadModel(config)
model.to(torch.device("cuda"))
# Initialize TensorBoard SummaryWriter



num_epochs = 1
lr = 0.001  # Learning rate
model.train()

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda")

        # Forward pass
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        # Manually zero the gradients of all parameters
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Backward pass to compute gradients
        loss.backward()

        # Manual SGD update
        with torch.no_grad():  # Ensure gradients are not computed for the update step
            for param in model.parameters():
                if param.grad is not None:
                    param -= lr * param.grad  # Update parameters

        # Log the loss
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)

        print(f"Loss: {loss.item()}")

# Close the TensorBoard writer
writer.close()