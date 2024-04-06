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

dataloader = DataLoader(model_inputs, batch_size=16, collate_fn=manual_collate_fn)

config = GPT2Config(vocab_size=len(tokenizer), n_positions=512)
model = GPT2LMHeadModel(config)
model.to(torch.device("cuda"))
optimizer = SGD(model.parameters(), lr=0.001)

num_epochs = 1
model.train()

for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to("cuda")
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)

        print(f"Loss: {loss.item()}")