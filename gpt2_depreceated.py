import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
print('subssampling')
subsample_size = int(0.001 * len(dataset['train']))

# Create a random subsample of the dataset
dataset = dataset['train'].shuffle(seed=42).select(range(subsample_size))

from transformers import DataCollatorForLanguageModeling

# Load the dataset
# dataset = load_dataset("wikitext", "wikitext-103-v1", split='train')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# Tokenize the dataset
def tokenize_function(examples):
    # Removed padding=True, truncation=True from here
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, return_tensors="pt", mlm=False)

# DataLoader
train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=4, collate_fn=data_collator)

# Initialize the model
config = GPT2Config(vocab_size=len(tokenizer), n_positions=512)
model = GPT2LMHeadModel(config)
model.to(torch.device("cuda"))

# Initialize the optimizer
optimizer = SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = {k: v.to(torch.device("cuda")) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

# Note: This is a simplified example. For actual training, you'd also want to include validation,
# handle device placement more robustly, and possibly use a more sophisticated optimizer like Adam.
