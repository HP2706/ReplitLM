
import einops
import torch
import torch.nn as nn
import numpy as np
import math
import tqdm.auto as tqdm
import numpy as np
from datasets import load_dataset
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from replitLM_spec.modeling_mpt import MPTModel
from transformers import DataCollatorForLanguageModeling
import multiprocessing as mp
from huggingface_hub import login, HfApi, Repository
from pynvml import *
from torch.nn.utils.rnn import pad_sequence
from google.cloud import storage
from io import BytesIO


def upload_blob(model, destination_blob_name, bucket_name='replit-code-bucket'):
    """Upload a PyTorch model directly from memory to a GCS bucket.
    
    Args:
    model (torch.nn.Module): The PyTorch model to upload.
    destination_blob_name (str): The destination blob name in the GCS bucket.
    bucket_name (str, optional): The name of the GCS bucket. Defaults to 'replit-code-bucket'.
    """
    
    # Instantiate a Google Cloud Storage client
    storage_client = storage.Client()
    
    # Get the GCS bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create a blob in the bucket
    blob = bucket.blob(destination_blob_name)

    # Save the PyTorch model to a bytes buffer
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    
    # Upload the bytes buffer content to GCS
    blob.upload_from_file(buffer, content_type='application/octet-stream')

    print(f'Model uploaded to {destination_blob_name}.')




def upload_huggingface(model, version:int):
    login("API_TOKEN") # change this!
    api = HfApi()
    with Repository("torch-model", clone_from="<user>/torch-model", token=True).commit(commit_message="My cool model :)"):
        torch.save(model.state_dict(), f"model_{version}.pt")

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

#!nvidia-smi


#downloading dataset

t0 = time.time()
token = "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"

dataset_name = "codeparrot/github-jupyter-code-to-text"

if dataset_name == "bigcode/the-stack-dedup":
    # full dataset (3TB of data)
    num_workers = 1
    dataset_train_raw = load_dataset("bigcode/the-stack-dedup", split = "train", token=token, streaming = True)
    test_dataset_raw = load_dataset("bigcode/the-stack-dedup", split = "test", token=token, streaming = True)
else:
    # small dataset
    num_workers = mp.cpu_count()
    dataset_train_raw = load_dataset("codeparrot/github-jupyter-code-to-text", split="train", token=token, streaming = True, num_workers=num_workers)
    test_dataset_raw = load_dataset("codeparrot/github-jupyter-code-to-text", split="test", token=token, streaming = True, num_workers=num_workers)
print(time.time() - t0)


tokenizer = AutoTokenizer.from_pretrained("replit/replit-code-v1-3b", trust_remote_code=True)

def encode(x):
    # Extract the 'content' field
    content = x['content']
    tokenization = tokenizer(content, padding=True, truncation=True, max_length=2048)
    print("len", len(tokenization['input_ids']))
    if 'labels' in x:
        tokenization['labels'] = x['labels']
        print("LABELS FOUND")

    # Tokenize the 'content'
    return tokenization


dataset_train = dataset_train_raw.map(encode, batched=True, num_proc=num_workers)
dataset_train = dataset_train.with_format(type='torch')
dataset_test = test_dataset_raw.map(encode, batched=True, num_proc=num_workers)
dataset_test = dataset_test.with_format(type='torch')

def collate_func(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


BATCH_SIZE = 64

train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, collate_fn=collate_func, num_workers=num_workers)
test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, collate_fn=collate_func, num_workers=num_workers)
# Load model directly
print_gpu_utilization()
model = MPTModel.from_pretrained("replit/replit-code-v1-3b")
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise ValueError('No GPU found, please run with --cuda')
model.to(device)
print(model.device)
print_gpu_utilization()
print(model.get_cc())



t0 = time.time()

steps = int(1000)
log = 100
lamb = 1e-3
swap_log = int(1e6) #1000
plot_log = 1000

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise ValueError('No GPU found, please run with --cuda')

model.to(device)

# Create an iterator for the training data
train_data_iter = iter(train_dataloader)

for step in tqdm(range(steps)):
    if step == int(steps * 1 / 4):
        lamb *= 10
    if step == int(steps * 3 / 4):
        lamb *= 10

    # Get the next batch of training data
    try:
        batch = next(train_data_iter)
    except StopIteration:
        # If the iterator runs out of data (end of epoch), create a new iterator to restart from the beginning of the dataset
        train_data_iter = iter(train_dataloader)
        batch = next(train_data_iter)

    # Move the batch to the GPU if available
    token_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_ids=token_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Compute the loss
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(logits, token_ids)
    cc = model.get_cc(no_penalize_last=False)
    total_loss = loss + lamb * cc
    total_loss.backward()

    # Update the parameters
    optimizer.step()

    # Compute the test loss, but not for every training step to save computation
    if step % log == 0:
        # Switch to evaluation mode for testing
        model.eval()
        with torch.no_grad():
            # Assume test_dataloader is defined similarly to train_dataloader
            test_batch = next(iter(test_dataloader))
            test_token_ids = test_batch['input_ids'].to(device)
            test_attention_mask = test_batch['attention_mask'].to(device)

            # Forward pass on test data
            test_outputs = model(input_ids=test_token_ids, attention_mask=test_attention_mask)
            test_logits = test_outputs.logits
            
            # Compute the test loss
            test_loss = loss_function(test_logits, test_token_ids)

        # Switch back to training mode
        model.train()
        
        print("step = %d | train loss: %.2e | train last: %.2e | test loss %.2e | test last: %.2e | cc: %.2e " %
              (step, loss.detach().cpu().numpy(), total_loss.detach().cpu().numpy(), 
               test_loss.detach().cpu().numpy(), cc.detach().cpu().numpy()))

    if (step+1) % swap_log == 0:
        model.relocate()

print("took in total: ", time.time() - t0)
model.save_pretrained("replit/replit-code-v1-3b-sparse")
