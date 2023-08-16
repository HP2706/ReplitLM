
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
from functools import partial
from io import BytesIO
import wandb
from torch.utils.data import IterableDataset
from replitLM_spec.configuration_mpt import MPTConfig



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
    login(os.environ.get("HUGGINGFACE_API_KEY")) # change this!
    with Repository("replit-code-model", clone_from="<user>/torch-model", token=True).commit(commit_message="My cool model :)"):
        torch.save(model.state_dict(), f"model_{version}.pt")

def gpu_utilization():
    """Get the current gpu memory usage.
    
    Returns:
        dict: The keys are 'used_memory (MiB)', 'total_memory (MiB)', and 'utilization (%)'.
    """
    # Initialize NVML
    nvmlInit()
    
    # Get GPU handle
    handle = nvmlDeviceGetHandleByIndex(0)
    
    # Get GPU memory info
    info = nvmlDeviceGetMemoryInfo(handle)
    
    # Calculate the memory usage percentage
    utilization = (info.used / info.total) * 100
    
    # Shutdown the NVML
    nvmlShutdown()
    
    # Return the memory info in a dictionary
    return {
        'used_memory (MiB)': info.used // 1024**2,
        'total_memory (MiB)': info.total // 1024**2,
        'utilization (%)': utilization
    }
#!nvidia-smi


#downloading dataset

def encode(x, tokenizer, config):
    # Extract the 'content' field
    content = x['content']
    tokenization = tokenizer(content, padding=True, truncation=True, max_length=config['max_seq_len'])
    print("len", len(tokenization['input_ids']))
    if 'labels' in x:
        tokenization['labels'] = x['labels']
        print("LABELS FOUND")

    # Tokenize the 'content'
    return tokenization


def collate_func(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}



def create_dataset(config, BATCH_SIZE = 16):
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
        num_workers =1 # mp.cpu_count()
        dataset_train_raw = load_dataset("codeparrot/github-jupyter-code-to-text", split="train", token=token, streaming = True)
        test_dataset_raw = load_dataset("codeparrot/github-jupyter-code-to-text", split="test", token=token, streaming = True)
    print(time.time() - t0)


    tokenizer = AutoTokenizer.from_pretrained("replit/replit-code-v1-3b", trust_remote_code=True)
    encode_partial = partial(encode, tokenizer=tokenizer, config=config)
    if isinstance(dataset_train_raw, IterableDataset):
        dataset_train = dataset_train_raw.map(encode_partial, batched=True)
        dataset_test = test_dataset_raw.map(encode_partial, batched=True)
     
    else: 
        print("type of dataset is not iterable", type(dataset_train_raw))
        dataset_train = dataset_train_raw.map(encode_partial, batched=True)
        dataset_test = test_dataset_raw.map(encode_partial, batched=True)
         
    dataset_train = dataset_train.with_format(type='torch')
    dataset_test = dataset_test.with_format(type='torch')
    
    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, collate_fn=collate_func)
    test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, collate_fn=collate_func)
    return train_dataloader, test_dataloader



def train(config, train_dataloader, test_dataloader):

    wandb.login(key = os.environ.get("WANDB_API_KEY"))
    wandb.init(
        # set the wandb project where this run will be logged
        project="bimt-pruning-replit-code-model",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-3,
        "architecture": "transformer",
        "dataset": "codeparrot/github-jupyter-code-to-text",
        "epochs": 10,
        }
    )

    #model = MPTModel.from_pretrained("replit/replit-code-v1-3b")
    mpt_config = MPTConfig(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        expansion_ratio=config['expansion_ratio'],
        max_seq_len=config['max_seq_len'],
        vocab_size=config['vocab_size'],
        resid_pdrop=config['resid_pdrop'],
        emb_pdrop=config['emb_pdrop'],
        learned_pos_emb=config['learned_pos_emb'],
        attn_config=config['attn_config'],
        init_device=config['init_device'],
        logit_scale=config['logit_scale'],
        no_bias=config['no_bias'],
        verbose=config['verbose'],
        embedding_fraction=config['embedding_fraction'],
        norm_type=config['norm_type'],
        use_cache=config['use_cache'],
        init_config=config['init_config']
    )
    
    model = MPTModel(mpt_config)

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda')
    else:
        raise ValueError('No GPU found, please run with --cuda')
        
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.to(device)
    print(model.device)
    print(model.get_cc())

    t0 = time.time() 

    steps = int(100)
    log = 5
    lamb = 1e-3
    swap_log = int(1e6) #1000
    plot_log = 1000
    version = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3*n_gpu)

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
        logits, outputs = model(input_ids=token_ids, attention_mask=attention_mask)
        # Compute the loss
        loss_function = torch.nn.CrossEntropyLoss()
        
        logits = logits.view(-1, logits.shape[-1])  # reshape to [16*50, 32768]
        token_ids = token_ids.view(-1)  # reshape to [16*50]        
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
                test_logits, test_outputs = model(input_ids=test_token_ids, attention_mask=test_attention_mask)
                # Compute the test loss
                cc = model.get_cc(no_penalize_last=False)
                test_logits = test_logits.view(-1, test_logits.shape[-1])  # reshape to [16*50, 32768]
                test_token_ids = test_token_ids.view(-1) 
                test_loss = loss_function(test_logits, test_token_ids)
                total_test_loss = test_loss + lamb * cc
                wandb.log({"test_loss": test_loss, "total_test_loss": total_test_loss})
                
            # Switch back to training mode
            model.train()
            
            print(f"step = {step}")
            print(f"train loss = {loss.detach().cpu().numpy()}")
            print(f"total train loss = {total_loss.detach().cpu().numpy()}")
            print(f"test loss = {test_loss.detach().cpu().numpy()}")
            print(f"total test loss = {total_test_loss.detach().cpu().numpy()}")
            print(f"connection cost = {cc.detach().cpu().numpy()}")
            print("gpu_utilization", gpu_utilization())
            
        wandb.log({
        "Step": step, 
        "Train Loss": loss.detach().cpu().numpy(), 
        "Total Train Loss": total_loss.detach().cpu().numpy(), 
        "Test Loss": test_loss.detach().cpu().numpy() , 
        "Connection Cost": cc.detach().cpu().numpy(),
        "gpu_utilization": gpu_utilization(),
        })

        if (step+1) % swap_log == 0:
            model.relocate()
            
        if steps% (steps/2):
            upload_blob(model, f"replit/replit-code-v1-3b-sparse_{version}")
            
    if n_gpu > 1:
        save_model = model.module
    else:
        save_model = model
    print("took in total: ", time.time() - t0)
    save_model.save_pretrained("replit/replit-code-v1-3b-sparse")
    upload_blob(save_model, f"replit/replit-code-v1-3b-sparse_{version}")
    upload_huggingface(save_model, version)
    

