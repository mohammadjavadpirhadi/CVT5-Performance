import os
import argparse
import json
import random
import math
import torch
import numpy as np

GLOBAL_SEED = 2024

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.use_deterministic_algorithms(True)

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from transformers import get_cosine_schedule_with_warmup

from dataloaders.compressed_video import get_dataset_and_data_loader
from models.cvt5 import *
from utils.losses import get_loss_fn
from utils.trainer import train_one_epoch, evaluate_one_epoch_sn
from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser()

parser.add_argument('--config',
                    required=True,type=str,
                    help='path to config file')

args = parser.parse_args()

with open(f"{os.getcwd()}/configs/{args.config}") as config_file:
    config = json.load(config_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"Number of CPU cores: {os.cpu_count()}")

train_dataset, train_dataloader = get_dataset_and_data_loader(config["train_dataset"])
test_dataset, test_dataloader = get_dataset_and_data_loader(config["test_dataset"])
spottting_loss_fn = get_loss_fn(train_dataset, config["spotting_loss_fn"])
model = CVT5Model(config["cvt5_config"], spottting_loss_fn).to(device)

num_params = sum(p.numel() for p in model.parameters())
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params:,d} (Trainable: {num_trainable_params:,d})")

optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
num_cycles = config["num_cosine_schedule_cycles"] if "num_cosine_schedule_cycles" in config else 0.5
scheduler = get_cosine_schedule_with_warmup(optimizer, config["num_warmup_steps"], config["epoches"]*len(train_dataloader), num_cycles=num_cycles)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as profiler:
    model.train(True)
    train_one_epoch(
        model,
        train_dataloader,
        config["cvt5_config"]["short_memory_len"],
        config["cvt5_config"]["umt5_enabled"],
        int(config["train_dataset"]["config"]["add_bg_label"]),
        optimizer,
        scheduler,
        device,
        profiler,
        config["verbose"]
    )
print("Train stats:")
print(profiler.key_averages().total_average())

if device == 'cuda':
    torch.cuda.empty_cache()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as profiler:
    model.train(False)
    with torch.no_grad():
        evaluate_one_epoch_sn(
            model,
            test_dataloader,
            test_dataset,
            config["test_dataset"]["splits"][0], # Should have one split
            config["cvt5_config"]["umt5_enabled"],
            config["test_dataset"]["config"]["base_dir"],
            device,
            profiler
        )
print("Inference stats:")
print(profiler.key_averages().total_average())

