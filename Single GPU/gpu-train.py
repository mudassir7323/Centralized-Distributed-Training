import os
import math
import time
import random
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from transformers import GPT2Config

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# =========================================================
# CONFIG
# =========================================================

SEED = 42

MAX_LENGTH = 256
BATCH_SIZE = 8
GRAD_ACCUM = 4

EPOCHS = 5

LR = 1e-4

DEVICE = "cuda"

SAVE_DIR = "checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =========================================================
# DISTRIBUTED SETUP
# =========================================================

distributed = "RANK" in os.environ

if distributed:
    torch.distributed.init_process_group(backend="nccl")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

else:
    rank = 0
    local_rank = 0
    world_size = 1

device = torch.device(f"cuda:{local_rank}")

print(f"Rank: {rank} | World Size: {world_size}")

# =========================================================
# GPU CHECK
# =========================================================

if rank == 0:
    os.system("nvidia-smi")

# =========================================================
# LOAD DATASET
# =========================================================

if rank == 0:
    print("\nLoading dataset...")

dataset = load_dataset("wikitext", "wikitext-103-v1")

train_texts = [
    x for x in dataset["train"]["text"][:15000]
    if len(x.strip()) > 10
]

val_texts = [
    x for x in dataset["validation"]["text"][:2000]
    if len(x.strip()) > 10
]

if rank == 0:
    print(f"Train Samples: {len(train_texts)}")
    print(f"Val Samples: {len(val_texts)}")

# =========================================================
# TOKENIZER
# =========================================================

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# DATASET CLASS
# =========================================================

class GPTDataset(Dataset):

    def __init__(self, texts):

        self.texts = texts

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": input_ids.clone()
        }

# =========================================================
# DATALOADERS
# =========================================================

train_dataset = GPTDataset(train_texts)
val_dataset = GPTDataset(val_texts)

if distributed:

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

else:

    train_sampler = None
    val_sampler = None

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# =========================================================
# MODEL
# =========================================================

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=MAX_LENGTH,
    n_embd=768,
    n_layer=12,
    n_head=12,
    use_cache=False
)

model = GPT2LMHeadModel(config)

model.to(device)

# torch compile = faster
model = torch.compile(model)

if distributed:

    model = DDP(
        model,
        device_ids=[local_rank]
    )

if rank == 0:

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nParameters: {total_params:,}")

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# =========================================================
# FP16
# =========================================================

scaler = torch.cuda.amp.GradScaler()

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_epoch():

    model.train()

    total_loss = 0

    start = time.time()

    bar = tqdm(
        train_loader,
        disable=(rank != 0)
    )

    optimizer.zero_grad()

    for step, batch in enumerate(bar):

        input_ids = batch["input_ids"].to(device, non_blocking=True)

        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True
        )

        with torch.cuda.amp.autocast():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:

            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM

        if rank == 0:

            bar.set_description(
                f"Loss: {loss.item() * GRAD_ACCUM:.4f}"
            )

    scheduler.step()

    elapsed = time.time() - start

    throughput = (
        len(train_dataset) * MAX_LENGTH
    ) / elapsed

    return total_loss / len(train_loader), throughput

# =========================================================
# VALIDATION
# =========================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0

    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_loader)

    perplexity = math.exp(min(avg_loss, 20))

    return avg_loss, perplexity

# =========================================================
# TRAIN LOOP
# =========================================================

best_val_loss = float("inf")

train_losses = []
val_losses = []

for epoch in range(EPOCHS):

    if distributed:
        train_sampler.set_epoch(epoch)

    if rank == 0:

        print("\n" + "=" * 60)
        print(f"Epoch {epoch+1}/{EPOCHS}")

    train_loss, throughput = train_epoch()

    val_loss, val_ppl = validate()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if rank == 0:

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Perplexity: {val_ppl:.2f}")
        print(f"Throughput: {throughput:.0f} tok/s")

    # SAVE BEST MODEL
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        save_path = os.path.join(
            SAVE_DIR,
            "best_model.pt"
        )

        model_to_save = (
            model.module
            if distributed
            else model
        )

        torch.save(
            model_to_save.state_dict(),
            save_path
        )

        if rank == 0:
            print(f"Saved Best Model -> {save_path}")

# =========================================================
# FINAL SAVE
# =========================================================

if rank == 0:

    final_path = os.path.join(
        SAVE_DIR,
        "final_model.pt"
    )

    model_to_save = (
        model.module
        if distributed
        else model
    )

    torch.save(
        model_to_save.state_dict(),
        final_path
    )

    print("\nTraining Complete")
    print(f"Final Model Saved -> {final_path}")


# =========================================================
# SAVE PLOTS
# =========================================================

if rank == 0:

    import matplotlib.pyplot as plt

    # TRAIN vs VAL LOSS
    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training vs Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "loss_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved -> loss_curve.png")

    # PERPLEXITY PLOT
    perplexities = [
        math.exp(min(v, 20))
        for v in val_losses
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(perplexities) + 1),
        perplexities
    )

    plt.xlabel("Epoch")

    plt.ylabel("Perplexity")

    plt.title("Validation Perplexity")

    plt.grid(True)

    plt.savefig(
        "perplexity_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved -> perplexity_curve.png")    

# =========================================================
# CLEANUP
# =========================================================

if distributed:
    torch.distributed.destroy_process_group()