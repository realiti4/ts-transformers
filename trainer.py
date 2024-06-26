import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import itertools
from typing import List, Iterator, Optional, Dict
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader

from utils.utils import yaml_loader

from gluonts.itertools import Cyclic, Map, Filter
from gluonts.dataset.common import FileDataset

# from model.chronos import ChronosConfig
from utils.dataset import ChronosDataset, has_enough_observations
from model.chronos import MeanScaleUniformBins
from model.t5 import T5Model, ChronosConfig

transformers.set_seed(seed=1337)

max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
warmup_steps = 100  # 715
max_steps = 19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens


def get_lr(it):
    # 1 linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2 if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3 in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def train(
    training_data_paths: str,
    probability: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    shuffle_buffer_length: int = 100,
    gradient_accumulation_steps: int = 2,
    model_id: str = "google/t5-efficient-tiny",
    model_type: str = "seq2seq",
    random_init: bool = False,
    tie_embeddings: bool = False,
    output_dir: str = "./output/",
    tf32: bool = True,
    torch_compile: bool = True,
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> None:
    config = ChronosConfig()

    # TODO - fix dataset by recreating array file with whole dataset without windows

    # Tokenizer
    tokenizer = MeanScaleUniformBins(low_limit=-15, high_limit=15, config=config)

    # Train Datasets
    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]

    # Train dataset
    shuffle_buffer_length = 10

    shuffled_train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=probability,
        tokenizer=tokenizer,
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        mode="training",
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

    # Params
    master_process = True
    total_batch_size = 32
    mini_batch = 16
    ddp_world_size = 1
    grad_accum_steps = total_batch_size // (mini_batch * ddp_world_size)
    lr = 1e-5

    assert (
        total_batch_size % (mini_batch * ddp_world_size) == 0
    ), "make sure total_batch_size is divisible by B * T * ddp_world_size"
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = "amazon/chronos-t5-tiny"

    # model = T5Model.from_pretrained(model)  # Fine-tuning
    model = T5Model(config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model_size = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(f"model size {model_size:.2f} M")

    train_loader = iter(shuffled_train_dataset)

    dataloader_params = {
        "batch_size": 16,
        "num_workers": 1,
        "pin_memory": True,
        "persistent_workers": False,
    }

    train_loader = DataLoader(shuffled_train_dataset, **dataloader_params)
    train_loader = iter(train_loader)
    # self.accelerator.prepare(DataLoader(shuffled_train_dataset, **dataloader_params))

    # Train
    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            inputs = next(train_loader)

            inputs['input_ids'], inputs['attention_mask'], inputs['labels'] = inputs['input_ids'].to(device), inputs['attention_mask'].to(device), inputs['labels'].to(device)

            # x, y = train_loader.next_batch()
            # x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(**inputs)
                loss = output.loss

            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        torch.cuda.synchronize()  # wait for the GPU to finish work

        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = inputs['input_ids'].size(0) * inputs['input_ids'].size(1) * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            # with open(log_file, "a") as f:
            #     f.write(f"{step} train {loss_accum.item():.6f}\n")

    print("Done")


if __name__ == "__main__":
    # Params
    model_name = "chronos-t5-base"
    conf_path = f"configs/{model_name}.yaml"

    conf = yaml_loader(conf_path)

    user_conf = {
        "training_data_paths": ["datasets/all_bitcoin.arrow"],
        "probability": [1.0],
        "model_id": f"amazon/{model_name}",
        "output_dir": "./output/",
        "max_steps": 10000,
        "log_steps": 100,
        # "save_steps": 2000,
        "learning_rate": 0.0001,
        "random_init": False,  # enable fine-tuning
        "per_device_train_batch_size": 12,
        "gradient_accumulation_steps": 1,
        "context_length": 768,
    }

    merged_conf = conf | user_conf

    train(**merged_conf)

    print("Done")
