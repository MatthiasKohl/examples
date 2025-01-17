import json
import os
import torch
import torch.distributed as dist
from datetime import datetime
import tqdm
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
    n_batches = min(len(train_loader), args.max_steps_per_epoch)

    if sampler:
        sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(n_batches), colour="blue", desc="r0 Training Epoch"
        )
    for i, batch in enumerate(train_loader):
        torch.cuda.nvtx.range_push(f"EP {epoch} / it {i}")
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        torch.cuda.nvtx.range_push(f"opt zero")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"forward")
        # import pdb;pdb.set_trace()
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
        loss = output["loss"]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"opt")
        optimizer.step()
        torch.cuda.nvtx.range_pop()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        torch.cuda.nvtx.range_pop()
        if rank==0:
            inner_pbar.update(1)
        if i + 1 >= n_batches:
            break

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]

    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, #epoch steps: {n_batches}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy


def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


def setup_model(model_name, **kwargs):
    if os.path.isfile(model_name):
        with open(model_name) as f:
            config = json.load(f)

        for key, val in kwargs.items():
            if key not in config:
                print(f"Invalid custom model arg: {key} ({val}). Ignoring")

        add_args = {k: v for k, v in kwargs.items() if k in config and config[k] != v}
        config.update(add_args)

        print(f"Setup model: using custom config from {model_name} "
              f"with updated args: {add_args}")
        t5_config = T5Config(**config)
        model = T5ForConditionalGeneration(t5_config)
        # for custom configs, just always use the flan-xl tokenizer
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    else:
        if kwargs:
            print(
                f"Setup model: ignoring kwargs {kwargs} because model config "
                f"{model_name} is chosen"
            )
        else:
            print(f"Setup model: using default model config {model_name}")
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer =  T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer
