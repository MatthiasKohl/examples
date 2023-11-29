import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

from torch.utils.data import DataLoader
from summarization_dataset import *
import policies
import model_checkpointing
from configs import fsdp_config, train_config
from utils import (bfloat_support, setup, setup_allocator,
                   cleanup, get_date_of_run,
                   train, validation, setup_model)
import time

def get_policies(cfg, rank):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.mixed_precision:
        bfloat_available = bfloat_support()
        if bfloat_available and not cfg.use_fp16:
            mixed_precision_policy = policies.bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = policies.fpSixteen
            if rank == 0:
                print(f"FP16 enabled. ")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    wrapping_policy = policies.get_t5_wrapper()

    return mixed_precision_policy, wrapping_policy


def fsdp_main(args):
    setup_allocator(train_config)

    model, tokenizer = setup_model(train_config.model_name)

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])


    dataset = load_dataset('wikihow', 'all', data_dir='data/')
    print(dataset.keys())
    print("Size of train dataset: ", dataset['train'].shape)
    print("Size of Validation dataset: ", dataset['validation'].shape)

   
    #wikihow(tokenizer, type_path, num_samples, input_length, output_length, print_text=False)
    train_dataset = wikihow(tokenizer, 'train', 1500, 512, 150, False) 
    val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False)

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    setup()

    train_kwargs = {'batch_size': train_config.batch_size_training, 'sampler': sampler1}
    test_kwargs = {'batch_size': train_config.batch_size_testing, 'sampler': sampler2}
    cuda_kwargs = {
        'num_workers': train_config.num_workers_dataloader,
        'pin_memory': True,
        'shuffle': False
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
 
    torch.cuda.set_device(local_rank)
    
    # Apply FSDP wrapping to the model
    if fsdp_config.enabled:
        # Set up FSDP parameters
        mixed_precision_policy, t5_auto_wrap_policy = get_policies(train_config, rank)
        if fsdp_config.cpu_offload is None:
            offload = None
        else:
            offload = CPUOffload(offload_params=fsdp_config.cpu_offload)
        model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=fsdp_config.limit_all_gathers,
            cpu_offload=offload)

        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    else:
        # move model to the GPU
        model = model.to(device=torch.cuda.current_device())

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "T5-model-"

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    mem_alloc_tracker = []
    mem_reserved_tracker = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        if args.run_validation:
            curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()
        
        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if args.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if args.track_memory and not train_config.alloc_type:
                mem_alloc_tracker.append(torch.cuda.memory_allocated())
                mem_reserved_tracker.append(torch.cuda.memory_reserved())

        if train_config.save_model and curr_val_loss < best_val_loss:
            if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                model_checkpointing.save_model_checkpoint(
                    model, optimizer, rank, fsdp_config, epoch=1
                )
            elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                model_checkpointing.save_model_and_optimizer_sharded(model, rank, fsdp_config)
                if fsdp_config.save_optimizer:
                    model_checkpointing.save_model_and_optimizer_sharded(model, rank, fsdp_config, optim=optimizer)

            if fsdp_config.save_optimizer:
                model_checkpointing.save_optimizer_checkpoint(
                    model, optimizer, rank, fsdp_config, epoch=1
                )

        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")
        if rank == 0 and mem_reserved_tracker and mem_alloc_tracker:
            print(f"Max memory reserved: {max(mem_reserved_tracker) / 1e9:.2f} GB, max memory allocated: {max(mem_alloc_tracker) / 1e9:.2f} GB")

    dist.barrier()
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    fsdp_main(args)
