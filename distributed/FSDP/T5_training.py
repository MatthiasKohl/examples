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
from utils import (bfloat_support, setup, setup_allocator, teardown_allocator,
                   DeviceType, cleanup, get_date_of_run,
                   train, validation, setup_model)
import time

from cuda import cudart

PAGE_SIZE = os.sysconf('SC_PAGE_SIZE')
LIBC_TUNABLES = os.getenv('GLIBC_TUNABLES', f'glibc.malloc.top_pad={PAGE_SIZE}')
LIBC_TUNABLES = {x.split('=')[0]: int(x.split('=')[-1]) for x in LIBC_TUNABLES.split(':')}
MALLOC_PAD = LIBC_TUNABLES['glibc.malloc.top_pad']


def ptr_round_down(x): return (x // MALLOC_PAD) * MALLOC_PAD
def ptr_round_up(x): return ((x + MALLOC_PAD - 1) // MALLOC_PAD) * MALLOC_PAD


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


def prefetch_to(t, device):
    stream = torch.cuda.current_stream().cuda_stream

    ptr_start = ptr_round_down(t.data_ptr())
    ptr_end = ptr_round_up(t.data_ptr() + t.element_size() * t.nelement())
    n_bytes = ptr_end - ptr_start
    status = cudart.cudaMemPrefetchAsync(ptr_start, n_bytes, device, stream)
    assert status[0] == cudart.cudaError_t.cudaSuccess,\
        ("cudart.cudaMemPrefetchAsync failed with " + repr(status[0]) + " for " +
         str(t) + " and device " + str(device) + ", ptr " + str(t.data_ptr()) +
         " / size " + str(t.element_size() * t.nelement()) + ", rounded ptr " +
         str(ptr_start) + " / rounded size " + str(n_bytes))


# TODO: should associate each tensor with a block and only prefetch to CPU
# if not last block, then use the hook to prefetch back to device, do nothing in unpack_hook
def pack_hook(t):
    prefetch_to(t, cudart.cudaCpuDeviceId)
    return t


def unpack_hook(t):
    prefetch_to(t, t.device.index)
    return t


def fsdp_main(model_kwargs):
    allocator = setup_allocator(train_config)

    torch.manual_seed(train_config.seed)

    model, tokenizer = setup_model(train_config.model_name, **model_kwargs)
    # get a mapping from layer ID to T5Block in order to add prefetching hooks
    layer_id, layer_map, layer_id_map = 0, {}, {}
    def block_pre_hook(block, args):
        i = layer_map[block]
        prev_block = layer_id_map.get(i - 1, None)
        next_block = layer_id_map.get(i + 1, None)
        if prev_block is not None and next_block is not None:
            # move parameters of previous block back to CPU
            # TODO use a separate stream for this (issue is sync)
            for p in prev_block.parameters():
                prefetch_to(p, cudart.cudaCpuDeviceId)
        if next_block is None:
            # we're already in the block for backward, just prefetch gradients
            # of this block
            for p in block.parameters():
                if p.grad:
                    prefetch_to(p.grad, p.grad.device.index)
        else:
            # prefetch parameters of next block
            for p in next_block.parameters():
                prefetch_to(p, p.device.index)

    def block_bw_pre_hook(block, grad_output):
        i = layer_map[block]
        prev_block = layer_id_map.get(i - 1, None)
        next_block = layer_id_map.get(i + 1, None)
        if next_block is not None and prev_block is not None:
            # move parameters and gradients of next block back to CPU
            # TODO potentially use different stream
            for p in next_block.parameters():
                prefetch_to(p, cudart.cudaCpuDeviceId)
                if p.grad:
                    prefetch_to(p.grad, cudart.cudaCpuDeviceId)
        if prev_block is None:
            # need to bring in all parameters and gradients to device for optimizer
            for block in layer_map.keys():
                for p in block.parameters():
                    prefetch_to(p, p.device.index)
                    if p.grad:
                        prefetch_to(p.grad, p.grad.device.index)
        else:
            # move parameters and gradients of previous block to device
            for p in prev_block.parameters():
                prefetch_to(p, p.device.index)
                if p.grad:
                    prefetch_to(p.grad, p.grad.device.index)

    for main_stack in ["encoder", "decoder"]:
        if not hasattr(model, main_stack):
            continue
        stack = getattr(model, main_stack)
        for i, block in enumerate(stack.block):
            layer_map[block] = layer_id + i
            layer_id_map[layer_id + i] = block
            block.register_forward_pre_hook(block_pre_hook)
            block.register_full_backward_pre_hook(block_bw_pre_hook)
        layer_id += len(stack.block)

    # TODO prefetch idea: model should be T5Model, containing encoder and decoder (both T5Stack), containing T5Block s
    # default, everything on host
    # then prefetch params of first block here to device.
    # Add forward hook to block to prefetch next block params before forward of current block
    # in hook of last layer, prefetch gradients of last layer (unclear if possible, but should work)
    # then in backward hook, prefetch params+gradients of previous block before backward of current block
    # in backward hook of first block, nothing needs to be done
    # mainly need to figure out how to get ID of block from these hooks, probably just a dict of module object to ID

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

    for epoch in range(1, train_config.epochs + 1):
        torch.cuda.nvtx.range_push(f"EP {epoch}")
        t0 = time.time()
        train_accuracy = train(train_config, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        train_time = time.time() - t0
        if train_config.run_validation:
            curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()
        torch.cuda.nvtx.range_pop()

        if rank == 0:
            dur.append(time.time() - t0)
            print(
                f"--> epoch {epoch} completed...entering save and stats zone. "
                f"Train time: {train_time} s. Full epoch time: {dur[-1]} s"
            )

            train_acc_tracking.append(train_accuracy.item())

            if train_config.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if train_config.track_memory and not train_config.alloc_type:
                mem_alloc_tracker.append(torch.cuda.max_memory_allocated())
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
    teardown_allocator(allocator)


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(
        description="Train T5, override default config")
    parser.add_argument_group("model")
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_positions", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument_group("fsdp")
    parser.add_argument("--use_fsdp", type=str2bool, default=None)
    parser.add_argument("--fsdp_activation_checkpointing", type=str2bool, default=None)
    parser.add_argument("--cpu_offload", type=str2bool, default=None)
    parser.add_argument_group("training")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--batch_size_training", type=int, default=None)
    parser.add_argument("--alloc_type", default=None)
    parser.add_argument("--alloc_max_pool_size", type=int, default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--pool_location", default="default")
    parser.add_argument("--pool_accessed_by", default="default")
    parser.add_argument("--pool_prefetch", default="default")
    args = parser.parse_args()

    model_kwargs = {
        a: getattr(args, a) for a in
        ["d_ff", "d_model", "n_positions", "num_heads", "num_layers"]
        if getattr(args, a) is not None
    }
    # decoder layers is always the same as num_layers for simplicity
    if "num_layers" in model_kwargs:
        model_kwargs["num_decoder_layers"] = model_kwargs["num_layers"]
    if args.use_fsdp is not None:
        print(
            f"Config: over-writing fsdp enabled: {args.use_fsdp}"
            f"(previously {fsdp_config.enabled})"
        )
        fsdp_config.enabled = args.use_fsdp
    group_args = [
        (fsdp_config, ["fsdp_activation_checkpointing", "cpu_offload"]),
        (train_config, [
            "model_name", "batch_size_training", "alloc_type",
            "alloc_max_pool_size", "max_steps_per_epoch", "pool_location",
            "pool_accessed_by", "pool_prefetch"
        ])
    ]
    for group, arg_keys in group_args:
        for arg_key in arg_keys:
            val = getattr(args, arg_key)
            if val is not None:
                print(
                    f"Config: over-writing {arg_key} with {val} (previously "
                    f"{getattr(group, arg_key)})"
                )
                setattr(group, arg_key, val)

    # allow calling directly without torchrun
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29400'

    fsdp_main(model_kwargs)
