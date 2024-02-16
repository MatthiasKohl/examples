import torch
import os
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    offload_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from transformers.models.t5.modeling_t5 import T5Block

from functools import partial

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    offload_to_cpu=False,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, T5Block)


def apply_fsdp_checkpointing(model, is_offload):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fdsp activation checkpointing (offload: {is_offload})...")

    wrapper = offload_wrapper if is_offload else non_reentrant_wrapper
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn
    )
