import bisect
from collections import OrderedDict
import gc

import torch
import torch.nn as nn

from torch.distributed.optim import _apply_optimizer_in_backward


def _release_storage(t):
    t.untyped_storage().resize_(0)


def _offload(t, cpu_t, non_blocking=True):
    cpu_t.copy_(t, non_blocking=non_blocking)


def _prefetch(t, cpu_t, non_blocking=True):
    t.untyped_storage().resize_(cpu_t.untyped_storage().size())
    t.copy_(cpu_t, non_blocking=non_blocking)


def _recursive_tensors(x):
    if isinstance(x, torch.Tensor):
        yield x
        return
    try:
        for sub_x in x.values():
            yield from _recursive_tensors(sub_x)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        try:
            for sub_x in x:
                yield from _recursive_tensors(sub_x)
        except (TypeError, ValueError, RuntimeError):
            pass


class OffloadPreHook(torch.autograd.Function):
    @staticmethod
    def forward(block, *args):
        # pre-forward
        block.stream.synchronize()
        with torch.cuda.stream(block.stream):
            with torch.no_grad():
                prev_block_act = block.id_map.get(block.block_id - block.num_blocks_act + 1)
                if prev_block_act is not None:
                    # release storage of the packed tensors
                    for i, key in enumerate(prev_block_act.packed_tensors):
                        # only release if not dirty
                        if not key in prev_block_act.dirty_tensors:
                            _release_storage(prev_block_act.packed_tensors[key][0])
                prev_block_p = block.id_map.get(block.block_id - block.num_blocks_params + 1)
                if prev_block_p is not None:
                    for p, _ in prev_block_p.params:
                        _release_storage(p)
                next_block_p = block.id_map.get(block.block_id + block.num_blocks_params - 1)
                if next_block_p is not None:
                    for p, cpu_param in next_block_p.params:
                        _prefetch(p, cpu_param)
        return args

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.block = inputs[0]

    @staticmethod
    def backward(ctx, *grad_args):
        # post-backward
        # the model must be used with _apply_optimizer_in_backward
        # from torch.distributed.optim !
        # this means that once we sync with the main stream here, the gradient
        # is already released and we can move our parameter to CPU
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(ctx.block.stream):
            with torch.no_grad():
                # no tensors (from any block) can be dirty in backward
                ctx.block.dirty_tensors.clear()
                ctx.block.packed_tensors.clear()
                prev_block_p = ctx.block.id_map.get(
                    ctx.block.block_id - ctx.block.num_blocks_params + 1)
                if prev_block_p is not None:
                    for p, cpu_param in ctx.block.params:
                        _offload(p, cpu_param)
        return None, *grad_args


class OffloadPostHook(torch.autograd.Function):
    @staticmethod
    def forward(block, *args):
        # post-forward
        with torch.cuda.stream(block.stream):
            with torch.no_grad():
                next_block_act = block.id_map.get(block.block_id + block.num_blocks_act - 1)
                if next_block_act is not None:
                    # mark any of the outputs as dirty
                    for tensor in _recursive_tensors(args):
                        key = hash(tensor)
                        if key in block.packed_tensors:
                            block.dirty_tensors[key] = block.block_id
                next_block_p = block.id_map.get(block.block_id + block.num_blocks_params - 1)
                if next_block_p is not None:
                    for p, cpu_param in block.params:
                        _offload(p, cpu_param)
        return args[0] if len(args) == 1 else args

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.block = inputs[0]

    @staticmethod
    def backward(ctx, *grad_args):
        # pre-backward
        # sync the stream to ensure that our parameters and gradients have
        # been brought in. This also ensure that we can delete the parameters
        # of the next_block (previous block in backward)
        ctx.block.stream.synchronize()
        with torch.cuda.stream(ctx.block.stream):
            with torch.no_grad():
                next_block_p = ctx.block.id_map.get(
                    ctx.block.block_id + ctx.block.num_blocks_params - 1)
                if next_block_p is not None:
                    for p, _ in next_block_p.params:
                        _release_storage(p)
                prev_block_p = ctx.block.id_map.get(
                    ctx.block.block_id - ctx.block.num_blocks_params + 1)
                if prev_block_p is not None:
                    for p, cpu_param in prev_block_p.params:
                        _prefetch(p, cpu_param)
                prev_block_act = ctx.block.id_map.get(
                    ctx.block.block_id - ctx.block.num_blocks_act + 1)
                if prev_block_act is not None:
                    for i, key in enumerate(prev_block_act.packed_tensors):
                        t, cpu_t = prev_block_act.packed_tensors[key]
                        # this tensor may have already been pre-fetched
                        # in this case, the storage contains something
                        if t.untyped_storage().size() <= 0:
                            _prefetch(t, cpu_t)
        return None, *grad_args


class OffloadBlockWrapper(nn.Module):
    def __init__(self, block, block_id, block_id_map, dirty_tensors, stream):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.id_map = block_id_map
        self.dirty_tensors = dirty_tensors
        self.stream = stream
        self.num_blocks_params = 0
        self.num_blocks_act = 0
        self.num_blocks = 0
        self.packed_tensors = dict()
        self.params = []
        self._unpack_warning_triggered = False

    def initialize(self, num_blocks_params, num_blocks_act, device):
        self.num_blocks_params = num_blocks_params
        self.num_blocks_act = num_blocks_act
        self.num_blocks = len(self.id_map)
        self.num_offload_p = self.num_blocks - self.num_blocks_params + 1
        self.block = self.block.to(device=device)
        # ensure that we have copies for both parameters and gradients
        for p in self.block.parameters():
            # release storage of parameters not (yet) required
            if self.block_id >= self.num_blocks_params - 1:
                _release_storage(p)
            cpu_p = None
            if (self.block_id < self.num_offload_p or
                    self.block_id >= self.num_blocks - self.num_offload_p):
                cpu_p = torch.empty_like(p, device="cpu", pin_memory=True)
            self.params.append((p, cpu_p))

        if self.num_blocks_params > self.num_blocks and self.num_blocks_act > self.num_blocks:
            self.forward = self._forward_none
        elif self.num_blocks_params > self.num_blocks:
            self.forward = self._forward_act
        elif self.num_blocks_act > self.num_blocks:
            self.forward = self._forward_param
        else:
            self.forward = self._forward_full

        # ensure any storage associated with original block is released
        gc.collect()

    def pack(self, t):
        if self.block_id > self.num_blocks - self.num_blocks_act or not t.is_cuda:
            # we're in the last device blocks, so don't pack anything anymore
            return t

        key = hash(t)
        if key in self.packed_tensors:
            # this tensor was already packed by this block
            # nothing to do here, we already offloaded
            pass
        elif key in self.dirty_tensors:
            # this tensor has been marked as dirty by a previous block
            # we have to add the reference to it in our packed tensors
            # but don't need to do anything else
            prev_block = self.id_map[self.dirty_tensors[key]]
            self.packed_tensors[key] = prev_block.packed_tensors[key]
            # also mark the tensor as non-dirty again
            del self.dirty_tensors[key]
        else:
            # we need to pack the tensor, offload to cpu
            # by default, the tensor is non-dirty
            with torch.cuda.stream(self.stream):
                cpu_t = torch.empty_like(t, device="cpu", pin_memory=True)
                _offload(t, cpu_t)
                self.packed_tensors[key] = (t, cpu_t)
        return key

    def unpack(self, key):
        if isinstance(key, torch.Tensor):
            return key

        # this should have already been moved back to device
        return self.packed_tensors[key][0]

    def _forward_none(self, *args, **kwargs):
        return self.block(*args, **kwargs)

    def _forward_act(self, *args, **kwargs):
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            return self.block(*args, **kwargs)

    def _forward_param(self, *args, **kwargs):
        args = OffloadPreHook.apply(self, *args)
        args = self.block(*args, **kwargs)
        args = [args] if isinstance(args, torch.Tensor) else args
        return OffloadPostHook.apply(self, *args)

    def _forward_full(self, *args, **kwargs):
        args = OffloadPreHook.apply(self, *args)
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            args = self.block(*args, **kwargs)
        args = [args] if isinstance(args, torch.Tensor) else args
        return OffloadPostHook.apply(self, *args)


class OffloadingWrapper(nn.Module):
    def __init__(self, wrapped_module, block_type,
                 device=None, num_blocks_params=2, num_blocks_act=2):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.block_type = block_type
        self.device = device or torch.cuda.current_device()
        self.stream = torch.cuda.Stream()
        self.block_id_map = OrderedDict()
        self.dirty_tensors = dict()
        block_id = 0
        current_stack = [(
            wrapped_module, None, None, isinstance(wrapped_module, block_type)
        )]
        current_item = 0
        while current_item < len(current_stack):
            m, name, parent, within_block = current_stack[current_item]
            is_block = isinstance(m, block_type)
            if is_block:
                if parent is None:
                    raise ValueError(
                        "Block type cannot be type of main wrapped module"
                    )
                new_module = OffloadBlockWrapper(
                    m, block_id, self.block_id_map, self.dirty_tensors,
                    self.stream
                )
                setattr(parent, name, new_module)
                self.block_id_map[block_id] = new_module
                block_id += 1
            elif not within_block:
                m._apply(lambda t: t.to(device=self.device), recurse=False)
            current_item += 1
            current_stack[current_item:current_item] = [
                (child, name, m, is_block or within_block)
                for name, child in m.named_children()
            ]
        # ensure that we don't keep references to the original blocks anywhere
        del current_stack[:]
        gc.collect()

        num_blocks_params = min(num_blocks_params, len(self.block_id_map) + 1)
        if num_blocks_params <= 0:
            num_blocks_params = len(self.block_id_map) + 1
        num_blocks_act = min(num_blocks_act, len(self.block_id_map) + 1)
        if num_blocks_act <= 0:
            num_blocks_act = len(self.block_id_map) + 1
        for block_wrapper in self.block_id_map.values():
            block_wrapper.initialize(num_blocks_params, num_blocks_act, self.device)

    def forward(self, *args, **kwargs):
        return self.wrapped_module(*args, **kwargs)


class TestBlock(nn.Module):
    def __init__(self, in_size, out_size, act=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = act() if act else None

    def forward(self, x):
        x = self.linear(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def main():
    batch_size, in_size, hidden_size, out_size = 1024, 128, 512, 256
    device = torch.device("cuda")

    model = nn.Sequential(
        TestBlock(in_size, hidden_size),
        TestBlock(hidden_size, hidden_size),
        TestBlock(hidden_size, hidden_size),
        TestBlock(hidden_size, out_size, act=None)
    )
    model = OffloadingWrapper(model, TestBlock, device=device)
    _apply_optimizer_in_backward(
        torch.optim.AdamW,
        params=model.parameters(),
        optimizer_kwargs={"lr": 2.0}
    )
    loss = nn.CrossEntropyLoss()

    for i in range(16):
        print(f"iteration {i}", end=" ", flush=True)
        # p_wrapped = next(model.wrapped_module.children()).params[0]
        # p_linear = next(model.wrapped_module.children()).block.linear.weight
        # print(p_wrapped[0] is p_linear)
        # print(p_wrapped)
        torch.cuda.nvtx.range_push(f"IT {i}")
        inputs = torch.randn(batch_size, in_size, device=device)
        targets = torch.randint(0, out_size, size=(batch_size,), device=device)
        torch.cuda.nvtx.range_push(f"forward")
        outputs = model(inputs)
        loss_value = loss(outputs, targets)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"backward")
        # import pdb; pdb.set_trace()
        loss_value.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        print(f"done")


if __name__ == "__main__":
    main()
