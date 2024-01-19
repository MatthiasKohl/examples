from collections import OrderedDict

import torch
import torch.nn as nn

from torch.distributed.optim import _apply_optimizer_in_backward


def _release_storage(t):
    t.untyped_storage().resize_(0)


def _offload(t, cpu_t):
    cpu_t.copy_(t, non_blocking=True)


def _prefetch(t, cpu_t):
    t.untyped_storage().resize_(cpu_t.untyped_storage().size())
    t.copy_(cpu_t, non_blocking=True)


class OffloadPreHook(torch.autograd.Function):
    @staticmethod
    def forward(offload_args, *args):
        # pre-forward
        stream, block, next_block, prev_block = offload_args
        device = torch.cuda.current_device()
        stream.synchronize()
        with torch.cuda.stream(stream):
            with torch.no_grad():
                if prev_block is not None:
                    # delete the references to the to-be-packed tensors
                    for i, pt in enumerate(prev_block.packed_tensors):
                        prev_block.packed_tensors[i] = (pt[0], None, pt[-1])
                    for p, _ in prev_block.params:
                        _release_storage(p)
                if next_block is not None:
                    for p, cpu_param in next_block.params:
                        _prefetch(p, cpu_param)
        return args

    @staticmethod
    def setup_context(ctx, inputs, output):
        offload_args = inputs[0]
        ctx.offload_args = offload_args

    @staticmethod
    def backward(ctx, *grad_args):
        # post-backward
        stream, block, _, prev_block = ctx.offload_args
        # the model must be used with _apply_optimizer_in_backward
        # from torch.distributed.optim !
        # this means that once we sync with the main stream here, the gradient
        # is already released and we can move our parameter to CPU
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(stream):
            with torch.no_grad():
                block.packed_tensors.clear()
                if prev_block is not None:
                    for p, cpu_param in block.params:
                        _offload(p, cpu_param)
        return None, *grad_args


class OffloadPostHook(torch.autograd.Function):
    @staticmethod
    def forward(offload_args, *args):
        # post-forward
        stream, block, next_block, _ = offload_args
        with torch.cuda.stream(stream):
            with torch.no_grad():
                if next_block is not None:
                    for p, cpu_param in block.params:
                        _offload(p, cpu_param)
        return args[0] if len(args) == 1 else args

    @staticmethod
    def setup_context(ctx, inputs, output):
        offload_args = inputs[0]
        ctx.offload_args = offload_args

    @staticmethod
    def backward(ctx, *grad_args):
        # pre-backward
        stream, _, next_block, prev_block = ctx.offload_args
        device = torch.cuda.current_device()
        # sync the stream to ensure that our parameters and gradients have
        # been brought in. This also ensure that we can delete the parameters
        # of the next_block (previous block in backward)
        stream.synchronize()
        with torch.cuda.stream(stream):
            with torch.no_grad():
                if next_block is not None:
                    for p, _ in next_block.params:
                        _release_storage(p)
                if prev_block is not None:
                    for p, cpu_param in prev_block.params:
                        _prefetch(p, cpu_param)
                    for i, pt in enumerate(prev_block.packed_tensors):
                        dev, t, cpu_t = pt
                        if t is None:
                            prev_block.packed_tensors[i] = (
                                dev,
                                cpu_t.to(device=dev, non_blocking=True),
                                cpu_t
                            )
                        else:
                            _prefetch(t, cpu_t)
        return None, *grad_args


class OffloadBlockWrapper(nn.Module):
    def __init__(self, block, block_id, block_id_map, stream):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.id_map = block_id_map
        self.stream = stream
        self.packed_tensors = []
        # ensure that we have copies for both parameters and gradients
        self.params = [(p, torch.empty_like(p, device="cpu", pin_memory=True))
            for p in block.parameters()
        ]

    def pack(self, t):
        if next(reversed(self.id_map)) == self.block_id or not t.is_cuda:
            # we're last, so don't pack anything anymore
            return t

        with torch.cuda.stream(self.stream):
            idx = len(self.packed_tensors)
            cpu_t = torch.empty_like(t, device="cpu", pin_memory=True)
            _offload(t, cpu_t)
            self.packed_tensors.append((t.device, t, cpu_t))
        return idx

    def unpack(self, idx):
        if isinstance(idx, torch.Tensor):
            return idx

        # this should have already been moved back to device
        dev, t, cpu_t = self.packed_tensors[idx]
        if t is None:
            return cpu_t.to(device=dev)
        return t

    def forward(self, *args, **kwargs):
        prev_block = self.id_map.get(self.block_id - 1, None)
        next_block = self.id_map.get(self.block_id + 1, None)

        args = OffloadPreHook.apply((self.stream, self, next_block, prev_block), *args)
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            args = self.block(*args, **kwargs)
        args = args if type(args) in [list, tuple] else [args]
        outputs = OffloadPostHook.apply((self.stream, self, next_block, prev_block), *args)
        return outputs


class OffloadingWrapper(nn.Module):
    def __init__(self, wrapped_module, block_type):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.block_type = block_type
        self.stream = torch.cuda.Stream()
        self.block_id_map = OrderedDict()
        block_id = 0
        current_stack = [(wrapped_module, None, None)]
        current_item = 0
        while current_item < len(current_stack):
            m, name, parent = current_stack[current_item]
            if isinstance(m, block_type):
                if parent is None:
                    raise ValueError(
                        "Block type cannot be type of main wrapped module"
                    )
                new_module = OffloadBlockWrapper(
                    m, block_id, self.block_id_map, self.stream
                )
                setattr(parent, name, new_module)
                self.block_id_map[block_id] = new_module
                block_id += 1
            current_item += 1
            current_stack[current_item:current_item] = [
                (child, name, m) for name, child in m.named_children()
            ]

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
    model = OffloadingWrapper(model, TestBlock)
    model.to(device=device)
    _apply_optimizer_in_backward(
        torch.optim.AdamW,
        params=model.parameters(),
        optimizer_kwargs={"lr": 0.002}
    )
    loss = nn.CrossEntropyLoss()

    for i in range(3):
        # TODO need to check whether anything actually gets updated,
        # doesn't seem like it for now !
        print(f"iteration {i}")
        print(next(model.wrapped_module.children()).params[0])
        print(next(model.wrapped_module.children()).block.linear.weight)
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
        print(f"iteration {i} done")


if __name__ == "__main__":
    main()
