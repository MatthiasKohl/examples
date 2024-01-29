import bisect
from collections import OrderedDict
from dataclasses import dataclass
import gc
import weakref

import torch
import torch.nn as nn
from torch.multiprocessing.reductions import StorageWeakRef
from torch._utils import _rebuild_tensor_v2, get_tensor_metadata

from torch.distributed.optim import _apply_optimizer_in_backward

# set this to True to get a fully blocking version
BLOCKING = False
# set this to True to get a fully-in-stream-sync version (this is more restrictive than BLOCKING)
SYNC_STREAM = False


def _release_storage(t):
    t.untyped_storage().resize_(0)


def _offload(t, cpu_t, non_blocking=not BLOCKING):
    cpu_t.copy_(t, non_blocking=non_blocking)


def _prefetch(t, cpu_t, non_blocking=not BLOCKING):
    t.untyped_storage().resize_(cpu_t.untyped_storage().size())
    t.copy_(cpu_t, non_blocking=non_blocking)


@dataclass
class OffloadRef:
    block_id: int
    key: int


@dataclass
class MainOffloadMeta:
    meta_args: tuple # args passed to _rebuild_tensor_v2
    dtype: torch.dtype
    device: torch.device # always the original device of tensor
    active_tensor: torch.Tensor # tensor having currently active values (may require sync)
    num_views: int=0


@dataclass
class ViewOffloadMeta:
    meta_args: tuple=() # args passed to _rebuild_tensor_v2
    dtype: torch.dtype=None


def _make_view_meta(t: torch.Tensor):
    backward_hooks = OrderedDict()  # we don't support hooks for now
    meta_args = (
        t.storage_offset(),
        tuple(t.size()),
        t.stride(),
        t.requires_grad,
        backward_hooks,
        get_tensor_metadata(t),
    )
    return ViewOffloadMeta(meta_args, t.dtype)


def _make_main_meta(t: torch.Tensor, device: torch.device, active_tensor: torch.Tensor):
    view = _make_view_meta(t)
    return MainOffloadMeta(view.meta_args, view.dtype, device, active_tensor)


def _preload_packed(block, non_blocking=not BLOCKING):
    for key, entry in block.packed_tensors.items():
        # first, we ignore entries that are simply references to a finalized entry
        if isinstance(entry, OffloadRef):
            continue
        # get the actual owning block and final key
        main_ref = block.all_packed[key]
        main_block = block.id_map[main_ref.block_id]
        # main entry is always the first one
        main_entry = main_block.packed_tensors[main_ref.key][0]
        assert isinstance(main_entry, MainOffloadMeta)
        if main_entry.device == main_entry.active_tensor.device:
            # already pre-loaded (e.g. by another block)
            continue

        t = torch.empty_like(main_entry.active_tensor, device=main_entry.device)
        t.copy_(main_entry.active_tensor, non_blocking=non_blocking)
        main_entry.active_tensor = t


def _cleanup_packed(block, log_domain):
    # clean-up any packed tensors
    if block.block_id == 0:
        block.all_packed.clear()
    if block.packed_tensors:
        # check if we had any remaining main entries
        main_entry = next((
            v[0] for k, v in block.packed_tensors.items()
            if isinstance(v, list) and isinstance(v[0], MainOffloadMeta)
        ), None)
        if main_entry:
            print(
                f"Warning: {log_domain} block {block.block_id}: found "
                f"main entry in packed tensors: {main_entry}"
            )
        block.packed_tensors.clear()


class OffloadPreHook(torch.autograd.Function):
    @staticmethod
    def forward(block, *args):
        torch.cuda.nvtx.range_push(f"forward {block.block_id}")
        # pre-forward
        # main stream must wait on our custom stream here to ensure that:
        # 1. we can release storage of offloaded parameters (previous block)
        # 2. parameters have been brought in for current block
        # we do this with host `synchronize` because releasing storage may not be stream-ordered
        if BLOCKING:
            block.stream.synchronize()
            torch.cuda.current_stream().synchronize()
        else:
            torch.cuda.current_stream().wait_stream(block.stream)
        # clean-up any packed tensors (post-backward may not be enough because
        # block 0 may not even have any post-backward)
        _cleanup_packed(block, "pre-forward")

        with torch.no_grad():
            with torch.cuda.stream(block.stream):
                prev_block_p = block.id_map.get(block.block_id - block.num_blocks_params + 1)
                if prev_block_p is not None and block.num_blocks_params > 1:
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
        # this means that once we wait on ops on the main stream here, the gradient
        # is already released and we can move our parameter to CPU
        if BLOCKING:
            ctx.block.stream.synchronize()
            torch.cuda.current_stream().synchronize()
        else:
            ctx.block.stream.wait_stream(torch.cuda.current_stream())
        # clean-up any packed tensors (we'll cleanup in pre-forward as well
        # since this may not be enough, but we can already free up the structure)
        _cleanup_packed(ctx.block, "post-backward")

        with torch.no_grad():
            with torch.cuda.stream(ctx.block.stream):
                prev_block_p = ctx.block.id_map.get(
                    ctx.block.block_id - ctx.block.num_blocks_params + 1)
                if prev_block_p is not None:
                    for p, cpu_param in ctx.block.params:
                        _offload(p, cpu_param)
        torch.cuda.nvtx.range_pop()
        return None, *grad_args


class OffloadPostHook(torch.autograd.Function):
    @staticmethod
    def forward(block, *args):
        # post-forward
        # no need to wait on anything here, since the parameters of this block
        # should not be modified by forward
        if BLOCKING:
            block.stream.synchronize()
            torch.cuda.current_stream().synchronize()

        with torch.no_grad():
            with torch.cuda.stream(block.stream):
                next_block_p = block.id_map.get(block.block_id + block.num_blocks_params - 1)
                if next_block_p is not None:
                    for p, cpu_param in block.params:
                        _offload(p, cpu_param)
        torch.cuda.nvtx.range_pop()
        return args[0] if len(args) == 1 else args

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.block = inputs[0]

    @staticmethod
    def backward(ctx, *grad_args):
        torch.cuda.nvtx.range_push(f"backward {block.block_id}")
        # pre-backward
        # wait on operations of our stream here:
        # 1. ensures that parameters and gradients have been brought in
        # 2. ensures that parameters of the next block (previous in backward order)
        #    can be released
        # we do this with host `synchronize` because releasing storage may not be stream-ordered
        if BLOCKING:
            ctx.block.stream.synchronize()
            torch.cuda.current_stream().synchronize()
        else:
            torch.cuda.current_stream().wait_stream(ctx.block.stream)
        with torch.no_grad():
            with torch.cuda.stream(ctx.block.stream):
                next_block_p = ctx.block.id_map.get(
                    ctx.block.block_id + ctx.block.num_blocks_params - 1)
                if next_block_p is not None and ctx.block.num_blocks_params > 1:
                    for p, _ in next_block_p.params:
                        _release_storage(p)
                prev_block_p = ctx.block.id_map.get(
                    ctx.block.block_id - ctx.block.num_blocks_params + 1)
                if prev_block_p is not None:
                    for p, cpu_param in prev_block_p.params:
                        _prefetch(p, cpu_param)
                prev_block_act = ctx.block.id_map.get(
                    ctx.block.block_id - ctx.block.num_blocks_act + 1)
                if prev_block_act is not None and ctx.block.num_blocks_act > 1:
                    _preload_packed(prev_block_act)
        if ctx.block.num_blocks_act == 1:
            _preload_packed(ctx.block, non_blocking=False)
        return None, *grad_args


class OffloadBlockWrapper(nn.Module):
    def __init__(self, block, block_id, block_id_map, all_params, all_packed, stream):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.id_map = block_id_map
        self.all_params = all_params
        self.all_packed = all_packed
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
                key = hash(StorageWeakRef(p.untyped_storage()))
                self.all_params[key] = self.block_id
            self.params.append((p, cpu_p))

        # for activation offloading, we need both pre-backward and post-backward
        # hooks, so we just use the full forward anyway
        if self.num_blocks_params > self.num_blocks and self.num_blocks_act > self.num_blocks:
            self.forward = self._forward_none
        else:
            self.forward = self._forward_full

        # ensure any storage associated with original block is released
        gc.collect()

    def pack(self, t):
        if self.block_id >= self.num_blocks - self.num_blocks_act or not t.is_cuda:
            # we're in the last device blocks, so don't pack anything anymore
            return t

        storage = t.untyped_storage()
        key = hash(StorageWeakRef(storage))
        # hashes must be strictly positive
        assert key > 0

        # if this is actually a parameter, don't pack it
        if key in self.all_params and self.num_blocks_params >= self.num_blocks_act:
            return t

        do_offload = True
        new_key = key
        while new_key in self.all_packed:
            ref = self.all_packed[new_key]
            if ref.key < 0:
                # this has been finalized already, we choose a different key
                new_key += 1
                continue
            # `ref` must point to a "main" storage, thus we don't need to offload
            do_offload = False
            break

        if do_offload:
            with torch.no_grad():
                with torch.cuda.stream(self.stream):
                    cpu_t = torch.empty_like(t, device="cpu", pin_memory=True)
                    _offload(t, cpu_t)
            view_idx = 0
            meta = _make_main_meta(t, t.device, cpu_t)
            main_meta = meta
            self.packed_tensors[new_key] = [meta]
            self.all_packed[new_key] = OffloadRef(self.block_id, new_key)
        else:
            # this must be a view to some existing (non-finalized) storage
            # we get that reference from `all_packed`, and add the view entry
            # to this block's packed tensors
            meta = _make_view_meta(t)
            main_ref = self.all_packed[new_key]
            # add the view entry
            entries = self.packed_tensors.get(new_key, [])
            view_idx = len(entries)
            entries.append(meta)
            self.packed_tensors[new_key] = entries
            # assign the main meta to increase `num_views` correctly
            block = self.id_map[main_ref.block_id]
            main_meta = block.packed_tensors[main_ref.key][0]

        # whether we are the "main" view or not, we always increase the number
        # of views by exactly 1
        main_meta.num_views += 1

        def on_storage_del(k):
            if k not in self.packed_tensors:
                return
            entries = self.packed_tensors[k]
            assert not isinstance(entries, OffloadRef)
            # it's important to assign a new key that can never clash with
            # any hash an actual storage could produce. This is why we use
            # the negative number range here
            new_k = -k
            while new_k in self.all_packed:
                new_k -= 1
            assert new_k not in self.packed_tensors
            new_ref = OffloadRef(self.block_id, new_k)
            self.all_packed[k] = new_ref
            self.all_packed[new_k] = new_ref
            self.packed_tensors[new_k] = entries
            # put the new ref in `packed_tensors` as well: anyone trying to
            # access main storage should go through `all_packed`
            self.packed_tensors[k] = new_ref

        # only need to add a finalize hook in case we are the main owner
        if do_offload:
            weakref.finalize(storage, on_storage_del, new_key)

        return new_key, view_idx

    def unpack(self, key_idx):
        if isinstance(key_idx, torch.Tensor):
            return key_idx

        key, view_idx = key_idx
        # get the actual block and key for this entry
        main_ref = self.all_packed[key]
        main_block = self.id_map[main_ref.block_id]
        main_entry = main_block.packed_tensors[main_ref.key][0]
        assert isinstance(main_entry, MainOffloadMeta)
        try:
            # this tensor has already been moved back to device
            main_t = main_entry.active_tensor
            if main_ref.block_id == self.block_id and view_idx == 0:
                # this is the special case where block ownership did not change
                # and we have the "main" view: can directly return the tensor
                return main_t
            # in all other cases, need to get our view meta and re-construct tensor
            # if the main entry is in another block, the key for our view entries
            # may not have changed, just check this here
            self_entries = self.packed_tensors[key]
            if isinstance(self_entries, OffloadRef):
                view_meta = self.packed_tensors[self_entries.key][view_idx]
            else:
                view_meta = self_entries[view_idx]
            storage = torch.storage.TypedStorage(
                wrap_storage=main_t._typed_storage()._untyped_storage,
                dtype=view_meta.dtype,
                _internal=True,
            )
            return _rebuild_tensor_v2(storage, *view_meta.meta_args)
        finally:
            # removing any meta entry always decreases the `num_views` of the 
            # main meta by exactly 1
            main_entry.num_views -= 1
            if main_entry.num_views == 0:
                del main_block.packed_tensors[main_ref.key]

    def _forward_none(self, *args, **kwargs):
        return self.block(*args, **kwargs)

    def _forward_full(self, *args, **kwargs):
        # import pdb; pdb.set_trace()
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
        if SYNC_STREAM:
            self.stream = torch.cuda.current_stream()
        else:
            self.stream = torch.cuda.Stream()
        self.block_id_map = OrderedDict()
        self.all_params = dict()
        self.all_packed = dict()
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
                    m, block_id, self.block_id_map, self.all_params,
                    self.all_packed, self.stream
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
