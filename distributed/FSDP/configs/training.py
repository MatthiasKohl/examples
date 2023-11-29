from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="configs/model_flan_xl.json"
    run_validation: bool=True
    batch_size_training: int=4
    batch_size_testing: int=4
    num_workers_dataloader: int=2
    lr: float=0.002
    weight_decay: float=0.0
    gamma: float= 0.85
    use_fp16: bool=False
    mixed_precision: bool=True
    save_model: bool=False

    # allocator options
    alloc_type: str="malloc_prefetch"
    alloc_max_pool_size: int=256 * (1024 ** 3)

# following model sizes to be tested:
# 2048 5120 (flan-t5-xl)
# 2304 5760
# 2560 6400
# 2816 7040 (OOM with batch size 4 and default allocator)
# 3072 7680 (heads 48)
# 3328 8320
# 3584 8960
# 3840 9600
# 4096 10240 (flan-t5-xxl)
