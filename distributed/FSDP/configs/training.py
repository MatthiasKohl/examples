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
    epochs: int=2
    max_steps_per_epoch: int=40
    seed: int=1
    track_memory: bool=True
    run_validation: bool=False

    # allocator options
    alloc_type: str="rmm"
    alloc_initial_pool_size: int=48 * (1024 ** 3)
    alloc_max_pool_size: int=96 * (1024 ** 3)
