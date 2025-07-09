from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


@dataclass
class Args:
    synth_path: str
    source_path: str
    target_type: Literal["acdc_night", "acdc_rain", "acdc_snow", "acdc_fog", "gta5"]
    target_path: str
    experiment_name: str
    stop_epoch: int = 0
    debug: bool = False
    load: Optional[str] = None
    resume: Optional[str] = None
    epochs: int = 100
    seed: int = 42
    sim_weight: float = 0.0
    model: str = "resnet-50"
    ssl: str = "sim"
    optimizer: str = "sgd"
    momentum: float = 0.9
    wd: float = 1e-4
    schedular: str = "poly"
    batch_size: int = 4
    lr_head: float = 0.001
    lr: float = 0.001
    target_size: Tuple[int, int] = (512, 1024)
    crop: str = "both_random"
    crop_size: Tuple[int, int] = (384, 768)
    blur: bool = False
    cutout: bool = False
    jitter: float = 0.5
    scale: float = 0.0
    xs: bool = False
    workers: int = 4

    def __post_init__(self):
        if self.debug:
            self.epochs = 2
            self.workers = 0

    def save(self):
        with open("exp_args.py", "a") as f:
            f.write("class Args:\n")
            for key, value in vars(self).items():
                f.write(f"    {key} = {repr(value)}\n")
