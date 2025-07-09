import os
from dataclasses import dataclass, field
from typing import List, Literal

import torch.multiprocessing as mp
import tyro


@dataclass
class Args:
    type: Literal["intermediate", "target"]
    setting: Literal["fog", "rain", "snow", "game", "night"]
    image_root: str
    label_root: str
    target_root: str
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])

    def __post_init__(self):
        self.num_gpus = len(self.gpu_ids)
        if self.setting == "fog":
            self.prompt = "Make it foggy"
            self.guidance_scale = 7.0
            self.image_guidance_scale = 2.0
        if self.setting == "rain":
            self.prompt = "make it rain"
            self.guidance_scale = 20
            self.image_guidance_scale = 2.0
        if self.setting == "snow":
            self.prompt = "What would it look like if it were snowing?"
            self.guidance_scale = 15
            self.image_guidance_scale = 1.1
        if self.setting == "game":
            self.prompt = "have it be a 3d game"
            self.guidance_scale = 12
            self.image_guidance_scale = 2.0
        if self.setting == "night":
            self.prompt = "Make it evening"
            self.guidance_scale = 20
            self.image_guidance_scale = 1.7


def process_images(args, gpu_id, image_paths, label_paths, target_paths):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    for image_path, label_path, target_path in zip(
        image_paths, label_paths, target_paths
    ):
        print(f"GPU {gpu_id}: generating {target_path}")
        generate_and_save(
            args.prompt,
            args.guidance_scale,
            args.image_guidance_scale,
            image_path,
            target_path,
        )


def split_tasks(image_paths, label_paths, target_paths, num_gpus):
    chunk_size = len(image_paths) // num_gpus
    remainder = len(image_paths) % num_gpus
    tasks = []
    start = 0
    for i in range(num_gpus):
        end = start + chunk_size + (1 if i < remainder else 0)
        tasks.append(
            (image_paths[start:end], label_paths[start:end], target_paths[start:end])
        )
        start = end
    return tasks


def main(args: Args):
    image_paths = []
    label_paths = []
    target_paths = []
    for city in sorted(os.listdir(args.image_root)):
        city_image_dir = os.path.join(args.image_root, city)
        city_label_dir = os.path.join(args.label_root, city)
        target_image_dir = os.path.join(args.target_root, "leftImg8bit", "train", city)
        os.makedirs(target_image_dir, exist_ok=True)
        for image_name in sorted(os.listdir(city_image_dir)):
            if image_name.endswith("_leftImg8bit.png"):
                image_path = os.path.join(city_image_dir, image_name)
                label_name = image_name.replace("_leftImg8bit.png", "_gtFine_color.png")
                label_path = os.path.join(city_label_dir, label_name)
                if os.path.exists(label_path):
                    target_image_path = os.path.join(target_image_dir, image_name)
                    if os.path.exists(target_image_path):
                        print(f"skipping {target_image_path}")
                    else:
                        image_paths.append(image_path)
                        label_paths.append(label_path)
                        target_paths.append(target_image_path)
                else:
                    print(f"Label file {label_path} does not exist.")
    print(f"len(image_paths):{len(image_paths)}")
    tasks = split_tasks(image_paths, label_paths, target_paths, args.num_gpus)
    processes = []
    for gpu_id, task in zip(args.gpu_ids, tasks):
        p = mp.Process(
            target=process_images,
            args=(args, gpu_id, *task),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = tyro.cli(Args)
    if args.type == "intermediate":
        from rand_gen import generate_and_save
    if args.type == "target":
        from target_gen import generate_and_save
    main(args)
