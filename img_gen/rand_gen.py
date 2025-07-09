import random

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

generator = torch.Generator("cuda").manual_seed(42)
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None
)
pipe = pipe.to("cuda")


def generate_and_save(
    prompt, guidance_scale, image_guidance_scale, image_path, target_path
):
    image = load_image(image_path)
    new_size = (1024, 512)
    image = image.resize(new_size)

    left = random.randint(0, new_size[0] - 512)
    top = random.randint(0, new_size[1] - 512)

    right = left + 512
    bottom = top + 512

    crop = image.crop((left, top, right, bottom))

    output = pipe(
        prompt=prompt,
        image=crop,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=50,
        generator=generator,
    ).images[0]

    image.paste(output, (left, top))

    image.save(target_path)
