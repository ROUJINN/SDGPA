import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image
from PIL import Image

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
    left_crop = image.crop((0, 0, 512, 512))
    right_crop = image.crop((512, 0, 1024, 512))
    left_output = pipe(
        prompt=prompt,
        image=left_crop,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=50,
        generator=generator,
    ).images[0]
    right_output = pipe(
        prompt=prompt,
        image=right_crop,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=50,
        generator=generator,
    ).images[0]
    merged_image = Image.new("RGB", (1024, 512))
    merged_image.paste(left_output, (0, 0))
    merged_image.paste(right_output, (512, 0))
    merged_image.save(target_path)
