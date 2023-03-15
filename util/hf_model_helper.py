import json
import os
import random
import numpy as np
import pytorch_lightning as pl
import torch

from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import transforms
from PIL import Image

from model.config.diffusion import DiffusionConfig

"""
This class is used to load the HF models, and initialise them if they don't exist
"""
class HFDiffuserModelHelper:
    # Preprocessing the datasets.
    img_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_image(image):
        image = image.convert("RGB")
        return HFDiffuserModelHelper.img_transforms(image)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(tokenizer : CLIPTokenizer, q : str | list | np.ndarray, is_train=True):
        captions = []
        for caption in q:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Expected caption to be a string or list of strings, but got {type(caption)}"
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def encode_prompt(tokenizer : CLIPTokenizer, encoder : CLIPTextModel, text : list[str] = None, has_attention : bool = False, batch_size : int = None):
        """
        Convert a text prompt or lack thereof into a latent vector.
        """
        # Generate an empty prompt if none is provided
        if text is None and batch_size is None:
            raise Exception("Must provide either text or batch_size")
        elif text is None:
            text = [""] * batch_size

        # Tokenise text
        text_embeds = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Do attention mask
        attention_mask = None
        if has_attention:
            attention_mask = text_embeds.attention_mask

        # Encode text
        return encoder(**text_embeds.to(encoder.device))
    
    def numpy_to_pil(images):
        r"""
        Convert a numpy array to a PIL image.
        Pulled from diffusers.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def decode_latents_float(vae : AutoencoderKL, latents):
        r"""
        Decode latents to images.
        Pulled from diffusers.
        """
        latents = 1 / 0.18215 * latents
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        
        return images

    def decode_latents(latents):
        r"""
        Decode latents to images.
        Pulled from diffusers.
        """
        images = HFDiffuserModelHelper.decode_latents_float(latents)
        images = images.cpu().permute(0, 2, 3, 1).float()
        return images.numpy()