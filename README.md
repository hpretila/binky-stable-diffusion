# binky-stable-diffusion
Implementing training loop &amp; inference with HF Diffusers, on smaller datasets. This code is heavily based on the example code provided by the HuggingFace team, in their Diffusion repository, for training [text to image models](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).

This repository aims to test multiple libraries for training Text-to-Image Diffusion models, specifically the following:
* [HuggingFace Diffusers](https://huggingface.co/docs/diffusers/main/en/index) ([GitHub](https://github.com/huggingface/diffusers)), which integrates well with the HuggingFace ecosystem, allowing for easier experimentation.
* [Lightning AI](https://lightning.ai/) ([GitHub](https://github.com/lightning-ai/lightning)), which allows for more readable code, and better boilerplate.
* [Collosal AI](https://colossalai.org/) ([GitHub](https://github.com/hpcaitech/ColossalAI)), a backend that leverages distributed training techniques to allow training on less resources.

## Requirements 📋

Assuming you have an NVIDIA GPU with CUDA support, you will need the following:

* Python 3.10
* PyTorch 1.13
* Lightning 1.9.4
* HuggingFace Transformers 4.26
* HuggingFace Diffusers 0.14.0
* colossalai 0.2.5

## Set Up 🛠️

The repo needs some standing up. The following steps are required to get it running.

1. You may install colossal AI with the `CUDA_EXT=1` flag which will build all of the CUDA extensions prior to installation. See their [documentation](https://colossalai.org/docs/get_started/installation/) for more details.
    ```bash
    CUDA_EXT=1 pip install colossalai
    ```

    Otherwise requirements are listed in `requirements.txt`. To install them, run:
    ```bash
    pip install -r requirements.txt
    ```

2. The code assumes that you have a `weights` folder with HF checkpoints of at least a pretrained:
    * CLIP Encoder (`CLIPTextEncoder`) at `weights/text_encoder/`, expecting a `pytorch_model.bin` and `config.json`.
    * CLIP Tokenizer (`CLIPTextTokenizer`) at `weights/text_tokenizer/`, expecting a `special_tokens_map`, `tokenizer_config.json` and `vocab.json`.
    * VAE (`UNet2D`) at `weights/vae/`, expecting a `diffusion_pytorch_model.bin` and `config.json`.
    * Scheduler at `weights/scheduler/`, expecting a `scheduler_config.json`.
    * Safety Checker at `weights/safety_checker/`, expecting a `pytorch_model.bin` and `config.json`.
    * Feature Extractor at `weights/feature_extractor/`, expecting a `preprocessor_config.json`.

    You may add the conditional UNet at `weights/unet/`, expecting a `diffusion_pytorch_model.bin` and `config.json`.

3. Definitions for the Scheduler, VAE and conditional UNet should also be mirrored in the `config/` folder.

## Progress 🚧
### Roadmap 🗺️
- [x] Implement training loop for UNet2DConditional with ColossalAI backend.
- [ ] Implement checkpoint saving; see [this issue](https://github.com/hpretila/binky-stable-diffusion/issues/1), where compatibility with the Diffusers library seems to be an issue. There is some ad-hoc changes made to the Diffusers library for it to work, but it breaks for other cases.
- [ ] Documentation on use and architecture.
- [ ] A fully trained model? 😂