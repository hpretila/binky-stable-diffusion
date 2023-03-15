import json
import os
import diffusers
import pytorch_lightning as pl
import torch

from colossalai.utils import load_checkpoint, save_checkpoint
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor

from model.config.diffusion import DiffusionConfig

"""
This class is used to load the HF models, and initialise them if they don't exist
"""
class HFDiffuserModelLoader:
    # Load the noise scheduler
    def load_hf_scheduler(config: DiffusionConfig) -> None:
        config_json = open(config.config_scheduler, "r").read()
        config_dict = json.loads(config_json)

        return PNDMScheduler.from_config(config_dict)

    # Load the safety models
    def load_hf_safety_models(config: DiffusionConfig) -> tuple[StableDiffusionSafetyChecker, CLIPFeatureExtractor]:
        base_path : str = config.pretrained_model_name_or_path
        
        # Load our config
        feature_extractor_config_json = open(os.path.join(base_path,"feature_extractor","preprocessor_config.json"), "r").read()
        feature_extractor_config_dict = json.loads(feature_extractor_config_json)
    
        with torch.no_grad():
            safety_checker : StableDiffusionSafetyChecker = StableDiffusionSafetyChecker.from_pretrained(base_path, subfolder='safety_checker')
            feature_extractor : CLIPFeatureExtractor = CLIPFeatureExtractor(**feature_extractor_config_dict)

        return safety_checker, feature_extractor

    # Load the language models
    def load_hf_clip_model(config: DiffusionConfig) -> tuple[CLIPTokenizer, CLIPTextModel]:
        base_path : str = config.pretrained_model_name_or_path

        tokenizer : CLIPTokenizer = CLIPTokenizer.from_pretrained(base_path, subfolder='tokenizer')
        text_encoder : CLIPTextModel = CLIPTextModel.from_pretrained(base_path, subfolder='text_encoder')

        return tokenizer, text_encoder

    # Load the diff model, initialise if there are any missing files
    def load_hf_diff_model(config: DiffusionConfig) -> tuple[AutoencoderKL, UNet2DConditionModel]:
        base_path : str = config.pretrained_model_name_or_path
        vae_path : str = base_path + "/vae/diffusion_pytorch_model.bin"
        unet_path : str = base_path + "/unet/diffusion_pytorch_model.bin"
        
        vae : AutoencoderKL = None
        unet : UNet2DConditionModel = None

        # Check if VAE exists, otherwise we're starting from scratch
        if not os.path.exists(vae_path):
            # Load our config
            vae_config_json = open(config.config_vae, "r").read()
            vae_config_dict = json.loads(vae_config_json)

            # Initialise model
            vae : AutoencoderKL = AutoencoderKL(**vae_config_dict)
        else:
            # Load the model off the street 🤭
            vae  = AutoencoderKL.from_pretrained(base_path, subfolder='vae')

        # Check if UNet exists, otherwise we're starting from scratch
        if not os.path.exists(unet_path):
            # Load our config
            unet_config_json = open(config.config_unet, "r").read()
            unet_config_dict = json.loads(unet_config_json)

            # Initialise model
            unet : UNet2DConditionModel = UNet2DConditionModel(**unet_config_dict)
        else:
            # Load our config
            unet_config_json = open(config.config_unet, "r").read()
            unet_config_dict = json.loads(unet_config_json)

            # Load the model off the street 🤭
            unet : UNet2DConditionModel = UNet2DConditionModel.from_pretrained(base_path, subfolder='unet')

        return vae, unet

    def pipeline_it(config: DiffusionConfig, vae: AutoencoderKL, unet: UNet2DConditionModel, 
                           tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, scheduler) -> None:
        # Load our safety models
        safety_checker, feature_extractor = HFDiffuserModelLoader.load_hf_safety_models(config)

        # Pipeline it all up
        pipeline : StableDiffusionPipeline = StableDiffusionPipeline(
            unet=unet, vae=vae, text_encoder=text_encoder, 
            tokenizer=tokenizer, safety_checker=safety_checker, 
            feature_extractor=feature_extractor, scheduler=scheduler
        )

        return pipeline
    
    def save_without_pipeline(config: DiffusionConfig, unet: UNet2DConditionModel) -> None:
        print("Saving model without pipeline")
        # ColossalAI save_checkpoint
        unet.save_pretrained(config.pretrained_model_name_or_path + "/unet")
        # save_checkpoint(config.pretrained_model_name_or_path + "/unet/diffusion_pytorch_model.bin", 1, unet)
        # Copy the config file to the new location
        #os.system("cp " + config.config_unet + " " + config.pretrained_model_name_or_path + "/unet/config.json")
