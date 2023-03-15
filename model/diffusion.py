import math
import diffusers
import pytorch_lightning as pl
import torch

from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.lr_scheduler import LinearWarmupLR

from model.config.diffusion import DiffusionConfig
from util.hf_model_loader import HFDiffuserModelLoader as ModelLoader
from util.hf_model_helper import HFDiffuserModelHelper as ModelHelper

class DiffusionModel(pl.LightningModule):
    # Initialize the model
    def __init__(self, config : DiffusionConfig) -> None:
        super().__init__()
        
        # Save our config
        self.config = config

    # Load the model
    def load_model(self) -> None:
        # Load our noise scheduler
        self.noise_scheduler = ModelLoader.load_hf_scheduler(self.config)

        # Load our language models
        self.tokenizer, self.text_encoder = ModelLoader.load_hf_clip_model(self.config)
        self.vae, self.unet = ModelLoader.load_hf_diff_model(self.config)

        # Freeze anything that's not needed
        self.text_encoder.requires_grad_(self.config.train_text_encoder)
        self.vae.requires_grad_(self.config.train_vae)
        self.unet.requires_grad_(self.config.train_unet)

        # Use memory efficient attention
        if self.config.use_xformers:
            self.unet.set_use_memory_efficient_attention_xformers(True)

    def configure_sharded_model(self) -> None:
        self.load_model()

    def forward(self, q, img, timestep = 1):
        raise NotImplementedError("This model does not support forward yet")
    
    def training_step(self, batch, batch_idx):
        img, q = batch

        # Call parameters
        batch_size = len(q)

        # Convert input images into latent space
        latents = self.vae.encode(img.to(self.vae.device)).latent_dist.sample()
        latents = latents * 0.18215 # self.vae.config.scaling_factor

        # Generate some noise to add later during training
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latent space
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Encode prompt
        text_tokens = ModelHelper.tokenize_captions(self.tokenizer, q)
        prompt_embeds = self.text_encoder.forward(text_tokens.to(self.text_encoder.device)).last_hidden_state

        # Move everything to the correct device
        timesteps = timesteps.to(self.unet.device)
        noise = noise.half().to(self.unet.device)
        prompt_embeds = prompt_embeds.half().to(self.unet.device)
        noisy_latents = noisy_latents.half().to(self.unet.device)
        latents = latents.half().to(self.unet.device)

        # Predict the noise residual and compute loss
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample
        target = self.noise_scheduler.add_noise(latents, noise, timesteps-1)       
        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        return loss
    
    def configure_optimizers(self):
        opt = HybridAdam(self.unet.parameters(), lr=0.00008)
        scheduler = LinearWarmupLR(opt, self.trainer.max_steps, 1000)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
    
    def train_dataloader(self):
        pass
    
    def save_model(self):
        # Save the model
        ModelLoader.save_without_pipeline(self.config, self.unet)
        
        # pipeline : StableDiffusionPipeline = ModelLoader.pipeline_it(config=self.config, text_encoder=self.text_encoder, 
        #                                                              vae=self.vae, unet=self.unet, tokenizer=self.tokenizer, 
        #                                                              scheduler=self.noise_scheduler)
                                                                     
        # pipeline.save_pretrained(self.config.pretrained_model_name_or_path)

        pass
