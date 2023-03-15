import os
import sys
import colossalai
import json
import pytorch_lightning as pl
import torch
import torch.utils.data

from torch.utils.data import DataLoader
from pytorch_lightning.strategies.colossalai import ColossalAIStrategy

from data.dirtycollate import dirty_collate
from data.persimmon import PersimmonDataset
from model.config.diffusion import DiffusionConfig
from model.diffusion import DiffusionModel
from util.hf_model_callback import HfModelCheckpoint

if __name__ == "__main__":
    # Training config
    BATCH_SIZE = 4
    DATA_DIR = os.environ["DATA_DIR"]
    IMG_DIM = 512

    # Load JSON and deserialise into DiffusionConfig
    config_json = open("config/config.json", "r").read()
    config_dict = json.loads(config_json)
    diffusion_config = DiffusionConfig.from_dict(config_dict)

    # Create DiffusionModel
    diffusion_model : DiffusionModel = DiffusionModel(diffusion_config)

    # Data
    dataset = PersimmonDataset(root_dir=DATA_DIR, img_dim=IMG_DIM)
    training_set, validation_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])

    train_loader = DataLoader(training_set, batch_size=BATCH_SIZE, collate_fn=dirty_collate)
    val_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, collate_fn=dirty_collate)

    denoiser_trainer = pl.Trainer(
        max_steps=600000,
        accelerator="gpu", 
        precision=16,
        strategy=ColossalAIStrategy(
                use_chunk=True,
                enable_distributed_storage=True,
                placement_policy='cuda',
                initial_scale=16
            ),
        callbacks= [
            HfModelCheckpoint()
        ])
    
    denoiser_trainer.fit(diffusion_model, train_loader, val_loader)