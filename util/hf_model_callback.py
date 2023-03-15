from pytorch_lightning.callbacks import ModelCheckpoint
from fsspec.core import url_to_fs
import pytorch_lightning as pl

class HfModelCheckpoint(ModelCheckpoint):
    def __init__(self, checkpoint_step = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # Checkpoint step
        self.checkpoint_step = checkpoint_step

        # Keep track of count
        self.batch = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # Iterate count
        self.batch += 1

        # Save model on checkpoint step
        if trainer.is_global_zero and self.batch % self.checkpoint_step == 0:
            trainer.lightning_module.save_model()