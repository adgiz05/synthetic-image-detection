from src.datamodules import NoiseDataModule
from src.modules import NoiseClassifier

import pytorch_lightning as pl
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def setup(config=None):
    pl.seed_everything(42)
    dm = NoiseDataModule(batch_size=64, num_workers=8, dataset_config={'patch_size':128}, cached=True)
    model = NoiseClassifier(lr=1e-3)

    logger = pl.loggers.WandbLogger(project='imaginet-noise', name='baseline')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/loss',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=400,
        # precision='16-mixed',
        accelerator='gpu',
        devices=[5],
        default_root_dir='imaginet-noise',
        logger=logger,
        callbacks=[checkpoint_callback],
        # check_val_every_n_epoch=config.check_val_every_n_epoch,
        accumulate_grad_batches=1, # adjust
        benchmark=True,
    )

    return dm, model, trainer

if __name__ == '__main__':
    dm, model, trainer = setup()
    trainer.fit(model, datamodule=dm)