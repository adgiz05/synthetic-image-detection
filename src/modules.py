from src.losses import MultiLabelLossImagiNet
from src.layers import ConResNet, CONRESNET_DEFAULT_CONFIG
from src.optimizers import OptimizerFactory
from src.schedulers import SchedulerFactory
from src.utils import load_resnet50_imagenet_weights

import torch
import pytorch_lightning as pl

class SelfContrastivePretrainingModule(pl.LightningModule):
    def __init__(self, model='conresnet', pretraining=True, optimizer_config={}, scheduler_config={}, config=None):
        super(SelfContrastivePretrainingModule, self).__init__()
        self.save_hyperparameters()
        match model:
            case 'conresnet':
                self.model = ConResNet(**CONRESNET_DEFAULT_CONFIG)
                if pretraining:
                    self.model = load_resnet50_imagenet_weights(self.model)
            case _:
                raise ValueError(f"Model {model} not recognized")

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.criterion = MultiLabelLossImagiNet()

    def forward(self, images, labels, return_features=False):
        """
        images: [B, N, C, H, W]
        labels: [B, num_classes]
        return_features: bool, if True, returns the features used for computing the loss
        """

        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W) # [B*N, C, H, W] (e.g [i_00, i_01, i_10, i_11, ...])

        hidden_states, last_hidden_state = self.model(images)  # S * [B*N, D], [B*N, D]

        # Positives for CL will be the same image in different views (N) and different hidden states (S + 1)
        S = len(hidden_states)
        features = torch.cat(hidden_states + [last_hidden_state], dim=0)  #[(S+1)*B*N, D]
        features = features.view((S + 1) * N, B, -1).permute(1, 0, 2)  #[B, N*(S+1), D]

        loss = self.criterion(features, labels)
        return {
            'loss': loss,
            'features': features if return_features else None,
        }
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        loss = self(images, labels)['loss']
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        loss = self(images, labels)['loss']
        self.log('val/loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = OptimizerFactory(**self.optimizer_config)(self.parameters())
        scheduler = SchedulerFactory(**self.scheduler_config)(optimizer)
        return [optimizer], [scheduler]