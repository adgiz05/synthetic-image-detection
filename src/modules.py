from src.losses import MultiLabelLossImagiNet
from src.layers import ConResNet, CONRESNET_DEFAULT_CONFIG, CLSHead
from src.optimizers import OptimizerFactory
from src.schedulers import SchedulerFactory
from src.utils import load_resnet50_imagenet_weights

import torch
import kornia
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from torchmetrics import Accuracy, F1Score, AUROC, MetricCollection
from src.losses import DualSyntheticLoss

# NUM_GENERATORS = 4
NUM_GENERATORS = 8
NUM_SPECIFIC_MODELS = 8

SYNTHETIC_LABEL_IDX = 0
GENERATOR_LABEL_IDX = 2
SPECIFIC_MODEL_LABEL_IDX = 3

class ImageModule(pl.LightningModule):
    def __init__(self, model_id='google/vit-base-patch16-224-in21k', optimizer_config={}, scheduler_config={}, loss_config={}, config=None):
        super(ImageModule, self).__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(model_id)

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.criterion = DualSyntheticLoss(**loss_config)

        # Classification Heads
        self.synthetic_head = CLSHead(self.model.config.hidden_size, 2, _type='mlp')
        self.model_head = CLSHead(self.model.config.hidden_size, NUM_GENERATORS, _type='mlp')

        # Synthetic vs Real Metrics
        synthetic_metrics_collection = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=2),
            "f1": F1Score(task="multiclass", num_classes=2, average="macro"),
            "auroc": AUROC(task="multiclass", num_classes=2),
        })

        self.synthetic_metrics = torch.nn.ModuleDict({
            'train_split' : synthetic_metrics_collection.clone(prefix="train/synthetic_"),
            'val_split' : synthetic_metrics_collection.clone(prefix="val/synthetic_"),
            'test_split' : synthetic_metrics_collection.clone(prefix="test/synthetic_")
        })

        # Model Detection Metrics
        model_metrics_collection = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=NUM_GENERATORS),
            "f1": F1Score(task="multiclass", num_classes=NUM_GENERATORS, average="macro"),
            "auroc": AUROC(task="multiclass", num_classes=NUM_GENERATORS),
        })

        self.model_metrics = torch.nn.ModuleDict({
            'train_split' : model_metrics_collection.clone(prefix="train/model_"),
            'val_split' : model_metrics_collection.clone(prefix="val/model_"),
            'test_split' : model_metrics_collection.clone(prefix="test/model_")
        })

    def forward(self, images, labels, return_features=False):
        """
        images: [B, C, H, W]
        labels: [B, num_classes]
        return_features: bool, if True, returns the features used for computing the loss
        """
        out = self.model(pixel_values=images).pooler_output # [B, hidden_size]
        synthetic_logits = self.synthetic_head(out) # [B, 2]
        model_logits = self.model_head(out) # [B, NUM_GENERATORS]

        synthetic_loss, model_loss = self.criterion(synthetic_logits, model_logits, labels)
        return {
            'synthetic_logits' : synthetic_logits,
            'model_logits' : model_logits,
            'synthetic_loss' : synthetic_loss,
            'model_loss' : model_loss,
            'loss' : synthetic_loss + model_loss,
            **({'features' : out} if return_features else {})
        }
    
    def _update_metrics(self, split, out, labels):
        split = f'{split}_split'

        synthetic_preds = out['synthetic_logits']
        synthetic_labels = labels[:, 0]

        self.synthetic_metrics[split].update(synthetic_preds, synthetic_labels)

        synthetic_mask = synthetic_labels == 1
        if synthetic_mask.any():
            model_preds = out['model_logits'][synthetic_mask]
            model_labels = labels[synthetic_mask, GENERATOR_LABEL_IDX]

            self.model_metrics[split].update(model_preds, model_labels)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        out = self(images, labels)

        # Compute metrics
        self._update_metrics('train', out, labels)
        self.log_dict({
            'train/synthetic_loss' : out['synthetic_loss'],
            'train/model_loss' : out['model_loss'],
            'train/loss' : out['loss']
        }, on_step=True, on_epoch=True)

        return out['loss']
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images, labels)
        self._update_metrics('val', out, labels)
        self.log_dict({
            'val/synthetic_loss' : out['synthetic_loss'],
            'val/model_loss' : out['model_loss'],
            'val/loss' : out['loss']
        }, on_step=False, on_epoch=True)
        return out['loss']
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images, labels)
        self._update_metrics('test', out, labels)
        self.log_dict({
            'test/synthetic_loss' : out['synthetic_loss'],
            'test/model_loss' : out['model_loss'],
            'test/loss' : out['loss']
        }, on_step=False, on_epoch=True)
        return out['loss']

    def configure_optimizers(self):
        optimizer = OptimizerFactory(**self.optimizer_config)(self.parameters())
        scheduler = SchedulerFactory(**self.scheduler_config)(optimizer)
        return [optimizer], [scheduler]
    
    def on_train_epoch_end(self):
        self.log_dict(self.synthetic_metrics['train_split'].compute(), prog_bar=True)
        self.log_dict(self.model_metrics['train_split'].compute(), prog_bar=True)
        self.synthetic_metrics['train_split'].reset()
        self.model_metrics['train_split'].reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.synthetic_metrics['val_split'].compute(), prog_bar=True)
        self.log_dict(self.model_metrics['val_split'].compute(), prog_bar=True)
        self.synthetic_metrics['val_split'].reset()
        self.model_metrics['val_split'].reset()

    def on_test_epoch_end(self):
        self.log_dict(self.synthetic_metrics['test_split'].compute(), prog_bar=True)
        self.log_dict(self.model_metrics['test_split'].compute(), prog_bar=True)
        self.synthetic_metrics['test_split'].reset()
        self.model_metrics['test_split'].reset()  

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

import kornia.augmentation as K
import kornia.geometry.transform as KT

class NoiseClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, kernel_size=(7,7), sigma=(1.5,1.5), patch_size=128):
        super().__init__()
        self.save_hyperparameters()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.patch_size = patch_size

        # CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        self.classifier = nn.Linear(64, 1)

        # --- Augmentación/crop en GPU ---
        self.random_crop = K.RandomCrop((patch_size, patch_size), p=1.0)
        self.center_crop = lambda x: KT.center_crop(x, (patch_size, patch_size))

    def crop_or_pad(self, x, train=True):
        B, C, H, W = x.shape
        ps = self.patch_size

        # Padding centrado si hace falta
        pad_h = max(ps - H, 0)
        pad_w = max(ps - W, 0)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = KT.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

        # Ahora siempre H >= ps y W >= ps
        if train and self.training:  # modo entrenamiento
            x = self.random_crop(x)
        else:  # validación/test
            x = self.center_crop(x)
        return x

    def preprocess_noise(self, x):
        # x: (B,1,H,W) float tensor
        denoised = kornia.filters.guided_blur(x, x, (5,5), eps=1e-2)  # window y regularización
        noise = (x - denoised).abs()
        return noise

    def forward(self, x):
        # Crop/pad en GPU
        x = self.crop_or_pad(x, train=self.training)

        # Residual
        x = self.preprocess_noise(x)

        # CNN + GAP
        x = self.conv_layers(x)
        x = x.mean(dim=[2,3])        # [B,64]
        logits = self.classifier(x)  # [B,1]
        return logits.squeeze(-1)    # [B]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)