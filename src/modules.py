from src.losses import MultiLabelLossImagiNet
from src.layers import ConResNet, CONRESNET_DEFAULT_CONFIG, CLSHead
from src.optimizers import OptimizerFactory
from src.schedulers import SchedulerFactory
from src.utils import load_resnet50_imagenet_weights

import torch
import pytorch_lightning as pl
from transformers import AutoModel
from torchmetrics import Accuracy, F1Score, AUROC, MetricCollection
from src.losses import DualSyntheticLoss

NUM_GENERATORS = 4
NUM_SPECIFIC_MODELS = 8
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

        metrics = MetricCollection({
            'synth_acc' : Accuracy(task='binary', num_classes=2),
            'synth_f1' : F1Score(task='binary', num_classes=2, average='macro'),
            'synth_auroc' : AUROC(task='binary', num_classes=2),
            'model_acc' : Accuracy(task='multiclass', num_classes=NUM_GENERATORS),
            'model_f1' : F1Score(task='multiclass', num_classes=NUM_GENERATORS, average='macro'),
            'model_auroc' : AUROC(task='multiclass', num_classes=NUM_GENERATORS),
        })

        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

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
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        out = self(images, labels)

        self.train_metrics.update(
            {
                "synth_acc": (out["synthetic_logits"], labels[:, 0]),
                "synth_f1": (out["synthetic_logits"], labels[:, 0]),
                "synth_auroc": (out["synthetic_logits"], labels[:, 0]),
                "model_acc": (out["model_logits"], labels[labels[:, 1] == 1, GENERATOR_LABEL_IDX]),
                "model_f1": (out["model_logits"], labels[labels[:, 1] == 1, GENERATOR_LABEL_IDX]),
                "model_auroc": (out["model_logits"], labels[labels[:, 1] == 1, GENERATOR_LABEL_IDX]),
            }
        )
        self.log_dict({
            'train/synthetic_loss' : out['synthetic_loss'],
            'train/model_loss' : out['model_loss'],
            'train/loss' : out['loss']
        }, on_step=True, on_epoch=True)

        return out['loss']
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images, labels)
        self.val_metrics.update(
            {
                "synth_acc": (out["synthetic_logits"], labels[:, 0]),
                "synth_f1": (out["synthetic_logits"], labels[:, 0]),
                "synth_auroc": (out["synthetic_logits"], labels[:, 0]),
                "model_acc": (out["model_logits"], labels[labels[:, 1] == 1, GENERATOR_LABEL_IDX]),
                "model_f1": (out["model_logits"], labels[labels[:, 1] == 1, GENERATOR_LABEL_IDX]),
                "model_auroc": (out["model_logits"], labels[labels[:, 1] == 1, GENERATOR_LABEL_IDX]),
            }
        )
        self.log_dict({
            'val/synthetic_loss' : out['synthetic_loss'],
            'val/model_loss' : out['model_loss'],
            'val/loss' : out['loss']
        }, on_step=False, on_epoch=True)
        return out['loss']
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images, labels)
        self.test_metrics.update(
            {
                "synth_acc": (out["synthetic_logits"], labels[:, 0]),
                "synth_f1": (out["synthetic_logits"], labels[:, 0]),
                "synth_auroc": (out["synthetic_logits"], labels[:, 0]),
                "model_acc": (out["model_logits"], labels[labels[:, 1] == 1, GENERATOR_LABEL_IDX]),
                "model_f1": (out["model_logits"], labels[labels[:, 1] == 1, GENERATOR_LABEL_IDX]),
                "model_auroc": (out["model_logits"], labels[labels[:, 1] == 1, GENERATOR_LABEL_IDX]),
            }
        )
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
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()    

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