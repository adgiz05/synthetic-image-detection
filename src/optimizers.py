import torch

class OptimizerFactory:
    def __init__(self, name='sgd', lr=5e-3, weight_decay=1e-4):
        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay

    def __call__(self, params):
        match self.name:
            case 'sgd':
                return torch.optim.SGD(
                    params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    momentum=0.9,
                    nesterov=True
                )
            case 'adamw':
                return torch.optim.AdamW(
                    params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case _:
                raise ValueError(f"Optimizer {self.name} not recognized")