import torch

class SchedulerFactory:
    def __init__(self, name='linear+cosine', max_epochs=1000, scheduler_skip=10):
        self.name = name
        self.T_max = max_epochs
        self.scheduler_skip = scheduler_skip

    def __call__(self, optimizer):
        match self.name:
            case 'linear':
                return torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    total_iters=self.T_max,
                )
            
            case 'cosine':
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.T_max,
                )
            
            case 'linear+cosine':
                linear_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    total_iters=self.scheduler_skip,
                )

                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.T_max - self.scheduler_skip,
                )

                return torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[linear_scheduler, cosine_scheduler],
                    milestones=[self.scheduler_skip]
                )
        
            case _:
                raise ValueError(f"Scheduler {self.name} not recognized")