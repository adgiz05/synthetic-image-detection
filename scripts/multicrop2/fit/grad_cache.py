"""
GradCache implementation for SelfCon training.

Memory-efficient gradient accumulation for contrastive learning.
Adapted to work without external dependencies (RandContext included inline).
"""
from typing import List, Callable, Any
from contextlib import nullcontext
from collections import UserDict
import logging

import torch
import torch.distributed
import torch.distributed.nn
from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


class RandContext:
    """Context manager to save and restore random states for gradient checkpointing.

    This allows replaying the forward pass with the exact same random operations
    (e.g., dropout) that were used in the first pass.
    """

    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices = []
        self.fwd_gpu_states = []

        for tensor in tensors:
            if tensor.is_cuda:
                device = tensor.get_device()
                if device not in self.fwd_gpu_devices:
                    self.fwd_gpu_devices.append(device)
                    self.fwd_gpu_states.append(torch.cuda.get_rng_state(device))

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        for device, state in zip(self.fwd_gpu_devices, self.fwd_gpu_states):
            torch.cuda.set_rng_state(state, device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)


class GradCache:
    """Gradient Cache for memory-efficient contrastive learning.

    Splits batch into chunks, computes forward pass without gradients,
    builds gradient cache, then applies gradients in chunks.

    This allows training with larger effective batch sizes than would
    fit in GPU memory by accumulating gradients across chunks.

    Args:
        model: The neural network model
        chunk_size: Size of each chunk for gradient accumulation
        loss_fn: Loss function to compute
        loss_type: Type of loss ("SelfCon", "SupCon", "SimCLR")
        fp16: Whether to use mixed precision
        scaler: GradScaler for mixed precision training
        distributed: Whether running in DDP mode (enables all_gather)
        lightning_module: Reference to LightningModule for no_sync() in DDP
    """

    def __init__(
            self,
            model: nn.Module,
            chunk_size: int,
            loss_fn: Callable[..., Tensor],
            loss_type: str = "SelfCon",
            fp16: bool = False,
            scaler: GradScaler = None,
            distributed: bool = False,
            lightning_module: nn.Module = None,
    ):
        self.model = model
        for param in model.parameters():
            param.requires_grad_(True)
        self.chunk_size = chunk_size

        self.loss_fn = loss_fn
        self.loss_type = loss_type

        if fp16:
            assert scaler is not None, "mixed precision training requires a gradient scaler passed in"

        self.fp16 = fp16
        self.scaler = scaler
        self.distributed = distributed
        self.lightning_module = lightning_module

        self._get_input_tensors_strict = False

    def __call__(self, *args, **kwargs):
        return self.cache_step(*args, **kwargs)

    def split_inputs(self, model_input, chunk_size: int) -> List:
        """Split input tensor into chunks."""
        if isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))
        else:
            raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')

    def get_input_tensors(self, model_input) -> List[Tensor]:
        """Extract all tensors from potentially nested input structure."""
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def compute_loss(self, reps: Tensor, labels: Tensor = None, **loss_kwargs) -> Tensor:
        """Compute loss from representations."""
        loss = self.loss_fn(reps, labels, **loss_kwargs)
        return loss

    def forward_no_grad(
            self,
            model: nn.Module,
            model_input: Tensor,
    ) -> [Tensor, List[RandContext]]:
        """Forward pass without gradients, saving random states for replay."""
        rnd_states = []
        model_reps = []

        with torch.no_grad():
            for x in model_input:
                rnd_states.append(RandContext(*self.get_input_tensors(x)))
                if self.loss_type == "SelfCon":
                    y1, y2 = model(x)
                    model_reps.append(torch.cat([f.unsqueeze(1) for f in y1] + [y2.unsqueeze(1)], dim=1))
                else:
                    features = model(x)
                    bsz = int(features.shape[0]/2)
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    model_reps.append(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1))
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states

    @staticmethod
    def _all_gather_labels(labels):
        """Gather labels from all ranks for distributed contrastive loss."""
        gathered = [torch.zeros_like(labels) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, labels)
        return torch.cat(gathered, dim=0)

    def build_cache(self, reps: Tensor, labels: Tensor = None, **loss_kwargs) -> [List[Tensor], Tensor]:
        """Build gradient cache from representations.

        In distributed mode, uses all_gather to collect representations from all
        ranks so the contrastive loss sees the full global batch. Uses
        torch.distributed.nn.all_gather which supports autograd, so reps.grad
        contains only the local gradients.
        """
        reps = reps.detach().requires_grad_()
        with autocast() if self.fp16 else nullcontext():
            if self.distributed:
                all_reps = torch.cat(torch.distributed.nn.all_gather(reps), dim=0)
                all_labels = self._all_gather_labels(labels) if labels is not None else None
                loss = self.compute_loss(all_reps, all_labels, **loss_kwargs)
            else:
                loss = self.compute_loss(reps, labels, **loss_kwargs)

        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        cache = reps.grad

        return cache, loss.detach()

    def forward_backward(
            self,
            model: nn.Module,
            model_input,
            cached_gradients: List[Tensor],
            random_states: List[RandContext],
            no_sync_except_last: bool = False
    ):
        """Replay forward pass and apply cached gradients."""
        if no_sync_except_last:
            sync_target = self.lightning_module if self.lightning_module is not None else model
            sync_contexts = [sync_target.no_sync for _ in range(len(model_input) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_input))]

        for x, state, gradient, sync_context in zip(model_input, random_states, cached_gradients, sync_contexts):
            with sync_context():
                if self.loss_type == "SelfCon":
                    with state:
                        y1, y2 = model(x)
                    reps = torch.cat([f.unsqueeze(1) for f in y1] + [y2.unsqueeze(1)], dim=1)
                else:
                    with state:
                        features = model(x)
                    bsz = int(features.shape[0]/2)
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    reps = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                surrogate.backward()

    def cache_step(
            self,
            model_input: Tensor,
            labels: Tensor = None,
            no_sync_except_last: bool = False,
            num_crops: int = 1,
            **loss_kwargs
    ) -> Tensor:
        """Complete gradient cache step.

        Args:
            model_input: Input tensor to the model
            labels: Labels for supervised contrastive learning
            no_sync_except_last: For DDP, only sync on last chunk
            num_crops: Number of concentric crops per sample (1 = standard)
            **loss_kwargs: Additional arguments for loss function

        Returns:
            Computed loss value
        """
        model_input = self.split_inputs(model_input, self.chunk_size)

        model_reps, rnd_states = self.forward_no_grad(self.model, model_input)
        # model_reps: (B*K, n_views, feat_dim)

        if num_crops > 1:
            BK, n_views, feat_dim = model_reps.shape
            B = BK // num_crops
            # Reshape for loss: (B, K*n_views, feat_dim)
            reps_for_loss = model_reps.view(B, num_crops * n_views, feat_dim)
        else:
            reps_for_loss = model_reps

        cache, loss = self.build_cache(reps_for_loss, labels, **loss_kwargs)
        # cache shape matches reps_for_loss

        if num_crops > 1:
            # Reshape back to (B*K, n_views, feat_dim) for per-chunk forward_backward
            cache = cache.view(BK, n_views, feat_dim)

        cache = cache.split(self.chunk_size)
        use_no_sync = no_sync_except_last or (self.distributed and self.lightning_module is not None)
        self.forward_backward(self.model, model_input, cache, rnd_states, no_sync_except_last=use_no_sync)
        return loss
