from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.constraints import BaseConstraints, Constraints
from quantem.core.ml.ddp import DDPMixin
from quantem.core.ml.loss_functions import get_loss_function
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.core.utils.rng import RNGMixin
from quantem.tomography.dataset_models import TomographyINRPretrainDataset


@dataclass(slots=True)
class DefaultConstraintsTomography(Constraints):
    """
    Data class for all constraints that can be applied to the object model.
    """

    # Hard Constraints
    positivity: bool = True
    shrinkage: float = 0.0
    circular_mask: bool = False
    fourier_filter: str | None = None  # Hamming, etc...

    # Soft Constraints
    tv_vol: float = 0.0

    soft_constraint_keys = ["tv_vol"]
    hard_constraint_keys = ["positivity", "shrinkage", "circular_mask", "fourier_filter"]


class ObjectBase(AutoSerialize, nn.Module, RNGMixin, OptimizerMixin):
    DEFAULT_LRS = {
        "object": 8e-6,
    }
    _token = object()
    """
    Base class for all ObjectModels to inherit from.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use a factory method to instantiate this class.")

        self._shape = shape

        # Initialize dependencies
        nn.Module.__init__(self)
        RNGMixin.__init__(self, rng=rng, device=device)
        OptimizerMixin.__init__(self)

        # Initialize a torch.zeros volume with the given shape
        self._obj = torch.zeros(self._shape, device=device, dtype=torch.float32)

        # --- Properties ---
        @property
        def shape(self) -> tuple[int, int, int]:
            """
            Shape of the object (x, y, z).
            """
            return self._shape

        @shape.setter
        def shape(self, shape: tuple[int, int, int]):
            self._shape = shape

        @property
        def obj(self) -> torch.Tensor:
            """
            Returns the object, should be implemented in subclasses.
            """
            raise NotImplementedError

        @abstractmethod
        def dtype(self) -> torch.dtype:
            """
            Returns the dtype of the object.
            """
            raise NotImplementedError

        @abstractmethod
        def forward(self, *args, **kwargs) -> torch.Tensor:
            """
            Forward pass, should be implemented in subclasses. Note for any nn.Module this is
            a required method.
            """
            raise NotImplementedError

        @abstractmethod
        def reset(self) -> None:
            """
            Reset the object, should be implemented in subclasses.
            """
            raise NotImplementedError

        @property
        def params(self) -> list[nn.Parameter]:
            """
            Get the parameters that should be optimized for this model.
            """
            raise NotImplementedError

        # --- Helper Functions ---
        def get_optimization_parameters(self) -> list[nn.Parameter]:
            """
            Get the parameters that should be optimized for this model.
            """
            return list[nn.Parameter](self.params())

        @abstractmethod  # Each subclass should implement this.
        def to(self, *args, **kwargs):
            """
            Move the object to a device
            """

            raise NotImplementedError


class ObjectConstraints(BaseConstraints, ObjectBase):
    DEFAULT_CONSTRAINTS = DefaultConstraintsTomography()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()

    def apply_hard_constraints(
        self,
        obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply hard constraints to the object model.

        Only hard constraint here is the positivity and shrinkage. TODO: Add the other hard constraints.
        """
        obj2 = obj.clone()
        if self.constraints.positivity:
            obj2 = torch.clamp(obj2, min=0.0, max=None)
        if self.constraints.shrinkage:
            obj2 = torch.max(obj2 - self.constraints.shrinkage, torch.zeros_like(obj2))

        # TODO: Need to implement the other hard constraints: Fourier Filter and Circular Mask.
        return obj2

    # def apply_soft_constraints(
    #     self,
    #     obj: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    # TODO: Already in BaseConstraints class.
    #     Apply soft constraints to the object model.

    #     Only soft constraint here is the TV loss.
    #     """
    #     return NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_tv_loss(self, obj: torch.Tensor, tv_weight: float = 0.0) -> torch.Tensor:
        """
        Get the TV loss for the object model. Must be implemented in each subclass.
        """
        raise NotImplementedError


class ObjectPixelated(ObjectConstraints):
    """
    Object model for pixelated objects.

    Supports: Conventional algorithms (SIRT, FBP), and AD-based reconstructions.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__(
            shape=shape,
            device=device,
            rng=rng,
            _token=self._token,
        )

    # --- Properties ----
    @property
    def obj(self) -> torch.Tensor:
        return self.apply_hard_constraints(self._obj)

    @obj.setter
    def obj(self, obj: torch.Tensor):
        self._obj = obj

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @shape.setter
    def shape(self, shape: tuple[int, int, int]):
        self._shape = shape

    @property
    def soft_loss(self) -> torch.Tensor:
        return self.apply_soft_constraints(self._obj)

    @property
    def name(self) -> str:
        return "ObjectPixelated"

    @property
    def obj_type(self) -> str:
        return "pixelated"

    def apply_soft_constraints(self, obj: torch.Tensor) -> torch.Tensor:
        soft_loss = torch.tensor(0.0, device=obj.device, dtype=obj.dtype, requires_grad=True)
        if self.constraints.tv_vol > 0:
            tv_loss = self.get_tv_loss(
                obj.unsqueeze(0).unsqueeze(0), factor=self.constraints.tv_vol
            )
            soft_loss += tv_loss
        return soft_loss

    # --- Forward method ---
    def forward(self, dummy_input=None) -> torch.Tensor:
        return self.obj

    # --- Defining the TV loss ---
    def get_tv_loss(self, obj: torch.Tensor, tv_weight: float = 1e-3) -> torch.Tensor:
        tv_d = torch.pow(obj[:, :, 1:, :, :] - obj[:, :, :-1, :, :], 2).sum()
        tv_h = torch.pow(obj[:, :, :, 1:, :] - obj[:, :, :, :-1, :], 2).sum()
        tv_w = torch.pow(obj[:, :, :, :, 1:] - obj[:, :, :, :, :-1], 2).sum()
        tv_loss = tv_d + tv_h + tv_w

        return tv_loss * tv_weight / (torch.prod(torch.tensor(obj.shape)))

    # --- Helper Functions ---
    def to(self, device: str):
        self._obj = self._obj.to(device)


class ObjectINR(ObjectConstraints, DDPMixin):
    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        model: nn.Module | None = None,
    ):
        super().__init__(
            shape=shape,
            device=device,
            rng=rng,
            _token=self._token,
        )
        self._pretrain_losses = []
        self._pretrain_lrs = []
        self.device = device

        # Register the network submodule (important: real nn.Module attribute)
        if model is not None:
            self.setup_distributed(device=device)
            self._model = self.build_model(model)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        obj_model = cls(
            shape=shape,
            device=device,
            rng=rng,
            model=model,  # ✅ build/register in __init__
        )

        obj_model.setup_distributed(device=device)
        obj_model.to(device)
        return obj_model

    # --- Properties ---

    @property
    def model(self) -> "nn.Module":
        """
        Returns the INR model.
        """
        return self._model

    # @model.setter
    # def model(self, model: "nn.Module"):
    #     """
    #     This doesn't work -- can't have setters for torch sub modules
    #     https://github.com/pytorch/pytorch/issues/52664

    #     For now, upon initialization private variable `._model` is set to the built model.
    #     """
    #     raise RuntimeError("\n\n\nsetting model, this shouldn't be reachable???\n\n\n")

    def apply_soft_constraints(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        soft_loss = torch.tensor(0.0, device=coords.device)
        if self.constraints.tv_vol > 0:
            num_tv_samples = min(10000, coords.shape[0])
            tv_indices = torch.randperm(coords.shape[0], device=coords.device)[:num_tv_samples]

            tv_coords = coords[tv_indices].detach().requires_grad_(True)

            tv_densities_recomputed = self.model(tv_coords)
            if tv_densities_recomputed.dim() > 1:
                tv_densities_recomputed = tv_densities_recomputed.squeeze(-1)

            grad_outputs = torch.autograd.grad(
                outputs=tv_densities_recomputed,
                inputs=tv_coords,
                grad_outputs=torch.ones_like(tv_densities_recomputed),
                create_graph=True,
            )[0]

            grad_norm = torch.norm(grad_outputs, dim=1)
            soft_loss += self.constraints.tv_vol * grad_norm.mean()

        return soft_loss

    @property
    def params(self):
        return self.model.parameters()

    # Pretraining
    @property
    def pretrained_weights(self) -> dict[str, torch.Tensor]:
        """get the pretrained weights of the INR model"""
        return self._pretrained_weights

    def _set_pretrained_weights(self, model: "torch.nn.Module"):
        """set the pretrained weights of the INR model"""
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Pretrained model must be a torch.nn.Module, got {type(model)}")
        self._pretrained_weights = deepcopy(model.state_dict())

    @property
    def pretrain_target(self) -> TomographyINRPretrainDataset:
        """get the pretrain target"""
        return self._pretrain_target

    @pretrain_target.setter
    def pretrain_target(self, target: TomographyINRPretrainDataset):
        """set the pretrain target"""
        self._pretrain_target = target

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the object.
        """
        # TODO: This is a temporary solution to get the dtype of the object.
        return torch.float32

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @shape.setter
    def shape(self, shape: tuple[int, int, int]):
        self._shape = shape

    # --- Helper Functions ---

    # Reset method that goes back to the pretrained weights.
    def reset(self):
        """reset the model to the pretrained weights"""
        self._model = self.build_model(
            self._model, self._pretrained_weights
        )  # Since loading the pretrained weights needs to be done in build_model.

    def get_optimization_parameters(self):
        return self.params

    # --- Forward Method ---

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """forward pass for the INR model"""

        all_densities = self.model(coords)

        if all_densities.dim() > 1:
            all_densities = all_densities.squeeze(-1)

        valid_mask = (
            (coords[:, 0] >= -1) & (coords[:, 0] <= 1) & (coords[:, 1] >= -1) & (coords[:, 1] <= 1)
        ).float()

        all_densities = all_densities * valid_mask

        return all_densities

    # Pretrain Loop

    def pretrain(
        self,
        pretrain_dataset: TomographyINRPretrainDataset,
        batch_size: int,
        reset: bool = False,
        num_iters: int = 10,
        num_workers: int = 0,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        loss_fn: Callable | str = "l1",
        apply_constraints: bool = False,
        show: bool = True,
    ):
        """
        Pretrain the INR model to fit target volume.
        """

        if (
            pretrain_dataset is not None
        ):  # Need to make a check if there's already a pretrain dataset to not go through with the setup again.
            self.pretrain_dataset = pretrain_dataset
            self.pretraining_dataloader, self.pretraining_sampler = self.setup_dataloader(
                pretrain_dataset, batch_size, num_workers=num_workers
            )

        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)
        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iters)

        if reset:
            raise NotImplementedError(
                "TODO: Resseting the model to the pretrained weights is not implemented yet. To make this work I would have to reinstantiate the model I think."
            )

        loss_fn = get_loss_function(loss_fn, self.dtype)

        self._pretrain(
            num_iters=num_iters,
            loss_fn=loss_fn,
            apply_constraints=apply_constraints,
        )

    def _pretrain(
        self,
        num_iters: int,
        loss_fn: Callable,
        apply_constraints: bool,
    ):
        self.model.train()
        optimizer = self.optimizer
        scheduler = self.scheduler

        for a0 in range(num_iters):
            epoch_loss = 0
            for batch_idx, batch in enumerate[Any](self.pretraining_dataloader):
                coords = batch["coords"].to(self.device, non_blocking=True)
                target = batch["target"].to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16, enabled=True
                ):
                    outputs = self.forward(coords)
                    loss = loss_fn(outputs, target)

                loss.backward()
                epoch_loss += loss.item()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

            self._pretrain_losses.append(epoch_loss / len(self.pretraining_dataloader))
            print(
                f"Epoch {a0 + 1}/{num_iters}, Pretrain Loss: {epoch_loss / len(self.pretraining_dataloader):.4f}"
            )
            self._pretrain_lrs.append(optimizer.param_groups[0]["lr"])

    def create_volume(
        self,
        return_vol: bool = False,
    ):
        N = max(self._shape)
        with torch.no_grad():
            coords_1d = torch.linspace(-1, 1, N)
            x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
            inputs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

            model = self.model.module if hasattr(self.model, "module") else self.model

            inference_batch_size = 5 * N * N
            total_samples = N**3
            samples_per_gpu = total_samples // self.world_size
            remainder = total_samples % self.world_size
            if self.global_rank < remainder:
                start_idx = self.global_rank * (samples_per_gpu + 1)
                end_idx = start_idx + samples_per_gpu + 1
            else:
                start_idx = self.global_rank * samples_per_gpu + remainder
                end_idx = start_idx + samples_per_gpu
            inputs_subset = inputs[start_idx:end_idx]
            num_samples = inputs_subset.shape[0]
            outputs_list = []
            for batch_start in range(0, num_samples, inference_batch_size):
                batch_end = min(batch_start + inference_batch_size, num_samples)
                batch_coords = inputs_subset[batch_start:batch_end].to(
                    self.device, non_blocking=True
                )
                batch_outputs = model(batch_coords)
                if batch_outputs.dim() > 1:
                    batch_outputs = batch_outputs.squeeze(-1)
                outputs_list.append(batch_outputs.cpu())

            outputs = torch.cat(outputs_list, dim=0)

            if self.world_size > 1:
                output_size = torch.tensor(outputs.shape[0], device=self.device, dtype=torch.long)
                all_sizes = [
                    torch.zeros(1, device=self.device, dtype=torch.long)
                    for _ in range(self.world_size)
                ]
                dist.all_gather(all_sizes, output_size)

                max_size = max(size.item() for size in all_sizes)

                if outputs.shape[0] < max_size:
                    padding = torch.zeros(
                        max_size - outputs.shape[0], device=outputs.device, dtype=outputs.dtype
                    )
                    outputs_padded = torch.cat([outputs, padding], dim=0).to(self.device)
                else:
                    outputs_padded = outputs.to(self.device)

                gathered_outputs = [
                    torch.empty(max_size, device=self.device, dtype=outputs.dtype)
                    for _ in range(self.world_size)
                ]
                dist.all_gather(gathered_outputs, outputs_padded.contiguous())

                trimmed_outputs = []
                for rank, size in enumerate(all_sizes):
                    trimmed_outputs.append(gathered_outputs[rank][: size.item()])

                pred_full = torch.cat(trimmed_outputs, dim=0).reshape(N, N, N).float()
            else:
                pred_full = outputs.reshape(N, N, N).float()

            if return_vol:
                return pred_full.detach().cpu()
            else:
                self._obj = pred_full.detach().cpu()

    def get_tv_loss(
        self,
        coords: torch.Tensor,
    ):
        tv_loss = torch.tensor(0.0, device=coords.device)

        num_tv_samples = min(10000, coords.shape[0])
        tv_indices = torch.randperm(coords.shape[0], device=coords.device)[:num_tv_samples]

        tv_coords = coords[tv_indices].detach().requires_grad_(True)

        tv_densities_recomputed = self.forward(tv_coords)

        if tv_densities_recomputed.dim() > 1:
            tv_densities_recomputed = tv_densities_recomputed.squeeze(-1)

        grad_outputs = torch.autograd.grad(
            outputs=tv_densities_recomputed,
            inputs=tv_coords,
            grad_outputs=torch.ones_like(tv_densities_recomputed),
            create_graph=True,
        )[0]

        grad_norm = torch.norm(grad_outputs, dim=1)

        tv_loss += self.constraints.tv_vol * grad_norm.mean()
        return tv_loss

    def to(self, device: str):
        # self._model = self._model.to(device)
        self._obj = self._obj.to(device)


ObjectModelType = ObjectPixelated | ObjectINR
