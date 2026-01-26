import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.ddp import DDPMixin
from quantem.core.utils.rng import RNGMixin
from quantem.tomography.dataset_models import DatasetModelType, TomographyDatasetBase
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.tomography.object_models import ConstraintsTomography, ObjectBase, ObjectPixelated


class TomographyBase(AutoSerialize, RNGMixin, DDPMixin):
    """
    A base class for performing electron tomography reconstructions.

    Should have all the default attributes needed for tomography reconstructions.
    """

    _token = object()

    def __init__(
        self,
        dset: DatasetModelType,
        obj_model: ObjectBase,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        # if _token is not self._token: # TODO: Idk why this isn't working.
        #     raise RuntimeError("Use Dataset.from_* to instantiate this class.")

        super().__init__()
        self.obj_model = obj_model

        self.dset = dset
        self.rng = rng
        self.device = device
        self.logger = logger

        # Loss
        self._epoch_losses: list[float] = []
        # DDP Initialization
        # print("Checking if obj_model is a ObjectPixelated: ", not isinstance(obj_model, ObjectPixelated))
        if not isinstance(obj_model, ObjectPixelated):
            print("Setting up DDP for obj_model")
            self.setup_distributed(device=device)
            # self._obj_model._model = self.build_model(obj_model) # Assuming when object is initialized it's already wrapped in DDP?
            # print("After DDP Setup", self._obj_model)

        self.dset = dset
        self.dset.to(device)

    # --- Properties ---
    @property
    def dset(self) -> DatasetModelType:
        return self._dset

    @dset.setter
    def dset(self, new_dset: DatasetModelType):
        if not isinstance(new_dset, TomographyDatasetBase) and "TomographyDataset" not in str(
            type(new_dset)
        ):
            raise TypeError(f"dset should be a TomographyDataset, got {type(new_dset)}")
        self._dset = new_dset

    @property
    def obj_type(self) -> str:
        return self.obj_model.obj_type

    @property
    def obj_model(self) -> ObjectBase:
        return self._obj_model

    @obj_model.setter
    def obj_model(self, obj_model: ObjectBase):
        # if not isinstance(obj_model, ObjectBase):
        #     raise TypeError(f"obj_model should be a ObjectBase, got {type(obj_model)}")
        self._obj_model = obj_model

    @property
    def constraints(self) -> ConstraintsTomography:
        return self.obj_model.constraints

    @constraints.setter
    def constraints(self, constraints: ConstraintsTomography):
        if not isinstance(constraints, ConstraintsTomography):
            raise TypeError(
                f"constraints should be a ConstraintsTomography, got {type(constraints)}"
            )
        self.obj_model.constraints = constraints

    @property
    def logger(self) -> LoggerTomography | None:
        return self._logger

    @logger.setter
    def logger(self, logger: LoggerTomography | None):
        if not isinstance(logger, LoggerTomography) and logger is not None:
            raise TypeError(f"logger should be a LoggerTomography, got {type(logger)}")
        self._logger = logger

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str):
        print("Device trying to set: ", device)
        # if not isinstance(device, str):
        #     raise TypeError(f"device should be a str, got {type(device)}")
        self._device = device
        self.to(device)

    @property
    def epoch_losses(self) -> NDArray:
        """
        Returns the fidelity loss for each epoch ran.
        """
        return np.array(self._epoch_losses)

    @property
    def num_epochs(self) -> int:
        return len(self._epoch_losses)

    # --- Helper Functions ---

    def to(self, device: str):
        self.obj_model.to(device)
        self.dset.to(device)
