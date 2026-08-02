import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.ddp import DDPMixin
from quantem.core.utils.rng import RNGMixin
from quantem.tomography.dataset_models import DatasetModelType, TomographyDatasetBase
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.tomography.object_models import (
    ObjConstraintParams,
    ObjConstraintsType,
    ObjectINR,
    ObjectModelType,
    ObjectTensorDecomp,
)


class TomographyBase(AutoSerialize, RNGMixin, DDPMixin):
    """
    A base class for performing electron tomography reconstructions.

    Should have all the default attributes needed for tomography reconstructions.
    """

    _token = object()

    def __init__(
        self,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        rng: np.random.Generator | int | None = None,
        verbose: int | bool = True,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use .from_* to instantiate this class.")

        super().__init__()
        self.obj_model = obj_model

        self.dset = dset
        self.verbose = verbose
        self.rng = rng
        self.device = device
        self.logger = logger

        # Loss
        self._epoch_losses: list[float] = []
        self._consistency_losses: list[float] = []
        self._val_losses: list[float] = []
        self._lrs: dict[str, list] = {}
        # DDP Initialization
        if isinstance(obj_model, ObjectINR) or isinstance(obj_model, ObjectTensorDecomp):
            self.setup_distributed(device=device)
            if self.global_rank == 0:
                print("Setting up DDP for obj_model")

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
    def verbose(self) -> int | bool:
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: int | bool):
        self._verbose = verbose

    @property
    def obj_model(self) -> ObjectModelType:
        return self._obj_model

    @obj_model.setter
    def obj_model(self, obj_model: ObjectModelType):
        self._obj_model = obj_model

    @property
    def constraints(self) -> ObjConstraintsType:  # TODO: Also looks at Dataset constraints
        return self.obj_model.constraints

    @constraints.setter
    def constraints(self, constraints: ObjConstraintsType | dict | None):
        if constraints is None:
            return
        elif isinstance(constraints, dict):
            self.obj_model.constraints = ObjConstraintParams.parse_dict(constraints)
        elif isinstance(constraints, ObjConstraintsType):
            self.obj_model.constraints = constraints
        else:
            raise ValueError(f"Invalid constraints type: {type(constraints)}")

    @property
    def logger(self) -> LoggerTomography | None:
        return self._logger

    @logger.setter
    def logger(self, logger: LoggerTomography | None):
        if not isinstance(logger, LoggerTomography) and logger is not None:
            raise TypeError(f"logger should be a LoggerTomography, got {type(logger)}")
        self._logger = logger

    @property
    def epoch_losses(self) -> NDArray:
        """
        Returns the fidelity loss for each epoch ran.
        """
        return np.array(self._epoch_losses)

    @property
    def consistency_losses(self) -> NDArray:
        """
        Returns the consistency loss for each epoch ran.
        """
        return np.array(self._consistency_losses)

    @property
    def learning_rates(self) -> dict[str, list]:
        """
        Returns the learning rates for each epoch ran.
        """
        return self._lrs

    def append_learning_rates(self, learning_rates: dict[str, float]):
        """
        Appends the learning rates for each epoch ran.
        """
        for key, value in learning_rates.items():
            if key not in self._lrs:
                self._lrs[key] = []
            self._lrs[key].append(float(value))

    @property
    def num_epochs(self) -> int:
        return len(self._epoch_losses)

    # --- Helper Functions ---

    def to(self, device: str):
        self.obj_model.to(device)
        self.dset.to(device)
        self.device = device
