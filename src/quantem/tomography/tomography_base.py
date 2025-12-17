import numpy as np

from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.rng import RNGMixin
from quantem.tomography.dataset_models import DatasetModelType
from quantem.tomography.logger_tomography import LoggerTomography


class TomographyBase(AutoSerialize, RNGMixin):
    """
    A base class for performing electron tomography reconstructions.

    Should have all the default attributes needed for tomography reconstructions.
    """

    _token = object()

    def __init__(
        self,
        dset: DatasetModelType,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        pass
