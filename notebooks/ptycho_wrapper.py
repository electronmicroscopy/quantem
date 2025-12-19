from pathlib import Path
import numpy as np

import quantem as em
from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.core.visualization import show_2d

from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.core.visualization.visualization import show_2d
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.object_models import ObjectDIP, ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbePixelated
from quantem.diffractive_imaging.ptychography import Ptychography
from quantem.core.utils.diffractive_imaging_utils import fit_probe_circle

config.set_device(0)
print(config.get("device"))

def ptycho_wrapper(
        dset,
        tv_weight=0,
        probe_semiangle=20,
        probe_energy=200e3,
        probe_defocus=30,
        num_probes=1,
        num_slices=1,
        slice_thickness=1,
        obj_padding=0,
        batch_size=128,
        obj_lr=5e-1,
        probe_lr=1e-3,
        num_iters=25
    ):
    probe_qy0, probe_qx0, probe_R = fit_probe_circle(dset.dp_mean.array, show=False)
    dset.sampling[2] = probe_semiangle / probe_R
    dset.sampling[3] = probe_semiangle / probe_R
    dset.units[2:] = ["mrad", "mrad"]
    probe_R = probe_semiangle / dset.sampling[2]
    # print(dset)


    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(
        com_fit_function="constant",
        plot_rotation=False,
        plot_com=False,
        probe_energy=probe_energy,
        force_com_rotation=0, 
        force_com_transpose=False,
    )    


    # create a pixelated ptychography first, to fit the dft basis to.
    probe_params = {
        "energy" : probe_energy,
        "defocus" : probe_defocus,
        "semiangle_cutoff" : probe_semiangle, 
    }
    detector_model = DetectorPixelated() 

    # Set up ptychography model for excited state
    probe_model = ProbePixelated.from_params(
        num_probes=num_probes,
        probe_params=probe_params,
    )

    obj_model = ObjectPixelated.from_uniform(
        num_slices=num_slices, 
        slice_thicknesses=slice_thickness,
        obj_type='pure_phase',
    )
    probe_model = ProbePixelated.from_params(
        num_probes=num_probes,
        probe_params=probe_params,
    )
    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=obj_model,
        probe_model=probe_model,
        detector_model=detector_model,
        device='cuda',
    )
    ptycho.preprocess( 
        obj_padding_px=(obj_padding, obj_padding),
        batch_size=batch_size,
    )


    opt_params = { # except type, all args are passed to the optimizer (of type type)
            "object": {
                "type": "adam", 
                "lr": obj_lr
                , 
            },
            "probe": {
                "type": "adam", 
                "lr": probe_lr, 
            },
            # "dataset": { ### for optimizing over descan shifts and probe positions
            #     "type": "adam",
            #     "lr": 1e-4,
            # }
    }

    scheduler_params = {
        "object": { ## scheduler kwargs are passed to the scheduler (of type type)
            "type": "exp",
            "factor": 0.1,
            # "type": "plateau", ## i like plateau for many cases
        },
        "probe": {
            # "type": "exp",
            "type": "plateau",
            # "threshold": 1e-2, # e.g. plateau kwargs 
            # "patience": 100,
            # "cooldown": 100,
        },
        # "dataset": { 
        #     "type": "exp",  ## exp is also frequently used 
        #     "factor": 0.1,
        # }
    }

    constraints = {
        "object": {
            "tv_weight_xy": tv_weight, ## these are mostly the defaults 
            "tv_weight_z": 0.,
            "fix_potential_baseline": False,
            "identical_slices": True, ## default for this is False 
            "apply_fov_mask": False,
        },
        "probe": {
            "center_probe": False,
            "orthogonalize_probe": True,
        },
        "dataset":{
            "descan_tv_weight": 0,
            "descan_shifts_constant": False, 
        }
    }


    # ptycho.remove_optimizer("probe")
    ptycho.reconstruct(
        num_iters=num_iters,
        reset=True,
        autograd=True, 
        device='cuda',
        constraints=constraints, 
        optimizer_params=opt_params,
        scheduler_params=scheduler_params,
        batch_size=batch_size,
    ) # .visualize()

    return ptycho