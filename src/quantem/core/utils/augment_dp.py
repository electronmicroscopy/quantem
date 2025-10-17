import os
from typing import TYPE_CHECKING, Union

import numpy as np
import scipy.ndimage as ndi
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.utils import array_funcs as af
from quantem.core.utils.utils import RNGMixin
from quantem.core.utils.validators import validate_tensor

if TYPE_CHECKING:
    import torch
    import torch.nn.functional as F

if config.get("has_torch"):
    import torch
    import torch.nn.functional as F

ArrayLike = Union[np.ndarray, "torch.Tensor"]

# TODO
# add dark background
# add gaussian noise


class DPAugmentor(RNGMixin):
    def __init__(
        self,
        add_bkg: bool = False,
        bkg_weight: list[float] | float = [0.001, 0.05],
        bkg_q: list[float] | float = [0.01, 0.1],
        add_shot: bool = False,
        e_dose: list[float] | float = [1e4, 1e7],
        add_shift: bool = False,
        xshift: list[float] | float = [0, 10],
        yshift: list[float] | float = [0, 10],
        add_ellipticity: bool = False,
        ellipticity_scale: list[float] | float = [0, 0.15],
        add_ellipticity_to_label: bool = True,
        add_salt_and_pepper: bool = False,
        salt_and_pepper: list[float] | float = [0, 5e-4],
        add_scale: bool = False,
        scale_factor: list[float] | float = [0.9, 1.1],
        add_blur: bool = False,
        blur_sigma: list[float] | float = [0.0, 1.5],
        add_flipshift: bool = False,
        free_rotation: bool = False,
        rotation_range: list[float] | float = [-180, 180],
        log_file: os.PathLike | None = None,
        rng: np.random.Generator | int | None = None,
        device: str = "cpu",
    ):
        """
        Initialize diffraction pattern augmentor with configurable transformations.

        Parameters
        ----------
        add_bkg : bool, default=False
            Enable inelastic plasmon background addition via convolution with probe.
        bkg_weight : list[float] | float, default=[0.001, 0.05]
            Range for background weight (fraction of total intensity).
        bkg_q : list[float] | float, default=[0.01, 0.1]
            Range for plasmon scattering parameter q₀ in 1/(q² + q₀²) form factor.

        add_shot : bool, default=False
            Enable Poisson shot noise based on electron dose.
        e_dose : list[float] | float, default=[1e4, 1e7]
            Range for electron dose (electrons per image) for shot noise.

        add_shift : bool, default=False
            Enable random translation of the diffraction pattern.
        xshift : list[float] | float, default=[0, 10]
            Range for horizontal shift in pixels (applied with random sign).
        yshift : list[float] | float, default=[0, 10]
            Range for vertical shift in pixels (applied with random sign).

        add_ellipticity : bool, default=False
            Enable elliptical distortion of the diffraction pattern.
        ellipticity_scale : list[float] | float, default=[0, 0.15]
            Range for ellipticity strength parameter (std dev of Gaussian distortion).
        add_ellipticity_to_label : bool, default=True
            Whether to apply ellipticity transforms to labels. If False, labels get
            shift/scale/flip/rotation but maintain exx=1, eyy=1, exy=0.

        add_salt_and_pepper : bool, default=False
            Enable salt and pepper (impulse) noise.
        salt_and_pepper : list[float] | float, default=[0, 5e-4]
            Range for fraction of pixels affected by salt and pepper noise.

        add_scale : bool, default=False
            Enable uniform scaling of the diffraction pattern.
        scale_factor : list[float] | float, default=[0.9, 1.1]
            Range for scaling factor. Use [1.0, max_val] for magnification only.

        add_blur : bool, default=False
            Enable Gaussian blur.
        blur_sigma : list[float] | float, default=[0.0, 1.5]
            Range for Gaussian blur standard deviation in pixels.

        add_flipshift : bool, default=False
            Enable random flips and rotations applied before other augmentations.
        free_rotation : bool, default=False
            If True, use continuous rotation within rotation_range.
            If False, use only 90-degree rotations (0°, 90°, 180°, 270°).
        rotation_range : list[float] | float, default=[-180, 180]
            Range for rotation angles in degrees (only used if free_rotation=True).

        log_file : os.PathLike | None, default=None
            Path to CSV file for logging augmentation parameters. If None, no logging.
        rng : np.random.Generator | int | None, default=None
            Random number generator or seed for reproducible augmentations.
        device : str, default="cpu"
            Device for computations ("cpu", "cuda", "cuda:0", etc.).

        Notes
        -----
        - Augmentations are applied in order: flipshift → background → elastic →
          shot noise → blur → salt & pepper
        - For labels, only geometric transforms (flipshift, elastic) are applied
        - Ellipticity creates anisotropic scaling via exx, eyy, exy parameters
        - All ranges can be single values, val, or [min, max] for uniform sampling
        """
        super().__init__(rng=rng)
        self._setup_device(device)
        self.log_file = log_file

        self.set_params(
            add_bkg,
            bkg_weight,
            bkg_q,
            add_shot,
            e_dose,
            add_shift,
            xshift,
            yshift,
            add_ellipticity,
            ellipticity_scale,
            add_ellipticity_to_label,
            add_salt_and_pepper,
            salt_and_pepper,
            add_scale,
            scale_factor,
            add_blur,
            blur_sigma,
            add_flipshift,
            free_rotation,
            rotation_range,
        )
        self.generate_params()
        self._init_log_file()

    def _setup_device(self, device: str) -> None:
        if device == "gpu" or device.startswith("cuda"):
            if not config.get("has_torch"):
                raise RuntimeError("torch required for GPU operations but not available")
            self.device = device if device.startswith("cuda") else "cuda"
            self.use_torch = True
        else:
            self.device = "cpu"
            self.use_torch = False

        if hasattr(self, "_rng_seed") and self._rng_seed is not None:
            self._rng_to_device(self.device)

    def _init_log_file(self) -> None:
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(
                    "bkg_weight,bkg_q,e_dose,xshift,yshift,exx,eyy,exy,"
                    "scale_factor,flip_horizontal,flip_vertical,rotation_angle,"
                    "blur_sigma,salt_and_pepper,rng_seed\n"
                )

    def set_params(
        self,
        add_bkg: bool = False,
        bkg_weight: list[float] | float = [0.01, 0.1],
        bkg_q: list[float] | float = [0.01, 0.1],
        add_shot: bool = False,
        e_dose: list[float] | float = [1e5, 1e10],
        add_shift: bool = False,
        xshift: list[float] | float = [0, 10],
        yshift: list[float] | float = [0, 10],
        add_ellipticity: bool = False,
        ellipticity_scale: list[float] | float = [0, 0.15],
        add_ellipticity_to_label: bool = True,
        add_salt_and_pepper: bool = False,
        salt_and_pepper: list[float] | float = [0, 1e-3],
        add_scale: bool = False,
        scale_factor: list[float] | float = [0.9, 1.1],
        add_blur: bool = False,
        blur_sigma: list[float] | float = [0.0, 1.5],
        add_flipshift: bool = False,
        free_rotation: bool = False,
        rotation_range: list[float] | float = [-180, 180],
    ) -> None:
        self.add_bkg = add_bkg
        self.add_shot = add_shot
        self.add_shift = add_shift
        self.add_ellipticity = add_ellipticity
        self.add_ellipticity_to_label = add_ellipticity_to_label
        self.add_salt_and_pepper = add_salt_and_pepper
        self.add_scale = add_scale
        self.add_blur = add_blur
        self.add_flipshift = add_flipshift

        self._bkg_weight_range = self._check_input(bkg_weight) if add_bkg else [0, 0]
        self._bkg_q_range = self._check_input(bkg_q) if add_bkg else [0, 0]
        self._e_dose_range = self._check_input(e_dose) if add_shot else [np.inf, np.inf]
        self._xshift_range = self._check_input(xshift) if add_shift else [0, 0]
        self._yshift_range = self._check_input(yshift) if add_shift else [0, 0]
        self._ellipticity_scale_range = (
            self._check_input(ellipticity_scale) if add_ellipticity else [0, 0]
        )
        self._salt_and_pepper_range = (
            self._check_input(salt_and_pepper) if add_salt_and_pepper else [0, 0]
        )
        self._scale_range = self._check_input(scale_factor) if add_scale else [0, 0]
        self._blur_range = self._check_input(blur_sigma) if add_blur else [0, 0]

        self.free_rotation = free_rotation
        self._rotation_range = self._check_input(rotation_range) if add_flipshift else [0, 0]

    def generate_params(self) -> None:
        self.bkg_weight = self._uniform_or_zero(self._bkg_weight_range, self.add_bkg)
        self.bkg_q = self._uniform_or_zero(self._bkg_q_range, self.add_bkg)
        self.e_dose = self._uniform_or_default(self._e_dose_range, self.add_shot, np.inf)
        self.salt_and_pepper = self._uniform_or_zero(
            self._salt_and_pepper_range, self.add_salt_and_pepper
        )
        self.blur_sigma = self._uniform_or_zero(self._blur_range, self.add_blur)
        self.xshift = self._uniform_with_sign(self._xshift_range, self.add_shift)
        self.yshift = self._uniform_with_sign(self._yshift_range, self.add_shift)
        self._generate_ellipticity_params()
        self._generate_flipshift_params()

        if self.add_scale:
            self.scale_factor = self.rng.uniform(self._scale_range[0], self._scale_range[1])
        else:
            self.scale_factor = 0

    def _uniform_or_zero(self, range_vals: list, enabled: bool) -> float:
        return self.rng.uniform(range_vals[0], range_vals[1]) if enabled else 0

    def _uniform_or_default(self, range_vals: list, enabled: bool, default) -> float:
        return self.rng.uniform(range_vals[0], range_vals[1]) if enabled else default

    def _uniform_with_sign(self, range_vals: list, enabled: bool) -> float:
        if not enabled:
            return 0
        return self.rng.uniform(range_vals[0], range_vals[1]) * self.rng.choice([1, -1])

    def _generate_ellipticity_params(self) -> None:
        if self.add_ellipticity:
            self.ellipticity_scale = self.rng.uniform(
                self._ellipticity_scale_range[0], self._ellipticity_scale_range[1]
            )
            exx = self.rng.normal(loc=1, scale=self.ellipticity_scale)
            eyy = self.rng.normal(loc=1, scale=self.ellipticity_scale)
            mval = (exx + eyy) / 2  # Normalize to preserve area
            self.exx = exx / mval
            self.eyy = eyy / mval
            self.exy = self.rng.normal(loc=0, scale=self.ellipticity_scale)
        else:
            self.ellipticity_scale = 0
            self.exx = self.eyy = 1
            self.exy = 0

    def _generate_flipshift_params(self) -> None:
        if self.add_flipshift:
            # Independent 0.5 probability for each flip direction
            self.flip_horizontal = self.rng.random() < 0.5
            self.flip_vertical = self.rng.random() < 0.5

            # Always apply rotation when flipshift is enabled
            if self.free_rotation:
                self.rotation_angle = self.rng.uniform(
                    self._rotation_range[0], self._rotation_range[1]
                )
            else:
                self.rotation_angle = self.rng.choice([0, 90, 180, 270])
        else:
            self.flip_horizontal = False
            self.flip_vertical = False
            self.rotation_angle = 0

    def print_params(self, print_all: bool = False) -> None:
        print("Augmentation summary:")
        print(">" * 32)

        params = [
            (
                "Inelastic background",
                self.add_bkg,
                f"Weight: {self.bkg_weight:.3f}, Q: {self.bkg_q:.3f}",
            ),
            ("Shot noise", self.add_shot, f"e-dose: {self.e_dose:.2e}"),
            ("Image shift", self.add_shift, f"(x,y): ({self.xshift:.2f}, {self.yshift:.2f})"),
            (
                "Elliptic scaling",
                self.add_ellipticity,
                f"(exx,eyy,exy): ({self.exx:.2f},{self.eyy:.2f},{self.exy:.2f})",
            ),
            ("Scaling", self.add_scale, f"Factor: {self.scale_factor:.2f}"),
            (
                "Flip/rotation",
                self.add_flipshift,
                f"Flip: H={self.flip_horizontal}, V={self.flip_vertical}, Rot: {self.rotation_angle:.1f}°",
            ),
            ("Salt & pepper", self.add_salt_and_pepper, f"Amount: {self.salt_and_pepper:.2e}"),
            ("Gaussian blur", self.add_blur, f"Sigma: {self.blur_sigma:.2f}"),
        ]

        for name, enabled, details in params:
            print(f"{name}: {enabled}")
            if (enabled or print_all) and details:
                print(f"\t{details}")

        print(f"Random seed: {self._rng_seed}")
        print("<" * 32)

    def augment(
        self, dp: ArrayLike, probe: ArrayLike | None = None, label: ArrayLike | None = None
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        """
        Apply augmentations to diffraction pattern(s)

        Args:
            dp: Diffraction pattern or stack (N, H, W) or (H, W)
            probe: Optional probe function or stack - not augmented
            label: Optional label or stack - only elastic transforms applied

        Returns:
            Augmented diffraction pattern(s) or tuple with transformed label
        """
        self._validate_inputs(dp, probe, label)
        self._maybe_switch_to_torch(dp, probe, label)

        dp = self._as_array(dp)
        if probe is not None:
            probe = self._as_array(probe)
        if label is not None:
            label = self._as_array(label)

        is_stack = len(dp.shape) == 3
        if is_stack:
            return self._augment_stack(dp, probe, label)
        else:
            return self._augment_single(dp, probe, label)

    def _augment_stack(
        self,
        dp_stack: ArrayLike,
        probe_stack: ArrayLike | None = None,
        label_stack: ArrayLike | None = None,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        batch_size = dp_stack.shape[0]

        if probe_stack is not None and probe_stack.shape[0] != batch_size:
            raise ValueError(f"Probe stack size {probe_stack.shape[0]} != DP size {batch_size}")
        if label_stack is not None and label_stack.shape[0] != batch_size:
            raise ValueError(f"Label stack size {label_stack.shape[0]} != DP size {batch_size}")

        augmented_dps = []
        augmented_labels = [] if label_stack is not None else None

        for i in tqdm(range(batch_size), desc="augmenting"):
            dp_single = dp_stack[i]
            probe_single = probe_stack[i] if probe_stack is not None else None
            label_single = label_stack[i] if label_stack is not None else None

            if label_single is not None:
                aug_dp, aug_label = self._augment_single(dp_single, probe_single, label_single)
                augmented_dps.append(aug_dp)
                augmented_labels.append(aug_label)  # type: ignore
            else:
                aug_dp = self._augment_single(dp_single, probe_single, None)
                augmented_dps.append(aug_dp)

        if self.use_torch:
            stacked_dps = torch.stack(augmented_dps)  # type: ignore
            if augmented_labels is not None:
                stacked_labels = torch.stack(augmented_labels)  # type: ignore
                return stacked_dps, stacked_labels
            return stacked_dps
        else:
            stacked_dps = np.stack(augmented_dps)
            if augmented_labels is not None:
                stacked_labels = np.stack(augmented_labels)
                return stacked_dps, stacked_labels
            return stacked_dps

    def _augment_single(
        self, dp: ArrayLike, probe: ArrayLike | None = None, label: ArrayLike | None = None
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        result = dp
        transformed_label = label

        if self.add_flipshift:
            result = self._apply_flipshift(result)
            if transformed_label is not None:
                transformed_label = self._apply_flipshift(transformed_label)
        if self.add_bkg:
            result = self._apply_bkg(result, probe)
        if self.add_ellipticity or self.add_shift or self.add_scale:
            result = self._apply_elastic(result)
            if transformed_label is not None:
                transformed_label = self._apply_elastic_to_label(transformed_label)
        if self.add_shot:
            result = self._apply_shot(result)
        if self.add_blur:
            result = self._apply_blur(result)
        if self.add_salt_and_pepper:
            result = self._apply_salt_and_pepper(result)

        self.write_logs()
        self.generate_params()

        if label is not None:
            assert transformed_label is not None  # Type guard
            return result, transformed_label
        return result

    def _validate_inputs(
        self, dp: ArrayLike, probe: ArrayLike | None, label: ArrayLike | None = None
    ) -> None:
        arrays_to_check = [dp]
        if probe is not None:
            arrays_to_check.append(probe)
        if label is not None:
            arrays_to_check.append(label)
            if "float" not in str(label.dtype):
                raise ValueError("Label must be a float array/tensor")

        for arr in arrays_to_check:
            if self._is_cupy_array(arr):
                raise ValueError("CuPy arrays not supported. Use numpy/torch arrays.")

    def _maybe_switch_to_torch(
        self, dp: ArrayLike, probe: ArrayLike | None, label: ArrayLike | None = None
    ) -> None:
        dp_is_torch = config.get("has_torch") and isinstance(dp, torch.Tensor)
        probe_is_torch = (
            probe is not None and config.get("has_torch") and isinstance(probe, torch.Tensor)
        )
        label_is_torch = (
            label is not None and config.get("has_torch") and isinstance(label, torch.Tensor)
        )

        if (dp_is_torch or probe_is_torch or label_is_torch) and not self.use_torch:
            if dp_is_torch:
                self.device = str(dp.device)  # type: ignore
            elif probe_is_torch:
                self.device = str(probe.device)  # type: ignore
            elif label_is_torch:
                self.device = str(label.device)  # type: ignore
            else:
                self.device = "cpu"
            self.use_torch = True
            self._rng_to_device(self.device)

    def _apply_shot(self, inputs: ArrayLike) -> ArrayLike:
        """Apply Poisson shot noise"""
        if self.use_torch:
            image = inputs.clone()  # type: ignore
            offset = image.min()
            image = (image - offset) / (image - offset).sum()
            return torch.poisson(image * self.e_dose, generator=self._rng_torch) + offset
        else:
            image = np.array(inputs)
            offset = image.min()
            image = (image - offset) / (image - offset).sum()
            return self.rng.poisson(image * self.e_dose) + offset

    def _apply_elastic(self, inputs: ArrayLike) -> ArrayLike:
        """Apply elastic transformations (scaling, rotation, translation)"""
        if self.use_torch:
            return self._apply_elastic_torch(inputs)  # type: ignore
        else:
            return self._apply_elastic_numpy(inputs)

    def _apply_elastic_torch(self, inputs: "torch.Tensor") -> "torch.Tensor":
        height, width = inputs.shape
        device = inputs.device

        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        y_center, x_center = height // 2, width // 2
        y, x = y.clone() - y_center, x.clone() - x_center

        if self.add_scale:
            y, x = y * self.scale_factor, x * self.scale_factor

        x_new = self.exx * x + self.exy * y + x_center
        y_new = self.exy * x + self.eyy * y + y_center

        if self.add_shift:
            x_new += self.xshift
            y_new += self.yshift

        x_norm = 2.0 * x_new / (width - 1) - 1.0
        y_norm = 2.0 * y_new / (height - 1) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

        image_batch = torch.as_tensor(inputs)[None, None, ...]
        distorted = F.grid_sample(
            image_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return distorted.squeeze(0).squeeze(0)

    def _apply_elastic_numpy(self, inputs: ArrayLike) -> np.ndarray:
        image = np.array(inputs)
        height, width = image.shape

        y, x = np.indices((height, width)).astype("float")
        y_center, x_center = height // 2, width // 2
        y, x = y - y_center, x - x_center

        if self.add_scale:
            y, x = y * self.scale_factor, x * self.scale_factor

        x_new = self.exx * x + self.exy * y + x_center
        y_new = self.exy * x + self.eyy * y + y_center

        if self.add_shift:
            x_new += self.xshift
            y_new += self.yshift

        return ndi.map_coordinates(image, [y_new, x_new], order=1, mode="constant")

    def _apply_bkg(self, inputs: ArrayLike, probe: ArrayLike | None = None) -> ArrayLike:
        """Apply inelastic plasmon background via convolution with probe"""
        height, width = inputs.shape

        qx = af.view(af.sort(af.fftfreq(height, 0.1, like=inputs), axis=0), (-1, 1))
        qy = af.view(af.sort(af.fftfreq(width, 0.1, like=inputs), axis=0), (1, -1))

        CBEDbg = 1.0 / (qx**2 + qy**2 + self.bkg_q**2)  # Plasmon form factor: 1/(q² + q₀²)
        CBEDbg = CBEDbg.squeeze() / af.sum(CBEDbg.squeeze())

        if probe is not None:
            CBEDbgConv = af.fftshift(af.ifft2(af.fft2(CBEDbg) * af.fft2(probe)))
        else:
            CBEDbgConv = CBEDbg

        inputs_float = af.as_type(inputs, torch.float32 if self.use_torch else np.float32)
        return inputs_float * (1 - self.bkg_weight) + CBEDbgConv.real * self.bkg_weight

    def _apply_blur(self, inputs: ArrayLike) -> ArrayLike:
        """Apply Gaussian blur"""
        if self.use_torch:
            return self._apply_blur_torch(validate_tensor(inputs, name="inputs"))
        else:
            return af.clip(ndi.gaussian_filter(inputs, sigma=self.blur_sigma), a_min=0)

    def _apply_blur_torch(self, inputs: "torch.Tensor") -> "torch.Tensor":
        kernel_size = int(2 * np.ceil(3 * self.blur_sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, dtype=torch.float32, device=inputs.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / self.blur_sigma) ** 2)
        kernel_1d = (kernel_1d / kernel_1d.sum())[None, None, ...]

        image = inputs[None, None, ...]
        image = F.conv2d(image, kernel_1d.unsqueeze(-1), padding=(kernel_size // 2, 0))
        image = F.conv2d(image, kernel_1d.unsqueeze(-2), padding=(0, kernel_size // 2))

        return torch.clamp(image.squeeze(0).squeeze(0), min=0)

    def _apply_salt_and_pepper(self, inputs: ArrayLike) -> ArrayLike:
        """Apply salt and pepper noise"""
        offset = inputs.min()
        return self._get_salt_and_pepper(inputs - offset, self.salt_and_pepper) + offset

    def _get_salt_and_pepper(
        self,
        image: ArrayLike,
        amount: float = 1e-3,
        salt_vs_pepper: float = 1.0,
        pepper_val: float = 0,
        salt_val: float | None = None,
    ) -> ArrayLike:
        if self.use_torch:
            out = image.clone()  # type: ignore
            salt_value = salt_val if salt_val is not None else float(image.max())  # type: ignore

            flipped = torch.rand(out.shape, device=out.device, generator=self._rng_torch) <= amount  # type: ignore
            salted = (
                torch.rand(out.shape, device=out.device, generator=self._rng_torch)
                <= salt_vs_pepper
            )  # type: ignore

            out[flipped & salted] = salt_value  # type: ignore
            out[flipped & ~salted] = pepper_val  # type: ignore
            return out
        else:
            out = np.array(image).copy()
            salt_value = salt_val if salt_val is not None else float(image.max())

            flipped = self.rng.random(out.shape) <= amount
            salted = self.rng.random(out.shape) <= salt_vs_pepper

            out[flipped & salted] = salt_value
            out[flipped & ~salted] = pepper_val
            return out

    def write_logs(self) -> None:
        if self.log_file is None:
            return
        with open(self.log_file, "a") as f:
            f.write(
                f"{self.bkg_weight},{self.bkg_q},{self.e_dose},{self.xshift},"
                f"{self.yshift},{self.exx},{self.eyy},{self.exy},"
                f"{self.scale_factor},{self.flip_horizontal},{self.flip_vertical},"
                f"{self.rotation_angle},{self.blur_sigma},{self.salt_and_pepper},"
                f"{self._rng_seed}\n"
            )

    def _is_cupy_array(self, arr) -> bool:
        return config.get("has_cupy") and hasattr(arr, "__module__") and "cupy" in arr.__module__

    @staticmethod
    def _check_input(inp: list[float] | float) -> list[float]:
        if isinstance(inp, list):
            assert len(inp) == 2 and inp[0] <= inp[1], f"Bad value range: {inp}"
            return inp
        return [inp, inp]

    def _as_array(self, ar) -> ArrayLike:
        if self.use_torch:
            if config.get("has_torch") and isinstance(ar, torch.Tensor):
                return ar.to(self.device)
            else:
                dummy_tensor = torch.tensor([], device=self.device)
                return af.match_device(np.array(ar), dummy_tensor)
        else:
            return np.asarray(ar)

    def _make_fourier_coord(
        self, Nx: int, Ny: int, pixelSize: float | list[float]
    ) -> tuple[ArrayLike, ArrayLike]:
        if isinstance(pixelSize, list):
            assert len(pixelSize) == 2, "pixelSize must be scalar or length 2"
            pixelSize_x, pixelSize_y = float(pixelSize[0]), float(pixelSize[1])
        else:
            pixelSize_x = pixelSize_y = float(pixelSize)

        dummy = torch.tensor([1.0], device=self.device) if self.use_torch else np.array([1.0])

        qx = af.fftfreq(Nx, pixelSize_x, like=dummy)
        qy = af.fftfreq(Ny, pixelSize_y, like=dummy)
        qy, qx = af.meshgrid(qy, qx, indexing="ij")
        return qx, qy

    def _apply_flipshift(self, inputs: ArrayLike) -> ArrayLike:
        """Apply random flips and rotations"""
        result = inputs

        # Apply flips
        if self.flip_horizontal or self.flip_vertical:
            if self.use_torch:
                if self.flip_horizontal:
                    result = torch.flip(result, dims=[1])  # type: ignore
                if self.flip_vertical:
                    result = torch.flip(result, dims=[0])  # type: ignore
            else:
                if self.flip_horizontal:
                    result = np.flip(result, axis=1)
                if self.flip_vertical:
                    result = np.flip(result, axis=0)

        # Apply rotation
        if self.rotation_angle != 0:
            if self.use_torch:
                result = self._rotate_torch(result, self.rotation_angle)  # type: ignore
            else:
                result = self._rotate_numpy(result, self.rotation_angle)

        return result

    def _rotate_torch(self, inputs: "torch.Tensor", angle: float) -> "torch.Tensor":
        """Torch implementation of rotation"""
        if angle in [90, 180, 270]:
            # Use efficient 90-degree rotations
            k = int(angle // 90)
            return torch.rot90(inputs, k=k, dims=[0, 1])  # type: ignore
        else:
            # Free rotation using affine transformation
            angle_rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            # Create rotation matrix
            rotation_matrix = torch.tensor(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32, device=inputs.device
            )

            # Apply rotation
            batch_input = inputs.unsqueeze(0).unsqueeze(0)
            grid = F.affine_grid(
                rotation_matrix.unsqueeze(0), list(batch_input.size()), align_corners=True
            )  # type: ignore
            rotated = F.grid_sample(
                batch_input, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )  # type: ignore
            return rotated.squeeze(0).squeeze(0)

    def _rotate_numpy(self, inputs: ArrayLike, angle: float) -> np.ndarray:
        """Numpy implementation of rotation"""
        image = np.array(inputs)
        if angle in [90, 180, 270]:
            # Use efficient 90-degree rotations
            k = int(angle // 90)
            return np.rot90(image, k=k)
        else:
            # Free rotation using scipy
            return ndi.rotate(image, angle, mode="constant", cval=0, reshape=False)

    def _apply_elastic_to_label(self, inputs: ArrayLike) -> ArrayLike:
        """Apply elastic transformations to label, optionally excluding ellipticity"""
        if not self.add_ellipticity_to_label:
            # Temporarily override ellipticity parameters for label processing
            orig_exx, orig_eyy, orig_exy = self.exx, self.eyy, self.exy
            self.exx, self.eyy, self.exy = 1.0, 1.0, 0.0

            result = self._apply_elastic(inputs)

            # Restore original ellipticity parameters
            self.exx, self.eyy, self.exy = orig_exx, orig_eyy, orig_exy
            return result
        else:
            return self._apply_elastic(inputs)
