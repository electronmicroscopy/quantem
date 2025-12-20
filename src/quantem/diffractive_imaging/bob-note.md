object_models - tomo

```python
class ObjectMultiplexed(ObjectPixelated):
    """
    Object model for multiplexed objects.
    This object will descrive a sample in more then one state,
    Usually the base state and it's response to external stimulis

    NOTE
    Alternative scan, turn on and turn off light.
    Two channels of data, line on and line off.
    For each diffraction pattern, so that we know light on and light off region


    OPTIMIZE THIS SECTION
    [ ] 40-70x
    [ ] 

    """

    def __init__(
        self,
        num_channels: int,
        patches_mask: torch.Tensor,
        num_slices: int = 1,
        **kwargs,
    ):
        super().__init__(
            num_slices=num_slices,
            **kwargs
        )
        self.num_channels = num_channels
        self._obj = nn.Parameter(torch.ones(num_channels, num_slices, 1, 1), requires_grad=True)

        self.patches_mask = patches_mask.reshape(-1)

    @property
    def obj(self):
        # applying the hard containts to each channel separately
        post_constraint_obj = torch.zeros_like(self._obj)
        for ch in range(self._obj.shape[0]):
            post_constraint_obj[ch] = self.apply_hard_constraints(
                self._obj[ch], mask=self.mask
            )
        return post_constraint_obj

    def _initialize_obj(
        self,
        shape: tuple[int, int, int] | np.ndarray,
        sampling: tuple[float, float] | np.ndarray | None = None,
    ) -> None:
        super()._initialize_obj(shape, sampling)
        # if self.obj.numel() > self.num_slices and np.array_equal(self.shape, shape):
            # return
        
        # adding num_channels here
        init_shape = (self.num_channels,) + tuple(int(x) for x in shape)
        if self._initialize_mode == "uniform":
            if self.obj_type in ["complex", "pure_phase"]:
                arr = torch.ones(init_shape) * torch.exp(1.0j * torch.zeros(init_shape))
            else:
                arr = torch.zeros(init_shape)
        elif self._initialize_mode == "random":
            ph = (
                torch.randn(init_shape, dtype=torch.float32, generator=self._rng_torch) - 0.5
            ) * 1e-6
            if self.obj_type == "potential":
                arr = ph
            else:
                arr = torch.exp(1.0j * ph)
        elif self._initialize_mode == "array":
            arr = self._initial_obj
        else:
            raise ValueError(f"Invalid initialize mode: {self._initialize_mode}")

        self._initial_obj = arr.type(self.dtype)
        self.reset()

    @classmethod
    def from_uniform(
        cls,
        num_channels: int,
        patches_mask: torch.Tensor,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | None | np.ndarray = None,
        device: str = "cpu",
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
        rng: np.random.Generator | int | None = None,
    ):
        """
        Create ObjectPixelated from a uniform initialization.
        """
        obj_model = cls(
            num_channels=num_channels,
            patches_mask=patches_mask,
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            device=device,
            obj_type=obj_type,
            initialize_mode="uniform",
            rng=rng,
            _token=cls._token,
        )

        return obj_model

    def forward(self, patch_indices: torch.Tensor, batch_indices: torch.Tensor):
        """Get patch indices of the object"""
        # using the batch indicies to select which channel to use for each patch
        patches = []
        # return self._get_obj_patches(self.obj[0,...], patch_indices)
        for i, batch_id in enumerate(batch_indices):
            # find the correct object channel
            ch = int(self.patches_mask[batch_id])
            patch = self._get_obj_patches(self.obj[ch,...], patch_indices[i])
            patches.append(patch)
        return torch.stack(patches, dim=0).transpose(0,1)

# class ObjectImplicit(ObjectBase):
#     """
#     Object model for implicit objects. Importantly, the forward call from scan positions
#     for this model will not require subpixel shifting of the object probe, as subpixel shifting
#     will be done in the object model itself, so it is properly aligned around the probe positions
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._obj = None
#         self._obj_shape = None
#         self._num_slices = None

#     def pretrain(self, *args, **kwargs):


#     ### here the forward call will take the batch indices and create the appropriate
#     ### input (which maybe is just the raw patch indices? tbd) for the implicit input
#     ### so it will be parallelized inference across the batches rather than inference once
#     ### and then patching that, like it will be for DIP

# constraints are going to be tricky, specifically the TV and filtering if we want to allow
# multiscale reconstructions


ObjectModelType = ObjectPixelated | ObjectDIP | ObjectMultiplexed # | ObjectImplicit
```