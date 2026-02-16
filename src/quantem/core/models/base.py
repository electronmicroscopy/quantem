from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

__all__ = [
    "Overlay",
    "Parameter",
    "ModelContext",
    "Component",
    "OriginND",
    "Model",
    "PreparedModel",
]


@dataclass(frozen=True)
class Overlay:
    kind: str
    data: dict[str, Any]


class Parameter:
    def __init__(
        self,
        value: float | Sequence[float],
        *,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        tags: Mapping[str, Any] | None = None,
    ):
        v0, lb, ub = self._parse(value, lower_bound, upper_bound)
        self.initial_value = float(v0)
        self.lower_bound = float(lb)
        self.upper_bound = float(ub)
        self.tags: dict[str, Any] = dict(tags) if tags is not None else {}
        self._index: int | None = None

    @staticmethod
    def _parse(
        value: float | Sequence[float],
        lower_bound: float | None,
        upper_bound: float | None,
    ) -> tuple[float, float, float]:
        if isinstance(value, (list, tuple, np.ndarray)):
            seq = list(value)
            if len(seq) == 2:
                v0 = float(seq[0])
                dev = float(seq[1])
                return v0, v0 - dev, v0 + dev
            if len(seq) == 3:
                v0 = float(seq[0])
                lb = -np.inf if seq[1] is None else float(seq[1])
                ub = np.inf if seq[2] is None else float(seq[2])
                return v0, lb, ub
            raise ValueError("Parameter sequences must have length 2 or 3.")
        v0 = float(value)
        lb = -np.inf if lower_bound is None else float(lower_bound)
        ub = np.inf if upper_bound is None else float(upper_bound)
        return v0, lb, ub

    def bind(self, index: int) -> None:
        self._index = int(index)

    def value(self, x: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._index is None:
            return torch.as_tensor(self.initial_value, device=device, dtype=dtype).reshape(())
        return x[int(self._index)].to(device=device, dtype=dtype).reshape(())


class ModelContext:
    def __init__(
        self,
        *,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
        mask: torch.Tensor | None = None,
        fields: Mapping[str, Any] | None = None,
    ):
        self.H = int(H)
        self.W = int(W)
        self.device = device
        self.dtype = dtype
        self.mask = mask
        self.fields: dict[str, Any] = dict(fields) if fields is not None else {}
        self.origins: dict[str, tuple[torch.Tensor, ...]] = {}

    def __getattr__(self, name: str) -> Any:
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(name)


class Component:
    def __init__(self, *, name: str):
        self.name = str(name)

    def parameters(self) -> list[Parameter]:
        out: list[Parameter] = []

        def add(obj: Any) -> None:
            if obj is None:
                return
            if isinstance(obj, Parameter):
                out.append(obj)
                return
            if isinstance(obj, (list, tuple)):
                for z in obj:
                    add(z)

        for v in self.__dict__.values():
            add(v)
        return out

    def prepare(self, ctx: ModelContext) -> Any:
        raise NotImplementedError


class OriginND(Component):
    def __init__(self, *, key: str, coords: Sequence[Parameter]):
        super().__init__(name=f"Origin[{key}]")
        self.key = str(key)
        self.coords = list(coords)

    @classmethod
    def from_row_col(
        cls,
        *,
        key: str,
        origin_row: float | Sequence[float],
        origin_col: float | Sequence[float],
    ) -> "OriginND":
        p_row = Parameter(origin_row, tags={"role": "origin_row", "origin_key": str(key)})
        p_col = Parameter(origin_col, tags={"role": "origin_col", "origin_key": str(key)})
        return cls(key=key, coords=[p_row, p_col])

    def prepare(self, ctx: ModelContext) -> Any:
        key = self.key
        p0 = self.coords[0]
        p1 = self.coords[1]

        r0 = torch.as_tensor(p0.initial_value, device=ctx.device, dtype=ctx.dtype).reshape(())
        c0 = torch.as_tensor(p1.initial_value, device=ctx.device, dtype=ctx.dtype).reshape(())
        ctx.origins[key] = (r0, c0)

        class PreparedOriginND:
            origin_key = key

            def render(self, out: torch.Tensor, x: torch.Tensor, ctx: ModelContext) -> None:
                r = p0.value(x, device=ctx.device, dtype=ctx.dtype)
                c = p1.value(x, device=ctx.device, dtype=ctx.dtype)
                ctx.origins[key] = (r, c)

            def overlays(self) -> list[Overlay]:
                r = float(p0.initial_value)
                c = float(p1.initial_value)
                return [
                    Overlay(
                        kind="points_rc",
                        data={"r": np.array([r]), "c": np.array([c]), "marker": "x", "s": 80.0, "color": "tab:blue"},
                    )
                ]

        return PreparedOriginND()


class Model:
    def __init__(self) -> None:
        self._components: list[Component] = []

    def add(self, items: Iterable[Component]) -> "Model":
        for obj in items:
            if not isinstance(obj, Component):
                raise TypeError("Model.add expects Component objects.")
            self._components.append(obj)
        return self

    def compile(self, ctx: ModelContext) -> "PreparedModel":
        seen: set[int] = set()
        params: list[Parameter] = []
        for c in self._components:
            for p in c.parameters():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                params.append(p)

        for i, p in enumerate(params):
            p.bind(i)

        x0 = torch.as_tensor([p.initial_value for p in params], device=ctx.device, dtype=ctx.dtype)

        prepared_components: list[Any] = []
        for c in self._components:
            prepared_components.append(c.prepare(ctx))

        return PreparedModel(
            components=prepared_components,
            ctx=ctx,
            params=params,
            x0=x0,
        )


class PreparedModel:
    def __init__(
        self,
        *,
        components: list[Any],
        ctx: ModelContext,
        params: list[Parameter],
        x0: torch.Tensor,
    ):
        self.components = components
        self.ctx = ctx
        self.params = params
        self.x0 = x0

    def render(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((self.ctx.H, self.ctx.W), device=self.ctx.device, dtype=self.ctx.dtype)
        for c in self.components:
            if hasattr(c, "render"):
                c.render(out, x, self.ctx)
        return out

    def render_initial(self) -> torch.Tensor:
        return self.render(self.x0)

    def overlays(self) -> list[Overlay]:
        out: list[Overlay] = []
        for c in self.components:
            if hasattr(c, "overlays"):
                ov = c.overlays()
                if ov:
                    out.extend(list(ov))
        return out
