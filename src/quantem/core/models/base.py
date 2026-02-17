from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


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
        self.tags: dict[str, Any] = {} if tags is None else dict(tags)
        self.initial_value, self.lower_bound, self.upper_bound = self._parse_value(
            value, lower_bound, upper_bound
        )
        self._index: int | None = None

    @staticmethod
    def _parse_value(
        value: float | Sequence[float],
        lower_bound: float | None,
        upper_bound: float | None,
    ) -> tuple[float, float | None, float | None]:
        if isinstance(value, (list, tuple, np.ndarray)):
            v = list(value)
            if len(v) == 2:
                v0 = float(v[0])
                dev = float(v[1])
                lb = v0 - dev if lower_bound is None else float(lower_bound)
                ub = v0 + dev if upper_bound is None else float(upper_bound)
                return v0, lb, ub
            if len(v) == 3:
                v0 = float(v[0])
                lb = None if v[1] is None else float(v[1])
                ub = None if v[2] is None else float(v[2])
                if lower_bound is not None:
                    lb = float(lower_bound)
                if upper_bound is not None:
                    ub = float(upper_bound)
                return v0, lb, ub
            raise ValueError("Parameter sequences must have length 2 or 3.")
        v0 = float(value)
        return v0, lower_bound, upper_bound

    def bind(self, index: int) -> None:
        self._index = int(index)

    @property
    def index(self) -> int:
        if self._index is None:
            raise RuntimeError("Parameter is not bound. Call Model.compile(...) first.")
        return self._index


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
        self.fields: dict[str, Any] = {} if fields is None else dict(fields)

    def __getattr__(self, name: str) -> Any:
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(name)


class Component:
    def __init__(self, *, name: str):
        self.name = str(name)

    def parameters(self) -> list[Parameter]:
        return []

    def prepare(self, ctx: ModelContext) -> Any:
        raise NotImplementedError


class PreparedModel:
    def __init__(
        self,
        *,
        components: list[Any],
        ctx: ModelContext,
        params: list[Parameter],
        x0: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ):
        self.components = components
        self.ctx = ctx
        self.params = params
        self.x0 = x0
        self.lb = lb
        self.ub = ub

    def render(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((self.ctx.H, self.ctx.W), device=self.ctx.device, dtype=self.ctx.dtype)
        for c in self.components:
            c.render(out, x, self.ctx)
        return out

    def render_initial(self) -> torch.Tensor:
        return self.render(self.x0)

    def overlays(self, x: torch.Tensor | None = None) -> list[Overlay]:
        out: list[Overlay] = []
        for c in self.components:
            fn = getattr(c, "overlays", None)
            if fn is None:
                continue
            ov = fn(self.x0 if x is None else x, self.ctx)
            if ov:
                out.extend(list(ov))
        return out


class Model:
    def __init__(self):
        self._components: list[Component] = []

    def add(self, items: Iterable[Component]) -> "Model":
        for obj in items:
            if not isinstance(obj, Component):
                raise TypeError("Model.add expects Component objects.")
            self._components.append(obj)
        return self

    def parameter_list(self) -> list[Parameter]:
        params: list[Parameter] = []
        for c in self._components:
            params.extend(c.parameters())
        return params

    def compile(self, ctx: ModelContext) -> PreparedModel:
        params = self.parameter_list()
        for i, p in enumerate(params):
            p.bind(i)

        x0 = torch.zeros((len(params),), device=ctx.device, dtype=ctx.dtype)
        lb = torch.full((len(params),), -torch.inf, device=ctx.device, dtype=ctx.dtype)
        ub = torch.full((len(params),), torch.inf, device=ctx.device, dtype=ctx.dtype)

        for p in params:
            x0[p.index] = torch.as_tensor(p.initial_value, device=ctx.device, dtype=ctx.dtype)
            if p.lower_bound is not None:
                lb[p.index] = torch.as_tensor(p.lower_bound, device=ctx.device, dtype=ctx.dtype)
            if p.upper_bound is not None:
                ub[p.index] = torch.as_tensor(p.upper_bound, device=ctx.device, dtype=ctx.dtype)

        prepared_components: list[Any] = []
        for c in self._components:
            prepared_components.append(c.prepare(ctx))

        return PreparedModel(components=prepared_components, ctx=ctx, params=params, x0=x0, lb=lb, ub=ub)
