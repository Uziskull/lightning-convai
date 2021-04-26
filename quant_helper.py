import functools
from typing import Any, Callable, Optional, Sequence, Union

import torch

import pytorch_lightning as pl

def _multiarg_wrap_qat(
    quant_cb,
    model: pl.core.LightningModule,
    func: Callable,
    trigger_condition: Optional[Union[Callable, int]] = None
) -> Callable:
    @functools.wraps(func)
    def wrapper(*data) -> Any:
        _is_func_true = isinstance(trigger_condition, Callable) and trigger_condition(model.trainer)
        _is_count_true = isinstance(trigger_condition, int) and quant_cb._forward_calls < trigger_condition
        _quant_run = trigger_condition is None or _is_func_true or _is_count_true
        # apply custom trigger
        if _quant_run:
            quant_cb._forward_calls += 1
            data = [model.quant(d.float()).long() if isinstance(d, torch.Tensor) else d for d in data]
        data = func(*data)
        # apply custom trigger
        if _quant_run:
            data = model.dequant(data)
        return data

    return wrapper


def _multiarg_wrap_quantize(model: pl.core.LightningModule, func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*data) -> Any:
        data = [model.quant(d) if isinstance(d, torch.Tensor) else d for d in data]
        data = func(*data)
        data = model.dequant(data)
        return data

    return wrapper