# -*- coding: utf-8 -*-
r"""
Lightning Trainer Setup
=======================
   Setup logic for the lightning trainer.
"""
import os
from argparse import Namespace
from datetime import datetime

import torch
import pytorch_lightning as pl

#############################

import pytorch_lightning.callbacks.quantization

from quant_helper import _multiarg_wrap_qat, _multiarg_wrap_quantize

pytorch_lightning.callbacks.quantization.wrap_qat_forward_context = _multiarg_wrap_qat
pytorch_lightning.callbacks.quantization.wrap_quantize_forward_context = _multiarg_wrap_quantize

#############################

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    QuantizationAwareTraining
)

#############################

def _conv1d_to_linear(module):
    in_size, out_size = module.weight.shape
    linear = torch.nn.Linear(in_size, out_size)
    linear.weight.data = module.weight.data.T.contiguous()
    linear.bias.data = module.bias.data
    return linear

def conv1d_to_linear(model):
    """in-place
    This is for Dynamic Quantization, as Conv1D is not recognized by PyTorch, convert it to nn.Linear
    """
    for name in list(model._modules):
        module = model._modules[name]
        if isinstance(module, torch.nn.Conv1d):
            linear = _conv1d_to_linear(module)
            model._modules[name] = linear
        else:
            conv1d_to_linear(module)

class CustomQAT(QuantizationAwareTraining):
    def on_fit_end(self, trainer, pl_module):
        # pytorch doesn't support conv1d quantization, so we have to convert it to linear
        conv1d_to_linear(pl_module)
        super(CustomQAT, self).on_fit_end(trainer, pl_module)
        
#############################

from pytorch_lightning.loggers import TensorBoardLogger
from utils import Config


class TrainerConfig(Config):
    """
    The TrainerConfig class is used to define default hyper-parameters that
    are used to initialize our Lightning Trainer. These parameters are then overwritted
    with the values defined in the YAML file.

    -------------------- General Parameters -------------------------

    :param seed: Training seed.

    :param deterministic: If true enables cudnn.deterministic. Might make your system
        slower, but ensures reproducibility.

    :param verbose: verbosity mode.

    :param overfit_batches: Uses this much data of the training set. If nonzero, will use
        the same training set for validation and testing. If the training dataloaders
        have shuffle=True, Lightning will automatically disable it.

    -------------------- Model Checkpoint & Early Stopping -------------------------

    :param early_stopping: If true enables EarlyStopping.

    :param save_top_k: If save_top_k == k, the best k models according to the metric
        monitored will be saved.

    :param monitor: Metric to be monitored.

    :param save_weights_only: Saves only the weights of the model.

    :param metric_mode: One of {min, max}. In min mode, training will stop when the
        metric monitored has stopped decreasing; in max mode it will stop when the
        metric monitored has stopped increasing.

    :param min_delta: Minimum change in the monitored metric to qualify as an improvement.

    :param patience: Number of epochs with no improvement after which training will be stopped.

    :param accumulate_grad_batches: Gradient accumulation steps.
    """

    seed: int = 3
    deterministic: bool = True
    verbose: bool = False
    overfit_batches: float = 0.0

    # Model Checkpoint & Early Stopping
    early_stopping: bool = True
    save_top_k: int = 1
    monitor: str = "val_loss"
    save_weights_only: bool = False
    metric_mode: str = "min"
    min_delta: float = 0.0
    patience: int = 1
    accumulate_grad_batches: int = 1

    ## Compression
    quantize: bool = False

    def __init__(self, initial_data: dict) -> None:
        trainer_attr = pl.Trainer.default_attributes()
        for key in trainer_attr:
            setattr(self, key, trainer_attr[key])

        for key in initial_data:
            if hasattr(self, key):
                setattr(self, key, initial_data[key])


def build_trainer(hparams: Namespace) -> pl.Trainer:
    """
    :param hparams: Namespace

    Returns:
        - pytorch_lightning Trainer
    """
    # Early Stopping Callback
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=hparams.min_delta,
        patience=hparams.patience,
        verbose=hparams.verbose,
        mode=hparams.metric_mode,
    )

    # TestTube Logger Callback
    tb_logger = TensorBoardLogger(
        save_dir="experiments/",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )

    # Model Checkpoint Callback
    ckpt_path = os.path.join(
        "experiments/",
        tb_logger.version,
        "checkpoints",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=hparams.verbose,
        monitor=hparams.monitor,
        save_weights_only=hparams.save_weights_only,
        period=0,  # Always allow saving checkpoint even within the same epoch
        mode=hparams.metric_mode,
    )

    callback_list = [
        LearningRateMonitor()
    ]
    if hparams.quantize:
        torch.backends.quantized.engine = 'qnnpack'
        callback_list.append(
            #QuantizationAwareTraining(
            CustomQAT(
                input_compatible=True,
                qconfig="qnnpack"
                # qconfig=torch.quantization.QConfig(
                    # activation=torch.quantization.FakeQuantize.with_args(
                        # observer=torch.quantization.MovingAverageMinMaxObserver,
                        # quant_min=0,
                        # quant_max=255,
                        # reduce_range=False
                    # ),
                    # weight=torch.quantization.FakeQuantize.with_args(
                        # observer=torch.quantization.MovingAverageMinMaxObserver,
                        # quant_min=0,
                        # quant_max=255,
                        # reduce_range=False
                    # )
                # )
            )
        )

    trainer = pl.Trainer(
        logger=tb_logger,
        checkpoint_callback=True,#checkpoint_callback,
        #early_stop_callback=early_stop_callback,
        callbacks=callback_list,
        gradient_clip_val=hparams.gradient_clip_val,
        gpus=hparams.gpus,
        log_gpu_memory="all",
        deterministic=hparams.deterministic,
        overfit_batches=hparams.overfit_batches,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
        val_check_interval=hparams.val_check_interval,
        #log_save_interval=hparams.log_save_interval,
        distributed_backend="dp",
        precision=hparams.precision,
        weights_summary="top",
        profiler=hparams.profiler,
    )
    return trainer
