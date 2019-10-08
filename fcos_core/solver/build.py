# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import logging
from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    logger = logging.getLogger("fcos_core.trainer")
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if key.endswith(".offset.weight") or key.endswith(".offset.bias"):
            logger.info("set lr factor of {} as {}".format(
                key, cfg.SOLVER.DCONV_OFFSETS_LR_FACTOR
            ))
            lr *= cfg.SOLVER.DCONV_OFFSETS_LR_FACTOR
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
