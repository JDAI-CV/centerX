# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.
The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import logging
import time

from detectron2.data import build_detection_test_loader

from detectron2.engine.defaults import DefaultTrainer
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from torch.nn.parallel import DistributedDataParallel
from detectron2.config import get_cfg
from configs import add_centernet_config


# from detectron2.modeling import build_model
from modeling import build_model
from data import DatasetMapper, build_detection_train_loader
from torchtools.optim import RangerLars
from solver import WarmupCosineAnnealingLR
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from detectron2.engine import hooks
from . import additional_hooks

import torch


__all__ = ["CenterTrainer"]


class CenterTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            logger = setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        # Assume these objects must be constructed in this order.

        data_loader = self.build_train_loader(cfg)
        cfg = self.auto_scale_hyperparams(cfg, data_loader)

        model = self.build_model(cfg)
        # KD or not
        self.kd = cfg.MODEL.CENTERNET.KD.ENABLED
        self.model_t = None
        if self.kd:
            self.model_t = self.build_teacher_model(cfg)

        optimizer = self.build_optimizer(cfg, model)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0

        if cfg.SOLVER.SWA.ENABLED:
            self.max_iter = cfg.SOLVER.MAX_ITER + cfg.SOLVER.SWA.ITER
        else:
            self.max_iter = cfg.SOLVER.MAX_ITER

        self.cfg = cfg
        self.skip_loss = cfg.MODEL.CENTERNET.LOSS.SKIP_LOSS
        self.history_loss = 10e8
        self.skip_weight = cfg.MODEL.CENTERNET.LOSS.SKIP_WEIGHT

        self.communism = cfg.MODEL.CENTERNET.LOSS.COMMUNISM.ENABLE
        self.communism_cls_loss = cfg.MODEL.CENTERNET.LOSS.COMMUNISM.CLS_LOSS
        self.communism_wh_loss = cfg.MODEL.CENTERNET.LOSS.COMMUNISM.WH_LOSS
        self.communism_off_loss = cfg.MODEL.CENTERNET.LOSS.COMMUNISM.OFF_LOSS

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_teacher_model(cls, cfg):
        teacher_cfgs = cfg.MODEL.CENTERNET.KD.TEACHER_CFG
        teacher_weights = cfg.MODEL.CENTERNET.KD.TEACHER_WEIGTHS
        assert len(teacher_cfgs) == len(teacher_weights)
        model_ts = []

        for t_cfg,t_weight in zip(teacher_cfgs, teacher_weights):
            teacher_cfg = get_cfg()
            add_centernet_config(teacher_cfg)
            teacher_cfg.merge_from_file(t_cfg)

            model_t = build_model(teacher_cfg)
            for param in model_t.parameters():
                param.requires_grad = False

            # Load pre-trained teacher model
            logger = logging.getLogger("detectron2")
            logger.info("Loading teacher model ...")
            DetectionCheckpointer(model_t).load(t_weight)

            # Load pre-trained student model
            # logger.info("Loading student model ...")
            # Checkpointer(self.model, self.data_loader.dataset).load(cfg.MODEL.STUDENT_WEIGHTS)

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            model_t.apply(set_bn_eval)
            model_t.eval()

            model_ts.append(model_t)

        return model_ts

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module in model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if isinstance(module, norm_module_types):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        assert cfg.SOLVER.OPTIM_NAME in ["RangerLars", "Adam", "SGD"]
        if cfg.SOLVER.OPTIM_NAME == "RangerLars":
            optimizer = RangerLars(params, lr=cfg.SOLVER.BASE_LR)
        if cfg.SOLVER.OPTIM_NAME == "Adam":
            optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR)
        if cfg.SOLVER.OPTIM_NAME == "SGD":
            optimizer = torch.optim.SGD(
                params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV
            )

        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        if cfg.SOLVER.LR_SCHEDULER_NAME == "WarmupCosineAnnealingLR":
            decay_iter = int(cfg.SOLVER.MAX_ITER * cfg.SOLVER.COSINE_DECAY_ITER)
            return WarmupCosineAnnealingLR(
                optimizer,
                max_iters=cfg.SOLVER.MAX_ITER,
                delay_iters=decay_iter,
                eta_min_lr=cfg.SOLVER.MIN_LR,
                warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.WARMUP_METHOD,
            )
        return build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger(__name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        if cfg.SOLVER.SWA.ENABLED:
            ret.append(
                additional_hooks.SWA(
                    cfg.SOLVER.MAX_ITER,
                    cfg.SOLVER.SWA.PERIOD,
                    cfg.SOLVER.SWA.LR_START,
                    cfg.SOLVER.SWA.ETA_MIN_LR,
                    cfg.SOLVER.SWA.LR_SCHED,
                )
            )

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(self.model):
            logger.info("Prepare precise BN dataset")
            ret.append(hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ))
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), 20))

        return ret

    def run_step(self):
        """
        Implement the moco training logic described above.
        """
        assert self.model.training, "[KDTrainer] base model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start


        loss_dict = self.model(data, self.model_t)

        if self.communism:
            losses = 0.0
            for loss_name in loss_dict:
                if 'cls' in loss_name:
                    alpha = self.communism_cls_loss / loss_dict[loss_name].detach()
                    losses = losses + alpha * loss_dict[loss_name]
                elif 'wh' in loss_name:
                    alpha = self.communism_wh_loss / loss_dict[loss_name].detach()
                    losses = losses + alpha * loss_dict[loss_name]
                elif 'off' in loss_name:
                    alpha = self.communism_off_loss / loss_dict[loss_name].detach()
                    losses = losses + alpha * loss_dict[loss_name]
                else:
                    losses = losses + loss_dict[loss_name]

        else:
            losses = sum(loss for loss in loss_dict.values())
        #self._detect_anomaly(losses, loss_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        if self.skip_loss:
            if self.history_loss * self.skip_weight > losses.item():
                losses.backward()
                self.history_loss = losses.item()
            else:
                losses = 0.0 * losses
                losses.backward()
        else:
            losses.backward()
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        metrics_dict = loss_dict
        try:
            self._write_metrics(metrics_dict, data_time)
        except:
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)

        self.optimizer.step()

    @staticmethod
    def auto_scale_hyperparams(cfg, data_loader):
        r"""
        This is used for auto-computation actual training iterations,
        because some hyper-param, such as MAX_ITER, means training epochs rather than iters,
        so we need to convert specific hyper-param to training iterations.
        """

        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        iters_per_epoch = len(data_loader.dataset.dataset) // cfg.SOLVER.IMS_PER_BATCH
        cfg.SOLVER.MAX_ITER *= iters_per_epoch
        cfg.SOLVER.WARMUP_ITERS *= iters_per_epoch
        cfg.SOLVER.WARMUP_FACTOR = 1.0 / cfg.SOLVER.WARMUP_ITERS
        cfg.SOLVER.STEPS = list(cfg.SOLVER.STEPS)
        for i in range(len(cfg.SOLVER.STEPS)):
            cfg.SOLVER.STEPS[i] *= iters_per_epoch
        cfg.SOLVER.STEPS = tuple(cfg.SOLVER.STEPS)
        cfg.SOLVER.SWA.ITER *= iters_per_epoch
        cfg.SOLVER.SWA.PERIOD *= iters_per_epoch
        cfg.SOLVER.CHECKPOINT_PERIOD *= iters_per_epoch

        # Evaluation period must be divided by 200 for writing into tensorboard.
        num_mod = (200 - cfg.TEST.EVAL_PERIOD * iters_per_epoch) % 200
        cfg.TEST.EVAL_PERIOD = cfg.TEST.EVAL_PERIOD * iters_per_epoch + num_mod

        logger = logging.getLogger(__name__)
        logger.info(
            f"max_Iter={cfg.SOLVER.MAX_ITER}, wamrup_Iter={cfg.SOLVER.WARMUP_ITERS}, "
            f"step_Iter={cfg.SOLVER.STEPS}, ckpt_Iter={cfg.SOLVER.CHECKPOINT_PERIOD}, "
            f"eval_Iter={cfg.TEST.EVAL_PERIOD}."
        )

        if frozen: cfg.freeze()

        return cfg
