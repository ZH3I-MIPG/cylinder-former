"""
Scheduler

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

SCHEDULERS = Registry("schedulers")


@SCHEDULERS.register_module()
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            milestones=[int(rate * total_steps) for rate in milestones],
            gamma=gamma,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class MultiStepWithWarmupLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        warmup_rate=0.05,
        warmup_scale=1e-6,
        last_epoch=-1,
    ):
        milestones = [rate * total_steps for rate in milestones]

        def multi_step_with_warmup(s):
            factor = 1.0
            for i in range(len(milestones)):
                if s < milestones[i]:
                    break
                factor *= gamma

            if s <= warmup_rate * total_steps:
                warmup_coefficient = 1 - (1 - s / warmup_rate / total_steps) * (
                    1 - warmup_scale
                )
            else:
                warmup_coefficient = 1.0
            return warmup_coefficient * factor

        super().__init__(
            optimizer=optimizer,
            lr_lambda=multi_step_with_warmup,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class PolyLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        total_steps,
        power=0.9,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: (1 - s / (total_steps + 1)) ** power,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class ExpLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        total_steps,
        gamma=0.9,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: gamma ** (s / total_steps),
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    def __init__(
        self,
        optimizer,
        total_steps,
        eta_min=0,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    r"""
    torch.optim.lr_scheduler.OneCycleLR, Block total_steps
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
        )

@SCHEDULERS.register_module()
class ResetOneCycleLR:
    def __init__(self,
                 optimizer,
                 max_lr,
                 steps_per_epoch,  # 每个 epoch 的步数
                 epochs_per_cycle=1,  # 每个周期的 epoch 数
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.0,
                 final_div_factor=1e4,
                 three_phase=False):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch  # 每个 epoch 的步数
        self.epochs_per_cycle = epochs_per_cycle
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        self.current_epoch = 0
        self.scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=steps_per_epoch * epochs_per_cycle,  # 每个 epoch 的步数
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=-1
        )

    def step(self):
        self.scheduler.step()

    def reset(self):
        self.current_epoch += 1
        print(f"Current epoch: {self.current_epoch },steps per epoch: {self.epochs_per_cycle}.")
        if self.current_epoch == self.epochs_per_cycle:
            # 在每个 epoch 结束时重置
            self.scheduler = OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.max_lr,
                total_steps=self.steps_per_epoch * self.epochs_per_cycle,  # 每个 epoch 的步数
                pct_start=self.pct_start,
                anneal_strategy=self.anneal_strategy,
                cycle_momentum=self.cycle_momentum,
                base_momentum=self.base_momentum,
                max_momentum=self.max_momentum,
                div_factor=self.div_factor,
                final_div_factor=self.final_div_factor,
                three_phase=self.three_phase,
                last_epoch=-1
            )
            self.current_epoch = 0


    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        """返回调度器的状态字典"""
        return {
            'current_epoch': self.current_epoch,
            'scheduler_state_dict': self.scheduler.state_dict(),
            'max_lr': self.max_lr,
            'steps_per_epoch': self.steps_per_epoch,
            'epochs_per_cycle': self.epochs_per_cycle,
            'pct_start': self.pct_start,
            'anneal_strategy': self.anneal_strategy,
            'cycle_momentum': self.cycle_momentum,
            'base_momentum': self.base_momentum,
            'max_momentum': self.max_momentum,
            'div_factor': self.div_factor,
            'final_div_factor': self.final_div_factor,
            'three_phase': self.three_phase
        }

    def load_state_dict(self, state_dict):
        """从状态字典加载调度器的状态"""
        self.current_epoch = state_dict['current_epoch']
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        self.max_lr = state_dict['max_lr']
        self.steps_per_epoch = state_dict['steps_per_epoch']
        self.epochs_per_cycle = state_dict['epochs_per_cycle']
        self.pct_start = state_dict['pct_start']
        self.anneal_strategy = state_dict['anneal_strategy']
        self.cycle_momentum = state_dict['cycle_momentum']
        self.base_momentum = state_dict['base_momentum']
        self.max_momentum = state_dict['max_momentum']
        self.div_factor = state_dict['div_factor']
        self.final_div_factor = state_dict['final_div_factor']
        self.three_phase = state_dict['three_phase']


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        start_value=0,
        warmup_iters=0,
        freeze_value=None,
        freeze_iters=0,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters

        warmup_schedule = np.linspace(start_value, base_value, warmup_iters)

        if freeze_value is None:
            freeze_value = final_value
        freeze_schedule = np.ones(freeze_iters) * freeze_value

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((warmup_schedule, schedule, freeze_schedule))
        self.iter = 0

    def get(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]

    def step(self):
        value = self.get(self.iter)
        self.iter += 1
        return value

    def reset(self):
        self.iter = 0

    def __getitem__(self, it):
        return self.get(it)


def build_scheduler(cfg, optimizer):
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg=cfg)
