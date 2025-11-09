"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry
from collections import OrderedDict

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if self.cfg.resume:
            if os.path.isfile(self.cfg.weight):
                self.logger.info(f"Loading weight at: {self.cfg.weight}")
                checkpoint = torch.load(self.cfg.weight)
                weight = OrderedDict()
                for key, value in checkpoint["state_dict"].items():
                    if key.startswith("module."):
                        if comm.get_world_size() == 1:
                            key = key[7:]  # module.xxx.xxx -> xxx.xxx
                    else:
                        if comm.get_world_size() > 1:
                            key = "module." + key  # xxx.xxx -> module.xxx.xxx
                    weight[key] = value
                model.load_state_dict(weight, strict=True)
                self.logger.info(
                    "=> Loaded weight '{}' (epoch {})".format(
                        self.cfg.weight, checkpoint["epoch"]
                    )
                )
            else:
                raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        if self.cfg.scheduler.type == "ResetOneCycleLR":
            self.cfg.scheduler.steps_per_epoch = len(self.train_loader)
        else:
            self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("KDTrainer")
class KDTrainer(TrainerBase):
    def __init__(self, cfg):
        super(KDTrainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building Teacher model ...")
        self.t_model = self.build_t_model()
        self.logger.info("=> Building Student model ...")
        self.s_model = self.build_s_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        self.alpha1 = cfg.alpha1  # t_dec_feat & s_dec_feat 蒸馏权重
        self.alpha2 = cfg.alpha2  # t_emb_feat & s_emb_feat 蒸馏权重
        self.alpha3 = cfg.alpha3  # 输出层软标签蒸馏权重
        self.tau = cfg.tau  # 温度参数

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.t_model.eval()
                self.s_model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with torch.no_grad():
            t_output_dict = self.t_model(input_dict)

        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            s_output_dict = self.s_model(input_dict)


        t_emb_feat = t_output_dict["emb_feat"]
        s_emb_feat = s_output_dict["emb_feat"]

        t_seg_logits = t_output_dict["seg_logits"]
        s_seg_logits = s_output_dict["seg_logits"]
        loss_seg = s_output_dict["loss"]

        output = dict()
        # 1. 定义权重系数与温度参数（可根据验证集调整）
        alpha1 = self.alpha1  # t_dec_feat & s_dec_feat 蒸馏权重
        alpha2 = self.alpha2  # t_emb_feat & s_emb_feat 蒸馏权重
        alpha3 = self.alpha3  # 输出层软标签蒸馏权重
        tau = self.tau  # 温度参数

        # 2. 中间特征直接蒸馏（L2范数，参考论文Mask Distillation的L2损失形式，移除掩码对齐，）
        # dec_feat 特征蒸馏
        loss_kd_med = 0

        def encode_coords(coords, max_vals):
            """将三维整数坐标编码为唯一整数"""
            x, y, z = coords.T  # 拆分坐标分量
            # 编码公式：x * (max_y+1) * (max_z+1) + y * (max_z+1) + z
            encoded = x * (max_vals[1] + 1) * (max_vals[2] + 1) + y * (max_vals[2] + 1) + z
            return encoded

        def get_max_vals(t_coords, s_coords):
            """从t和s的坐标中动态计算各维度最大值"""
            # 转换为CPU计算，避免占用GPU资源
            t_coords_cpu = t_coords.cpu()
            s_coords_cpu = s_coords.cpu()

            # 计算每个维度的最大值（t和s中的较大者）
            max_x = max(t_coords_cpu[:, 0].max().item(), s_coords_cpu[:, 0].max().item())
            max_y = max(t_coords_cpu[:, 1].max().item(), s_coords_cpu[:, 1].max().item())
            max_z = max(t_coords_cpu[:, 2].max().item(), s_coords_cpu[:, 2].max().item())

            return (max_x, max_y, max_z)

        def compute_matching_loss(t_feat, s_feat, t_coords, s_coords):
            """基于动态max_vals的哈希匹配损失计算"""
            t_feat = F.normalize(t_feat, p=2, dim=-1)  # 教师特征归一化
            s_feat = F.normalize(s_feat, p=2, dim=-1)  # 学生特征归一化

            # 确保坐标是整数类型
            t_coords = t_coords.long()  # [N, 3]
            s_coords = s_coords.long()  # [M, 3]

            # 动态计算当前block的坐标范围最大值
            max_vals = get_max_vals(t_coords, s_coords)

            # 编码s坐标并构建哈希映射（坐标→索引）
            s_encoded = encode_coords(s_coords.cpu(), max_vals).numpy()  # [M]
            s_coord_map = {code: idx for idx, code in enumerate(s_encoded)}  # {编码: 索引}

            # 编码t坐标并查找匹配的s索引
            t_encoded = encode_coords(t_coords.cpu(), max_vals).numpy()  # [N]
            match_indices = []  # 存储t中匹配到的s索引
            for code in t_encoded:
                if code in s_coord_map:
                    match_indices.append(s_coord_map[code])
                else:
                    match_indices.append(-1)  # 标记不匹配

            # 生成mask（仅保留匹配的坐标）
            match_indices = torch.tensor(match_indices, device=t_feat.device)  # [N]
            mask = match_indices != -1  # [N]

            if not mask.any():
                return torch.tensor(0.0, device=t_feat.device)

            # 提取匹配的特征（批量操作）
            t_feat_filtered = t_feat[mask]  # [K, C]
            s_feat_filtered = s_feat[match_indices[mask]]  # [K, C]

            # 计算L2损失
            return torch.mean(torch.norm(t_feat_filtered - s_feat_filtered, p=2, dim=-1))

        # 处理编码器部分（0-4层）
        for i in range(5):
            t_m_feat = t_output_dict["layer_features"][f"block_enc{i}"]
            s_m_feat = s_output_dict["layer_features"][f"block_enc{i}"]
            t_coords = t_output_dict["layer_coord"][f"block_enc{i}_coord"]
            s_coords = s_output_dict["layer_coord"][f"block_enc{i}_coord"]

            loss_kd_med += compute_matching_loss(t_m_feat, s_m_feat, t_coords, s_coords)

        # 处理解码器部分（0-3层）
        for i in range(4):
            t_m_feat = t_output_dict["layer_features"][f"block_dec{i}"]
            s_m_feat = s_output_dict["layer_features"][f"block_dec{i}"]
            t_coords = t_output_dict["layer_coord"][f"block_dec{i}_coord"]
            s_coords = s_output_dict["layer_coord"][f"block_dec{i}_coord"]

            loss_kd_med += compute_matching_loss(t_m_feat, s_m_feat, t_coords, s_coords)

        # emb_feat 特征蒸馏
        t_emb_feat_norm = F.normalize(t_emb_feat, p=2, dim=-1)
        s_emb_feat_norm = F.normalize(s_emb_feat, p=2, dim=-1)
        loss_kd_emb = torch.mean(torch.norm(t_emb_feat_norm - s_emb_feat_norm, p=2, dim=-1))

        # 3. 输出层软标签蒸馏（KL散度，教师引导学生学习软分布）
        # 软化教师与学生的logits，计算概率分布
        t_soft = F.softmax(t_seg_logits / tau, dim=1)
        s_soft = F.softmax(s_seg_logits / tau, dim=1)
        # KL散度（教师分布对学生分布的 divergence，确保学生拟合教师）
        loss_kd_logits = F.kl_div(s_soft.log(), t_soft, reduction='batchmean') * (tau ** 2)  # 乘tau²平衡梯度

        # 4. 最终总损失（原任务损失 + 三类蒸馏损失加权叠加，参考论文总损失结构，）
        loss = loss_seg + alpha1 * loss_kd_med + alpha2 * loss_kd_emb + alpha3 * loss_kd_logits
        # if loss_seg < 0.5 and alpha1 * loss_kd_med < 0.5 * loss:
        #     self.alpha1 = 1
        # if loss_seg < 0.5 and alpha2 * loss_kd_emb < 0.5 * loss:
        #     self.alpha2 = 5
        # if loss_seg < 0.5 and alpha3 * loss_kd_logits < 0.5 * loss:
        #     self.alpha3 = 5
        output["loss_all"] = loss
        output["loss_seg"] = loss_seg
        output["loss_kd_med"] = loss_kd_med
        output["loss_kd_emb"] = loss_kd_emb
        output["loss_kd_logits"] = loss_kd_logits
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_t_model(self):
        model = build_model(self.cfg.teacher_model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Teacher Num params: {n_parameters}")

        if getattr(self.cfg, "teacher_weight", None):
            teacher_path = self.cfg.teacher_weight
            if os.path.isfile(teacher_path):
                self.logger.info(f"=> Loading teacher model weights from {teacher_path}")
                state_dict = torch.load(teacher_path, map_location="cpu", weights_only=False)
                # 处理 DataParallel/DistributedDataParallel 多包一层的情况
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # 去掉 key 前缀中的 "module."，适配 DDP 保存的权重
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k[7:] if k.startswith("module.") else k
                    new_state_dict[new_key] = v
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                if missing:
                    self.logger.warning(f"=> Missing keys: {missing}")
                if unexpected:
                    self.logger.warning(f"=> Unexpected keys: {unexpected}")
            else:
                self.logger.warning(f"=> teacher_model path not exist: {teacher_path}")

        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_s_model(self):
        model = build_model(self.cfg.student_model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Student Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.s_model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")

        if self.cfg.scheduler.type == "ResetOneCycleLR":
            self.cfg.scheduler.steps_per_epoch = len(self.train_loader)
        else:
            self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("PreTedTrainer")
class PreTedTrainer(TrainerBase):
    def __init__(self, cfg):
        super(PreTedTrainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        # 1. 根据配置构建模型结构
        model = build_model(self.cfg.model)

        # 2. 可选：SyncBatchNorm
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # 3. 统计可训练参数量
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")

        # 4. 如果提供了预训练权重，则加载
        if getattr(self.cfg, "pretrained_model", None):
            pretrained_path = self.cfg.pretrained_model
            if os.path.isfile(pretrained_path):
                self.logger.info(f"=> Loading pretrained weights from {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location="cpu")
                # 处理 DataParallel/DistributedDataParallel 多包一层的情况
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # 去掉 key 前缀中的 "module."，适配 DDP 保存的权重
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k[7:] if k.startswith("module.") else k
                    # 直接丢弃 recon_head.* 的权重
                    if new_key.startswith("recon_head."):
                        continue
                    if new_key.startswith("backbone.dec"):
                        continue
                    new_state_dict[new_key] = v
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                if missing:
                    self.logger.warning(f"=> Missing keys: {missing}")
                if unexpected:
                    self.logger.warning(f"=> Unexpected keys: {unexpected}")
            else:
                self.logger.warning(f"=> pretrained_model path not exist: {pretrained_path}")

        # 5. 移动到 GPU 并封装为 DDP
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")

        if self.cfg.scheduler.type == "ResetOneCycleLR":
            self.cfg.scheduler.steps_per_epoch = len(self.train_loader)
        else:
            self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
