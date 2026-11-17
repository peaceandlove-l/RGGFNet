import json
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import amp
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from sam.lsy.mymodelnew.Method import get_dataset, get_model
from sam.lsy.mymodelnew.toolbox.log import get_logger
from sam.lsy.mymodelnew.toolbox.loss_agent import (
    BoundaryAwareRLReward,
    BoundaryAwareSegmentationLoss,
    PPOActorCriticLoss,
)
from sam.lsy.mymodelnew.toolbox.metrics import averageMeter, runningScore
from sam.lsy.mymodelnew.toolbox.utils import ClassWeight, save_ckpt


torch.manual_seed(123)
np.random.seed(123)
cudnn.benchmark = True


def _reduce_scalar(value, distributed, world_size):
    if not distributed:
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def _get_geometry_depth(sample, depth, device):
    geometry_depth = sample.get("geometry_depth", None)
    if geometry_depth is None:
        # 兼容其他旧数据集；NYUv2 新数据管线会显式返回 geometry_depth。
        geometry_depth = depth[:, :1]
    return geometry_depth.to(device, non_blocking=True)


def _get_intrinsics(sample, device):
    intrinsics = sample.get("intrinsics", None)
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.to(device, non_blocking=True)
    return intrinsics


def run(args):
    with open(args.config, "r") as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}-NYU-paper4all'
    args.logdir = logdir
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(args.config, logdir)
    logger = get_logger(logdir)

    args.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend="nccl")
        args.world_size = dist.get_world_size()
        device = torch.device("cuda", args.gpu)
    else:
        args.gpu = 0
        args.world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(cfg).to(device)
    trainset, *testset = get_dataset(cfg)

    train_sampler = None
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    # 在 DDP 包装之前取得参数引用。warmup 阶段 policy 分支未参与 loss，因此需要
    # find_unused_parameters=True。
    core_model = model
    if hasattr(core_model, "get_policy_parameters"):
        policy_params = core_model.get_policy_parameters()
        main_params = core_model.get_main_parameters()
    else:
        policy_params = []
        main_params = [p for p in core_model.parameters() if p.requires_grad]

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=True,
        )

    train_loader = DataLoader(
        trainset,
        batch_size=cfg["ims_per_gpu"],
        shuffle=(train_sampler is None),
        num_workers=cfg["num_workers"],
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        testset[0],
        batch_size=cfg["ims_per_gpu"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.SGD(
        main_params,
        lr=cfg["lr_start"],
        weight_decay=cfg["weight_decay"],
        momentum=cfg["momentum"],
    )
    policy_optimizer = None
    if policy_params:
        policy_optimizer = torch.optim.AdamW(
            policy_params,
            lr=cfg.get("rl_lr", 1e-4),
            weight_decay=cfg.get("rl_weight_decay", 1e-4),
        )

    epochs = int(cfg["epochs"])
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda ep: max(0.0, 1.0 - ep / max(epochs, 1)) ** cfg.get("lr_power", 0.9),
    )
    policy_scheduler = None
    if policy_optimizer is not None:
        policy_scheduler = LambdaLR(
            policy_optimizer,
            lr_lambda=lambda ep: max(0.0, 1.0 - ep / max(epochs, 1)) ** 0.9,
        )

    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    if hasattr(trainset, "class_weight"):
        class_weight = trainset.class_weight
    else:
        class_weight = ClassWeight(cfg["class_weight"]).get_weight(
            train_loader, cfg["n_classes"]
        )
    class_weight = torch.from_numpy(np.asarray(class_weight)).float().to(device)
    class_weight[cfg["id_unlabel"]] = 0.0

    criterion = BoundaryAwareSegmentationLoss(
        weight=class_weight,
        ignore_index=cfg["id_unlabel"],
        dice_weight=cfg.get("dice_loss_weight", 0.25),
        boundary_weight=cfg.get("boundary_loss_weight", 0.20),
    ).to(device)
    reward_fn = BoundaryAwareRLReward(
        ignore_index=cfg["id_unlabel"],
        boundary_weight=cfg.get("reward_boundary_f1", 1.0),
        small_iou_weight=cfg.get("reward_small_iou", 0.5),
        overlap_weight=cfg.get("reward_overlap", 0.20),
        coverage_weight=cfg.get("reward_coverage", 0.30),
        latency_weight=cfg.get("reward_latency", 0.10),
        small_object_ratio=cfg.get("small_object_ratio", 0.02),
        coverage_target=cfg.get("coverage_target", 0.85),
        boundary_tolerance=cfg.get("boundary_tolerance", 1),
    ).to(device)
    ppo_criterion = PPOActorCriticLoss(
        clip_ratio=cfg.get("ppo_clip", 0.2),
        value_coef=cfg.get("ppo_value_coef", 0.5),
        entropy_coef=cfg.get("ppo_entropy_coef", 0.01),
    ).to(device)

    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    running_metrics_val = runningScore(
        cfg["n_classes"], ignore_index=cfg["id_unlabel"]
    )

    rl_start_epoch = int(cfg.get("rl_start_epoch", 30))
    rg_enabled_in_model = bool(cfg.get("use_rgra_agent", False))
    rgra_start_epoch = max(int(cfg.get("rgra_start_epoch", rl_start_epoch)), rl_start_epoch)
    rgra_usage_penalty = float(cfg.get("rgra_usage_penalty", 0.02))
    rl_weight = float(cfg.get("rl_weight", 0.05))
    ppo_epochs = max(int(cfg.get("ppo_epochs", 3)), 1)
    latency_every = max(int(cfg.get("latency_reward_every", 10)), 1)
    latency_ema = None
    latency_reference_ms = None
    best_miou = 0.0

    for ep in range(epochs):
        if args.distributed:
            train_sampler.set_epoch(ep)

        rl_active = ep >= rl_start_epoch and policy_optimizer is not None
        policy_mode = "sample" if rl_active else "warmup"
        rgra_active = bool(rg_enabled_in_model and ep >= rgra_start_epoch and rl_active)
        model.train()
        train_loss_meter.reset()
        epoch_metrics = defaultdict(float)
        epoch_metric_steps = 0

        for i, sample in enumerate(train_loader):
            image = sample["image"].to(device, non_blocking=True)
            depth = sample["depth"].to(device, non_blocking=True)
            geometry_depth = _get_geometry_depth(sample, depth, device)
            label = sample["label"].to(device, non_blocking=True)
            intrinsics = _get_intrinsics(sample, device)

            optimizer.zero_grad(set_to_none=True)
            if policy_optimizer is not None:
                policy_optimizer.zero_grad(set_to_none=True)

            measure_latency = device.type == "cuda" and (
                latency_ema is None or i % latency_every == 0
            )
            if measure_latency:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            elif device.type != "cuda":
                cpu_start = time.perf_counter()

            with amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(
                    image,
                    depth,
                    geometry_depth=geometry_depth,
                    intrinsics=intrinsics,
                    return_auxiliary=True,
                    policy_mode=policy_mode,
                    use_rgra_agent=rgra_active,
                )
                logits = outputs["logits"]
                seg_loss, seg_parts = criterion(
                    logits, label, return_components=True
                )

            if measure_latency:
                end_event.record()
                torch.cuda.synchronize(device)
                current_latency_ms = start_event.elapsed_time(end_event)
                latency_ema = (
                    current_latency_ms
                    if latency_ema is None
                    else 0.90 * latency_ema + 0.10 * current_latency_ms
                )
            elif device.type != "cuda":
                current_latency_ms = (time.perf_counter() - cpu_start) * 1000.0
                latency_ema = (
                    current_latency_ms
                    if latency_ema is None
                    else 0.90 * latency_ema + 0.10 * current_latency_ms
                )

            if rl_active and latency_reference_ms is None:
                # 以监督 warmup 的实际 forward latency 作为归一化基线，而不是写死毫秒数。
                latency_reference_ms = float(latency_ema or 1.0)

            reward = None
            if rl_active:
                reward, reward_metrics = reward_fn(
                    logits.detach(),
                    label,
                    outputs,
                    latency_ms=latency_ema,
                    latency_reference_ms=latency_reference_ms,
                )
                if rgra_active:
                    # Only penalize actually active RGRA levels, not inactive HRFR levels.
                    ratios = torch.stack(outputs["level_rgra_active_ratios"], dim=1)
                    keep = outputs["hrfr_level_keep"].to(dtype=ratios.dtype)
                    active_ratio = (ratios * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
                    reward = reward - rgra_usage_penalty * active_ratio.detach()
                    epoch_metrics["rgra_active_ratio"] += float(active_ratio.mean().item())
                for key, value in reward_metrics.items():
                    epoch_metrics[key] += float(value.detach().item())

            for key, value in seg_parts.items():
                epoch_metrics[f"seg_{key}"] += float(value.item())
            epoch_metrics["latency_ms"] += float(latency_ema or 0.0)
            epoch_metric_steps += 1

            # 1) 主网络更新：监督损失训练 NGFC / Gaussian proposal / decoder /
            #    RAGS continuous adjustment heads。离散 actor/critic 参数被排除在 main optimizer 外。
            scaler.scale(seg_loss).backward()
            if cfg.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(main_params, cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            # 2) 严格 PPO replay：缓存 rollout state/action，不重跑 DINO/NGFC；
            #    每个 minibatch 对同一 old policy rollout 做多个 clipped PPO epoch。
            if rl_active and policy_optimizer is not None and reward is not None:
                rollout = outputs.get("policy_rollout", None)
                if rollout is None:
                    raise RuntimeError("RL 阶段缺少 policy_rollout")
                policy_core = model.module if hasattr(model, "module") else model
                ppo_loss_total = 0.0
                ppo_part_total = defaultdict(float)
                for _ in range(ppo_epochs):
                    policy_optimizer.zero_grad(set_to_none=True)
                    current_policy_stats = policy_core.evaluate_policy_rollout(rollout)
                    ppo_loss, ppo_parts = ppo_criterion(
                        current_policy_stats,
                        reward.detach(),
                        return_components=True,
                    )
                    (rl_weight * ppo_loss).backward()

                    # 自定义 evaluator 不经过 DDP.forward，因此显式同步 policy gradients。
                    if args.distributed:
                        for parameter in policy_params:
                            if parameter.grad is not None:
                                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                                parameter.grad /= args.world_size
                    if cfg.get("policy_grad_clip", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            policy_params, cfg.get("policy_grad_clip", 1.0)
                        )
                    policy_optimizer.step()
                    ppo_loss_total += float(ppo_loss.detach().item())
                    for key, value in ppo_parts.items():
                        ppo_part_total[key] += float(value.detach().item())

                epoch_metrics["ppo_loss"] += ppo_loss_total / ppo_epochs
                for key, value in ppo_part_total.items():
                    epoch_metrics[f"ppo_{key}"] += value / ppo_epochs

            reduced_loss = _reduce_scalar(seg_loss, args.distributed, args.world_size)
            train_loss_meter.update(reduced_loss.item())

        scheduler.step()
        if policy_scheduler is not None and rl_active:
            policy_scheduler.step()

        # validation 使用 deterministic policy。warmup 完成后，这对应真正部署策略。
        model.eval()
        running_metrics_val.reset()
        val_loss_meter.reset()
        with torch.no_grad():
            for sample in val_loader:
                image = sample["image"].to(device, non_blocking=True)
                depth = sample["depth"].to(device, non_blocking=True)
                geometry_depth = _get_geometry_depth(sample, depth, device)
                label = sample["label"].to(device, non_blocking=True)
                intrinsics = _get_intrinsics(sample, device)

                logits = model(
                    image,
                    depth,
                    geometry_depth=geometry_depth,
                    intrinsics=intrinsics,
                    policy_mode="deterministic" if ep >= rl_start_epoch else "warmup",
                    use_rgra_agent=rg_enabled_in_model and ep >= rgra_start_epoch,
                )
                loss = criterion(logits, label)
                val_loss_meter.update(loss.item())
                prediction = logits.argmax(dim=1).cpu().numpy()
                running_metrics_val.update(label.cpu().numpy(), prediction)

        if args.local_rank == 0:
            scores = running_metrics_val.get_scores()[0]
            newmiou = scores["mIou: "]
            newpa = scores["pixel_acc: "]
            newca = scores["class_acc: "]
            mode_name = "RL" if rl_active else "SUP-WARMUP"
            metric_text = ""
            if epoch_metric_steps > 0:
                selected_keys = [
                    "seg_ce",
                    "seg_dice",
                    "seg_boundary",
                    "reward",
                    "boundary_f1",
                    "small_iou",
                    "gaussian_overlap",
                    "coverage",
                    "ppo_loss",
                    "rgra_active_ratio",
                    "latency_ms",
                ]
                pieces = []
                for key in selected_keys:
                    if key in epoch_metrics:
                        pieces.append(
                            f"{key}={epoch_metrics[key] / epoch_metric_steps:.4f}"
                        )
                metric_text = " | " + ", ".join(pieces)

            logger.info(
                f'Epoch [{ep + 1:3d}/{epochs}] {mode_name} '
                f'train/val={train_loss_meter.avg:.5f}/{val_loss_meter.avg:.5f}, '
                f'PA={newpa:.3f}, CA={newca:.3f}, mIoU={newmiou:.3f}, '
                f'best={best_miou:.3f}{metric_text}'
            )
            save_ckpt(logdir, model, kind="end")
            if newmiou > best_miou:
                save_ckpt(logdir, model, kind="best")
                best_miou = newmiou

    if args.local_rank == 0:
        save_ckpt(logdir, model, kind="end")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGS + HRFR training")
    parser.add_argument(
        "--config",
        type=str,
        default="/data/Lsy/sam/lsy/mymodelnew/configs/nyuv2_agent.json",
        # default="/home/yangenquan/Disk/lsy/mymodelnew/configs/nyuv2.json",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    run(args)
