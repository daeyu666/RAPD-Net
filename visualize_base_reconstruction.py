"""
HSI 重建结果可视化工具。

使用方式：
    1. 构建并加载你自己训练的模型；
    2. 调用 visualize_reconstruction(model, cfg, ...) 即可。

不再依赖具体的模型类，用户可以传入任意满足接口
``pred = model(lr_hsi, hr_msi)`` 的模型。
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TrainConfig, get_dataset_configs, make_dirs
from data_loader import build_loaders
from metrics import calc_metrics
from srf_utils import load_hsi_wavelengths
from utils import (
    get_device,
    load_checkpoint,
    move_to_device,
    set_seed,
    tensor_to_numpy,
    save_mat,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def load_analysis_wavelengths(
    cfg: TrainConfig,
    info: dict,
    n_bands: int,
) -> Optional[np.ndarray]:
    """加载波长信息。"""
    wavelengths = info.get("hsi_wavelengths", None)
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
        if wavelengths.size == n_bands:
            return wavelengths

    if cfg.wavelength_path:
        wavelength_path = cfg.wavelength_path
    else:
        wavelength_path = os.path.join(cfg.wavelength_root, f"{cfg.dataset}.txt")

    if os.path.exists(wavelength_path):
        return load_hsi_wavelengths(wavelength_path=wavelength_path, n_bands=n_bands)

    return None


def choose_rgb_indices(
    n_bands: int,
    wavelengths: Optional[np.ndarray] = None,
    targets: Tuple[float, float, float] = (650.0, 550.0, 470.0),
) -> List[int]:
    """选择 RGB 显示用的波段索引。"""
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
        return [int(np.argmin(np.abs(wavelengths - t))) for t in targets]

    # fallback: 使用高/中/低波段位置
    return [
        int(round((n_bands - 1) * 0.70)),
        int(round((n_bands - 1) * 0.50)),
        int(round((n_bands - 1) * 0.30)),
    ]


def percentile_stretch(
    rgb: np.ndarray,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """百分位拉伸，用于 RGB 显示。"""
    rgb = np.asarray(rgb, dtype=np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[2]):
        band = rgb[:, :, c]
        lo = np.percentile(band, low)
        hi = np.percentile(band, high)
        if hi - lo < 1e-8:
            out[:, :, c] = np.clip(band, 0.0, 1.0)
        else:
            out[:, :, c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def to_rgb(hsi_hwc: np.ndarray, rgb_indices: List[int]) -> np.ndarray:
    """H×W×C -> RGB 图像。"""
    rgb = hsi_hwc[:, :, rgb_indices]
    return percentile_stretch(rgb)


def calc_sam_map(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算逐像素 SAM 误差图（单位：度）。"""
    dot = torch.sum(pred * gt, dim=1)
    pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1) + eps)
    gt_norm = torch.sqrt(torch.sum(gt * gt, dim=1) + eps)
    cos = dot / (pred_norm * gt_norm + eps)
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos) * 180.0 / np.pi
    return angle


def get_point_list(
    pred: torch.Tensor,
    gt: torch.Tensor,
    sam_map: torch.Tensor,
    max_points: int = 4,
) -> List[Tuple[int, int, str]]:
    """选取用于绘制光谱曲线的代表性像素点。"""
    _, _, h, w = pred.shape
    error_map = torch.mean(torch.abs(pred - gt), dim=1)[0]
    sam = sam_map[0]

    points = []
    points.append((h // 2, w // 2, "center"))

    sam_idx = int(torch.argmax(sam).item())
    points.append((sam_idx // w, sam_idx % w, "max_sam"))

    err_idx = int(torch.argmax(error_map).item())
    points.append((err_idx // w, err_idx % w, "max_l1"))

    points.append((random.randint(0, h - 1), random.randint(0, w - 1), "random"))

    unique = []
    seen = set()
    for y, x, name in points:
        key = (int(y), int(x))
        if key not in seen:
            seen.add(key)
            unique.append((int(y), int(x), name))
        if len(unique) >= max_points:
            break
    return unique


# ---------------------------------------------------------------------------
# 绑图函数
# ---------------------------------------------------------------------------
def plot_composite(
    save_path: str,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    lr_up_np: np.ndarray,
    rgb_indices: List[int],
    sam_np: np.ndarray,
    abs_err_np: np.ndarray,
    title: str,
):
    """绘制综合对比图（2×3 子图）。"""
    gt_rgb = to_rgb(gt_np, rgb_indices)
    pred_rgb = to_rgb(pred_np, rgb_indices)
    lr_rgb = to_rgb(lr_up_np, rgb_indices)
    diff_rgb = np.abs(pred_rgb - gt_rgb)
    diff_rgb = percentile_stretch(diff_rgb, low=0.0, high=99.5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title("GT RGB")
    axes[0, 1].imshow(pred_rgb)
    axes[0, 1].set_title("Reconstruction RGB")
    axes[0, 2].imshow(lr_rgb)
    axes[0, 2].set_title("Upsampled LR-HSI RGB")

    axes[1, 0].imshow(diff_rgb)
    axes[1, 0].set_title("RGB Absolute Difference")

    im1 = axes[1, 1].imshow(abs_err_np)
    axes[1, 1].set_title("Mean Absolute Error Map")
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 2].imshow(sam_np)
    axes[1, 2].set_title("SAM Map / degree")
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.reshape(-1):
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_spectra(
    save_path: str,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    lr_up_np: np.ndarray,
    wavelengths: Optional[np.ndarray],
    points: List[Tuple[int, int, str]],
    rgb_indices: List[int],
):
    """绘制代表性像素的光谱曲线对比。"""
    n_bands = gt_np.shape[2]
    if wavelengths is None:
        x_axis = np.arange(n_bands)
        x_label = "Band index"
    else:
        x_axis = wavelengths
        x_label = "Wavelength / nm"

    fig, axes = plt.subplots(len(points), 1, figsize=(10, 3.2 * len(points)))
    if len(points) == 1:
        axes = [axes]

    for ax, (y, x, name) in zip(axes, points):
        ax.plot(x_axis, gt_np[y, x, :], label="GT")
        ax.plot(x_axis, pred_np[y, x, :], label="Reconstruction")
        ax.plot(x_axis, lr_up_np[y, x, :], label="LR up", linestyle="--")

        if wavelengths is not None:
            for idx in rgb_indices:
                ax.axvline(float(wavelengths[idx]), linestyle=":", linewidth=0.8)

        ax.set_title(f"Spectrum at ({y}, {x}) - {name}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Reflectance / normalized")
        ax.grid(True, linewidth=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 可视化参数
# ---------------------------------------------------------------------------
@dataclass
class VisArgs:
    """可视化参数（独立于 TrainConfig 的可视化专用参数）。"""
    split: str = "test"
    start_index: int = 0
    num_samples: int = 1
    num_spectrum_points: int = 4


# ---------------------------------------------------------------------------
# 主可视化函数（模型通过参数传入，不再硬编码具体类）
# ---------------------------------------------------------------------------
@torch.no_grad()
def visualize_reconstruction(
    model: nn.Module,
    cfg: TrainConfig,
    vis_args: Optional[VisArgs] = None,
    device: Optional[torch.device] = None,
):
    """
    对重建结果进行可视化。

    参数：
        model:    已加载权重的模型，需支持 ``pred = model(lr_hsi, hr_msi)``。
        cfg:      配置。
        vis_args: 可视化参数。
        device:   推理设备。
    """
    if vis_args is None:
        vis_args = VisArgs()

    if device is None:
        device = get_device(cfg.device)

    model.to(device)
    model.eval()

    train_loader, test_loader, info = build_loaders(cfg)

    n_bands = info["n_bands"]
    wavelengths = load_analysis_wavelengths(cfg, info, n_bands)

    loader = test_loader if vis_args.split == "test" else train_loader
    rgb_indices = choose_rgb_indices(n_bands, wavelengths=wavelengths)

    save_root = os.path.join(cfg.output_root, "visualizations", cfg.dataset)
    os.makedirs(save_root, exist_ok=True)

    print("=" * 80)
    print("HSI Reconstruction Visualization")
    print("=" * 80)
    print(f"Dataset: {cfg.dataset}")
    print(f"Split: {vis_args.split}")
    print(f"HSI bands: {n_bands}")
    print(f"MSI bands: {info['n_select_bands']}")
    print(f"RGB indices: {rgb_indices}")
    if wavelengths is not None:
        print("RGB wavelengths:", [float(wavelengths[i]) for i in rgb_indices])
    print(f"Save root: {save_root}")
    print("=" * 80)

    saved = 0
    for idx, batch in enumerate(loader):
        if idx < vis_args.start_index:
            continue
        if saved >= vis_args.num_samples:
            break

        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        pred = model(lr_hsi, hr_msi)
        pred = torch.clamp(pred, 0.0, 1.0)

        lr_up = F.interpolate(
            lr_hsi,
            size=gt.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        lr_up = torch.clamp(lr_up, 0.0, 1.0)

        metrics_pred = calc_metrics(pred=pred, target=gt, scale_ratio=cfg.scale_ratio)
        metrics_lr = calc_metrics(pred=lr_up, target=gt, scale_ratio=cfg.scale_ratio)

        sam_map = calc_sam_map(pred, gt)
        abs_err_map = torch.mean(torch.abs(pred - gt), dim=1)

        gt_np = tensor_to_numpy(gt)
        pred_np = tensor_to_numpy(pred)
        lr_up_np = tensor_to_numpy(lr_up)
        hr_msi_np = tensor_to_numpy(hr_msi)
        sam_np = sam_map[0].detach().cpu().numpy()
        abs_err_np = abs_err_map[0].detach().cpu().numpy()

        sample_name = f"{vis_args.split}_sample{idx}"
        composite_path = os.path.join(save_root, f"{sample_name}_composite.png")
        spectra_path = os.path.join(save_root, f"{sample_name}_spectra.png")
        mat_path = os.path.join(save_root, f"{sample_name}_arrays.mat")
        txt_path = os.path.join(save_root, f"{sample_name}_metrics.txt")

        title = (
            f"{cfg.dataset} {sample_name} | "
            f"PSNR={metrics_pred['PSNR']:.3f}, SAM={metrics_pred['SAM']:.3f}"
        )
        plot_composite(
            save_path=composite_path,
            gt_np=gt_np,
            pred_np=pred_np,
            lr_up_np=lr_up_np,
            rgb_indices=rgb_indices,
            sam_np=sam_np,
            abs_err_np=abs_err_np,
            title=title,
        )

        points = get_point_list(
            pred, gt, sam_map, max_points=vis_args.num_spectrum_points
        )
        plot_spectra(
            save_path=spectra_path,
            gt_np=gt_np,
            pred_np=pred_np,
            lr_up_np=lr_up_np,
            wavelengths=wavelengths,
            points=points,
            rgb_indices=rgb_indices,
        )

        save_mat(
            mat_path,
            {
                "pred": pred_np,
                "gt": gt_np,
                "lr_up": lr_up_np,
                "hr_msi": hr_msi_np,
                "sam_map": sam_np,
                "abs_err_map": abs_err_np,
                "rgb_indices": np.asarray(rgb_indices, dtype=np.int64),
                "wavelengths": np.asarray(
                    [] if wavelengths is None else wavelengths, dtype=np.float32
                ),
            },
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {cfg.dataset}\n")
            f.write(f"Split: {vis_args.split}\n")
            f.write(f"Sample index: {idx}\n")
            f.write(f"Checkpoint: {cfg.resume}\n")
            f.write(f"HSI bands: {n_bands}\n")
            f.write(f"MSI bands: {info['n_select_bands']}\n")
            f.write(f"RGB indices: {rgb_indices}\n")
            if wavelengths is not None:
                f.write(
                    f"RGB wavelengths: "
                    f"{[float(wavelengths[i]) for i in rgb_indices]}\n"
                )
            f.write("\nReconstruction metrics:\n")
            for key, value in metrics_pred.items():
                f.write(f"  {key}: {value:.6f}\n")
            f.write("\nLR-up baseline metrics:\n")
            for key, value in metrics_lr.items():
                f.write(f"  {key}: {value:.6f}\n")
            f.write("\nSpectrum points:\n")
            for y, x, name in points:
                f.write(f"  {name}: y={y}, x={x}\n")

        print(
            f"[{sample_name}] "
            f"Recon PSNR={metrics_pred['PSNR']:.4f}, "
            f"SAM={metrics_pred['SAM']:.4f}"
        )
        print(
            f"[{sample_name}] "
            f"LR-up PSNR={metrics_lr['PSNR']:.4f}, "
            f"SAM={metrics_lr['SAM']:.4f}"
        )
        print(f"  Saved: {composite_path}")
        print(f"  Saved: {spectra_path}")
        print(f"  Saved: {mat_path}")
        print(f"  Saved: {txt_path}")

        saved += 1

    if saved == 0:
        raise RuntimeError(
            "No samples were visualized. Check --start_index and --num_samples."
        )


# ---------------------------------------------------------------------------
# 命令行入口（示例：需要用户提供自己的模型）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize HSI reconstruction results."
    )

    parser.add_argument("--dataset", type=str, default="Chikusei")
    parser.add_argument("--data_root", type=str, default="./data/raw")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    parser.add_argument("--log_root", type=str, default="./logs")
    parser.add_argument("--output_root", type=str, default="./outputs")

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--scale_ratio", type=int, default=4)
    parser.add_argument("--n_select_bands", type=int, default=-1)

    parser.add_argument("--msi_mode", type=str, default="srf",
                        choices=["uniform", "srf"])
    parser.add_argument("--srf_path", type=str,
                        default="./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv")
    parser.add_argument("--wavelength_root", type=str, default="./data/wavelengths")
    parser.add_argument("--wavelength_path", type=str, default="")
    parser.add_argument("--srf_interp", type=str, default="pchip",
                        choices=["pchip", "linear"])
    parser.add_argument("--srf_band_set", type=str, default="wv2_all8",
                        choices=["wv2_visible5", "wv2_visible6", "wv2_all8"])

    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=10)

    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "train"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_spectrum_points", type=int, default=4)

    args = parser.parse_args()

    # ---- 构建 cfg ----
    cfg = TrainConfig()
    cfg.datasets = get_dataset_configs()
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if args.n_select_bands > 0:
        cfg.n_select_bands = args.n_select_bands
    else:
        ds = cfg.datasets.get(cfg.dataset)
        if ds is not None:
            cfg.n_select_bands = ds.n_select_bands
    make_dirs(cfg)

    set_seed(cfg.seed)

    # ==================================================================
    # 用户需要在这里：
    #   1. 导入自己的模型类
    #   2. 实例化模型
    #   3. 加载训练好的权重
    #   4. 调用 visualize_reconstruction(model, cfg, vis_args)
    # ==================================================================
    #
    # 示例：
    #   from your_model import YourModel
    #   from utils import load_checkpoint
    #
    #   _, __, info = build_loaders(cfg)
    #   model = YourModel(
    #       n_bands=info["n_bands"],
    #       n_select_bands=info["n_select_bands"],
    #       scale_ratio=cfg.scale_ratio,
    #   )
    #   load_checkpoint(model, cfg.resume, strict=False)
    #
    #   vis_args = VisArgs(split=args.split, ...)
    #   visualize_reconstruction(model, cfg, vis_args)
    #
    # ==================================================================

    print(
        "\n请在 __main__ 块中提供你自己的模型后重新运行。\n"
        "详见文件末尾的注释示例。"
    )
