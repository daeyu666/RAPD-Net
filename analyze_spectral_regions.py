"""
按光谱区域分析模型重建质量。

使用方式：
    1. 构建并加载你自己训练的模型；
    2. 调用 analyze_regions(model, cfg) 即可。

不再依赖具体的模型类（如 ContentAdaptiveUnfoldNet），
用户可以传入任意满足接口 ``pred = model(lr_hsi, hr_msi)`` 的模型。
"""

import csv
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import TrainConfig, parse_args, print_config
from data_loader import build_loaders
from srf_utils import load_hsi_wavelengths
from utils import get_device, move_to_device, set_seed
from metrics import calc_metrics


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def load_analysis_wavelengths(
    cfg: TrainConfig,
    info: dict,
    n_bands: int,
) -> np.ndarray:
    """加载波长信息：优先使用 data_loader 中已缓存的波长。"""
    wavelengths = info.get("hsi_wavelengths", None)
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
        if wavelengths.size == n_bands:
            return wavelengths

    if cfg.wavelength_path:
        wavelength_path = cfg.wavelength_path
    else:
        wavelength_path = os.path.join(cfg.wavelength_root, f"{cfg.dataset}.txt")

    return load_hsi_wavelengths(wavelength_path=wavelength_path, n_bands=n_bands)


def make_region_specs(
    wavelengths: np.ndarray,
    min_bands: int = 2,
) -> List[Tuple[str, float, float, int]]:
    """根据波长范围生成分析区间。"""
    wl_min = float(np.min(wavelengths))
    wl_max = float(np.max(wavelengths))

    candidate_regions = [
        ("visible_430_700", 430.0, 700.0),
        ("rededge_700_760", 700.0, 760.0),
        ("nir1_760_900", 760.0, 900.0),
        ("nir2_900_max", 900.0, wl_max),
        ("nir_all_700_max", 700.0, wl_max),
        ("all", wl_min, wl_max),
    ]

    valid_regions = []
    for name, start, end in candidate_regions:
        start = max(start, wl_min)
        end = min(end, wl_max)
        if end <= start:
            continue
        mask = (wavelengths >= start) & (wavelengths <= end)
        band_count = int(mask.sum())
        if band_count >= min_bands:
            valid_regions.append((name, start, end, band_count))

    return valid_regions


def select_band_region(
    x: torch.Tensor,
    wavelengths: np.ndarray,
    wl_min: float,
    wl_max: float,
):
    """按波长范围选取波段。"""
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    indices_np = np.where(mask)[0]
    if indices_np.size == 0:
        raise ValueError(
            f"No bands found in wavelength range {wl_min:.2f}-{wl_max:.2f} nm"
        )
    indices = torch.from_numpy(indices_np).long().to(x.device)
    return x.index_select(1, indices), indices


def calc_gt_complexity(region_gt: torch.Tensor) -> Dict[str, float]:
    """计算 GT 光谱曲线复杂度（与模型无关）。"""
    region_gt = region_gt.detach().float()

    if region_gt.size(1) < 2:
        return {
            "spectral_tv": 0.0,
            "spectral_curvature": 0.0,
            "spectral_range": float(
                (region_gt.amax(dim=1) - region_gt.amin(dim=1)).mean().item()
            ),
        }

    first_diff = region_gt[:, 1:, :, :] - region_gt[:, :-1, :, :]
    spectral_tv = torch.mean(torch.abs(first_diff)).item()

    if region_gt.size(1) >= 3:
        second_diff = (
            region_gt[:, 2:, :, :]
            - 2.0 * region_gt[:, 1:-1, :, :]
            + region_gt[:, :-2, :, :]
        )
        spectral_curvature = torch.mean(torch.abs(second_diff)).item()
    else:
        spectral_curvature = 0.0

    spectral_range = torch.mean(
        region_gt.amax(dim=1) - region_gt.amin(dim=1)
    ).item()

    return {
        "spectral_tv": float(spectral_tv),
        "spectral_curvature": float(spectral_curvature),
        "spectral_range": float(spectral_range),
    }


def average_dicts(dict_list: List[Dict]) -> Dict:
    keys = dict_list[0].keys()
    return {
        key: float(sum(d[key] for d in dict_list) / len(dict_list))
        for key in keys
    }


# ---------------------------------------------------------------------------
# 主分析函数（模型通过参数传入，不再硬编码具体类）
# ---------------------------------------------------------------------------
@torch.no_grad()
def analyze_regions(
    model: nn.Module,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
):
    """
    对指定模型按光谱区域分析重建质量。

    参数：
        model: 已加载权重的模型，需支持 ``pred = model(lr_hsi, hr_msi)``。
        cfg:   配置。
        device: 推理设备，默认按 cfg.device 自动选择。
    """
    if device is None:
        device = get_device(cfg.device)

    model.to(device)
    model.eval()

    _, test_loader, info = build_loaders(cfg)

    n_bands = info["n_bands"]
    wavelengths = load_analysis_wavelengths(cfg, info, n_bands)

    regions = make_region_specs(wavelengths)
    region_results = {name: [] for name, _, _, _ in regions}
    complexity_results = {name: [] for name, _, _, _ in regions}

    for batch in test_loader:
        batch = move_to_device(batch, device)

        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        pred = model(lr_hsi, hr_msi)
        pred = torch.clamp(pred, 0.0, 1.0)

        for name, wl_min, wl_max, _ in regions:
            pred_region, indices = select_band_region(
                pred, wavelengths, wl_min, wl_max
            )
            gt_region = gt.index_select(1, indices)

            metrics = calc_metrics(
                pred=pred_region,
                target=gt_region,
                scale_ratio=cfg.scale_ratio,
            )
            complexity = calc_gt_complexity(gt_region)

            region_results[name].append(metrics)
            complexity_results[name].append(complexity)

    # ---- 打印 & 保存 ----
    print("=" * 80)
    print("Spectral Region Analysis")
    print("=" * 80)
    print(f"Dataset: {cfg.dataset}")
    print(
        f"Wavelength range: {float(wavelengths.min()):.2f}-"
        f"{float(wavelengths.max()):.2f} nm"
    )
    print(f"HSI bands: {n_bands}")
    print(f"MSI bands: {info['n_select_bands']}")
    print("-" * 80)

    output_rows = []
    for name, wl_min, wl_max, band_count in regions:
        avg = average_dicts(region_results[name])
        avg_complexity = average_dicts(complexity_results[name])

        row = {
            "dataset": cfg.dataset,
            "region": name,
            "wl_min": wl_min,
            "wl_max": wl_max,
            "band_count": band_count,
            **avg,
            **avg_complexity,
        }
        output_rows.append(row)

        print(f"\nRegion: {name}")
        print(f"  Range : {wl_min:.2f}-{wl_max:.2f} nm | bands={band_count}")
        print(f"  PSNR  : {avg['PSNR']:.4f}")
        print(f"  RMSE  : {avg['RMSE']:.6f}")
        print(f"  SAM   : {avg['SAM']:.4f}")
        print(f"  ERGAS : {avg['ERGAS']:.4f}")
        print(f"  SSIM  : {avg['SSIM']:.4f}")
        print(f"  CC    : {avg['CC']:.4f}")
        print(f"  GT_TV : {avg_complexity['spectral_tv']:.6f}")
        print(f"  GT_CUR: {avg_complexity['spectral_curvature']:.6f}")
        print(f"  GT_RNG: {avg_complexity['spectral_range']:.6f}")

    save_dir = os.path.join(cfg.output_root, "metrics")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{cfg.dataset}_spectral_regions.csv")

    fieldnames = [
        "dataset", "region", "wl_min", "wl_max", "band_count",
        "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC",
        "spectral_tv", "spectral_curvature", "spectral_range",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print("=" * 80)
    print(f"Saved region analysis to: {save_path}")
    print("=" * 80)

    return output_rows


# ---------------------------------------------------------------------------
# 命令行入口（示例：需要用户提供自己的模型）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = parse_args()
    print_config(cfg)
    set_seed(cfg.seed)

    # ==================================================================
    # 用户需要在这里：
    #   1. 导入自己的模型类
    #   2. 实例化模型
    #   3. 加载训练好的权重
    #   4. 调用 analyze_regions(model, cfg)
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
    #   analyze_regions(model, cfg)
    #
    # ==================================================================

    print(
        "\n请在 __main__ 块中提供你自己的模型后重新运行。\n"
        "详见文件末尾的注释示例。"
    )
