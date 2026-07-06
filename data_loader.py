# data_loader.py
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from srf_utils import (
    WV2_VISIBLE_5_BANDS,
    WV2_VISIBLE_6_BANDS,
    WV2_ALL_8_BANDS,
    load_hsi_wavelengths,
    build_srf_weights,
    hsi_to_msi_numpy,
    print_srf_summary,
)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import scipy.io as scio
except ImportError:
    scio = None

try:
    import hdf5storage
except ImportError:
    hdf5storage = None

try:
    import h5py
except ImportError:
    h5py = None


def read_hsi_mat(file_path: str, candidate_keys: List[str]) -> np.ndarray:
    """
    读取.mat格式高光谱数据，返回H×W×C格式。
    优先按candidate_keys读取，读取失败时自动寻找第一个三维数组。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find data file: {file_path}")

    mat_data = None

    if hdf5storage is not None:
        try:
            mat_data = hdf5storage.loadmat(file_path)
        except Exception:
            mat_data = None

    if mat_data is None and scio is not None:
        try:
            mat_data = scio.loadmat(file_path)
        except Exception:
            mat_data = None

    if mat_data is not None:
        for key in candidate_keys:
            if key in mat_data and isinstance(mat_data[key], np.ndarray):
                img = mat_data[key]
                return fix_hsi_shape(img)

        for key, value in mat_data.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray) and value.ndim == 3:
                return fix_hsi_shape(value)

    if h5py is not None:
        with h5py.File(file_path, "r") as f:
            for key in candidate_keys:
                if key in f:
                    img = np.array(f[key])
                    return fix_hsi_shape(img)

            for key in f.keys():
                value = np.array(f[key])
                if value.ndim == 3:
                    return fix_hsi_shape(value)

    raise RuntimeError(f"No valid 3D HSI array found in {file_path}")


def fix_hsi_shape(img: np.ndarray) -> np.ndarray:
    """
    将输入统一为H×W×C。
    部分v7.3 mat文件读出后可能是C×W×H或C×H×W，需要做简单判断。
    """
    img = np.array(img)
    img = np.squeeze(img)

    if img.ndim != 3:
        raise ValueError(f"HSI data must be 3D, but got shape: {img.shape}")

    # 若第一维像波段数，且后两维明显像空间尺寸，则转为H×W×C
    if img.shape[0] <= 256 and img.shape[1] > 256 and img.shape[2] > 256:
        img = np.transpose(img, (1, 2, 0))

    # 若中间维像波段数，则转为H×W×C
    elif img.shape[1] <= 256 and img.shape[0] > 256 and img.shape[2] > 256:
        img = np.transpose(img, (0, 2, 1))

    img = img.astype(np.float32)
    return img


def normalize_hsi(img: np.ndarray) -> np.ndarray:
    """
    归一化到[0,1]。
    """
    img = img.astype(np.float32)
    min_value = float(np.min(img))
    max_value = float(np.max(img))

    if max_value - min_value < 1e-8:
        return np.zeros_like(img, dtype=np.float32)

    img = (img - min_value) / (max_value - min_value)
    return img.astype(np.float32)


def crop_to_scale(img: np.ndarray, scale_ratio: int) -> np.ndarray:
    """
    裁掉不能被scale_ratio整除的边缘，避免下采样和上采样尺寸对不上。
    """
    h, w, c = img.shape
    new_h = h // scale_ratio * scale_ratio
    new_w = w // scale_ratio * scale_ratio
    return img[:new_h, :new_w, :]


def gaussian_blur_bandwise(img: np.ndarray, kernel_size: int = 5, sigma: float = 2.0) -> np.ndarray:
    """
    对每个光谱波段分别做高斯模糊，避免OpenCV对多通道数量的限制。
    """
    if cv2 is None:
        return img

    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[2]):
        blurred[:, :, i] = cv2.GaussianBlur(img[:, :, i], (kernel_size, kernel_size), sigma)
    return blurred


def resize_hsi(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    对H×W×C格式HSI逐波段resize。
    """
    if cv2 is None:
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)

    out = np.zeros((target_h, target_w, img.shape[2]), dtype=np.float32)
    for i in range(img.shape[2]):
        out[:, :, i] = cv2.resize(
            img[:, :, i],
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC,
        )
    return out


def make_lr_hsi(hr_hsi: np.ndarray, scale_ratio: int) -> np.ndarray:
    """
    按照旧data_loader.py的方式：
    HR-HSI -> GaussianBlur -> resize，生成LR-HSI。
    """
    h, w, _ = hr_hsi.shape
    blurred = gaussian_blur_bandwise(hr_hsi, kernel_size=5, sigma=2.0)
    lr_hsi = resize_hsi(blurred, h // scale_ratio, w // scale_ratio)
    return lr_hsi.astype(np.float32)


def make_hr_msi(hr_hsi: np.ndarray, n_select_bands: int) -> np.ndarray:
    """
    没有光谱响应函数时，按照旧data_loader.py方式从HR-HSI均匀抽取波段生成HR-MSI。
    """
    n_bands = hr_hsi.shape[2]

    if n_select_bands > n_bands:
        raise ValueError(
            f"n_select_bands={n_select_bands} is larger than HSI bands={n_bands}"
        )

    band_indices = np.linspace(0, n_bands - 1, n_select_bands).round().astype(np.int64)
    hr_msi = hr_hsi[:, :, band_indices]
    return hr_msi.astype(np.float32)


def hsi_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    H×W×C -> C×H×W
    """
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()


def get_center_test_rect(h: int, w: int, test_size: int) -> Tuple[int, int, int, int]:
    top = max((h - test_size) // 2, 0)
    left = max((w - test_size) // 2, 0)
    bottom = min(top + test_size, h)
    right = min(left + test_size, w)
    return top, left, bottom, right


def intersects(rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> bool:
    t1, l1, b1, r1 = rect1
    t2, l2, b2, r2 = rect2
    return not (r1 <= l2 or r2 <= l1 or b1 <= t2 or b2 <= t1)


def build_patch_coords(
    h: int,
    w: int,
    patch_size: int,
    stride: int,
    test_rect: Tuple[int, int, int, int],
    split: str,
) -> List[Tuple[int, int]]:
    coords = []

    if split == "test":
        top, left, bottom, right = test_rect
        if bottom - top < patch_size or right - left < patch_size:
            top = max((h - patch_size) // 2, 0)
            left = max((w - patch_size) // 2, 0)
        return [(top, left)]

    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            patch_rect = (top, left, top + patch_size, left + patch_size)
            if not intersects(patch_rect, test_rect):
                coords.append((top, left))

    if len(coords) == 0:
        for top in range(0, h - patch_size + 1, stride):
            for left in range(0, w - patch_size + 1, stride):
                coords.append((top, left))

    return coords


class HSIHSRDataset(Dataset):
    """
    HSI-MSI融合超分数据集。
    返回：
        lr_hsi:  C×h×w
        hr_msi:  c×H×W
        gt:      C×H×W
    """

    def __init__(
        self,
        img: np.ndarray,
        dataset_name: str,
        patch_size: int,
        stride: int,
        scale_ratio: int,
        n_select_bands: int,
        split: str = "train",
        test_size: int = 128,
        augment: bool = True,
        srf_weights=None,
    ):
        super().__init__()

        self.img = img
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.stride = stride
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.split = split
        self.augment = augment and split == "train"
        self.srf_weights = srf_weights

        h, w, _ = img.shape
        self.test_rect = get_center_test_rect(h, w, test_size)
        self.coords = build_patch_coords(
            h=h,
            w=w,
            patch_size=patch_size,
            stride=stride,
            test_rect=self.test_rect,
            split=split,
        )

    def __len__(self):
        return len(self.coords)

    def random_augment(self, patch: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            patch = np.flip(patch, axis=0)
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1)
        if random.random() < 0.5:
            patch = np.rot90(patch, k=random.randint(1, 3), axes=(0, 1))
        return np.ascontiguousarray(patch)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        top, left = self.coords[index]
        gt = self.img[
            top:top + self.patch_size,
            left:left + self.patch_size,
            :,
        ].copy()

        if self.augment:
            gt = self.random_augment(gt)

        lr_hsi = make_lr_hsi(gt, self.scale_ratio)
        if self.srf_weights is not None:
            hr_msi = hsi_to_msi_numpy(gt, self.srf_weights)
        else:
            hr_msi = make_hr_msi(gt, self.n_select_bands)

        sample = {
            "lr_hsi": hsi_to_tensor(lr_hsi),
            "hr_msi": hsi_to_tensor(hr_msi),
            "gt": hsi_to_tensor(gt),
            "dataset_id": torch.tensor(0, dtype=torch.long),
            "n_bands": torch.tensor(gt.shape[2], dtype=torch.long),
        }
        return sample


def build_datasets(cfg):
    dataset_cfg = cfg.datasets[cfg.dataset]
    file_path = os.path.join(cfg.data_root, dataset_cfg.file_name)

    img = read_hsi_mat(file_path, dataset_cfg.mat_keys)
    img = normalize_hsi(img)
    img = crop_to_scale(img, cfg.scale_ratio)

    n_bands = img.shape[2]
    print(f"Loaded {cfg.dataset}: shape={img.shape}, bands={n_bands}")

    srf_weights = None
    srf_band_names = None
    hsi_wavelengths = None

    if getattr(cfg, "msi_mode", "uniform") == "srf":
        if cfg.wavelength_path:
            wavelength_path = cfg.wavelength_path
        else:
            wavelength_path = os.path.join(cfg.wavelength_root, f"{cfg.dataset}.txt")

        hsi_wavelengths = load_hsi_wavelengths(
            wavelength_path=wavelength_path,
            n_bands=n_bands,
        )

        if cfg.srf_band_set == "wv2_visible5":
            selected_bands = WV2_VISIBLE_5_BANDS
        elif cfg.srf_band_set == "wv2_visible6":
            selected_bands = WV2_VISIBLE_6_BANDS
        elif cfg.srf_band_set == "wv2_all8":
            selected_bands = WV2_ALL_8_BANDS
        else:
            raise ValueError(f"Unsupported srf_band_set: {cfg.srf_band_set}")

        srf_weights, srf_band_names = build_srf_weights(
            srf_path=cfg.srf_path,
            hsi_wavelengths=hsi_wavelengths,
            selected_bands=selected_bands,
            interp_kind=cfg.srf_interp,
            normalize=True,
        )

        print_srf_summary(
            srf_weights=srf_weights,
            band_names=srf_band_names,
            hsi_wavelengths=hsi_wavelengths,
        )

        n_select_bands = srf_weights.shape[0]

    else:
        n_select_bands = cfg.n_select_bands

    train_set = HSIHSRDataset(
        img=img,
        dataset_name=cfg.dataset,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        scale_ratio=cfg.scale_ratio,
        n_select_bands=n_select_bands,
        srf_weights=srf_weights,
        split="train",
        test_size=cfg.image_size,
        augment=True,
    )

    test_set = HSIHSRDataset(
        img=img,
        dataset_name=cfg.dataset,
        patch_size=cfg.image_size,
        stride=cfg.image_size,
        scale_ratio=cfg.scale_ratio,
        n_select_bands=n_select_bands,
        srf_weights=srf_weights,
        split="test",
        test_size=cfg.image_size,
        augment=False,
    )

    info = {
        "dataset": cfg.dataset,
        "n_bands": n_bands,
        "n_select_bands": n_select_bands,
        "scale_ratio": cfg.scale_ratio,
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "msi_mode": getattr(cfg, "msi_mode", "uniform"),
        "srf_weights": srf_weights,
        "srf_band_names": srf_band_names,
        "hsi_wavelengths": hsi_wavelengths,
    }

    return train_set, test_set, info


def build_loaders(cfg):
    train_set, test_set, info = build_datasets(cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader, info