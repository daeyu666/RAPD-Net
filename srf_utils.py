# srf_utils.py
import os
import numpy as np
import pandas as pd


# 为了保证不同HSI数据集生成的MSI通道一致，第一版只使用WV2前5个可见光波段
WV2_VISIBLE_5_BANDS = [
    "WV2 Coastal Blue",
    "WV2 Blue",
    "WV2 Green",
    "WV2 Yellow",
    "WV2 Red",
]

WV2_VISIBLE_6_BANDS = [
    "WV2 Coastal Blue",
    "WV2 Blue",
    "WV2 Green",
    "WV2 Yellow",
    "WV2 Red",
    "WV2 RedEdge",
]

WV2_ALL_8_BANDS = [
    "WV2 Coastal Blue",
    "WV2 Blue",
    "WV2 Green",
    "WV2 Yellow",
    "WV2 Red",
    "WV2 RedEdge",
    "WV2 NIR1",
    "WV2 NIR2",
]
def load_hsi_wavelengths(wavelength_path: str, n_bands: int) -> np.ndarray:
    """
    读取 HSI 中心波长。

    支持：
        .txt：每行一个波长
        .csv：第一列或列名包含 wavelength / wave / wl / lambda / center
        .npy：一维数组

    单位：
        如果最大值小于10，认为单位是微米，自动转为nm；
        否则认为单位是nm。
    """
    if not os.path.exists(wavelength_path):
        raise FileNotFoundError(f"Cannot find wavelength file: {wavelength_path}")

    ext = os.path.splitext(wavelength_path)[1].lower()

    if ext == ".npy":
        wavelengths = np.load(wavelength_path).astype(np.float32)

    elif ext in [".txt", ".dat"]:
        wavelengths = np.loadtxt(wavelength_path).astype(np.float32)

    elif ext == ".csv":
        df = pd.read_csv(wavelength_path)

        lower_cols = [c.lower() for c in df.columns]
        selected_col = None

        for key in ["wavelength", "wave", "wl", "lambda", "center"]:
            for col, lower_col in zip(df.columns, lower_cols):
                if key in lower_col:
                    selected_col = col
                    break
            if selected_col is not None:
                break

        if selected_col is None:
            selected_col = df.columns[0]

        wavelengths = df[selected_col].values.astype(np.float32)

    else:
        raise ValueError(f"Unsupported wavelength file type: {ext}")

    wavelengths = np.asarray(wavelengths).reshape(-1).astype(np.float32)

    if wavelengths.size != n_bands:
        raise ValueError(
            f"Wavelength number mismatch: got {wavelengths.size}, "
            f"but HSI has {n_bands} bands."
        )

    if np.nanmax(wavelengths) < 10:
        wavelengths = wavelengths * 1000.0

    return wavelengths.astype(np.float32)


def estimate_band_widths(wavelengths: np.ndarray) -> np.ndarray:
    """
    根据 HSI 中心波长估计每个波段的积分宽度。
    用于将 SRF 响应值转换成离散积分权重。
    """
    wavelengths = np.asarray(wavelengths).astype(np.float32).reshape(-1)

    if wavelengths.size == 1:
        return np.ones_like(wavelengths, dtype=np.float32)

    edges = np.zeros(wavelengths.size + 1, dtype=np.float32)

    edges[1:-1] = 0.5 * (wavelengths[:-1] + wavelengths[1:])
    edges[0] = wavelengths[0] - 0.5 * (wavelengths[1] - wavelengths[0])
    edges[-1] = wavelengths[-1] + 0.5 * (wavelengths[-1] - wavelengths[-2])

    widths = edges[1:] - edges[:-1]
    widths = np.maximum(widths, 1e-6)

    return widths.astype(np.float32)


def interp_srf_to_hsi_wavelengths(
    srf_wavelengths: np.ndarray,
    response_values: np.ndarray,
    hsi_wavelengths: np.ndarray,
    interp_kind: str = "pchip",
) -> np.ndarray:
    """
    将 SRF 曲线重采样到 HSI 的所有中心波长点。

    默认使用 PCHIP 插值，比高阶多项式拟合更稳定，不容易出现振荡。
    如果 scipy 不可用，自动退化为线性插值。
    """
    srf_wavelengths = np.asarray(srf_wavelengths).astype(np.float32)
    response_values = np.asarray(response_values).astype(np.float32)
    hsi_wavelengths = np.asarray(hsi_wavelengths).astype(np.float32)

    order = np.argsort(srf_wavelengths)
    srf_wavelengths = srf_wavelengths[order]
    response_values = response_values[order]

    if interp_kind == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator

            curve = PchipInterpolator(
                srf_wavelengths,
                response_values,
                extrapolate=False,
            )

            sampled_response = curve(hsi_wavelengths)

        except Exception:
            sampled_response = np.interp(
                hsi_wavelengths,
                srf_wavelengths,
                response_values,
                left=0.0,
                right=0.0,
            )

    elif interp_kind == "linear":
        sampled_response = np.interp(
            hsi_wavelengths,
            srf_wavelengths,
            response_values,
            left=0.0,
            right=0.0,
        )

    else:
        raise ValueError(f"Unsupported interp_kind: {interp_kind}")

    sampled_response = np.nan_to_num(
        sampled_response,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sampled_response = np.maximum(sampled_response, 0.0)

    return sampled_response.astype(np.float32)


def build_srf_weights(
    srf_path: str,
    hsi_wavelengths: np.ndarray,
    selected_bands=None,
    interp_kind: str = "pchip",
    normalize: bool = True,
    eps: float = 1e-12,
):
    """
    根据 WorldView-2 SRF 和 HSI 中心波长生成 MSI 权重矩阵。

    默认只使用5个可见光波段：
        WV2 Coastal Blue
        WV2 Blue
        WV2 Green
        WV2 Yellow
        WV2 Red

    输入：
        srf_path: WorldView-2 SRF csv 文件路径
        hsi_wavelengths: HSI 所有中心波长，单位 nm
        selected_bands: 需要使用的 WV2 波段名称
        interp_kind: pchip 或 linear
        normalize: 是否对每个 MSI 波段的权重归一化

    输出：
        weights: M×C
            M 为 MSI 波段数，默认5；
            C 为 HSI 波段数；
            每一行对应一个 MSI 波段的光谱响应权重。
        band_names: list[str]
            MSI 波段名称。

    生成公式：
        response_c = SRF_m(lambda_c)
        raw_weight_c = response_c * delta_lambda_c
        weight_c = raw_weight_c / sum(raw_weight_c)

    因为 HSI 已经归一化到 [0,1]，且每个 MSI 波段权重和为 1，
    所以生成的 MSI 仍然保持在合理的 [0,1] 范围附近。
    """
    if not os.path.exists(srf_path):
        raise FileNotFoundError(f"Cannot find SRF file: {srf_path}")

    df = pd.read_csv(srf_path)

    if "WL(nm)" not in df.columns:
        raise ValueError("SRF file must contain column: WL(nm)")

    if selected_bands is None:
        selected_bands = WV2_VISIBLE_5_BANDS

    srf_wavelengths = df["WL(nm)"].values.astype(np.float32)
    hsi_wavelengths = np.asarray(hsi_wavelengths).astype(np.float32).reshape(-1)
    hsi_widths = estimate_band_widths(hsi_wavelengths)

    all_weights = []
    band_names = []

    for band in selected_bands:
        if band not in df.columns:
            raise ValueError(f"SRF file does not contain band column: {band}")

        response_values = df[band].values.astype(np.float32)

        sampled_response = interp_srf_to_hsi_wavelengths(
            srf_wavelengths=srf_wavelengths,
            response_values=response_values,
            hsi_wavelengths=hsi_wavelengths,
            interp_kind=interp_kind,
        )

        raw_weight = sampled_response * hsi_widths
        weight_sum = float(np.sum(raw_weight))

        if weight_sum < eps:
            raise ValueError(
                f"SRF band {band} has no valid overlap with HSI wavelengths. "
                f"HSI wavelength range: {hsi_wavelengths.min():.2f}-"
                f"{hsi_wavelengths.max():.2f} nm."
            )

        if normalize:
            weight = raw_weight / (weight_sum + eps)
        else:
            weight = raw_weight

        weight = weight.astype(np.float32)

        all_weights.append(weight)
        band_names.append(band)

    weights = np.stack(all_weights, axis=0).astype(np.float32)

    return weights, band_names


def hsi_to_msi_numpy(
    hsi: np.ndarray,
    srf_weights: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    """
    使用 SRF 权重将 HSI 转成 MSI。

    输入：
        hsi: H×W×C
        srf_weights: M×C

    输出：
        msi: H×W×M
    """
    if hsi.ndim != 3:
        raise ValueError(f"HSI must be H×W×C, but got shape: {hsi.shape}")

    if srf_weights.ndim != 2:
        raise ValueError(f"srf_weights must be M×C, but got shape: {srf_weights.shape}")

    h, w, c = hsi.shape
    m, c2 = srf_weights.shape

    if c != c2:
        raise ValueError(
            f"Band mismatch: HSI has {c} bands, but SRF weights expect {c2} bands."
        )

    msi = np.tensordot(hsi, srf_weights.T, axes=([2], [0]))
    msi = np.asarray(msi, dtype=np.float32)

    if clip:
        msi = np.clip(msi, 0.0, 1.0)

    return msi


def print_srf_summary(
    srf_weights: np.ndarray,
    band_names,
    hsi_wavelengths: np.ndarray,
):
    """
    打印每个 MSI 波段在 HSI 波长上的重采样和归一化情况。
    """
    print("=" * 80)
    print("SRF weight summary")
    print("=" * 80)
    print(
        f"HSI wavelength range: "
        f"{float(np.min(hsi_wavelengths)):.2f} - "
        f"{float(np.max(hsi_wavelengths)):.2f} nm"
    )

    for i, band in enumerate(band_names):
        weight = srf_weights[i]

        peak_idx = int(np.argmax(weight))
        peak_wl = float(hsi_wavelengths[peak_idx])

        nonzero = weight > weight.max() * 0.01

        if np.any(nonzero):
            wl_min = float(hsi_wavelengths[nonzero].min())
            wl_max = float(hsi_wavelengths[nonzero].max())
        else:
            wl_min = peak_wl
            wl_max = peak_wl

        print(
            f"{band}: "
            f"peak={peak_wl:.2f} nm, "
            f"main_range={wl_min:.2f}-{wl_max:.2f} nm, "
            f"weight_sum={float(weight.sum()):.6f}, "
            f"max_weight={float(weight.max()):.6f}"
        )

    print("=" * 80)