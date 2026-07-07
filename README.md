# RAPD-Net：HSI–MSI Fusion / Hyperspectral Super-Resolution

本仓库提供 RAPD-Net 的数据读取、SRF 构建、训练辅助函数和分阶段模型实现。

当前主线为：

```text
LR-HSI 场景自适应光谱基提取
        ↓
频率可靠性感知的光谱系数残差注入
        ↓
双域不确定性感知的基内—正交补残差扩散精修
        ↓
物理闭环与不确定度校准联合微调
```

## 主要文件

| 文件 | 说明 |
|------|------|
| `data_loader.py` | HSI 数据读取、归一化、训练/测试区域划分、LR-HSI 与 HR-MSI 构建 |
| `srf_utils.py` | WorldView-2 SRF 加载、插值和 HSI→MSI 投影 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC |
| `losses.py` | SAM 等基础损失 |
| `models/stage1_spectral_basis.py` | 新阶段一：仿射正交光谱基提取模型 |
| `train_stage1_basis.py` | 新阶段一完整训练、验证、检查点和导出 |
| `inspect_stage1_basis.py` | 新阶段一正交性、残差泄漏、系数使用和可视化检查 |
| `inspect_stage1_basis_hr_ceiling.py` | LR 自投影、HR 基空间 Oracle 和系数上采样基线 |
| `models/stage2_frequency_reliability.py` | SFSR 风格 Shared Encoder + SSP + NSP 频率可靠性模块 |
| `models/stage2_coefficient_residual.py` | 新阶段二：可靠 MSI 引导的有符号光谱系数残差注入 |
| `train_stage2_coefficients.py` | 新阶段二训练、Oracle 对照、Zero-MSI 与可靠性监控 |
| `models/stage1_unmixing.py` | 旧端元—丰度阶段一，仅保留用于历史实验复现 |
| `train_stage1_unmix.py` | 旧阶段一训练脚本，仅保留用于历史实验复现 |
| `models/stage2_physical_fusion.py` | 旧丰度/非负锥阶段二，仅保留用于历史实验复现 |

# 阶段一：场景自适应光谱基提取

阶段一只使用 LR-HSI，不读取 HR-MSI，也不使用 HR-HSI 监督。

模型采用仿射正交子空间：

\[
C_{\mathrm{lr}}=U_r^\top(Y_{\mathrm{lr}}-\mu),
\]

\[
\widehat Y_{\mathrm{lr}}=\mu+U_rC_{\mathrm{lr}},
\]

其中：

- \(\mu\) 为训练 LR-HSI 估计得到的场景均值光谱；
- \(U_r\in\mathbb R^{B\times r}\) 为场景自适应正交光谱基；
- \(U_r^\top U_r=I\) 由前向 QR 正交化严格保证；
- \(C_{\mathrm{lr}}\) 为允许正负值的低分辨率光谱系数；
- 光谱基是数学坐标，不应按真实端元曲线解释。

## PCA 初始化

训练开始前，从训练 LR-HSI 中采样光谱，计算场景均值、协方差矩阵和前 \(r\) 个 PCA 方向。PCA 初始化检查点会在优化前单独保存，避免后续 L1/SAM 微调反而损害最优子空间。

## 阶段一训练

Smoke test：

```bash
python train_stage1_basis.py \
  --dataset PaviaU \
  --basis_rank 32 \
  --basis_init_pixels 100000 \
  --epochs 2 \
  --batch_size 4 \
  --lr 5e-5
```

正式训练：

```bash
python train_stage1_basis.py \
  --dataset PaviaU \
  --basis_rank 32 \
  --basis_init_pixels 100000 \
  --epochs 300 \
  --batch_size 4 \
  --lr 5e-5
```

默认训练目标：

\[
L_{\mathrm{basis}}
=
L_1
+0.5L_{\mathrm{SAM}}
+0.1L_{\nabla\lambda}
+0.05L_{\nabla^2\lambda}
+0.001L_{\mathrm{projector-anchor}}.
\]

投影锚定约束比较 \(U_rU_r^\top\)，而不是逐列约束基向量，避免受到基旋转和符号不唯一性的影响。

## 阶段一输出

```text
checkpoints/stage1_basis/<dataset>/
├── basis_pca_init.pth
├── basis_best.pth
├── basis_best_sam.pth
├── basis_best_psnr.pth
├── basis_last.pth
└── basis_for_stage2.pth
```

后续阶段固定优先使用：

```text
checkpoints/stage1_basis/<dataset>/basis_for_stage2.pth
```

导出文件：

```text
outputs/stage1_basis/<dataset>/
├── spectral_basis.npy
├── mean_spectrum.npy
├── coefficient_scale.npy
├── coefficient_mean.npy
├── basis_projector.npy
├── pca_eigenvalues.npy
├── stage1_basis_test_outputs.npz
├── basis_statistics.json
└── final_metrics.json
```

不得对导出的基向量重新排序，也不能仅根据曲线外观删除某一基维度。

## 阶段一检查

```bash
python inspect_stage1_basis.py \
  --dataset PaviaU \
  --compare_all
```

HR 表达上限与阶段二基础结果：

```bash
python inspect_stage1_basis_hr_ceiling.py \
  --dataset PaviaU \
  --checkpoint checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth
```

该脚本分别报告：

- LR self-projection：仅表示 LR-HSI 子空间投影精度；
- HR basis oracle：当前光谱基对 HR-HSI 的表示上限；
- LR-coefficient upsampling base：不使用 MSI 的阶段二起点。

# 阶段二：频率可靠性感知的光谱系数残差注入

阶段二冻结完整阶段一，只允许 MSI 通过 SSP/NSP 频率可靠性链路修正高分辨率光谱系数。

\[
C_{\mathrm{up}}
=
\operatorname{Bicubic}(C_{\mathrm{lr}}),
\]

\[
X_{\mathrm{base}}
=
\mu+U_rC_{\mathrm{up}},
\]

\[
\Delta C_{\mathrm{rel}}
=
G_C(C_{\mathrm{up}},F_{\mathrm{phy}},F_{\mathrm{LF-diff}},F^{MF},F_{\mathrm{HF}}^{\mathrm{rel}},Q),
\]

\[
X_2
=
\mu+U_r(C_{\mathrm{up}}+\Delta C_{\mathrm{rel}}).
\]

其中 \(\Delta C_{\mathrm{rel}}\) 是允许正负值的光谱系数残差，不再受丰度非负和和为 1 约束。

## 可靠性路径

阶段二保留完整 SFSR 风格结构：

```text
shared MSI encoder
      ↓
20-band channel-wise SSP
      ↓
LF / MF / HF
      ↓
LM Sobel evidence + adaptive NSP
      ↓
MF + reliable HF + reliability map Q
```

原始 HR-MSI 或未经筛选的高频残差没有独立旁路进入系数预测器，避免网络绕过可靠性模块。

## 系数尺度

不同正交坐标的方差差异很大，因此网络预测标准化残差：

\[
\Delta \widetilde C_k
=
\frac{\Delta C_k}{s_k},
\]

其中 \(s_k\) 来自阶段一部署检查点的 `coefficient_scale`。默认采用：

\[
\Delta \widetilde C
=
6\tanh(\widehat{\Delta \widetilde C}),
\]

即默认允许每个系数坐标在约 \(\pm6\) 个标准差范围内修正，并记录饱和比例。

## 阶段二训练

Smoke test：

```bash
python train_stage2_coefficients.py \
  --dataset PaviaU \
  --stage1_basis_checkpoint checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth \
  --epochs 2 \
  --batch_size 2 \
  --lr 1e-4 \
  --msi_mode srf \
  --srf_band_set wv2_visible6
```

正式训练：

```bash
python train_stage2_coefficients.py \
  --dataset PaviaU \
  --stage1_basis_checkpoint checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth \
  --epochs 300 \
  --batch_size 4 \
  --lr 1e-4 \
  --msi_mode srf \
  --srf_band_set wv2_visible6
```

## 阶段二监督

训练目标包含：

- HR-HSI L1、SAM、一阶和二阶光谱形态；
- 标准化光谱系数残差监督；
- HR 系数重建监督；
- LR-HSI 退化闭环；
- LR 系数退化闭环；
- SRF/MSI 闭环；
- 系数残差幅度与空间 TV；
- SSP 低频对齐和 NSP 噪声最小化；
- 相对 Base 改进约束；
- Zero-MSI 使用约束。

## 阶段二监控

每轮验证同时报告：

- `base_psnr / base_sam`：系数 bicubic 上采样基线；
- `stage2_psnr / stage2_sam`：完整阶段二；
- `oracle_psnr / oracle_sam`：当前光谱基 HR Oracle；
- `remaining_psnr_to_oracle`：预测器距离基空间上限的差距；
- `recoverable_error_fraction`：阶段二利用了多少可恢复误差空间；
- `zero_msi_psnr_drop`：MSI 分支真实贡献；
- `noise_ratio`：NSP 筛除比例；
- `freq_low / freq_mid / freq_high`：三频带能量比例；
- `residual_saturation_ratio`：系数残差是否撞到 \(\pm6\sigma\) 上限；
- `tau_low / tau_high`：SSP 学习后的频率边界。

阶段二输出：

```text
checkpoints/stage2_coefficients/<dataset>/
├── coefficient_best.pth
├── coefficient_best_sam.pth
├── coefficient_best_psnr.pth
└── coefficient_last.pth
```

```text
outputs/stage2_coefficients/<dataset>/
├── stage2_coefficient_outputs.npz
└── final_metrics.json
```

# 旧阶段兼容性

新的 `basis_for_stage2.pth` 与旧的端元—丰度检查点结构完全不同。以下旧脚本不能加载新的阶段一检查点：

```text
train_stage2_physical.py
models/stage2_physical_fusion.py
```

它们仅用于历史消融，不属于当前主线。

# 数据目录

```text
project/
├── data/
│   ├── raw/
│   ├── wavelengths/
│   ├── srf/
│   └── srf_weights/
├── checkpoints/
├── logs/
├── outputs/
└── models/
```
