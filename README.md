# RAPD-Net：HSI–MSI Fusion / Hyperspectral Super-Resolution

本仓库提供 RAPD-Net 的数据读取、SRF 构建、训练辅助函数和分阶段模型实现。

当前主线已由“端元—丰度”改为：

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
| `models/stage2_frequency_reliability.py` | SFSR 风格 Shared Encoder + SSP + NSP 频率可靠性模块 |
| `models/stage1_unmixing.py` | 旧端元—丰度阶段一，仅保留用于历史实验复现 |
| `train_stage1_unmix.py` | 旧阶段一训练脚本，仅保留用于历史实验复现 |
| `models/stage2_physical_fusion.py` | 旧丰度/非负锥阶段二，等待新系数框架替换，不作为当前主线使用 |

# 新阶段一：场景自适应光谱基提取

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

## 训练命令

建议先运行两轮 smoke test：

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

默认训练目标包括：

\[
L_{\mathrm{basis}}
=
L_1
+0.5L_{\mathrm{SAM}}
+0.1L_{\nabla\lambda}
+0.05L_{\nabla^2\lambda}
+0.001L_{\mathrm{projector-anchor}}.
\]

投影锚定约束比较的是：

\[
U_rU_r^\top
\]

而不是逐列约束基向量，避免受到基旋转和符号不唯一性的影响。

## 阶段一输出

检查点：

```text
checkpoints/stage1_basis/<dataset>/
├── basis_pca_init.pth
├── basis_best.pth
├── basis_best_sam.pth
├── basis_best_psnr.pth
├── basis_last.pth
└── basis_for_stage2.pth
```

其中：

- `basis_pca_init.pth`：未经梯度优化的 PCA 初始化；
- `basis_best.pth`：按 L1、SAM 和光谱形态综合分数选择；
- `basis_best_sam.pth`：验证 SAM 最低；
- `basis_best_psnr.pth`：验证 PSNR 最高；
- `basis_for_stage2.pth`：重新统计最终系数尺度后的阶段二部署检查点，后续阶段优先使用。

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

阶段二必须使用同一个检查点中的：

\[
\mu,\quad U_r,\quad C_{\mathrm{lr}}.
\]

不得对导出的基向量重新排序，也不能仅根据曲线外观删除某一基维度。

## 阶段一检查

检查部署检查点：

```bash
python inspect_stage1_basis.py \
  --dataset PaviaU
```

比较全部检查点：

```bash
python inspect_stage1_basis.py \
  --dataset PaviaU \
  --compare_all
```

检查结果位于：

```text
outputs/stage1_basis_inspection/<dataset>/
```

重点关注：

- `orthogonality_error`：应接近数值精度；
- `projector_idempotence_error`：应接近数值精度；
- `residual_basis_leakage_ratio`：投影残差重新落入基空间的比例，应接近 0；
- `PSNR / SAM`：比较 PCA 初始化和微调检查点；
- `coefficient_energy_share`：检查是否存在几乎无能量的坐标；
- `per_band_rmse.png`：检查难重建波段是否集中在特定光谱区域。

# 当前阶段兼容性

新的 `basis_for_stage2.pth` 与旧的：

```text
checkpoints/stage1_unmix/<dataset>/unmixing_best.pth
```

结构完全不同。

旧 `train_stage2_physical.py` 仍依赖 `Stage1UnmixingNet`，不能加载新阶段一检查点。新的阶段二将改为：

\[
X_2
=
\mu+U_r(C_{\mathrm{up}}+\Delta C_{\mathrm{rel}}),
\]

其中 \(\Delta C_{\mathrm{rel}}\) 是由 SSP/NSP 可靠 MSI 细节引导的有符号光谱系数残差。

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
