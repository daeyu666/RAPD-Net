# HSI Super-Resolution 项目模板

高光谱图像超分辨率（HSI-MSI Fusion）项目的通用代码模板。

提供可直接复用的 dataloader、损失函数、评估指标、SRF 工具和通用训练辅助函数，
不依赖任何具体的模型结构。

## 模板内容

| 文件 | 说明 |
|------|------|
| `data_loader.py` | HSI 数据读取（.mat / h5）、预处理、patch 构建、DataLoader |
| `losses.py` | 光谱重建损失：SAM、光谱梯度、数据一致性、VQ commitment 等 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC 评估指标 |
| `srf_utils.py` | 光谱响应函数（SRF）加载、插值、权重构建、HSI→MSI 转换 |
| `prepare_srf_weights.py` | 预计算并保存 SRF 权重矩阵 |
| `utils.py` | 通用工具：随机种子、设备选择、checkpoint 存取、日志、CSV logger |
| `config.py` | 训练配置 dataclass + 命令行解析（不含模型参数） |
| `main.py` | 模板入口示例，展示如何串联各组件 |
| `models/stage1_unmixing.py` | RAPD-Net 第一阶段端元字典与丰度估计模型 |
| `train_stage1_unmix.py` | 第一阶段训练、验证、检查点及端元/丰度导出 |
| `analyze_spectral_regions.py` | 按光谱区域分析模型重建质量（模型通过参数传入） |
| `visualize_base_reconstruction.py` | 重建结果可视化：RGB 对比图、光谱曲线、误差图 |

## 使用方式

1. 将本项目复制为新项目的起点。
2. 在 `config.py` 的 `get_dataset_configs()` 中注册你的数据集。
3. 实现你自己的模型（例如 `models/your_model.py`）。
4. 在 `main.py`（或你自己的入口脚本）中串联 dataloader、模型、损失和训练循环。

```python
# 最小示例
from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss, DataConsistencyLoss
from utils import set_seed, get_device

cfg = parse_args()
set_seed(cfg.seed)
train_loader, test_loader, info = build_loaders(cfg)

# model = YourModel(...)
# criterion = ...
# train loop ...
```

## RAPD-Net 第一阶段：物理解混

第一阶段只使用 LR-HSI，学习场景级端元字典和低分辨率丰度图。端元先从真实 LR-HSI 像素中通过光谱最远点采样初始化，再与丰度估计器联合训练。

```bash
python train_stage1_unmix.py \
  --dataset PaviaU \
  --epochs 300 \
  --batch_size 4 \
  --lr 1e-4 \
  --unmix_num_endmembers 32
```

常用阶段参数：

- `--unmix_num_endmembers`：端元/光谱原子数，默认 32；
- `--unmix_hidden_channels`：丰度估计器宽度，默认 64；
- `--unmix_num_blocks`：空间残差块数量，默认 3；
- `--unmix_init_pixels`：端元初始化最多使用的 LR-HSI 像素数；
- `--unmix_lambda_sam`：训练阶段 SAM 权重，默认 0.5；
- `--unmix_lambda_sgrad`：训练阶段一阶光谱梯度权重，默认 0.1；
- `--unmix_selection_sam_weight`：默认最佳检查点选择中的 SAM 权重，默认 1.0；
- `--unmix_selection_sgrad_weight`：默认最佳检查点选择中的光谱梯度权重，默认 0.2；
- `--lambda_endmember_div`：近重复端元惩罚权重；
- `--lambda_abundance_tv`：丰度空间 TV 权重；
- `--lambda_abundance_entropy`：丰度低熵约束权重。

训练阶段的 SAM 权重与检查点选择权重彼此独立。默认训练保持 L1 与光谱角之间的平衡，而检查点选择更偏向光谱保真，避免仅因 L1 较低而错过 SAM 更好的模型。

输出位置：

```text
checkpoints/stage1_unmix/<dataset>/unmixing_best.pth
checkpoints/stage1_unmix/<dataset>/unmixing_best_sam.pth
checkpoints/stage1_unmix/<dataset>/unmixing_best_l1.pth
checkpoints/stage1_unmix/<dataset>/unmixing_last.pth
outputs/stage1_unmix/<dataset>/endmembers_model_order.npy
outputs/stage1_unmix/<dataset>/endmembers_sorted.npy
outputs/stage1_unmix/<dataset>/stage1_test_outputs.npz
logs/stage1_unmix/<dataset>.csv
```

- `unmixing_best.pth`：按光谱优先综合分数选择，作为后续阶段默认检查点；
- `unmixing_best_sam.pth`：验证集 SAM 最低的检查点；
- `unmixing_best_l1.pth`：验证集 L1 最低的检查点；
- `endmembers_sorted.npy`：仅用于曲线查看和人工检查，不用于替换模型内部端元顺序。

后续阶段应优先比较 `unmixing_best.pth` 与 `unmixing_best_sam.pth`，确认后续丰度注入是否更受益于综合重建精度还是最低光谱角。

## 目录结构约定

```
project/
├── data/               # 数据和 SRF 权重
│   ├── raw/            # 原始 HSI .mat 文件
│   ├── wavelengths/    # 各数据集波长文件
│   ├── srf/            # 原始 SRF CSV
│   └── srf_weights/    # 预计算的 SRF 权重
├── checkpoints/        # 模型权重
├── logs/               # 训练日志
├── outputs/            # 预测结果、指标、可视化
├── models/             # 用户自己的模型定义
└── code_template/      # 本模板（可作为 git submodule）
```

## 扩展原则

- 本模板只包含**与模型结构无关**的通用组件。
- 具体模型实现、训练逻辑、对比实验等请放在模板外的独立模块中。
- 对比模型建议统一放入 `models/baselines/`，配置统一放入 `configs/baselines/`。
