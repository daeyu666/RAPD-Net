"""
HSI Super-Resolution 项目模板入口。

本文件展示如何使用模板中的 dataloader、losses、metrics、utils 等工具。
用户需要根据具体项目实现自己的模型、训练逻辑和阶段控制。
"""

from config import parse_args, print_config
from data_loader import build_loaders
from utils import set_seed, get_device


def main():
    cfg = parse_args()
    print_config(cfg)
    set_seed(cfg.seed)

    # ---- 1. 构建数据加载器 ----
    train_loader, test_loader, info = build_loaders(cfg)

    print(f"\nDataset info:")
    for k, v in info.items():
        if k not in ("srf_weights", "hsi_wavelengths"):
            print(f"  {k}: {v}")

    device = get_device(cfg.device)

    # ---- 2. 模型构建（用户自行实现） ----
    # from your_model import YourModel
    # model = YourModel(
    #     n_bands=info["n_bands"],
    #     n_select_bands=info["n_select_bands"],
    #     scale_ratio=cfg.scale_ratio,
    # ).to(device)

    # ---- 3. 损失函数（按需选用模板中的 loss） ----
    # from losses import SAMLoss, SpectralGradientLoss, DataConsistencyLoss
    # criterion = ...

    # ---- 4. 训练 / 测试循环（用户自行实现） ----
    # for epoch in range(cfg.epochs):
    #     train_one_epoch(...)
    #     evaluate(...)

    # ---- 5. 保存 checkpoint ----
    # from utils import save_checkpoint
    # save_checkpoint(model, optimizer, epoch, best_metric, path)

    print("\nTemplate setup complete. Implement your model and training loop above.")


if __name__ == "__main__":
    main()
