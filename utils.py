# utils.py
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "cuda"):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _to_checkpoint_safe(obj):
    """将 extra 中的自定义配置对象转成基础 Python 类型。

    PyTorch 2.6 以后 torch.load 默认更偏向 weights_only 安全加载，
    如果 checkpoint 里保存了 DatasetConfig 等自定义对象，可能触发反序列化限制。
    这里在保存端尽量把 extra 转成 dict/list/str/int/float/bool/None/Tensor。
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_checkpoint_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_checkpoint_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _to_checkpoint_safe(vars(obj))
    return str(obj)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_metric: float,
    path: str,
    extra: Optional[Dict] = None,
):
    ensure_dir(os.path.dirname(path))

    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model": model.state_dict(),
    }

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    if extra is not None:
        state["extra"] = _to_checkpoint_safe(extra)

    torch.save(state, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
    map_location: str = "cpu",
    load_optimizer: bool = True,
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # PyTorch 2.6 起部分环境下 torch.load 默认使用 weights_only=True，
    # 旧 checkpoint 中若包含 DatasetConfig 等自定义对象会报 Unsupported global。
    # 本项目加载的是自己训练保存的本地 checkpoint，设置 weights_only=False 可以兼容旧文件。
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 兼容较老版本 PyTorch，没有 weights_only 参数。
        state = torch.load(path, map_location=map_location)

    if "model" in state:
        missing_keys, unexpected_keys = model.load_state_dict(state["model"], strict=strict)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=strict)

    if strict is False:
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    if optimizer is not None and load_optimizer and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except ValueError as e:
            print("Optimizer state not loaded because parameter groups do not match.")
            print(str(e))

    epoch = state.get("epoch", 0)
    best_metric = state.get("best_metric", 0.0)

    return epoch, best_metric


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    B×C×H×W 或 C×H×W -> H×W×C
    """
    x = x.detach().float().cpu()

    if x.dim() == 4:
        x = x[0]

    x = torch.clamp(x, 0.0, 1.0)
    x = x.permute(1, 2, 0).numpy()
    return x


def save_mat(path: str, data: Dict):
    ensure_dir(os.path.dirname(path))

    try:
        import scipy.io as scio
        scio.savemat(path, data)
    except ImportError:
        raise ImportError("Please install scipy to save .mat files.")


def get_time_string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def write_log(log_path: str, text: str, print_text: bool = True):
    ensure_dir(os.path.dirname(log_path))

    line = f"[{get_time_string()}] {text}"

    if print_text:
        print(line)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class CSVLogger:
    """
    每个epoch自动写入一行训练日志，方便后续用Excel、pandas或matplotlib画曲线。
    """

    def __init__(self, csv_path: str, fieldnames: list):
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        ensure_dir(os.path.dirname(csv_path))

        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def write(self, row: dict):
        clean_row = {}

        for key in self.fieldnames:
            value = row.get(key, "")

            if isinstance(value, float):
                clean_row[key] = f"{value:.6f}"
            else:
                clean_row[key] = value

        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean_row)
