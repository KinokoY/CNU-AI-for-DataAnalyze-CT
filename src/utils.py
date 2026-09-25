"""公共工具：仓库根定位、配置读取与 --set 覆盖、随机种子、日志、报告落盘、指纹计算。

整体功能：被 scripts/ 与 src/ 下所有脚本复用的一层薄工具，保证路径、随机性与产物格式跨轮次一致。
前后接口：上游读 configs/*.yaml 与 json 清单；下游给 dataset/unet/train/evaluate 提供 cfg 字典、REPO_ROOT 与报告写入函数。
用法：``from src.utils import load_config, REPO_ROOT, set_seed, save_report, config_fingerprint, cache_fingerprint``。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------------------
# 仓库根定位
# --------------------------------------------------------------------------------------

# src/utils.py -> parents[1] = 仓库根。所有相对路径都以仓库根为基准，
# 这样无论从哪个目录调用脚本（python scripts/x.py 或 python -m src.x）结果都一致。
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def resolve_path(path: str | os.PathLike, must_exist: bool = False) -> Path:
    """把可能是相对的路径解析成绝对路径；``~`` 会展开。

    must_exist=True 时路径不存在直接抛 FileNotFoundError，避免下游拿到一个空目录后静默跑出错误结果。
    """
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"路径不存在：{p}")
    return p


def rel_to_root(path: str | os.PathLike) -> str:
    """把路径转成相对仓库根的字符串（用于报告与日志，避免打印远程绝对路径）。"""
    p = Path(str(path)).expanduser()
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:  # noqa: BLE001 - 不在仓库内就原样返回
        return str(p)


# --------------------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------------------

def _coerce_scalar(text: str) -> Any:
    """把 --set 传入的字符串转成 YAML 风格标量（bool / None / int / float / list / str）。"""
    low = text.strip().lower()
    if low in ("null", "none", "~", ""):
        return None
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return json.loads(text)  # 能解析出 int/float/list/dict 就走 json
    except Exception:  # noqa: BLE001
        return text


def deep_update(base: dict, overrides: dict) -> dict:
    """递归合并字典，overrides 优先；返回新字典，不改动入参。"""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_set_overrides(pairs: Iterable[str] | None) -> dict:
    """把 ``train.batch_size=4`` 形式的覆盖串解析成嵌套字典；非法串直接抛错。"""
    result: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--set 参数必须是 key=value 形式，收到：{item!r}")
        dotted, raw = item.split("=", 1)
        keys = [k for k in dotted.strip().split(".") if k]
        if not keys:
            raise ValueError(f"--set 的 key 为空：{item!r}")
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _coerce_scalar(raw)
    return result


def load_yaml(path: str | os.PathLike) -> dict:
    """读取 YAML 配置文件为 dict；文件不存在或不是映射时抛错。"""
    import yaml  # 延迟导入，便于报错信息更清楚

    p = resolve_path(path, must_exist=True)
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML 映射（key: value），实际得到 {type(data).__name__}：{p}")
    return data


def load_config(path: str | os.PathLike = "configs/default.yaml",
                overrides: Iterable[str] | None = None) -> dict:
    """读取配置并应用 ``--set`` 覆盖，返回一个可直接使用的 dict。"""
    cfg = load_yaml(path)
    extra = parse_set_overrides(overrides)
    if extra:
        cfg = deep_update(cfg, extra)
    return cfg


def config_fingerprint(cfg: dict, drop: Sequence[str] = ()) -> str:
    """对配置内容算稳定 hash（用于 cache 与 checkpoint 的兼容性校验）。

    drop 里的点分键会在计算前剔除（例如 paths.reports 这类与结果无关的路径）。
    """
    cfg = copy.deepcopy(cfg)
    for dotted in drop:
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if isinstance(node, dict):
            node.pop(keys[-1], None)
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


#: 「缓存指纹」只由这些配置节决定（目前就是 preprocess）：
#: cache/ 与 cache_manifest.json 的内容只取决于预处理参数，别的节怎么加、怎么改都不该让缓存失效。
CACHE_FINGERPRINT_SECTIONS = ("preprocess",)


def cache_fingerprint(cfg: dict) -> str:
    """算「预处理 / 缓存指纹」：**只取 ``preprocess`` 节**，用于 cache_manifest 的一致性校验。

    preprocess.py / fetch_manifest.py 写清单、selfcheck_data.py / train.py 读清单，四处必须同一口径。
    **不要改回「排除若干节」的写法**：第 3 轮新增了与预处理完全无关的 ``loss`` 节，
    排除式写法会把它算进指纹里，于是缓存明明没变、训练启动校验却报
    「缓存来自别的预处理参数」——那正是这个函数存在的理由。
    """
    subset = {name: (cfg or {}).get(name) or {} for name in CACHE_FINGERPRINT_SECTIONS}
    return config_fingerprint(subset)


# --------------------------------------------------------------------------------------
# JSON / 报告
# --------------------------------------------------------------------------------------

def load_json(path: str | os.PathLike, default: Any = None) -> Any:
    """读取 JSON；文件不存在时返回 default（默认 None），解析失败则抛错。"""
    p = resolve_path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> Path:
    """写 JSON（UTF-8、保留中文），自动建父目录；返回写入的绝对路径。"""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent, sort_keys=False)
        fh.write("\n")
    return p


def save_report(obj: dict, path: str | os.PathLike, md_builder=None) -> tuple[Path, Path | None]:
    """把一个 dict 报告落盘为 JSON；md_builder 非空时再写一份同名 .md。

    md_builder 签名：``md_builder(obj) -> str``。返回 (json 路径, md 路径或 None)。
    """
    json_path = save_json(obj, path)
    md_path = None
    if md_builder is not None:
        md_path = json_path.with_suffix(".md")
        md_path.write_text(md_builder(obj), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------------------
# 随机性
# --------------------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定 random / numpy / torch 的种子。

    deterministic=True 时额外开启 cudnn.deterministic（会变慢，且个别算子可能报错），默认关闭。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = bool(deterministic)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
    except Exception:  # noqa: BLE001
        pass


def make_generator(seed: int):
    """返回一个只用于 DataLoader 采样的 torch.Generator（保证每折采样可复现）。"""
    import torch

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


# --------------------------------------------------------------------------------------
# 混合精度（AMP）：把「torch 版本差异」和「bf16 不要 GradScaler」这两件事收在一处
# --------------------------------------------------------------------------------------

#: amp 配置允许的规范化取值
AMP_CHOICES = ("bf16", "fp16", "off")


def resolve_amp(amp: str | None) -> tuple:
    """把配置里的 ``train.amp`` / ``eval.amp`` 解析成 ``(规范化名, torch.dtype 或 None)``。

    支持 ``bf16`` / ``bfloat16``、``fp16`` / ``float16`` / ``half``、``off`` / ``none`` / ``fp32``；
    dtype 为 ``None`` 表示不做 autocast（纯 fp32）。

    **bf16 与 GradScaler 的关系**：bf16 与 fp32 的指数范围相同（8 位指数），不存在 fp16 那种
    下溢问题，PyTorch 官方口径是 bf16 **不需要** loss scaling。所以本项目只在 ``fp16`` 时启用
    GradScaler，``bf16`` 只用 autocast（见 ``make_grad_scaler`` 与 src/train.py 的日志）。
    """
    import torch

    key = str(amp if amp is not None else "off").strip().lower()
    if key in ("bf16", "bfloat16"):
        return "bf16", torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return "fp16", torch.float16
    if key in ("off", "none", "fp32", "no", "false", ""):
        return "off", None
    raise ValueError(f"amp 只支持 {AMP_CHOICES}，收到 {amp!r}")


def autocast_context(device_type: str, dtype, enabled: bool = True):
    """返回 autocast 上下文管理器：优先 ``torch.amp.autocast``，回退 ``torch.cuda.amp.autocast``。

    torch 2.x 起新 API 是 ``torch.amp.autocast(device_type=..., dtype=...)``，
    老版本只有 ``torch.cuda.amp.autocast(dtype=...)``；这里统一成一处，训练与推理共用，
    避免每个脚本各写一遍 try/except。``dtype=None`` 或 ``enabled=False`` 时返回空上下文。
    """
    import contextlib

    import torch

    if not enabled or dtype is None:
        return contextlib.nullcontext()
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "autocast"):
        try:
            return amp.autocast(device_type=str(device_type), dtype=dtype)
        except TypeError:      # 老版本签名不同 → 落回 cuda.amp
            pass
    return torch.cuda.amp.autocast(dtype=dtype)


class NullGradScaler:
    """``GradScaler`` 的直通替身（非 CUDA 设备用）：接口一致，什么都不做。"""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, *args, **kwargs) -> None:
        return None

    def is_enabled(self) -> bool:
        return False

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state) -> None:
        return None


def make_grad_scaler(device_type: str, enabled: bool = True):
    """返回 ``GradScaler``：CUDA 上优先 ``torch.amp.GradScaler``，其余情况返回直通替身。

    * ``enabled=False``（bf16 / fp32）时 scaler 的 ``scale``/``step``/``update`` 都退化为原操作，
      于是训练循环只需要写一条路径；
    * 非 CUDA 设备（本地 CPU 调试）没有 GradScaler，直接给 ``NullGradScaler``，
      不依赖「None buffer / 空 scaler」这类版本相关的边界行为。
    """
    import torch

    device_type = str(device_type)
    if device_type != "cuda":
        return NullGradScaler()
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler(device_type, enabled=bool(enabled))
        except TypeError:      # 老版本没有 device_type 参数 → 落回 cuda.amp
            pass
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


# --------------------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------------------

def setup_logger(name: str = "ct", level: int = logging.INFO, log_file: str | os.PathLike | None = None):
    """配置并返回 logger：stdout 打 INFO，文件（可选）打 DEBUG，重复调用不会叠加 handler。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        path = resolve_path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def format_kv_table(rows: Sequence[dict], columns: Sequence[str]) -> str:
    """把若干 dict 渲染成 markdown 表格（缺失键渲染成 NA），用于报告里的逐 case 明细。"""
    def cell(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (list, tuple)):
            return "(" + ",".join(cell(v) for v in value) + ")"
        return str(value)

    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join([head, sep] + body)
