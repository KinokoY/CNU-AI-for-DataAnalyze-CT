"""整卷评估入口：载入 ``best.pt`` → 该折验证集整卷推理 → 3D 后处理 → 指标 → 报告（第 5 轮新增）。

整体流程（``python -m src.evaluate --fold 0 [--run-dir runs/smoke_fold0_fixed]``）：

    1. **前置校验**：读 ``data/splits.json`` / ``cache/cache_manifest.json``（口径与
       ``src.train.check_prerequisites`` 一致：病例集合、预处理指纹、cache 文件齐不齐），
       再核对 checkpoint 的 ``fold`` 与 ``cfg_hash`` —— 不一致直接以码 2 退出并列出差异；
    2. **挂权重**：``build_unet(cfg)`` 建同结构网络 → ``load_state_dict`` → ``eval()`` → 搬设备。
       基础版不含预训练编码器，所以「同一个 cfg + 同一个 state_dict」必然对得上
       （新增的预训练键会被 ``strict=False`` 忽略，但那种情况另有告警）；
    3. **逐例整卷推理**：复用训练期同一套 ``src.infer.predict_volume``（2.5D 三层窗 →
       ``/65535`` → 居中补边 → 逐层前向 → 按 ``pad_offset`` 裁回原始面内尺寸），
       保证「报告里的数字」与「训练日志里的数字」出自同一条链路；
    4. **后处理**：``src.postprocess.remove_small_lesions``（默认删 <50 mm³ 的孤立块）；
    5. **指标**：``src.metrics.case_metrics`` 一次给出后处理前 / 后两套
       （体素级 Dice/IoU/精确率/召回率 + 病灶级检出 + 假阳性统计）；
    6. **报告**：``reports/eval_fold<k>.json`` 逐折、``reports/eval_summary.{json,md}`` 汇总。

**报告口径（第 5 轮拍板，改前先读 ``docs/preprocess_notes.md`` 第三节「三·补」）**：

  * **主表用后处理之后的数字**（``clean``）；后处理**之前**（``raw``）的同一组指标一并写进 JSON，
    便于回答「删掉那些小块到底值多少 Dice / 召回」——但主表不并排两套，避免读者挑着看；
  * 一律给 **mean ± std 与逐例值**：肿瘤体积跨 3 个数量级（``data/splits.json`` 的
    ``tumor_volume_mm3`` 0.65–435 cm³），只看均值会被大病灶主导（fold 0 的 case 33 单独就能到
    0.76，而 57/59 长期 0）；
  * 汇总时区分 macro（逐例平均，与训练期早停同口径）与池化（所有体素合并后再算比率）——
    两者在小病灶占比高时差别很大，一起给；
  * 病灶级给**两套**：原判据（``detection_rate``）与**覆盖质量**（``overlap_frac`` = 交集/GT 体素）。
    ⚠️ 原判据里的「该 GT 病灶 >= ``eval.detect_min_mm3``（10 mm³）即算检出」在本数据集上**恒真**
    （20 例含肿瘤病人的肿瘤最小 652 mm³）⇒ 那一列实测报 19/19，**没有区分度**；
    报告会显式打 ``criterion_b_is_trivial`` 警告，并把覆盖比例分档表作为主口径。
    逐例表另给 ``best_cover``（该例覆盖得最好的那个病灶的比例），与 Dice 并排即可区分
    「没找到」（两者都低）与「找歪了/撒太大」（cover 高、Dice 低）。

**5 例仅肝脏病人（32/34/38/41/47）**：它们不在任何折的验证集里（跨轮约束：只进训练集），
所以它们的假阳性数字**只能**用一个具体权重跑一遍。本版统一用 fold 0 的权重（``--fp-ckpt`` 可换），
报告里显式标注权重来源——这些数字是「同一模型在无病灶病人上的假阳性率」，
不是 5 折交叉验证的结果。

**退出码**：``0`` 正常；``2`` 前置校验 / checkpoint 不匹配；``1`` 运行期异常（如形状不一致）。

前后接口：上游是 ``src.train`` 产出的 ``runs/<run>/best.pt`` 与 ``cache/``；
    下游是第 6 轮的调参依据（按病灶大小分层的检出率）。
用法：
    ```bash
    python -m src.evaluate --fold 0 --run-dir runs/smoke_fold0_fixed   # 开发期：12 轮烟测权重
    python -m src.evaluate --fold 0 --save-pred                        # 附带预测卷（叠图核对）
    python -m src.evaluate --all                                       # 5 折汇总（需要 5 折都有 best.pt）
    ```
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.dataset import data_config, format_hw, reset_open_cache

try:
    from src.infer import load_label_volume, predict_volume, resolve_threshold
    from src.metrics import (
        DEFAULT_DETECT_MIN_MM3,
        DEFAULT_SIZE_BINS,
        OVERLAP_BINS,
        case_metrics,
        summarize,
        voxel_spacing,      # 定义在 src.postprocess，由 src.metrics 显式再导出（导入口径只有一处）
    )
    from src.postprocess import remove_small_lesions
    from src.train import (
        build_model,
        cfg_for_hash,
        cfg_diff,
        check_prerequisites,
        file_digest,
        load_checkpoint,
        resolve_device,
    )
    from src.unet import count_parameters
    from src.utils import (
        cache_file,
        cache_fingerprint,
        config_fingerprint,
        format_kv_table,
        load_config,
        rel_to_root,
        resolve_path,
        save_report,
        set_seed,
        setup_logger,
    )
except ModuleNotFoundError:  # pragma: no cover - 兜底：把仓库根塞进 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset import data_config, format_hw, reset_open_cache  # type: ignore
    from src.infer import load_label_volume, predict_volume, resolve_threshold  # type: ignore
    from src.metrics import (  # type: ignore
        DEFAULT_DETECT_MIN_MM3,
        DEFAULT_SIZE_BINS,
        OVERLAP_BINS,
        case_metrics,
        summarize,
        voxel_spacing,
    )
    from src.postprocess import remove_small_lesions  # type: ignore
    from src.train import (  # type: ignore
        build_model,
        cfg_for_hash,
        cfg_diff,
        check_prerequisites,
        file_digest,
        load_checkpoint,
        resolve_device,
    )
    from src.unet import count_parameters  # type: ignore
    from src.utils import (  # type: ignore
        cache_file,
        cache_fingerprint,
        config_fingerprint,
        format_kv_table,
        load_config,
        rel_to_root,
        resolve_path,
        save_report,
        set_seed,
        setup_logger,
    )

LOGGER = setup_logger("evaluate")

#: 退出码（与 ``src.train`` 对齐：0 正常 / 2 前置校验失败）
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PREREQ = 2

#: 报告的 schema 版本（改字段时同步 +1，便于跨轮次对照旧报告）
REPORT_SCHEMA = "ct-liver-tumor-eval/1"

#: 逐例指标表的固定列（markdown 与 JSON 用同一份口径）
CASE_COLUMNS = ["case", "gt_mm3", "dice", "iou", "prec", "recall",
                "pred_mm3", "gt_lesions", "detected", "best_cover", "fp_mm3", "fp_lesions"]

#: 浮点数的报告精度（JSON 里保留 6 位小数足够，避免报告充满无意义的尾数）
ROUND = 6


# --------------------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CT 肝脏肿瘤分割整卷评估（后处理 + 体素级/病灶级指标 + 报告）；"
                    "运行手册见 docs/baseline.md")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件")
    parser.add_argument("--fold", type=int, default=None, help="评估哪一折（默认 0；--all 时忽略）")
    parser.add_argument("--all", action="store_true",
                        help="评估全部 5 折并汇总（需要每折都有 best.pt；缺的折只打印「跳过」）")
    parser.add_argument("--folds", default=None,
                        help="只评估指定折，逗号分隔：--folds 0,2（与 --all 二选一）")
    parser.add_argument("--run-dir", default=None,
                        help="checkpoint 所在目录，默认 <paths.runs>/fold<k>；"
                             "开发期指向短跑产物：--run-dir runs/smoke_fold0_fixed")
    parser.add_argument("--ckpt", default=None,
                        help="直接指定 checkpoint 文件（优先于 --run-dir），如 runs/smoke_fold0_fixed/best.pt")
    parser.add_argument("--cache-dir", default=None, help="覆盖 cache/ 位置")
    parser.add_argument("--out-dir", default=None, help="报告目录，默认 <paths.reports>（reports/）")
    parser.add_argument("--device", default=None, help="cuda / cpu；默认自动")
    parser.add_argument("--save-pred", action="store_true",
                        help="把预测卷写进 <run-dir>/pred/<case>.nii.gz（保留 affine；含 GT 与后处理后预测）")
    parser.add_argument("--save-raw", action="store_true",
                        help="配合 --save-pred：额外保存后处理前的预测与概率图（uint16 量化，体积较大）")
    parser.add_argument("--fp-ckpt", default=None,
                        help="仅肝脏病人假阳性统计用的 checkpoint；默认取本次评估的第一折权重")
    parser.add_argument("--skip-fp", action="store_true", help="跳过 5 例仅肝脏病人的假阳性统计")
    parser.add_argument("--limit-cases", type=int, default=0,
                        help="只评估每折前 N 例（开发期语法自检用，报告会标注不完整）")
    parser.add_argument("--skip-config-check", action="store_true",
                        help="允许 checkpoint 的 cfg_hash 与当前配置不一致（默认拒绝，返回码 2）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要评估的折/病例/权重路径，不推理")
    parser.add_argument("--set", dest="overrides", action="append", default=None,
                        help="覆盖配置项，可多次：--set eval.threshold=0.4")
    return parser.parse_args(argv)


def parse_folds(args) -> list:
    """解析要评估的折号列表：``--all`` / ``--folds 0,2`` / ``--fold k``（默认 0）。"""
    if args.folds:
        values = [int(part) for part in str(args.folds).replace(" ", "").split(",") if part]
        if not values:
            raise ValueError(f"--folds 解析不出折号：{args.folds!r}")
        return values
    if args.all:
        return [0, 1, 2, 3, 4]
    return [int(args.fold if args.fold is not None else 0)]


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------

def git_commit() -> str:
    """当前仓库的短 commit（写进报告，便于把数字与代码版本对上）；拿不到就返回空串。"""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(Path(__file__).resolve().parents[1]),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - 没有 git / 命令不可用都归到这里，不是致命问题
        return ""


def round_floats(value, digits: int = ROUND):
    """递归地把 dict / list 里的 float 收成固定精度（JSON 报告更干净，也更好 diff）。"""
    if isinstance(value, dict):
        return {k: round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v, digits) for v in value]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return round(float(value), digits)
    if isinstance(value, np.floating):
        return round(float(value), digits)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return round_floats(value.tolist(), digits)
    return value


def slim_lesion(record: dict) -> dict:
    """把 ``lesion_detection`` 的返回压成报告用的精简版（去掉逐病灶的 bbox，保留判定细节）。"""
    slim = {k: v for k, v in (record or {}).items() if k not in ("per_lesion", "detected")}
    slim["lesions"] = [
        {k: v for k, v in lesion.items() if k != "bbox"}
        for lesion in (record or {}).get("per_lesion", [])
    ]
    return slim


def slim_fp(stats: dict) -> dict:
    """假阳性统计的精简版（``largest_bbox`` 只留 z 范围，报告里够用且不冗长）。"""
    out = {k: v for k, v in (stats or {}).items() if k != "largest_bbox"}
    bbox = (stats or {}).get("largest_bbox")
    out["largest_z_range"] = [int(bbox[0]), int(bbox[1])] if bbox else None
    return out


def checkpoint_run_name(path) -> str:
    """从 checkpoint 路径里取 run 名字（``runs/smoke_fold0_fixed/best.pt`` → ``smoke_fold0_fixed``）。"""
    return Path(path).parent.name


# --------------------------------------------------------------------------------------
# 前置校验与 checkpoint
# --------------------------------------------------------------------------------------

def resolve_checkpoint(args, cfg: dict, fold: int | None = None) -> Path:
    """定位 checkpoint：``--ckpt`` > ``--run-dir/best.pt`` > ``<paths.runs>/fold<k>/best.pt``。"""
    if args.ckpt:
        return resolve_path(args.ckpt)
    if args.run_dir:
        return resolve_path(Path(args.run_dir) / "best.pt")
    runs = ((cfg.get("paths") or {}).get("runs", "runs"))
    index = 0 if fold is None else int(fold)
    return resolve_path(Path(runs) / f"fold{index}" / "best.pt")


def load_checkpoint_model(path, cfg: dict, device: torch.device, skip_config_check: bool = False) -> dict:
    """载入 ``best.pt`` 并把权重挂到 ``build_unet(cfg)`` 上，返回元信息 dict。

    校验三件事（任何一条不对就以 ``ValueError`` 抛出，由 ``main`` 转成退出码 2）：
      1. checkpoint 里有 ``model_state``（旧版本只有裸 ``state_dict`` 时也兼容）；
      2. ``fold`` 与当前 ``--fold`` 一致（防止把 fold 2 的权重当 fold 0 报出来）；
      3. ``cfg_hash`` 与当前配置一致（差异逐条打印；``--skip-config-check`` 可跳过）。

    ``strict=False`` 只针对「新增的预训练编码器键」这类情况：缺键会告警（随机初始化的部分
    会让指标失去意义），多余的键（当前配置里没有的模块）静默忽略。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 checkpoint {path}：先训练（python -m src.train --fold k），"
                                f"或用 --run-dir / --ckpt 指向实际的权重文件")
    payload = load_checkpoint(path, device)
    state = payload.get("model_state") or payload.get("state_dict")
    if not state:
        raise ValueError(f"{path} 里没有 model_state（也不是裸 state_dict）：无法恢复权重")

    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"{path} 与当前配置建出的网络不匹配：缺 {len(missing)} 个键"
                         f"（如 {list(missing)[:5]}）。请核对 --config 与训练时的配置是否一致。")
    if unexpected:
        LOGGER.warning("权重里有 %d 个当前网络用不到的键（如 %s）：已忽略，"
                       "通常是进阶版（预训练编码器）留下的参数。",
                       len(unexpected), list(unexpected)[:5])
    model.to(device)
    model.eval()

    current_hash = config_fingerprint(cfg_for_hash(cfg))
    ckpt_hash = payload.get("cfg_hash")
    if ckpt_hash and ckpt_hash != current_hash and not skip_config_check:
        LOGGER.error("checkpoint 与当前配置不一致（cfg_hash %s vs %s）；差异（最多 8 处）：",
                     ckpt_hash, current_hash)
        for line in cfg_diff(payload.get("cfg") or {}, cfg_for_hash(cfg), limit=8):
            LOGGER.error("  - %s", line)
        raise ValueError("配置不一致：报告里的数字会与权重来源对不上。"
                         "改回训练时的配置，或显式加 --skip-config-check 承担这个风险。")
    if ckpt_hash and ckpt_hash != current_hash:
        LOGGER.warning("已跳过配置一致性校验（--skip-config-check）：cfg_hash %s vs %s",
                       ckpt_hash, current_hash)

    # 两个「不会进 cfg_hash 但会让整卷推理错位」的配置：数据集尺寸与输入通道数
    saved = payload.get("cfg") or {}
    saved_data = saved.get("data") or {}
    now_data = cfg.get("data") or {}
    for key in ("target_hw", "z_context"):
        old, new = saved_data.get(key), now_data.get(key)
        if old is not None and new is not None and list(np.atleast_1d(old)) != list(np.atleast_1d(new)):
            raise ValueError(f"checkpoint 记录的 data.{key}={old} 与当前配置 {new} 不一致："
                             f"整卷推理的补边/通道口径会错位，报告不可信。")

    info = {
        "path": rel_to_root(path),
        "run": checkpoint_run_name(path),
        "fold": int(payload.get("fold", -1)),
        "epoch": int(payload.get("epoch", 0)),
        "metric_name": (payload.get("metric") or {}).get("name"),
        "metric_value": (payload.get("metric") or {}).get("value"),
        "best": payload.get("best") or {},
        "cfg_hash": ckpt_hash,
        "preprocess_cfg_hash": payload.get("preprocess_cfg_hash"),
        "splits_digest": payload.get("splits_digest", ""),
        "seed": payload.get("seed"),
        "amp": payload.get("amp"),
        "torch": payload.get("torch"),
        "saved_at": payload.get("saved_at"),
        "params": int(count_parameters(model)),
        "config_checked": bool(ckpt_hash == current_hash),
    }
    return {"model": model, "info": info, "payload_cfg": saved}


# --------------------------------------------------------------------------------------
# 单折评估
# --------------------------------------------------------------------------------------

def to_unit16(prob) -> np.ndarray:
    """浮点概率 → uint16（``round(p × 65535)``），用于把概率图落盘。

    与 preprocess 的 ``image_out_dtype=uint16`` 同一思路：512×512×488 的 float32 是 512 MB，
    压成 uint16 只有 256 MB（再 gzip 更小），精度 1/65535 远高于「阈值 0.5」需要的分辨率。
    """
    array = np.asarray(prob, dtype=np.float32)
    return np.rint(np.clip(array, 0.0, 1.0) * 65535.0).astype(np.uint16)


def save_volume(volume, cache_path, out_path, description: str) -> Path:
    """按 cache 里 GT 的 affine/header 保存一卷预测，返回写入路径。

    **必须复用同一个 affine**：``predict_volume`` 已经把预测裁回原始面内尺寸，与
    ``cache/label/<case>.nii`` 逐体素对齐，所以叠图时不会错位（自己造仿射会出现「看起来对上了、
    实际平移了几十毫米」这种最难查的错）。dtype 按传入数组走（预测是 uint8、概率是 uint16）。
    """
    import nibabel as nib

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reference = nib.load(str(cache_path))
    array = np.asarray(volume)
    image = nib.Nifti1Image(array, reference.affine, reference.header)
    image.header.set_data_dtype(array.dtype)
    image.header["descrip"] = np.asarray([description[:79]], dtype="S80")
    nib.save(image, str(out_path))
    return out_path


def _case_positions(ds, case: int, z: int) -> list:
    """返回某病例某层在 ``ds.index`` 里的位置（调试用；正常只有 1 个）。"""
    return [i for i, (c, zz) in enumerate(ds.index) if int(c) == int(case) and int(zz) == int(z)]


def run_fold(*, cfg: dict, fold: int, ckpt_path, cache_dir, split_record: dict, args,
             device: torch.device) -> dict:
    """评估一折：整卷推理 → 后处理 → 指标 → （可选）写预测卷，返回报告片段 dict。"""
    from src.dataset import CTSliceDataset   # 局部导入：只在真正评估时才需要

    val_cases = sorted(int(c) for c in split_record.get("val", []))
    if args.limit_cases:
        val_cases = val_cases[:int(args.limit_cases)]
    threshold = resolve_threshold(cfg, None)
    min_lesion_mm3 = float(((cfg.get("eval") or {}).get("min_lesion_mm3", 50.0)) or 0.0)
    detect_min_mm3 = float(((cfg.get("eval") or {}).get("detect_min_mm3", DEFAULT_DETECT_MIN_MM3)) or 0.0)
    spacing = voxel_spacing(cfg)

    loaded = load_checkpoint_model(ckpt_path, cfg, device,
                                   skip_config_check=bool(args.skip_config_check))
    model, ckpt_info = loaded["model"], loaded["info"]
    if int(ckpt_info.get("fold", -1)) != int(fold):
        raise ValueError(f"{ckpt_info['path']} 属于 fold {ckpt_info.get('fold')}，"
                         f"与本次要评估的 fold {fold} 不一致（用 --fold 指定正确的折）")
    LOGGER.info("=" * 78)
    LOGGER.info("fold %d 评估：权重 %s（epoch %s，%s = %s，%s 参数）；验证 %d 例 %s",
                fold, ckpt_info["path"], ckpt_info.get("epoch"),
                ckpt_info.get("metric_name"), ckpt_info.get("metric_value"),
                f"{ckpt_info['params'] / 1e6:.3f} M", len(val_cases), val_cases)
    LOGGER.info("口径：阈值 %.2f；后处理删 <%.1f mm³ 孤立块；病灶检出判据 = 与 GT 有重叠 或 "
                "该 GT 病灶 >= %.1f mm³；体素物理尺寸 %s", threshold, min_lesion_mm3,
                detect_min_mm3, tuple(spacing))

    dataset = CTSliceDataset(val_cases, cache_dir, "val", cfg, augment=False, fold=int(fold))
    del dataset     # 这里只借它做一次「cache / 划分 / 形状」的前置校验；逐例读取由 predict_volume 负责
    records: list = []
    pred_dir = Path(ckpt_path).parent / "pred"
    for position, case in enumerate(val_cases, start=1):
        started = time.perf_counter()
        prob, pred_raw, meta = predict_volume(model, case, cache_dir, cfg, device,
                                              pad_to_multiple=int(((cfg.get("model") or {})
                                                                   .get("pad_to_multiple", 16) or 16)),
                                              batch_slices=int(((cfg.get("eval") or {})
                                                                .get("infer_batch_slices", 8) or 8)),
                                              threshold=threshold,
                                              amp=((cfg.get("eval") or {}).get("amp")
                                                   or (cfg.get("train") or {}).get("amp")))
        gt = load_label_volume(case, cache_dir, cfg)
        if tuple(pred_raw.shape) != tuple(gt.shape):
            raise RuntimeError(f"case {case}：预测卷 {tuple(pred_raw.shape)} 与 GT 卷 "
                               f"{tuple(gt.shape)} 形状不一致（轴序或 pad_offset 裁回出错）")
        cleaned, n_removed, removed_mm3 = remove_small_lesions(pred_raw, min_lesion_mm3, spacing)
        metrics = case_metrics(pred_raw, cleaned, gt, spacing, cfg)
        gt_volume = float((metrics["gt_lesions"] or {}).get("total_mm3", 0.0))
        is_tumor = gt_volume > 0.0
        record = {
            "case": int(case),
            "is_tumor": bool(is_tumor),
            "gt_mm3": round(gt_volume, 3),
            # 该例**覆盖得最好**的那个 GT 病灶的比例（交集/GT）：与 Dice 并排就能区分
            # 「没找到」（两者都低）与「找歪了/撒太大」（cover 高、Dice 低）
            "best_cover": (max((float(item.get("overlap_frac", 0.0))
                                for item in metrics["lesion_clean"].get("per_lesion", [])), default=None)
                           if is_tumor else None),
            "run": ckpt_info["run"],
            "checkpoint": ckpt_info["path"],
            "fold": int(fold),
            "threshold": float(threshold),
            "z_context": int(meta.get("z_context", 0) or 0),
            "in_channels": int(meta.get("in_channels", 0) or 0),
            "orig_hw": [int(v) for v in meta.get("orig_hw", [])],
            "pad_offset": [int(v) for v in meta.get("pad_offset", [])],
            "n_slices": int(meta.get("n_slices", 0)),
            "prob_peak": float(meta.get("prob_peak", 0.0) or 0.0),
            "infer_seconds": round(float(meta.get("seconds", 0.0) or 0.0), 3),
            "voxel_raw": metrics["voxel_raw"],
            "voxel_clean": metrics["voxel_clean"],
            "lesion_raw": {**slim_lesion(metrics["lesion_raw"]),
                           "detected": metrics["lesion_raw"]["detected"]},
            "lesion_clean": {**slim_lesion(metrics["lesion_clean"]),
                             "detected": metrics["lesion_clean"]["detected"]},
            "fp_raw": slim_fp(metrics["fp_raw"]),
            "fp_clean": slim_fp(metrics["fp_clean"]),
            "gt_lesions": {
                "n": int((metrics["gt_lesions"] or {}).get("n", 0)),
                "volumes_mm3": (metrics["gt_lesions"] or {}).get("volumes_mm3", []),
                "total_mm3": round(gt_volume, 3),
                "by_size_bin": (metrics["gt_lesions"] or {}).get("by_size_bin", {}),
            },
            "pred_lesions_raw": metrics["pred_lesions_raw"],
            "pred_lesions_clean": metrics["pred_lesions_clean"],
            "postprocess": {"min_lesion_mm3": min_lesion_mm3, "n_removed": int(n_removed),
                            "volume_removed_mm3": round(float(removed_mm3), 3),
                            "seconds": round(time.perf_counter() - started - float(meta.get("seconds", 0.0)), 3)},
        }
        if args.save_pred:
            reference = cache_file(cache_dir, "label", case)
            saved = [str(save_volume(gt, reference, pred_dir / f"{int(case)}_gt.nii.gz",
                                     f"GT case {case} fold {fold}"))]
            saved.append(str(save_volume(cleaned, reference,
                                         pred_dir / f"{int(case)}_pred.nii.gz",
                                         f"pred(clean) case {case} fold {fold} thr={threshold:g}")))
            if args.save_raw:
                saved.append(str(save_volume(pred_raw, reference,
                                             pred_dir / f"{int(case)}_pred_raw.nii.gz",
                                             f"pred(raw) case {case} fold {fold} thr={threshold:g}")))
                saved.append(str(save_volume(to_unit16(prob), reference,
                                             pred_dir / f"{int(case)}_prob_u16.nii.gz",
                                             f"prob(u16/65535) case {case} fold {fold}")))
            record["saved"] = [rel_to_root(Path(p)) for p in saved]

        records.append(record)
        LOGGER.info("[%d/%d] case %d（GT %s mm³，%s）｜Dice %.4f IoU %.4f 精确率 %.4f 召回率 %.4f"
                    "｜病灶 %d/%d 检出%s｜预测 %d mm³（后处理删 %d 块 / %s mm³）｜峰值概率 %.4f｜%.1f s",
                    position, len(val_cases), case, f"{gt_volume:.1f}",
                    "含肿瘤" if is_tumor else "**仅肝脏**",
                    record["voxel_clean"]["dice"], record["voxel_clean"]["iou"],
                    record["voxel_clean"]["precision"], record["voxel_clean"]["recall"],
                    record["lesion_clean"]["n_detected"], record["lesion_clean"]["n_gt"],
                    "".join("✓" if item["detected"] else "✗"
                            for item in record["lesion_clean"].get("lesions", [])),
                    record["voxel_clean"]["pred_voxels"], int(n_removed), f"{removed_mm3:.1f}",
                    record["prob_peak"], record["infer_seconds"])

    assessment = assess_fold(records, fold=fold, ckpt_info=ckpt_info, threshold=threshold,
                             min_lesion_mm3=min_lesion_mm3, detect_min_mm3=detect_min_mm3,
                             spacing=spacing, val_cases=val_cases,
                             data_cfg=data_config(cfg), incomplete=bool(args.limit_cases))
    fold_report = {
        "fold": int(fold),
        "run": ckpt_info["run"],
        "checkpoint": ckpt_info,
        "threshold": float(threshold),
        "min_lesion_mm3": float(min_lesion_mm3),
        "detect_min_mm3": float(detect_min_mm3),
        "spacing": [float(v) for v in spacing],
        "val_cases": [int(c) for c in val_cases],
        "incomplete": bool(args.limit_cases),
        "assessment": assessment,
        "cases": records,
    }
    reset_open_cache()
    return fold_report


def assess_fold(records, *, fold: int, ckpt_info: dict, threshold: float, min_lesion_mm3: float,
                detect_min_mm3: float, spacing, val_cases, data_cfg: dict,
                incomplete: bool = False) -> dict:
    """把一折的逐例记录汇总成「折级」结论（macro / 池化 / 分层检出 / 后处理效果）。"""
    clean = [rec["voxel_clean"] for rec in records]
    raw = [rec["voxel_raw"] for rec in records]
    tumor_records = [rec for rec in records if rec["is_tumor"]]

    def macro(key: str, pool) -> dict:
        """逐例等权平均（macro）：``pool`` 直接给体素指标 dict 列表（clean 或 raw）。"""
        return summarize([float(item[key]) for item in pool])

    totals_clean = {key: int(sum(int(rec["voxel_clean"][key]) for rec in records))
                    for key in ("tp", "fp", "fn", "pred_voxels", "gt_voxels")}
    totals_raw = {key: int(sum(int(rec["voxel_raw"][key]) for rec in records))
                  for key in ("tp", "fp", "fn", "pred_voxels", "gt_voxels")}
    lesions_clean = [rec["lesion_clean"] for rec in tumor_records]
    lesions_raw = [rec["lesion_raw"] for rec in tumor_records]

    from src.metrics import confusion_summary, detection_stats   # 局部导入，避免循环引用的错觉

    assessment = {
        "n_cases": int(len(records)),
        "n_cases_tumor": int(len(tumor_records)),
        "val_cases": [int(c) for c in val_cases],
        "val_cases_tumor": [int(rec["case"]) for rec in tumor_records],
        "dice_mean": macro("dice", clean)["mean"],
        "dice_std": macro("dice", clean)["std"],
        "dice_min": macro("dice", clean)["min"],
        "dice_max": macro("dice", clean)["max"],
        "macro_clean": {key: macro(key, clean) for key in ("dice", "iou", "precision", "recall")},
        "macro_raw": {key: macro(key, raw) for key in ("dice", "iou", "precision", "recall")},
        "pooled_clean": confusion_summary({**totals_clean, "cases": len(records)}, spacing),
        "pooled_raw": confusion_summary({**totals_raw, "cases": len(records)}, spacing),
        "lesion_clean": detection_stats(lesions_clean),
        "lesion_raw": detection_stats(lesions_raw),
        "gt_volume_mm3": round(float(sum(float(rec["gt_mm3"]) for rec in tumor_records)), 3),
        "pred_volume_mm3_clean": round(float(sum(float(rec["voxel_clean"]["pred_voxels"])
                                                for rec in records)) * float(np.prod(spacing)), 3),
        "postprocess": {
            "min_lesion_mm3": float(min_lesion_mm3),
            "n_removed_total": int(sum(int(rec["postprocess"]["n_removed"]) for rec in records)),
            "volume_removed_mm3": round(float(sum(float(rec["postprocess"]["volume_removed_mm3"])
                                                  for rec in records)), 3),
            "dice_gain_mean": float(np.mean([float(rec["voxel_clean"]["dice"])
                                             - float(rec["voxel_raw"]["dice"]) for rec in records]))
            if records else 0.0,
        },
        "threshold": float(threshold),
        "detect_min_mm3": float(detect_min_mm3),
        "checkpoint": ckpt_info["path"],
        "run": ckpt_info["run"],
        "epoch": ckpt_info.get("epoch"),
        "incomplete": bool(incomplete),
        "n_slices_evaluated": int(sum(int(rec["n_slices"]) for rec in records)),
        "infer_seconds": round(float(sum(float(rec["infer_seconds"]) for rec in records)), 3),
        # 与训练日志对称的三个「一眼看出链路对不对」的量（报告要能自证输入口径）
        "z_context": int((data_cfg or {}).get("z_context", 0) or 0),
        "in_channels": int(2 * int((data_cfg or {}).get("z_context", 0) or 0) + 1),
        "target_hw": [int(v) for v in (data_cfg or {}).get("target_hw", [])],
    }
    LOGGER.info("-" * 78)
    LOGGER.info("fold %d 折级结论（后处理之后，macro = 逐例等权）：Dice %.4f ± %.4f"
                "（min %.4f max %.4f）；IoU %.4f；精确率 %.4f；召回率 %.4f",
                fold, assessment["dice_mean"], assessment["dice_std"], assessment["dice_min"],
                assessment["dice_max"], assessment["macro_clean"]["iou"]["mean"],
                assessment["macro_clean"]["precision"]["mean"],
                assessment["macro_clean"]["recall"]["mean"])
    LOGGER.info("池化（所有体素合并）：Dice %.4f / IoU %.4f / 精确率 %.4f / 召回率 %.4f；"
                "预测 %s mm³，GT %s mm³",
                assessment["pooled_clean"]["dice"], assessment["pooled_clean"]["iou"],
                assessment["pooled_clean"]["precision"], assessment["pooled_clean"]["recall"],
                assessment["pooled_clean"].get("pred_mm3"), assessment["pooled_clean"].get("gt_mm3"))
    lesion_now = assessment["lesion_clean"]
    if lesion_now.get("status") == "ok":
        bins = "；".join(f"{name}: {slot['n_detected']}/{slot['n_gt']}"
                         for name, slot in lesion_now["by_size_bin"].items()) or "（无 GT 病灶）"
        LOGGER.info("病灶级检出（后处理之后）：%d/%d = %.3f（逐例平均 %.3f）；按体积分档 %s",
                    lesion_now["n_detected"], lesion_now["n_gt"], lesion_now["detection_rate"],
                    lesion_now["detection_rate_per_case_mean"], bins)
        # 「与 GT 有重叠」「病灶 >= detect_min_mm3」这两条判据在当前数据尺度下可能恒真：
        # 恒真时 detection_rate 一定是 1.000，必须把重叠比例分档一起打出来，否则这一列是噪声。
        if lesion_now.get("criterion_b_is_trivial"):
            LOGGER.warning("注意：%d/%d 个 GT 病灶都 >= detect_min_mm3=%.1f mm³ ⇒ 上面那个检出率"
                           "**恒为 1.000、没有区分度**（模型一个体素都不预测也会是这个数）。"
                           "请看下面的重叠比例分档与 n_covered。",
                           lesion_now["n_gt_trivially_detected"], lesion_now["n_gt"],
                           float(lesion_now.get("detect_min_mm3", 0.0)))
        LOGGER.info("病灶覆盖质量（每病灶 交集/GT 体素；这才是有区分度的口径）：均值 %.3f 中位 %.3f；"
                    "有重叠 %d/%d、覆盖 >=%.2f 的 %d/%d、>=%.2f 的 %d/%d",
                    lesion_now["mean_overlap_frac"], lesion_now["median_overlap_frac"],
                    lesion_now["n_hit_overlap"], lesion_now["n_gt"],
                    float(lesion_now.get("cover_frac", 0.2)),
                    lesion_now["n_covered"], lesion_now["n_gt"],
                    float(lesion_now.get("cover_frac_strict", 0.5)),
                    lesion_now["n_covered_strict"], lesion_now["n_gt"])
    LOGGER.info("后处理：删掉 %d 块 / %s mm³；逐例 Dice 平均变化 %+.5f（后处理 - 原始）",
                assessment["postprocess"]["n_removed_total"],
                assessment["postprocess"]["volume_removed_mm3"],
                assessment["postprocess"]["dice_gain_mean"])
    LOGGER.info("%s", format_kv_table(
        [{"case": rec["case"], "gt_mm3": rec["gt_mm3"],
          "dice": round(float(rec["voxel_clean"]["dice"]), 4),
          "iou": round(float(rec["voxel_clean"]["iou"]), 4),
          "prec": round(float(rec["voxel_clean"]["precision"]), 4),
          "recall": round(float(rec["voxel_clean"]["recall"]), 4),
          "pred_mm3": rec["voxel_clean"]["pred_voxels"],
          "gt_lesions": rec["lesion_clean"]["n_gt"],
          "detected": rec["lesion_clean"]["n_detected"],
          "fp_mm3": round(float(rec["fp_clean"]["fp_mm3"]), 1),
          "fp_lesions": rec["fp_clean"]["n_lesions"]} for rec in records],
        CASE_COLUMNS))
    return assessment


# --------------------------------------------------------------------------------------
# 假阳性队列（5 例仅肝脏病人）
# --------------------------------------------------------------------------------------

def run_fp_cohort(*, cfg: dict, ckpt_path, cache_dir, cases, device, args) -> dict:
    """对「仅肝脏」病人跑一遍推理，报告假阳性（FP 体积 / 孤立块数 / 最大块）。

    这些病例的 GT 全空（``docs/data.md`` 第 2 节：按"无病灶"处理），因此
    预测的**任何**前景都是假阳性 —— 这正是「约束假阳性」这条跨轮约束需要的数字。
    它们不在任何折的验证集里，所以必须用一个具体权重跑；本版统一用 fold 0 的权重
    （``--fp-ckpt`` 可换），报告里的 ``checkpoint`` 字段显式标注来源。
    """
    cases = sorted(int(c) for c in cases)
    if not cases:
        return {"status": "跳过（data/splits.json 里没有 liver_only_cases）", "cases": []}
    threshold = resolve_threshold(cfg, None)
    min_lesion_mm3 = float(((cfg.get("eval") or {}).get("min_lesion_mm3", 50.0)) or 0.0)
    spacing = voxel_spacing(cfg)
    loaded = load_checkpoint_model(ckpt_path, cfg, device,
                                   skip_config_check=bool(args.skip_config_check))
    model, ckpt_info = loaded["model"], loaded["info"]
    LOGGER.info("=" * 78)
    LOGGER.info("仅肝脏病人假阳性统计：%d 例 %s；权重 %s（**不在任何折的验证集里**，"
                "这组数字只代表该权重在无病灶病人上的表现）",
                len(cases), cases, ckpt_info["path"])
    records: list = []
    for case in cases:
        prob, pred_raw, meta = predict_volume(model, case, cache_dir, cfg, device,
                                              pad_to_multiple=int(((cfg.get("model") or {})
                                                                   .get("pad_to_multiple", 16) or 16)),
                                              batch_slices=int(((cfg.get("eval") or {})
                                                                .get("infer_batch_slices", 8) or 8)),
                                              threshold=threshold,
                                              amp=((cfg.get("eval") or {}).get("amp")
                                                   or (cfg.get("train") or {}).get("amp")))
        gt = load_label_volume(case, cache_dir, cfg)
        cleaned, n_removed, removed_mm3 = remove_small_lesions(pred_raw, min_lesion_mm3, spacing)
        metrics = case_metrics(pred_raw, cleaned, gt, spacing, cfg)
        record = {
            "case": int(case),
            "gt_mm3": 0.0,
            "is_tumor": False,
            "run": ckpt_info["run"],
            "checkpoint": ckpt_info["path"],
            "fp_raw": slim_fp(metrics["fp_raw"]),
            "fp_clean": slim_fp(metrics["fp_clean"]),
            "pred_lesions_raw": metrics["pred_lesions_raw"],
            "pred_lesions_clean": metrics["pred_lesions_clean"],
            "postprocess": {"min_lesion_mm3": min_lesion_mm3, "n_removed": int(n_removed),
                            "volume_removed_mm3": round(float(removed_mm3), 3)},
            "prob_peak": float(meta.get("prob_peak", 0.0) or 0.0),
            "infer_seconds": round(float(meta.get("seconds", 0.0) or 0.0), 3),
        }
        if args.save_pred:
            reference = cache_file(cache_dir, "label", case)
            saved = [str(save_volume(cleaned, reference,
                                     Path(ckpt_path).parent / "pred" / f"{int(case)}_pred.nii.gz",
                                     f"pred(clean) liver-only case {case}"))]
            if args.save_raw:
                saved.append(str(save_volume(pred_raw, reference,
                                             Path(ckpt_path).parent / "pred" / f"{int(case)}_pred_raw.nii.gz",
                                             f"pred(raw) liver-only case {case}")))
            record["saved"] = [rel_to_root(Path(p)) for p in saved]
        records.append(record)
        LOGGER.info("case %d（仅肝脏）：预测体积 %s mm³（孤立块 %d 个，最大 %s mm³）；"
                    "后处理删 %d 块 / %s mm³ → 剩余 %s mm³；峰值概率 %.4f",
                    case, record["fp_raw"]["fp_mm3"], record["fp_raw"]["n_lesions"],
                    record["fp_raw"]["largest_mm3"], int(n_removed), f"{removed_mm3:.1f}",
                    record["fp_clean"]["fp_mm3"], record["prob_peak"])
    summary = summarize_fp(records)
    LOGGER.info("仅肝脏 %d 例：命中（有孤立块）%d 例 = %.3f；逐例 FP 体积 %s mm³"
                "（中位 %s，最大 %s）；后处理之后 %s mm³（中位 %s）",
                summary["n_cases"], summary["n_cases_with_lesion"], summary["rate_with_lesion"],
                summary["fp_volume_mm3"]["mean"], summary["fp_volume_mm3"]["median"],
                summary["fp_volume_mm3"]["max"], summary["fp_volume_mm3_clean"]["mean"],
                summary["fp_volume_mm3_clean"]["median"])
    reset_open_cache()
    return {"status": "ok", "checkpoint": ckpt_info["path"], "run": ckpt_info["run"],
            "epoch": ckpt_info.get("epoch"), "threshold": float(threshold),
            "min_lesion_mm3": float(min_lesion_mm3), "cases": records, "summary": summary}


def summarize_fp(records) -> dict:
    """把假阳性队列压成汇总数字（``status != ok`` 时调用方直接用状态串）。"""
    usable = [rec for rec in (records or []) if rec.get("fp_clean") is not None]
    if not usable:
        return {"status": "无可用病例", "n_cases": 0}

    def stats(values) -> dict:
        array = np.asarray([float(v) for v in values], dtype=np.float64)
        return {"mean": round(float(array.mean()), 3), "std": round(float(array.std(ddof=1)), 3)
                if array.size > 1 else 0.0,
                "min": round(float(array.min()), 3), "max": round(float(array.max()), 3),
                "median": round(float(np.median(array)), 3),
                "p95": round(float(np.percentile(array, 95)), 3)}

    with_lesion = [rec for rec in usable if int(rec["fp_clean"]["n_lesions"]) > 0]
    return {
        "status": "ok",
        "n_cases": int(len(usable)),
        "cases": [int(rec["case"]) for rec in usable],
        "n_cases_with_lesion": int(len(with_lesion)),
        "rate_with_lesion": round(float(len(with_lesion) / len(usable)), 4),
        "n_cases_with_any_fp_voxels": int(sum(1 for rec in usable
                                              if int(rec["fp_clean"]["fp_voxels"]) > 0)),
        "fp_volume_mm3": stats([rec["fp_raw"]["fp_mm3"] for rec in usable]),
        "fp_volume_mm3_clean": stats([rec["fp_clean"]["fp_mm3"] for rec in usable]),
        "fp_lesions_raw": stats([rec["fp_raw"]["n_lesions"] for rec in usable]),
        "fp_lesions_clean": stats([rec["fp_clean"]["n_lesions"] for rec in usable]),
        "largest_lesion_mm3": stats([rec["fp_clean"]["largest_mm3"] for rec in usable]),
        "checkpoint": usable[0].get("checkpoint"),
        "run": usable[0].get("run"),
    }


# --------------------------------------------------------------------------------------
# 汇总与报告
# --------------------------------------------------------------------------------------

def build_summary(*, cfg, args, folds: list, fold_reports: list, skipped: list,
                  fp_report: dict, cache_dir, out_dir: Path, minutes: float) -> dict:
    """把逐折报告合并成总报告 dict（JSON）与 markdown。"""
    all_cases = [rec for report in fold_reports for rec in report["cases"]]
    tumor_cases = [rec for rec in all_cases if rec["is_tumor"]]
    clean = [rec["voxel_clean"] for rec in all_cases]
    raw = [rec["voxel_raw"] for rec in all_cases]
    spacing = voxel_spacing(cfg)

    from src.metrics import confusion_summary, detection_stats   # 局部导入（同上）

    totals_clean = {key: int(sum(int(rec[key]) for rec in clean))
                    for key in ("tp", "fp", "fn", "pred_voxels", "gt_voxels")}
    totals_raw = {key: int(sum(int(rec[key]) for rec in raw))
                  for key in ("tp", "fp", "fn", "pred_voxels", "gt_voxels")}
    per_case = summarize([float(rec["dice"]) for rec in clean])
    macro = {key: summarize([float(rec[key]) for rec in clean])
             for key in ("dice", "iou", "precision", "recall")}
    macro_raw = {key: summarize([float(rec[key]) for rec in raw])
                 for key in ("dice", "iou", "precision", "recall")}

    fold_rows = []
    for report in fold_reports:
        assessment = report["assessment"]
        fold_rows.append({
            "fold": int(report["fold"]),
            "run": report["run"],
            "checkpoint": report["checkpoint"]["path"],
            "epoch": report["checkpoint"].get("epoch"),
            "n_cases": assessment["n_cases"],
            "dice_mean": assessment["dice_mean"],
            "dice_std": assessment["dice_std"],
            "dice_min": assessment["dice_min"],
            "dice_max": assessment["dice_max"],
            "iou_mean": assessment["macro_clean"]["iou"]["mean"],
            "precision_mean": assessment["macro_clean"]["precision"]["mean"],
            "recall_mean": assessment["macro_clean"]["recall"]["mean"],
            "pooled_dice": assessment["pooled_clean"]["dice"],
            "pooled_iou": assessment["pooled_clean"]["iou"],
            "pooled_precision": assessment["pooled_clean"]["precision"],
            "pooled_recall": assessment["pooled_clean"]["recall"],
            "detection_rate": assessment["lesion_clean"].get("detection_rate", 0.0),
            "n_gt_lesions": assessment["lesion_clean"].get("n_gt", 0),
            "n_detected_lesions": assessment["lesion_clean"].get("n_detected", 0),
            "n_covered": assessment["lesion_clean"].get("n_covered", 0),
            "mean_overlap_frac": assessment["lesion_clean"].get("mean_overlap_frac", 0.0),
            "criterion_b_is_trivial": bool(assessment["lesion_clean"].get("criterion_b_is_trivial")),
            "gt_volume_mm3": assessment["gt_volume_mm3"],
            "pred_volume_mm3": assessment["pred_volume_mm3_clean"],
            "sec_per_case": (round(float(assessment["infer_seconds"]) / max(1, assessment["n_cases"]), 2)),
        })

    summary = {
        "schema": REPORT_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(),
        "stage": "第 5 轮（整卷推理 → 3D 后处理 → 指标 → 评估报告）",
        "config": rel_to_root(resolve_path(args.config)),
        "config_overrides": list(args.overrides or []),
        "cfg_hash": config_fingerprint(cfg_for_hash(cfg)),
        "preprocess_cfg_hash": cache_fingerprint(cfg),
        "seed": int(((cfg.get("train") or {}).get("seed", 42))),
        "cache_dir": rel_to_root(cache_dir),
        "reports_dir": rel_to_root(out_dir),
        "splits": rel_to_root(resolve_path(((cfg.get("paths") or {}).get("splits", "data/splits.json")))),
        "splits_digest": file_digest(resolve_path(((cfg.get("paths") or {}).get("splits",
                                                                               "data/splits.json")))),
        "folds_requested": [int(f) for f in folds],
        "folds_evaluated": [int(report["fold"]) for report in fold_reports],
        "folds_skipped": skipped,
        "incomplete": bool(args.limit_cases) or bool(skipped),
        "settings": {
            "threshold": float(resolve_threshold(cfg, None)),
            "min_lesion_mm3": float(((cfg.get("eval") or {}).get("min_lesion_mm3", 50.0)) or 0.0),
            "detect_min_mm3": float(((cfg.get("eval") or {}).get("detect_min_mm3",
                                                                 DEFAULT_DETECT_MIN_MM3)) or 0.0),
            "spacing": [float(v) for v in spacing],
            "size_bins": [str(name) for _, _, name in DEFAULT_SIZE_BINS],
            "overlap_bins": [float(v) for v in OVERLAP_BINS],
            "z_context": int(data_config(cfg).get("z_context", 0) or 0),
            "target_hw": [int(v) for v in data_config(cfg).get("target_hw", [])],
            "save_pred": bool(args.save_pred),
            "save_raw": bool(args.save_raw),
            "limit_cases": int(args.limit_cases or 0),
        },
        "aggregate": {
            "n_cases": int(len(all_cases)),
            "n_cases_tumor": int(len(tumor_cases)),
            "n_folds": int(len(fold_reports)),
            "macro_clean": macro,
            "macro_raw": macro_raw,
            "pooled_clean": confusion_summary({**totals_clean, "cases": len(all_cases)}, spacing),
            "pooled_raw": confusion_summary({**totals_raw, "cases": len(all_cases)}, spacing),
            "per_case_clean": per_case,
            "lesion_clean": detection_stats([rec["lesion_clean"] for rec in tumor_cases]),
            "lesion_raw": detection_stats([rec["lesion_raw"] for rec in tumor_cases]),
            "voxel_seconds": round(float(sum(float(rec["infer_seconds"]) for rec in all_cases)), 3),
            "wall_minutes": round(float(minutes), 2),
        },
        "folds": fold_rows,
        "fp_cohort": fp_report,
        "cases": all_cases,
    }
    summary["aggregate"]["lesion_macro"] = {
        "detection_rate_per_case_mean": summary["aggregate"]["lesion_clean"].get(
            "detection_rate_per_case_mean", 0.0),
        "detection_rate_per_case_std": summarize(
            [float(rec["lesion_clean"].get("detection_rate", 0.0)) for rec in tumor_cases])["std"]
        if tumor_cases else 0.0,
    }
    return round_floats(summary)


def render_markdown(summary: dict) -> str:
    """把汇总 dict 渲染成 ``reports/eval_summary.md``（口径与判读见 docs/baseline.md 第 2.6 节）。"""
    aggregate = summary["aggregate"]
    pooled = aggregate["pooled_clean"]
    lesion = aggregate["lesion_clean"]
    fp = summary["fp_cohort"]
    settings = summary["settings"]
    lines: list = []
    push = lines.append

    push("# 整卷评估报告（第 5 轮）")
    push("")
    push(f"- 生成时间：{summary['generated_at']}｜代码版本：`{summary['git_commit'] or 'NA'}`")
    push(f"- 配置：`{summary['config']}`（cfg_hash `{summary['cfg_hash']}`）"
         f"｜预处理指纹 `{summary['preprocess_cfg_hash']}`｜种子 {summary['seed']}")
    push(f"- 评估折：**{summary['folds_evaluated']}**"
         + (f"；**跳过 {summary['folds_skipped']}**" if summary["folds_skipped"] else "")
         + f"｜病例 {aggregate['n_cases']} 例（含肿瘤 {aggregate['n_cases_tumor']} 例）")
    push(f"- 口径：阈值 {settings['threshold']}；后处理删 <{settings['min_lesion_mm3']} mm³ 孤立块；"
         f"病灶检出 = 与 GT 有重叠 或 该 GT 病灶 ≥{settings['detect_min_mm3']} mm³；"
         f"体素 {tuple(settings['spacing'])} mm")
    if summary["incomplete"]:
        push("- ⚠️ **本轮评估不完整**（有折被跳过，或用了 `--limit-cases`）："
             "下面的数字只能用于验证链路，不能作为基线结论。")
    push("")
    push("## 1. 逐折结果（后处理之后；macro = 逐例等权）")
    push("")
    push("| 折 | run | epoch | 病例 | Dice mean±std | Dice min/max | IoU | 精确率 | 召回率 "
         "| 检出率 | 覆盖≥0.2 | 秒/例 |")
    push("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary["folds"]:
        push(f"| {row['fold']} | `{row['run']}` | {row['epoch']} | {row['n_cases']} "
             f"| **{row['dice_mean']:.4f} ± {row['dice_std']:.4f}** "
             f"| {row['dice_min']:.4f} / {row['dice_max']:.4f} "
             f"| {row['iou_mean']:.4f} | {row['precision_mean']:.4f} | {row['recall_mean']:.4f} "
             f"| {row['n_detected_lesions']}/{row['n_gt_lesions']} "
             f"| {row['n_covered']}/{row['n_gt_lesions']} "
             f"| {row['sec_per_case']:.2f} |")
    if not summary["folds"]:
        push("| — | — | — | — | — | — | — | — | — | — | — |")
    push("")
    push("## 2. 汇总（macro 与池化两种口径都给）")
    push("")
    push("| 口径 | Dice | IoU | 精确率 | 召回率 | 说明 |")
    push("| --- | --- | --- | --- | --- | --- |")
    push(f"| macro（逐例等权，后处理之后） | {aggregate['macro_clean']['dice']['mean']:.4f} "
         f"± {aggregate['macro_clean']['dice']['std']:.4f} | "
         f"{aggregate['macro_clean']['iou']['mean']:.4f} | "
         f"{aggregate['macro_clean']['precision']['mean']:.4f} | "
         f"{aggregate['macro_clean']['recall']['mean']:.4f} | 与训练期早停同口径，"
         f"{aggregate['per_case_clean']['n']} 例 |")
    push(f"| macro（后处理之前） | {aggregate['macro_raw']['dice']['mean']:.4f} "
         f"± {aggregate['macro_raw']['dice']['std']:.4f} | "
         f"{aggregate['macro_raw']['iou']['mean']:.4f} | "
         f"{aggregate['macro_raw']['precision']['mean']:.4f} | "
         f"{aggregate['macro_raw']['recall']['mean']:.4f} | 原始预测，供对照 |")
    push(f"| 池化（所有体素合并，后处理之后） | {pooled['dice']:.4f} | {pooled['iou']:.4f} "
         f"| {pooled['precision']:.4f} | {pooled['recall']:.4f} | 由大病灶主导 |")
    push("")
    if lesion.get("status") == "ok":
        push(f"**病灶级检出（后处理之后）**：{lesion['n_detected']}/{lesion['n_gt']} = "
             f"**{lesion['detection_rate']:.3f}**（逐例平均 {lesion['detection_rate_per_case_mean']:.3f}）；"
             f"漏检 {lesion['n_missed']} 个。"
             f"只看 ≥{settings['detect_min_mm3']} mm³ 的病灶："
             f"{lesion['n_detected_big']}/{lesion['n_gt_big']} = {lesion['detection_rate_big']:.3f}。")
        if lesion.get("criterion_b_is_trivial"):
            push("")
            push(f"> ⚠️ **上面这个检出率当前没有区分度**：{lesion['n_gt_trivially_detected']}/"
                 f"{lesion['n_gt']} 个 GT 病灶的体积都 ≥ `eval.detect_min_mm3`"
                 f"（{settings['detect_min_mm3']} mm³）⇒ 「体积达标即算检出」这条判据恒真，"
                 f"**模型一个体素都不预测也会报 {lesion['n_gt']}/{lesion['n_gt']}**。"
                 f"请只看下面的重叠比例分档（第 2.2 节）与逐例 Dice。")
        push("")
        push(f"**病灶覆盖质量（每病灶 ``交集 / GT 病灶体素``，= 病灶级灵敏度）**："
             f"均值 **{lesion['mean_overlap_frac']:.3f}**、中位 {lesion['median_overlap_frac']:.3f}；"
             f"有任意重叠 {lesion['n_hit_overlap']}/{lesion['n_gt']}、"
             f"覆盖 ≥{lesion.get('cover_frac', 0.2):.2f} 的 {lesion['n_covered']}/{lesion['n_gt']}、"
             f"覆盖 ≥{lesion.get('cover_frac_strict', 0.5):.2f} 的 "
             f"{lesion['n_covered_strict']}/{lesion['n_gt']}。")
    else:
        push(f"**病灶级检出**：{lesion.get('status')}")
    push("")
    push("### 2.1 按病灶体积分档（后处理之后）")
    push("")
    push("| 体积档 (mm³) | GT 病灶 | 判据检出 | 检出率 | 有任意重叠 | 重叠率 |")
    push("| --- | --- | --- | --- | --- | --- |")
    by_bin = lesion.get("by_size_bin") or {}
    for name in settings["size_bins"]:
        slot = by_bin.get(name)
        if slot:
            push(f"| {name} | {slot['n_gt']} | {slot['n_detected']} | {slot['rate']:.3f} "
                 f"| {slot['n_hit_overlap']} | {slot['overlap_rate']:.3f} |")
        else:
            push(f"| {name} | 0 | 0 | — | 0 | — |")
    push("")
    push("### 2.2 按病灶覆盖比例分档（判据恒真时**只看这张表**）")
    push("")
    push("| 覆盖比例 = 交集/GT | GT 病灶 | 占比 |")
    push("| --- | --- | --- |")
    grades = lesion.get("by_overlap_frac") or []
    for grade in grades:
        high = "1.0" if grade["high"] is None else f"{grade['high']:.2f}"
        label = f"[{grade['low']:.2f}, {high})"
        push(f"| {label} | {grade['n_lesions']} | {grade['rate']:.3f} |")
    if not grades:
        push("| — | 0 | — |")
    push("")
    push("> 解读：覆盖比例落在 `[0.00, 0.10)` 基本等于「模型撒的大片正好压到了一点」；"
         "要到 `[0.50, 1.0)` 才算真的把病灶标出来了。"
         "`mean_overlap_frac` 与逐例 Dice 一起看，才能区分「没找到」和「找歪了」。")
    push("")
    push("## 3. 逐例指标（后处理之后）")
    push("")
    rows = []
    for rec in summary["cases"]:
        cover = rec.get("best_cover")
        rows.append({
            "case": rec["case"],
            "gt_mm3": rec["gt_mm3"] if rec["is_tumor"] else "仅肝脏",
            "dice": round(float(rec["voxel_clean"]["dice"]), 4),
            "iou": round(float(rec["voxel_clean"]["iou"]), 4),
            "prec": round(float(rec["voxel_clean"]["precision"]), 4),
            "recall": round(float(rec["voxel_clean"]["recall"]), 4),
            "pred_mm3": rec["voxel_clean"]["pred_voxels"],
            "gt_lesions": rec["lesion_clean"]["n_gt"],
            "detected": rec["lesion_clean"]["n_detected"],
            "best_cover": (None if cover is None else round(float(cover), 3)),
            "fp_mm3": round(float(rec["fp_clean"]["fp_mm3"]), 1),
            "fp_lesions": rec["fp_clean"]["n_lesions"],
        })
    push(format_kv_table(rows, CASE_COLUMNS))
    push("")
    push("> `pred_mm3` / `fp_mm3` 在 spacing=(1,1,1) 下与体素数相同；"
         "`detected` 的分母 `gt_lesions` 是**该例的 GT 病灶数**（不是体素级的）；"
         "`best_cover` = 该例**覆盖得最好的那个** GT 病灶的 `交集/GT`（有病灶时才给），"
         "把它和 `dice` 并排看就能区分「没找到」（两者都低）与「找歪了/撒太大」（cover 高、dice 低）。")
    push("")
    push("## 4. 仅肝脏病人假阳性")
    push("")
    if fp.get("status") == "ok":
        fp_sum = fp["summary"]
        push(f"病例 {'/'.join(str(c) for c in fp_sum['cases'])}（共 {fp_sum['n_cases']} 例，"
             f"GT 全空）｜权重 `{fp['run']}`（epoch {fp.get('epoch')}）")
        push("")
        push(f"- 预测出现孤立病灶的病例：**{fp_sum['n_cases_with_lesion']}/{fp_sum['n_cases']} "
             f"= {fp_sum['rate_with_lesion']:.3f}**（约束假阳性看这一项）")
        push(f"- 逐例假阳性体积（后处理之后）：均值 {fp_sum['fp_volume_mm3_clean']['mean']} mm³、"
             f"中位 {fp_sum['fp_volume_mm3_clean']['median']} mm³、"
             f"最大 {fp_sum['fp_volume_mm3_clean']['max']} mm³"
             f"（后处理之前：均值 {fp_sum['fp_volume_mm3']['mean']} mm³、"
             f"最大 {fp_sum['fp_volume_mm3']['max']} mm³）")
        push(f"- 逐例孤立块个数（后处理之后）：均值 {fp_sum['fp_lesions_clean']['mean']}、"
             f"最大 {fp_sum['fp_lesions_clean']['max']}；最大单块体积 "
             f"{fp_sum['largest_lesion_mm3']['max']} mm³")
        push(f"- 后处理删掉 {fp_sum['fp_lesions_raw']['mean']} 块/例（均值）")
        push("")
        push("> 这 5 例不在任何折的验证集里（跨轮约束：只进训练集），所以只能用**一个**权重评估 ——"
             "上面的数字是「该模型在无病灶病人上的假阳性」，不是 5 折交叉验证结果。")
    else:
        push(f"{fp.get('status')}")
    push("")
    push("## 5. 后处理效果")
    push("")
    push("| 项 | 后处理之前 | 后处理之后 |")
    push("| --- | --- | --- |")
    push(f"| 汇总 Dice（池化） | {aggregate['pooled_raw']['dice']:.4f} "
         f"| {aggregate['pooled_clean']['dice']:.4f} |")
    push(f"| macro Dice | {aggregate['macro_raw']['dice']['mean']:.4f} "
         f"| {aggregate['macro_clean']['dice']['mean']:.4f} |")
    push(f"| macro 精确率 | {aggregate['macro_raw']['precision']['mean']:.4f} "
         f"| {aggregate['macro_clean']['precision']['mean']:.4f} |")
    push(f"| macro 召回率 | {aggregate['macro_raw']['recall']['mean']:.4f} "
         f"| {aggregate['macro_clean']['recall']['mean']:.4f} |")
    lesion_raw = aggregate.get("lesion_raw") or {}
    if lesion_raw.get("status") == "ok" and lesion.get("status") == "ok":
        push(f"| 病灶检出率（判据） | {lesion_raw['detection_rate']:.3f} "
             f"({lesion_raw['n_detected']}/{lesion_raw['n_gt']}) "
             f"| {lesion['detection_rate']:.3f} ({lesion['n_detected']}/{lesion['n_gt']}) |")
        push(f"| 病灶覆盖 ≥{lesion.get('cover_frac', 0.2):.2f}（有区分度） "
             f"| {lesion_raw['n_covered']}/{lesion_raw['n_gt']} "
             f"| {lesion['n_covered']}/{lesion['n_gt']} |")
        push(f"| 平均覆盖比例 | {lesion_raw['mean_overlap_frac']:.3f} "
             f"| {lesion['mean_overlap_frac']:.3f} |")
    push("")
    push("各折后处理明细见 `reports/eval_fold<k>.json` 的 `assessment.postprocess`"
         "（逐例删了几块、删掉多少 mm³ 见 `cases[].postprocess`）。")
    push("")
    push("## 6. 生成命令与产物")
    push("")
    push("```bash")
    push(evaluate_command(summary))
    push("```")
    push("")
    push(f"- 逐折报告：`reports/eval_fold{{{'|'.join(str(f) for f in summary['folds_evaluated'])}}}.json`")
    push("- 汇总报告：`reports/eval_summary.json` + 本文件")
    push(f"- 推理耗时合计 {aggregate['voxel_seconds']:.1f} s，"
         f"整次评估墙钟 {aggregate['wall_minutes']:.1f} 分钟")
    push("")
    return "\n".join(lines) + "\n"


def evaluate_command(summary: dict) -> str:
    """复原一条可复制的 ``python -m src.evaluate`` 命令（报告里要能自证是怎么跑出来的）。"""
    folds = [int(f) for f in summary["folds_evaluated"]]
    settings = summary["settings"]
    parts = ["python -m src.evaluate"]
    runs = {row["run"] for row in summary["folds"]}
    if len(folds) == 1:
        parts.append(f"--fold {folds[0]}")
        if runs and next(iter(runs)) != f"fold{folds[0]}":
            parts.append(f"--run-dir runs/{next(iter(runs))}")
    else:
        parts.append("--folds " + ",".join(str(f) for f in folds))
        if any(row["run"] != f"fold{row['fold']}" for row in summary["folds"]):
            parts.append("# 各折权重来自非默认 run 目录，逐折重跑时需分别指定 --run-dir")
    if summary.get("config_overrides"):
        parts.extend(f"--set {item}" for item in summary["config_overrides"])
    if settings.get("save_pred"):
        parts.append("--save-pred")
    if settings.get("save_raw"):
        parts.append("--save-raw")
    if settings.get("limit_cases"):
        parts.append(f"--limit-cases {settings['limit_cases']}")
    return " ".join(parts)


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def error_hint(exc: BaseException) -> str:
    """把异常映射成一句「先看哪里」的处置提示（贴进日志，省一次来回）。"""
    text = f"{type(exc).__name__}: {exc}"
    if "形状不一致" in str(exc):
        return ("预测卷与 GT 卷形状不一致 → 轴序或 pad_offset 裁回出错，"
                "按 docs/baseline.md 第 3 节贴整段 traceback（最优先修的一类）。")
    if "not found" in str(exc).lower() or isinstance(exc, FileNotFoundError):
        return ("缺文件（checkpoint / cache）→ 先训练或用 --run-dir/--ckpt 指到实际产物；"
                "cache 缺失先跑 preprocess.py。")
    if "不一致" in str(exc):
        return ("权重与配置/折号对不上 → 用训练时的同一份配置与 --fold；"
                "确要强行评估可加 --skip-config-check（报告会记下）。")
    if isinstance(exc, KeyError):
        return "报告字段缺失（多半是 checkpoint 版本较旧）→ 贴回该 traceback。"
    return f"未归类的运行期错误 → 贴回该行与前后 20 行。原始信息：{text[:200]}"


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    cfg = load_config(args.config, args.overrides)
    if args.cache_dir:
        cfg.setdefault("paths", {})["cache"] = args.cache_dir
    set_seed(int(((cfg.get("train") or {}).get("seed", 42))))
    device = resolve_device(args.device)
    paths = cfg.get("paths") or {}
    cache_dir = resolve_path(paths.get("cache", "cache"))
    out_dir = resolve_path(args.out_dir or paths.get("reports", "reports"))
    data_cfg = data_config(cfg)
    folds = parse_folds(args)          # 只解析一次：后面定位 checkpoint、写报告都用它

    LOGGER.info("=" * 78)
    LOGGER.info("第 5 轮整卷评估：%s｜设备 %s｜阈值 %s｜后处理删 <%s mm³",
                rel_to_root(resolve_path(args.config)), device,
                ((cfg.get("eval") or {}).get("threshold", 0.5)),
                ((cfg.get("eval") or {}).get("min_lesion_mm3", 50.0)))
    LOGGER.info("输入：%s（2.5D %s 通道）→ 输出 %s",
                format_hw(data_cfg.get("target_hw", (512, 512))),
                data_cfg.get("z_context", 0), rel_to_root(out_dir))
    LOGGER.info("=" * 78)

    # ---- 前置校验（复用 src.train 的口径：splits / manifest / cache 文件） ----
    info, problems = check_prerequisites(cfg, folds[0])
    if problems:
        for item in problems:
            LOGGER.error("  - %s", item)
        LOGGER.error("前置校验失败：发现 %d 个问题，评估未启动（处置见 docs/baseline.md 第 3 节）",
                     len(problems))
        return EXIT_PREREQ
    splits = info["splits"]
    LOGGER.info("前置校验通过：cache %s / splits %s（指纹 %s）",
                rel_to_root(cache_dir), info["splits_path"], info.get("splits_digest") or "NA")

    # ---- 定位各折的 checkpoint ----
    plan: list = []
    skipped: list = []
    for fold in folds:
        record = next((r for r in splits.get("folds", []) if int(r.get("fold", -1)) == int(fold)), None)
        if record is None:
            skipped.append({"fold": int(fold), "reason": f"splits 里没有 fold {fold}"})
            continue
        try:
            ckpt = resolve_checkpoint(args, cfg, fold)
        except Exception as exc:  # noqa: BLE001 - 路径解析失败按「跳过」处理并说明原因
            skipped.append({"fold": int(fold), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if not ckpt.exists():
            reason = (f"找不到 {rel_to_root(ckpt)}：先训练 python -m src.train --fold {fold}"
                      f"，或用 --run-dir 指向实际产物（开发期可用 runs/smoke_fold0_fixed）")
            if len(folds) > 1:
                LOGGER.warning("fold %d 跳过：%s", fold, reason)
                skipped.append({"fold": int(fold), "reason": reason})
                continue
            LOGGER.error("fold %d：%s", fold, reason)
            return EXIT_PREREQ
        plan.append({"fold": int(fold), "record": record, "ckpt": ckpt})

    if not plan:
        LOGGER.error("没有任何一折可以评估：%s", skipped)
        return EXIT_PREREQ
    LOGGER.info("评估计划：%s", "；".join(
        f"fold {item['fold']} ← {rel_to_root(item['ckpt'])}（val {len(item['record'].get('val', []))} 例）"
        for item in plan))
    if skipped:
        LOGGER.warning("跳过的折：%s", skipped)

    if args.dry_run:
        for item in plan:
            LOGGER.info("--dry-run：fold %d / val %s / 权重 %s / 报告 %s",
                        item["fold"], sorted(int(c) for c in item["record"].get("val", [])),
                        rel_to_root(item["ckpt"]),
                        rel_to_root(out_dir / f"eval_fold{item['fold']}.json"))
        fp_cases = [int(c) for c in (splits.get("liver_only_cases") or [])]
        LOGGER.info("--dry-run：仅肝脏假阳性队列 %s（%s）", fp_cases,
                    "跳过（--skip-fp）" if args.skip_fp else "用 fold %d 的权重" % plan[0]["fold"])
        LOGGER.info("--dry-run 结束：没有做任何推理，也没有写报告。")
        return EXIT_OK

    # ---- 逐折评估 ----
    try:
        fold_reports: list = []
        for item in plan:
            report = run_fold(cfg=cfg, fold=item["fold"], ckpt_path=item["ckpt"],
                              cache_dir=cache_dir, split_record=item["record"], args=args,
                              device=device)
            fold_reports.append(report)
            save_report({"schema": REPORT_SCHEMA, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "git_commit": git_commit(), "settings": {"threshold": report["threshold"],
                                                                  "min_lesion_mm3": report["min_lesion_mm3"],
                                                                  "detect_min_mm3": report["detect_min_mm3"],
                                                                  "spacing": report["spacing"]},
                         **report},
                        out_dir / f"eval_fold{item['fold']}.json")
            LOGGER.info("fold %d 报告已写入 %s", item["fold"],
                        rel_to_root(out_dir / f"eval_fold{item['fold']}.json"))

        # ---- 仅肝脏病人的假阳性（单独一个权重，报告里标注来源） ----
        fp_report: dict = {"status": "跳过（--skip-fp）", "cases": []}
        if not args.skip_fp:
            fp_cases = [int(c) for c in (splits.get("liver_only_cases") or [])]
            fp_ckpt = resolve_path(args.fp_ckpt) if args.fp_ckpt else plan[0]["ckpt"]
            if not fp_cases:
                fp_report = {"status": "跳过（splits 里没有 liver_only_cases）", "cases": []}
            elif not Path(fp_ckpt).exists():
                fp_report = {"status": f"跳过（找不到 {rel_to_root(fp_ckpt)}）", "cases": []}
                LOGGER.warning("仅肝脏假阳性统计跳过：%s", fp_report["status"])
            else:
                fp_report = run_fp_cohort(cfg=cfg, ckpt_path=fp_ckpt, cache_dir=cache_dir,
                                          cases=fp_cases, device=device, args=args)
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        LOGGER.error("评估失败（%s）：%s", type(exc).__name__, exc)
        LOGGER.error("%s", error_hint(exc))
        LOGGER.error("不要让这样的报告进入结论（处置口径见 docs/baseline.md 第 3 节）。")
        return EXIT_ERROR

    minutes = (time.perf_counter() - started) / 60.0
    summary = build_summary(cfg=cfg, args=args, folds=folds, fold_reports=fold_reports,
                            skipped=skipped, fp_report=fp_report, cache_dir=cache_dir,
                            out_dir=out_dir, minutes=minutes)
    json_path, md_path = save_report(summary, out_dir / "eval_summary.json", md_builder=render_markdown)
    LOGGER.info("=" * 78)
    aggregate = summary["aggregate"]
    LOGGER.info("评估结束：%d 折 / %d 例（含肿瘤 %d 例）；macro Dice %.4f ± %.4f，"
                "池化 Dice %.4f；病灶检出 %s/%s",
                aggregate["n_folds"], aggregate["n_cases"], aggregate["n_cases_tumor"],
                aggregate["macro_clean"]["dice"]["mean"], aggregate["macro_clean"]["dice"]["std"],
                aggregate["pooled_clean"]["dice"],
                (aggregate["lesion_clean"] or {}).get("n_detected", 0),
                (aggregate["lesion_clean"] or {}).get("n_gt", 0))
    LOGGER.info("报告：%s（JSON）与 %s（markdown）", rel_to_root(json_path),
                rel_to_root(md_path) if md_path else "NA")
    LOGGER.info("下一步：按病灶大小分层的检出率决定第 6 轮先动哪个杠杆"
                "（见 docs/todo.md 第 6 轮）。")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
