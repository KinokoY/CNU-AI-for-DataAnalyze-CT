"""把压缩缓存 ``.nii.gz`` 就地转成未压缩 ``.nii``（**不改目录结构、不动原始数据**）。

为什么需要它（远程实测，第 3 轮 --debug 之后）：
    缓存写成 ``.nii.gz`` 时 nibabel 无法 mmap（未装 indexed_gzip），``a[:, :, z]`` 会把**整卷解压一遍**：
    **557 ms/层**。后果是训练取数的纯开销 8.7 分钟/epoch（GPU 只要 1.2 分钟）、
    整卷推理 76 s/例（4 例验证 5 分钟/epoch）。转成未压缩 ``.nii`` 后 nibabel 走 mmap，
    逐层读取变成页缓存读：**0.00 ms/层**，与文件大小无关。

做什么：
    对 ``<cache>/image/<case>.nii.gz`` 与 ``<cache>/label/<case>.nii.gz``
    → 在**同一个子目录**写 ``<case>.nii``（文件名只差扩展名），
    然后逐个体素比对（shape / dtype / affine / 全数组相等）确认无损；``--remove-gz`` 再删掉压缩原件。

安全约定：
    * 默认**不覆盖**已存在的 ``.nii``（要覆盖得显式 ``--overwrite``）；
    * 校验不通过就报错退出并且不删任何东西；
    * ``--dry-run`` 只打印计划；``--cases 33,57`` 只处理指定病例；
    * 只读写 ``cache/``，不碰 ``data/`` 下的原始影像。

前后接口：上游是 ``scripts/preprocess.py``（老版本写的 ``.nii.gz``）；
    下游是所有读取端——``src/dataset.py`` / ``src/infer.py`` 已按 ``src.utils.cache_file``
    「``.nii`` 优先、兼容 ``.nii.gz``」解析，所以转换后**不需要改配置、不需要重跑预处理、
    指纹与清单都保持不变**。
用法：仓库根目录执行 ``python scripts/inflate_cache.py --remove-gz``（先看一眼再加 --remove-gz）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    from src.utils import (
        cache_cases,
        cache_file,
        load_config,
        rel_to_root,
        resolve_path,
        setup_logger,
    )
except ModuleNotFoundError:  # pragma: no cover - 兜底：把仓库根塞进 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import (  # type: ignore
        cache_cases,
        cache_file,
        load_config,
        rel_to_root,
        resolve_path,
        setup_logger,
    )

LOGGER = setup_logger("inflate_cache")

#: 参与转换的子目录（顺序固定，日志好读）；目录结构保持不变
KINDS = ("image", "label")

#: ``.gz`` 之外可能残留的索引文件（装过 indexed_gzip 时会有），删原件时一并清掉
SIDECAR_SUFFIXES = (".gzidx",)


def inflate_one(src: Path, dst: Path, overwrite: bool = False) -> dict:
    """把 ``src``（.nii.gz）转成未压缩的 ``dst``（.nii），返回统计 dict；失败抛异常。

    逐体素校验：shape / dtype / affine / 整数组相等。任何一项不符就报错，绝不删原件。
    """
    import nibabel as nib

    if dst.exists() and not overwrite:
        return {"status": "skip", "reason": "目标已存在（加 --overwrite 可覆盖）"}
    src_bytes = src.stat().st_size
    started = time.perf_counter()
    image = nib.load(str(src))
    src_arr = np.asanyarray(image.dataobj)
    src_affine = np.asarray(image.affine)
    nib.save(image, str(dst))                     # 扩展名为 .nii → nibabel 写未压缩
    written = nib.load(str(dst), mmap=True)
    dst_arr = np.asanyarray(written.dataobj)
    if tuple(dst_arr.shape) != tuple(src_arr.shape):
        raise RuntimeError(f"{dst.name}：shape {tuple(dst_arr.shape)} != 源 {tuple(src_arr.shape)}")
    if str(dst_arr.dtype) != str(src_arr.dtype):
        raise RuntimeError(f"{dst.name}：dtype {dst_arr.dtype} != 源 {src_arr.dtype}")
    if not np.array_equal(np.asarray(written.affine), src_affine):
        raise RuntimeError(f"{dst.name}：affine 与源不一致")
    if not np.array_equal(dst_arr, src_arr):
        raise RuntimeError(f"{dst.name}：体素与源不一致（{int((dst_arr != src_arr).sum())} 个不同）")
    return {
        "status": "ok",
        "src_bytes": int(src_bytes),
        "dst_bytes": int(dst.stat().st_size),
        "voxels": int(dst_arr.size),
        "dtype": str(dst_arr.dtype),
        "shape": [int(s) for s in dst_arr.shape],
        "seconds": round(time.perf_counter() - started, 3),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="把 cache 里的 .nii.gz 就地转成未压缩 .nii（nibabel 可 mmap，逐层读取快两个数量级）")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件")
    parser.add_argument("--cache-dir", default=None, help="覆盖 cache/ 位置")
    parser.add_argument("--cases", default=None, help="只处理这些病例，如 33,57（默认全部）")
    parser.add_argument("--remove-gz", action="store_true", help="校验通过后删掉 .nii.gz 原件（省磁盘）")
    parser.add_argument("--overwrite", action="store_true", help="目标 .nii 已存在时覆盖它")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写任何文件")
    parser.add_argument("--set", dest="overrides", action="append", default=None, help="覆盖配置项")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    paths = cfg.get("paths", {}) or {}
    cache_dir = resolve_path(args.cache_dir or paths.get("cache", "cache"))
    if not cache_dir.is_dir():
        LOGGER.error("cache 目录不存在：%s（先跑 python scripts/preprocess.py）", rel_to_root(cache_dir))
        return 2

    wanted = None
    if args.cases:
        wanted = {int(part) for part in str(args.cases).replace(" ", "").split(",") if part}
    LOGGER.info("=" * 78)
    LOGGER.info("把压缩缓存转成未压缩 .nii：%s（%s）", rel_to_root(cache_dir),
                "dry-run，不写文件" if args.dry_run else
                ("转换后删除 .nii.gz" if args.remove_gz else "保留 .nii.gz"))
    LOGGER.info("=" * 78)

    problems: list = []
    n_ok = n_skip = n_plain = 0
    total_src = total_dst = 0
    for kind in KINDS:
        ids = cache_cases(cache_dir, kind)
        if wanted is not None:
            ids = [c for c in ids if c in wanted]
        compressed = [c for c in ids if str(cache_file(cache_dir, kind, c)).endswith(".gz")]
        n_plain += len(ids) - len(compressed)
        LOGGER.info("%s：%d 例，其中压缩的 %d 例%s", kind, len(ids), len(compressed),
                    "" if compressed else "（无需转换）")
        for case in compressed:
            src = cache_file(cache_dir, kind, case)             # 一定是 .nii.gz
            dst = src.with_suffix("")                           # <case>.nii.gz → <case>.nii
            if args.dry_run:
                LOGGER.info("  [dry-run] %s → %s（%.0f MB）", src.name, dst.name,
                            src.stat().st_size / 2 ** 20)
                continue
            try:
                info = inflate_one(src, dst, overwrite=bool(args.overwrite))
            except Exception as exc:  # noqa: BLE001 - 单文件失败不中断整体
                problems.append(f"{src.name}: {type(exc).__name__}: {exc}")
                LOGGER.error("  %s 转换失败：%s", src.name, exc)
                continue
            if info["status"] == "skip":
                n_skip += 1
                LOGGER.info("  %s 跳过：%s", src.name, info["reason"])
                continue
            n_ok += 1
            total_src += int(info["src_bytes"])
            total_dst += int(info["dst_bytes"])
            LOGGER.info("  %s → %s：%.1f MB → %.1f MB（%s %s，%.2f s）；校验通过（shape/dtype/"
                        "affine/体素全等）%s",
                        src.name, dst.name, info["src_bytes"] / 2 ** 20, info["dst_bytes"] / 2 ** 20,
                        info["dtype"], tuple(info["shape"]), info["seconds"],
                        "；已删除压缩原件" if args.remove_gz else "")
            if args.remove_gz:
                src.unlink()
                for extra in SIDECAR_SUFFIXES:
                    sidecar = Path(str(src) + extra)
                    if sidecar.exists():
                        sidecar.unlink()

    if args.dry_run:
        LOGGER.info("dry-run 结束：没有写任何文件。确认无误后去掉 --dry-run 重跑。")
        return 0

    # --remove-gz 的收尾：清掉「未压缩副本已存在」的残留 .nii.gz
    # （比如之前手动转换过、或本次因 --overwrite 未开而跳过的那些）
    if args.remove_gz:
        leftovers = 0
        for kind in KINDS:
            for gz in sorted((cache_dir / kind).glob("*.nii.gz")):
                if not gz.with_suffix("").exists():
                    continue                        # 没有未压缩副本 → 保留（可能是转换失败的）
                gz.unlink()
                leftovers += 1
                for extra in SIDECAR_SUFFIXES:
                    sidecar = Path(str(gz) + extra)
                    if sidecar.exists():
                        sidecar.unlink()
        if leftovers:
            LOGGER.info("另外清掉 %d 个「已有未压缩副本」的残留 .nii.gz", leftovers)

    LOGGER.info("-" * 78)
    LOGGER.info("转换完成：成功 %d 个，跳过 %d 个，失败 %d 个（另有 %d 个本来就是未压缩 .nii）",
                n_ok, n_skip, len(problems), n_plain)
    if n_ok:
        LOGGER.info("磁盘：压缩合计 %.2f GB → 未压缩合计 %.2f GB（未压缩约大 %.1f 倍，换来 mmap 逐层读取）",
                    total_src / 2 ** 30, total_dst / 2 ** 30,
                    (total_dst / max(1, total_src)))
    if problems:
        LOGGER.error("有 %d 个文件没转成功（原件一律保留）：", len(problems))
        for item in problems:
            LOGGER.error("  - %s", item)
        return 1

    # 性能抽检：随便挑一例刚转好的影像，量一下「逐层读取」的耗时（对比压缩缓存的 557 ms/层）
    sample = next((cache_file(cache_dir, "image", c) for c in cache_cases(cache_dir, "image")
                   if str(cache_file(cache_dir, "image", c)).endswith(".nii")), None)
    if sample is not None:
        import nibabel as nib

        image = nib.load(str(sample), mmap=True)
        array = np.asanyarray(image.dataobj)
        nz = min(32, int(array.shape[2]))
        started = time.perf_counter()
        for z in range(nz):
            _ = array[:, :, z]
        per_slice_ms = (time.perf_counter() - started) / max(1, nz) * 1000
        LOGGER.info("性能抽检（%s）：mmap 逐层读取 %.2f ms/层（压缩缓存实测 557 ms/层）",
                    sample.name, per_slice_ms)

    LOGGER.info("下一步：python -m src.selfcheck_data（确认缓存口径与指纹）；"
                "再跑 python -m src.train --fold 0 --debug 应看到取数耗时大幅下降。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
