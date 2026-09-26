========================================================
CT 肝脏肿瘤分割 · 基础版（2D 闭环）后续编码计划
（阶段一：预处理 + 划分 —— 已完成并验证，本计划不含它）
========================================================

【阶段一已确认的事实（后续代码直接依赖，勿再猜）】
- 缓存：nibabel 读回形状 (nx, ny, nz)，切片轴在最后一维（a[:, :, k] 是一层 (ny,nx) 切片）；
  SimpleITK 读回是其转置 (nz,ny,nx)，两库互为转置。影像 uint16 归一化值（/65535 得 [0,1]）；
  掩膜 uint8 只含 {0,1}；全部 1mm 各向同性。
- 面内尺寸：512×512 共 17 例；其余 8 例为 342–436 一族；最大 512。
  → **不按尺寸分桶**，统一补边到 512×512（见第 2 轮改动）。
- 切片数 nz：74–488，各例不同。
- 含肿瘤切片 766/5982 = 12.81% → 过采样基线（目标提到 batch 内 25–35%）。
- 划分：data/splits.json，5 折各 4 例验证 / 21 例训练（16 含肿瘤 + 5 仅肝脏），体积 CV 0.317。
- 5 例仅肝脏（32/34/38/41/47）只进训练集，永不进验证集。

--------------------------------------------------------
剩余 4 轮（第 3 轮已编码、待远程实测；第 4–6 轮待做），每轮都能独立运行 + 独立验证
--------------------------------------------------------

■ 第 2 轮：Dataset + 采样器 + 增强 —— 已完成（含一次口径简化），远程自检通过
  远程实测（fold 0）：21 例 / 4469 层切片、含肿瘤 617 层；batch 形状恒为 (8,1,512,512)；
  采样器 642 个 batch（阴性槽位为上界）、阳性层 617 个一轮恰好各一次、覆盖切片 4469/4469、
  全阴性 batch 25 个（当前参数下的下界）、同 epoch 可复现且相邻 epoch 不同；
  增强含肿瘤组 label 8/8 被改动、全背景组 8/8 保持全空；切片轴曲线（case 56）集中在 z≈264-439。
  交付：src/selfcheck_data.py、src/dataset.py（另在 configs/default.yaml 补了 data 节）
  【本轮改动：放弃按面内尺寸分桶，改为统一补边到 512×512】
    - 用户拍板：读取切片后把 image/label **居中补边**到 data.target_hw（默认 512×512，补 0），
      于是所有 batch 形状恒为 (B,1,512,512)，torch.stack 永远合法；
      512 是 16 的倍数 → 不需要模型内部 pad；pad_multiple / bucket_key 整条链路删除。
    - 每样本保留 orig_hw（补边前尺寸）与 pad_offset（内容在画布里的左上角）：
      第 4 轮整卷推理按 pad_offset 裁回原始面内尺寸（居中补边时内容不在原点）。
    - BucketBatchSampler → ProportionalBatchSampler：单一池 + 按比例定向抽正负样本，
      保留「阳性过采样比例控制」「一轮覆盖全部切片与全部阳性层」「同 epoch 可复现且相邻 epoch 不同」，
      删除 bucket_budget / positive_plan / 按桶配额 / 混尺寸校验。
    - 代价（写在 docs/preprocess_notes.md 6.1）：补边像素被白算；非 512 病例的补边区在增强后
      不再严格为 0，第 3 轮 loss 可考虑忽略补边区域。
  【先交付自检脚本】src/selfcheck_data.py（不依赖 torch 网络，只读缓存）
    - 跑 3 个 batch，打印：每个 batch 的 image/label shape、值域、batch 内含肿瘤切片比例实测值、
      补边样本数与补边像素占比、单 batch 耗时；
    - 再对 1 个 case 打印逐层肿瘤体素数曲线（--probe-case 可指定任意病例，含 val 侧），确认切片轴没搞反。
  src/dataset.py 接口：
    class CTSliceDataset(torch.utils.data.Dataset)
      __init__(cases, cache_dir, split, cfg, augment: bool, debug: bool=False, fold=None)
      __len__() -> int                 # 全部切片数
      __getitem__(i) -> {"image": FloatTensor[1,512,512], "label": LongTensor[512,512],
                         "case": str, "z": int, "orig_hw": (H,W)}
      属性：index = [(case, z)...]；pos_flags = bool 数组；case_hw / case_pad_offset / inplane_stats()
    def pad_to_target(arr, target_hw, align) -> (padded, (top, left))
    def pad_offset_of(orig_hw, target_hw, align) -> (top, left)   # 裁回原始尺寸的唯一口径
    def make_batch_sampler(ds, cfg, generator=None) -> BatchSampler
      # ProportionalBatchSampler：n_pos = min(bs, max(1, round(bs*pos_ratio_target)))，
      # 阳性层在整轮均摊、阴性层轮转池补满；一轮覆盖每层切片、每个阳性层恰好一次。
    def make_train_loader(splits, fold, cfg) -> DataLoader
    def make_val_loader(splits, fold, cfg) -> DataLoader   # 顺序、无增强（仍补边）
    def build_transforms(cfg, train: bool) -> ComposeSteps | None
  读取策略（避免 DataLoader 里开上千句柄）：
    __init__ 只 nib.load 读 header + label 体素（算 pos_flags/索引），不读 image 体素；
    __getitem__ 用每 worker 一份的 lru_cache(maxsize=8) 按 case 打开 nibabel 代理对象，
    memmap=True，只取 [:, :, z]。
  校验要点（必须打印/断言）：batch 形状恒为 (B,1,512,512)；值域 ∈ [0,1]；label ⊂ {0,1}；
    训练侧每批阳性数落在采样器计划区间（验证侧不判阳性数——旧版在这里误报过）。

■ 第 3 轮：模型 + 损失 + 训练 —— 已完成编码，远程 `--debug` 通过（**首折正式训练待跑**）
  交付：src/unet.py、src/losses.py、src/train.py
  src/unet.py：
    class DoubleConv2d(nn.Module)        # Conv3x3(+BN)+ReLU ×2
    class UNet2D(nn.Module)
      __init__(in_channels=1, out_channels=2, encoder_channels=(32,64,128,256),
               norm="batch", pad_to_multiple=16,
               encoder_pretrained=None, pretrained_weights_path=None)
      forward(x) -> logits [B,2,H,W]     # 内部 pad 到 16 的整数倍、输出 crop 回原 H/W
    def load_encoder_pretrained(model, name, path)   # 基础版 name=None 直接 return
    def count_parameters(model) -> int
  src/losses.py：
    def build_loss(cfg) -> DiceCELoss(softmax=True, to_onehot_y=False, batch=True,
                                      lambda_dice=1.0, lambda_ce=1.0)
  预训练方案（本版不用，但接口写死）：
    configs/default.yaml 里 model.encoder_pretrained=null、model.pretrained_weights_path=null；
    进阶版启 resnet18/34 时：本地下载 resnet18-f37072fd.pth（45MB，torchvision 官方）上传到远程
    ~/.cache/torch/hub/checkpoints/，或用 pretrained_weights_path 填绝对路径离线加载，
    首层 3 通道卷积核按通道均值压成 1 通道，解码器随机初始化——代码只优先读 path，不走网络。
  src/train.py：python -m src.train --fold 0 [--debug] [--set ...]
    - 启动即校验 data/splits.json 与 cache_manifest.json 的 case 集合/指纹，不一致直接退出；
    - AdamW(lr=1e-3, wd=1e-4) + CosineAnnealingLR + bf16 autocast + GradScaler；
      epochs=200，按"整卷肿瘤 Dice"早停 patience=20；
    - 每 epoch 的验证用整卷推理（复用第 4 轮的 infer）而不是逐层平均，避免空切片稀释；
    - 保存 runs/fold<k>/best.pt（含 model_state/epoch/metric/cfg 指纹）与 last.pt，
      写 runs/fold<k>/metrics.csv + TensorBoard；
    - 固定种子：random/numpy/torch/cuda + DataLoader generator；
    - --debug：batch_size=2、每折 2 例、3 个 iteration，打印每 batch shape、
      实测正样本比例、torch.cuda.max_memory_allocated()/reserved、单步耗时，然后退出不落盘。
      这是标定 batch_size（先按 8）的依据。
  逐折执行：for f in 0 1 2 3 4; do python -m src.train --fold $f; done

  【本轮实际交付与口径（用户拍板 + 编码时定稿，细节见 docs/baseline.md 第 4 节、
    docs/preprocess_notes.md 第七节）】
  交付文件：src/unet.py、src/losses.py、src/train.py，外加**提前落地的 src/infer.py**
    （predict_volume / seg_prob_to_label / load_label_volume）——每轮验证要整卷推理，
    第 4 轮只在它上面补 postprocess / metrics / evaluate。
  1. 网络：4 级下采样 U-Net（32/64/128/256 + 瓶颈 512），MaxPool 下采样 / ConvTranspose 上采样 +
     跳跃拼接，BatchNorm；前向按 pad_to_multiple=16 内部补 replicate 再裁回（512 下不触发）。
     预训练接口保留但基础版不实现：给 resnet18/34 直接报错说明原因，不静默忽略。
  2. 损失：手搓 DiceCELoss（softmax + CE + soft Dice），batch=True 在整 batch 上聚合，
     include_background=true、smooth=1e-5；**本轮不做补边区域 ignore mask**（第 6 轮再定）。
  3. 验证口径：每轮对 4 例做**整卷推理**，逐例算整卷肿瘤 Dice 再平均（macro，每例等权），
     用它选 best.pt 与早停；`eval.threshold=0.5`、`eval.infer_batch_slices=8`。
  4. AMP：bf16 只用 autocast，**不启用 GradScaler**（bf16 与 fp32 同指数范围，不需要 loss scaling）；
     fp16 才启用；off = 纯 fp32。跨版本入口在 src/utils.py。
  5. `--resume`：从 runs/fold<k>/last.pt 恢复 model/optimizer/scheduler/已完成轮数/best/patience，
     并用本轮新增的 ProportionalBatchSampler.set_epoch() 把采样序列接上；
     指纹只忽略 paths 与 train.epochs，其余配置改动一律拒绝续跑（退出码 4）。
  6. `--debug`：2 例 / batch_size=2（显式给 --set train.batch_size=N 就用 N）/ 3 个 iteration，
     打印 shape、实测阳性比例、分段耗时、显存峰值，再对 1 例做整卷推理自检，**不写 runs/**。
  7. 新增命令行：--fold/--debug/--debug-cases/--debug-val-cases/--debug-iters/--resume/
     --device/--out-dir/--set；退出码 0/2/3/4/130（前置校验/NaN/续跑不一致/中断）。
  8. 顺手修的一个坑：缓存指纹口径统一为 src.utils.cache_fingerprint（只取 preprocess 节）。
     原来的排除式写法会把本轮新增的 loss 节算进指纹，导致「缓存没变但训练拒绝启动」；
     现在四处（preprocess / fetch_manifest / selfcheck_data / train）共用同一函数，
     与 manifest 里已记录的 7b4c48b4dc7ef880 保持一致，远程**不需要重跑预处理**。
  9. 配置新增：loss 节（8 项）、train.min_lr_ratio / log_every / tensorboard、
     eval.threshold / infer_batch_slices。
  远程验证结果（fold 0，`--debug --set train.batch_size=16`，逐项已核对）：
    - 形状 `image (16,1,512,512)` / `label (16,512,512)` / `logits (16,2,512,512)`，值域 [0,1]、label ⊂ {0,1}；
    - 初始 loss 1.593（dice 0.683 + ce 0.909）——随机权重下的正常量级，随 step 下降；
    - 单步 0.183 s（前向 0.065 + 反向 0.117）→ 351 step ≈ 70 s/epoch；
    - 峰值显存 **10920 MB**（reserved 12152）；采样器 351 batch、每批 1~2 阳性、**全阴性 batch 0**；
    - 整卷推理 `prob/pred (512,512,135)` 与 GT 同形，GT 前景 434721 = splits.json 的 430k mm³ 口径吻合；
    - `selfcheck_data` 通过、指纹 7b4c48b4dc7ef880 不变；采样器 617/617 阳性层、覆盖 4469/4469 切片。
  仍在等待：首折前几轮的 train loss 与 val macro Dice 曲线、单 epoch 墙钟、早停轮数、5 折汇总。
  【首次正式训练（3 epoch 后 Ctrl-C）暴露的问题：损失坍缩 —— 已修】
  - 现象：`epoch 2: dice 0.4996 + ce 0.0078 = 0.507`、`epoch 3: 0.5046`，**验证整卷 Dice 4 例恒 0.0000**；
    训练"看着正常"（loss 在降、无 NaN、65 s/epoch、显存 10920 MB）但模型什么都没学到。
  - 根因：`loss.include_background=true` 使损失存在「全预测背景」的平凡最优解
    （dice 项 = 1-(1+0)/2 = 0.5，CE 可压到 ≈0），一个 batch 里肿瘤只占 0.2%~0.4%，
    背景项完全主导 → 2 个 epoch 就掉进去且出不来。
  - 修法：`loss.include_background: false`（dice 项 = 1 - 肿瘤 soft Dice，与上报指标同向）。
    配套把「dice 项贴 0.5 + ce < 0.01 = 已坍缩」写进 baseline 4.2 判读与 preprocess_notes 7.2。
  - 另修：全新开始（非 --resume）时把同目录上一轮的 best/last/metrics.csv/tensorboard 挪成 `.prev`
    （metrics.csv 是追加写的，否则重跑会出现同一个 epoch 两次）。
  - 新增长期约定：下游开发用的权重走 `--out-dir runs/smoke_fold0 --set train.epochs=12` 短跑烟测
    （见 baseline 4.2.1），不要污染 runs/fold<k>。
  【远程实测后追加的修复（第 3 轮 `--debug` 暴露）】
  - 症状：`predict_volume` 单例 76 s、训练 loader 纯取数 8.7 分钟/epoch，而 GPU 前向只要 0.019 s。
  - 根因：缓存是 `.nii.gz`，nibabel 无法 mmap，**每读一层都整卷解压**（probe 实测 557 ms/层）。
  - 修法（已实施）：缓存默认改**未压缩 `.nii`**（mmap，0.00 ms/层），读取端
    `src.utils.cache_file` 按「`.nii` 优先、`.nii.gz` 兼容」解析；
    新增 `scripts/inflate_cache.py` 把已有压缩缓存就地转换（逐体素校验后删原件，指纹/清单不变）；
    `preprocess.py` 默认写 `.nii`（`--compressed` 可退回）；dataset/infer/train/selfcheck/check_cache/
    fetch_manifest/probe_axis 七处的路径与「按文件清点病例」统一走 `cache_file` / `cache_cases`。
  - 另：`train.py` 每轮日志新增 `［取数 x s + 计算 y s］`，首轮数据成为瓶颈时会显式告警。

■ 第 4 轮：修「塌缩到全预测背景」—— 平衡采样 + Dice 只算前景 + 2.5D 三层输入（**已编码，待远程实测**）
  【为什么有这一轮】第 3 轮首折正式训练（12 轮短跑）暴露：`dice` 项 12 轮只从 0.93 降到 0.90、
  `ce` 掉到 0.009、**验证整卷 Dice 4 例恒 ≈0.000**（`33:0.001` 是偶然重叠），`prob 峰值` 只有 0.005
  —— 模型塌缩到全预测背景。根因不是管道（GT 434721 体素那条硬证据早已钉死），而是**训练分布**：
  老采样器每批只有 1~2 个含肿瘤切片、肿瘤像素占 0.2%~0.4%，CE 被背景彻底支配。
  完整复盘见 `docs/preprocess_notes.md` 8.1。
  交付：`src/dataset.py`（采样器重写 + 2.5D）、`src/losses.py`、`src/unet.py`、`src/train.py`、
  `src/selfcheck_data.py`、`configs/default.yaml`、`docs/*`（`src/infer.py` 只加了 meta 字段）
  1. **平衡采样 `BalancedBatchSampler`**（取代 `ProportionalBatchSampler`）：
     每批恒定 `n_pos = clamp(round(bs×data.pos_ratio_train), min_pos_per_batch, bs−1)` 个阳性 +
     `n_neg` 个阴性（默认 0.5 ⇒ bs=16 时 **8 正 + 8 阴**，旧口径是 1~2 正）；
     一轮规模由**槽位预算**决定（`data.epoch_samples=null` → 用数据集切片数，**epoch 墙钟与旧口径
     基本不变**）；阳性层用轮转池**重复**采样、上限 `data.max_pos_repeat`（默认 8）；
     阴性层不再要求一轮覆盖（覆盖率落到 ~36%，日志打印）。
     保留的两条：**每个 batch 都含阳性**（全阴性 batch 归零）、采样顺序仍由
     `(train.seed, epoch, 病例集合)` 派生（同 epoch 可复现、相邻 epoch 不同、`--resume` 可接上）。
     `train.pos_ratio_target` 自本轮起**不再生效**（保留键只为兼容旧命令/旧指纹，会打印提示）。
  2. **损失加 `loss.dice_positive_only: true`**：Dice 项只在 `Y.sum()>0` 的样本上聚合，
     全背景样本不参与 Dice；**CE 仍对整批所有样本算**（背景那一半继续压假阳性）。
     一个前景样本都没有时 dice 项记 0（不是 NaN）。
  3. **2.5D 三层输入**：`data.z_context=1` → `model.in_channels=3`，样本 = `[z−1, z, z+1]`，
     标签只监督中心层；越界用**端点复制**（`[0,0,1]` / `[nz−2,nz−1,nz−1]`），**绝不跨病人取层**。
     **当前实现**：中心层与相邻层先补边并叠成 `(3,H,W)`，几何增强同步作用于三通道与标签；
     强度增强（gamma/噪声）只动被监督的中心通道。此前只变换中心层、邻层不变的实现已修正。
     `z_context=0` 时退化成 `(1,H,W)`，与旧口径逐位一致。
     `predict_volume` **一行没改**（dataset 给几个通道就喂几个）。
     两处自洽校验：`unet.UNet2D.__init__` 与 `train.check_prerequisites`；collate 再兜一层。
  4. **验证侧补指标**：每轮整卷验证新增 IoU / 精确率 / 召回率（同一 `eval.threshold`）+
     `pred_voxels_total` + `prob_peak_max`；`metrics.csv` **末尾追加** 5 列（不改旧列）。
     预测为空时精确率记 0.0 并标 `precision_defined=False`（记 1.0 会骗人）。
     验证侧**分布不变**（顺序读、原始 ~10% 阳性，不做任何平衡采样）。
  5. 顺带修：`train.py` 原来**先 `scheduler.step()` 再打印 lr**，日志里的 lr 是"下一轮的值"，
     现在先取本轮 lr 再 step（`metrics.csv` 与 TensorBoard 同步修正）。
  6. 本地回归：`src/selfcheck_data.py` 新增「平衡采样器」与「2.5D 三层窗」两组断言，
     外加**病人级隔离**硬断言（一轮抽到的病例必须 ⊆ 本折 train）；`_selftest.py` 扩到 168 项断言
     （本地全通过，但只是逻辑/形状口径，真实数据以远程为准）。
  7. **不改** `lr` / 调度器、不启用 `ce_class_weights`、不做补边区域 ignore mask
     （前两个是下一级杠杆：若平衡采样后 `dice` 项仍长期横盘，先试
     `--set loss.ce_class_weights=[0.2,1.0]`）。
  远程验收顺序（详见 `docs/baseline.md` 第 3、4 节）：
    `python -m src.selfcheck_data` → `python -m src.train --fold 0 --debug`
    → `python -m src.train --fold 0 --out-dir runs/smoke_fold0 --set train.epochs=12`（短跑判读）
    → 达标后再 `--fold 0` 正式训练。
  短跑达标线：`dice` 项跌破 0.7、`预测体素 > 0`、`峰值概率` 明显抬起来、至少一例 val Dice > 0.05。
  仍在等待：平衡采样后的实测每批阳性数、2.5D 的显存/单步耗时、新口径下的曲线与早停轮数、5 折汇总。

■ 第 5 轮：整卷推理 + 3D 后处理 + 指标 + 评估
  交付：src/infer.py（第 3 轮已落地，本轮只加 meta 字段说明）、src/postprocess.py、
    src/metrics.py、src/evaluate.py
  src/infer.py：
    @torch.no_grad()
    def predict_volume(model, case, cache_dir, cfg, device, pad_to_multiple=16, batch_slices=8)
        -> (prob[H,W,Z] float32, pred[H,W,Z] uint8, meta)
      # 逐层前向（每层补边到 data.target_hw）→ 新建 np.zeros((H,W,Z)) 按 z 索引赋值拼回整卷
      # → **按 pad_offset 裁回原始面内尺寸**（居中补边时内容不在原点，不能写死 [:H,:W]）。
      # pad_offset 的来源：ds.pad_offset_of_case(case) 或 collate 出来的 batch["pad_offset"]。
    def seg_prob_to_label(prob, thr=0.5) -> uint8
  src/postprocess.py：
    def remove_small_lesions(binary3d, min_voxels, spacing) -> (cleaned, n_removed, removed_mm3)
    def label_lesions(binary3d) -> (int_labels, n_lesions, per_lesion_mm3)
      # scipy.ndimage.label，6 邻域（与预处理统计口径一致）
  src/metrics.py：
    def dice(pred, gt, eps=1e-6) / iou(pred, gt, eps=1e-6)
    def lesion_detection(pred_bin, gt_bin, spacing, detect_min_mm3) -> dict
      # 判据（两条 OR）：与 GT 有任意重叠，或单病灶体积 ≥ 10 mm³ 即算检出
    def evaluate_case(pred_bin, gt_bin, spacing, cfg) -> dict
  src/evaluate.py：python -m src.evaluate --fold 0 | --all
    - 载入 runs/fold<k>/best.pt → 该折 4 例验证病人：整卷推理 → 后处理（删 <50 mm³ 孤立块）→ 指标；
    - 5 例仅肝脏病人单独报告（口径）：FP 病例数/5，附每例 FP 体积 mm³ 与最大连通块 mm³（后处理前后各一份）；
    - 输出 reports/eval_fold<k>.json、reports/eval_summary.{json,md}：
      5 折 肿瘤 Dice / IoU / 病灶检出率 的 mean±std，以及 5 例仅肝脏的 FP 率；
    - --save-pred 时把预测卷存 runs/fold<k>/pred/<case>.nii.gz（保留 affine，便于叠图核对）。

■ 第 6 轮：文档与运行手册
  交付：更新 docs/baseline.md（补第 2–5 轮的真实命令与期望输出）、按需补 README
    - 每步写清"期望输出"与"出错时该贴回哪几行"；
    - 标注本版不含：肝脏通道/三分类、期相分层报告、ImageNet 预训练对照
      （~~2.5D 三层输入~~ 已在第 4 轮落地）。

■ 第 7 轮：按首次训练结果微调（预留）
    - ~~依据 --debug 的实测显存定稿 batch_size~~ → **第 3 轮已定稿：`train.batch_size=16`**
      （单通道实测 bs=2/bs=8 两点外推 ≈ 376 MB/样本 + 4.7 GB → 16 约 10.7 GB；第 4 轮 2.5D 之后
      以 `--debug` 实测为准，不做外推）；
    - ~~提高每批阳性数 / 消掉全阴性 batch~~ → **第 4 轮已解决**（平衡采样：bs=16 → 每批 8 正 8 阴、
      全阴性 batch 0 个）；老口径下"每批 1 个阳性"的成因见 `docs/preprocess_notes.md` 8.2；
    - 依据首折曲线决定是否调 `data.pos_ratio_train`（0.5 → 0.3 更保守）、`data.max_pos_repeat`、
      lr、早停耐心；欠检出仍存在时先试 `loss.ce_class_weights=[0.2,1.0]`；
    - 若 512×512 下 batch 上不去：退路是梯度累积（accumulate_grad）或按面内裁剪前景窗口，
      绝不改 spacing、绝不做 resize。

--------------------------------------------------------
进阶版（明确推迟：不在上面 5 轮里，等基础版闭环 + 第 4 轮评估跑通之后再说）
--------------------------------------------------------
■ 迁移学习 / 预训练编码器（用户已拍板：**先不做，放最后**）
  可行的做法：**ImageNet 预训练的 resnet18/34 当编码器**（torchvision 官方 resnet18-f37072fd.pth，45 MB），
  接回现有 U-Net 解码器微调 —— 也就是 d2l 里"从零训练 vs 预训练微调"那一套对照。
  最自然的搭配是**同时上 2.5D 三层输入**（相邻 3 层当 3 通道）：ImageNet 首层可以原样用（不必压通道），
  又补上 z 方向上下文；两件事本来就该一起做。
  实现要点（接口已预留：model.encoder_pretrained / model.pretrained_weights_path）：
    - resnet 的 stem（conv7x7/s2 + maxpool/s2）先降 4 倍 ⇒ **整体下采样变 32 倍**，
      model.pad_to_multiple 要改 32、解码器改成 5 级上采样；
    - 单通道版本要把首层 3 通道卷积核按通道均值压成 1 通道；解码器随机初始化，全网络端到端微调；
    - 权重**只从本地路径读**（model.pretrained_weights_path 绝对路径，或 ~/.cache/torch/hub/checkpoints/），
      代码绝不走网络；
    - 显存/耗时都要重新标定（编码器通道 64/128/256/512，batch_size=16 大概保不住），用 --debug 测。
  对照方式：独立 run 目录（如 runs/fold0_imagenet）+ 独立配置，报告里"基线 / 预训练"两栏并列。
  **【禁忌，务必遵守】不要使用公开的肝脏 / 肝肿瘤分割权重**（LiTS、MSD Task03 等）：
  本数据集的编号（31–60）、标签口径（1=肝脏、2=肿瘤且与肝脏互斥挖空）、以及 48–52 几何错位被剔除，
  都与 LiTS 家族高度吻合，很可能同源 ⇒ 用同类任务的公开权重等于把测试病人的标注喂进模型，
  5 折交叉验证直接失效，"提升"全是假的。只有**无关的外部数据（ImageNet 这类自然图像）**才安全。

--------------------------------------------------------
跨轮约束（不可省）
--------------------------------------------------------
1. 划分按病人，不按切片；data/splits.json 入库。
2. 5 例仅肝脏只进训练集，用于约束假阳性。
3. 重采样到 1mm 后不再对整图 resize；尺寸差异只靠**统一补边到 data.target_hw（512×512）**处理
   （补边偏移 pad_offset 要一路带到推理，裁回原始尺寸时用它）。
4. 每折划分、随机种子、指标口径写入产物，保证跨轮可复现。
5. 代码只写、不本地执行；需要远程执行时先提交再给命令；产物路径一律相对仓库根。
6. 病人级隔离（第 4 轮强化）：划分按病人；采样器只在本折 train 的切片索引内重排/重复；
   2.5D 三层窗只在本病例内索引（越界用端点复制）。任何一条被破坏都会让 Dice 虚高，
   src/selfcheck_data.py 里有对应的硬断言（见 docs/preprocess_notes.md 8.3）。
