========================================================
CT 肝脏肿瘤分割 · 基础版（2D 闭环）后续编码计划
（阶段一：预处理 + 划分 —— 已完成并验证，本计划不含它）
========================================================

【阶段一已确认的事实（后续代码直接依赖，勿再猜）】
- 缓存：nibabel 读回形状 (nx, ny, nz)，切片轴在最后一维（a[:, :, k] 是一层 (ny,nx) 切片）；
  SimpleITK 读回是其转置 (nz,ny,nx)，两库互为转置。影像 uint16 归一化值（/65535 得 [0,1]）；
  掩膜 uint8 只含 {0,1}；全部 1mm 各向同性。
- 面内尺寸：512×512 共 17 例；其余 8 例为 342–436 一族。按 16 对齐分 6 个桶：
  512×512(17)、448×448、432×432、416×416(2)、368×368、352×352(3)。
  → 不同面内尺寸不能同 batch，必须分桶。
- 切片数 nz：74–488，各例不同。
- 含肿瘤切片 766/5982 = 12.81% → 过采样基线（目标提到 batch 内 25–35%）。
- 划分：data/splits.json，5 折各 4 例验证 / 21 例训练（16 含肿瘤 + 5 仅肝脏），体积 CV 0.317。
- 5 例仅肝脏（32/34/38/41/47）只进训练集，永不进验证集。

--------------------------------------------------------
剩余 5 轮，每轮都能独立运行 + 独立验证
--------------------------------------------------------

■ 第 2 轮：Dataset + 分桶采样器 + 增强 —— 已完成
  交付：src/selfcheck_data.py、src/dataset.py（另在 configs/default.yaml 补了 data 节）
  【先交付自检脚本】src/selfcheck_data.py（不依赖 torch 网络，只读缓存）
    - 跑 3 个 batch，打印：每个 batch 的 image/label shape、分桶键、
      /65535 后的值域、batch 内含肿瘤切片比例实测值、单 batch 耗时；
    - 再对 1 个 case 打印逐层肿瘤体素数曲线，确认切片轴没搞反。
    - 目的：让你先确认"数据进模型的形态"正确，再往上搭训练。
  src/dataset.py 接口：
    class CTSliceDataset(torch.utils.data.Dataset)
      __init__(cases, cache_dir, split, cfg, augment: bool, debug: bool=False)
      __len__() -> int                 # 全部切片数
      __getitem__(i) -> {"image": FloatTensor[1,H,W], "label": LongTensor[H,W],
                         "case": str, "z": int, "orig_hw": (H,W)}
      属性：index = [(case, z)...]；pos_flags = bool 数组（该切片是否含肿瘤）
    def make_batch_sampler(ds, cfg, generator=None) -> BatchSampler
      # 自定义 BatchSampler：先按桶（ceil(H/16), ceil(W/16)）选桶，再在桶内
      # 按 n_pos = round(batch_size*pos_ratio_target)、n_neg = 其余 定向抽取；
      # 保证 batch 内肿瘤切片比例恒为 pos_ratio_target（默认 0.30）、batch 内尺寸一致。
      # 不用裸 WeightedRandomSampler：比例不可控，且不同尺寸无法同 batch。
    def make_train_loader(splits, fold, cfg) -> DataLoader
    def make_val_loader(splits, fold, cfg) -> DataLoader   # 顺序、无增强
    def build_transforms(cfg, train: bool) -> monai.transforms.Compose
  读取策略（避免 DataLoader 里开上千句柄）：
    __init__ 只 nib.load 读 header + label 体素（算 pos_flags/索引），不读 image 体素；
    __getitem__ 用每 worker 一份的 lru_cache(maxsize=8) 按 (case, key) 打开 nibabel 代理对象，
    memmap=True，只取 [:, :, z]。
  增强（仅训练，MONAI 2D，image/label 同步）：
    RandFlip(0.5, axis=0) + RandFlip(0.5, axis=1)、RandRotate90(0.5, (0,1))、
    RandAffine(0.5, 旋转±15°、缩放0.9–1.1、平移±10%，image=bilinear/label=nearest)、
    RandHistogramShift(0.2)、RandGaussianNoise(0.2, std=0.01)。
    不做弹性形变（无 2D 版，逐层施加会破坏 z 一致性）。
  校验要点（必须打印/断言）：batch 内正样本比例 ≈ 0.30；值域 ∈ [0,1]；label ⊂ {0,1}。

■ 第 3 轮：模型 + 损失 + 训练
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
    - --debug：batch_size=2、每折 2 例、3 个 iteration，打印每 batch shape、分桶键、
      实测正样本比例、torch.cuda.max_memory_allocated()/reserved、单步耗时，然后退出不落盘。
      这是标定 batch_size（先按 8）的依据。
  逐折执行：for f in 0 1 2 3 4; do python -m src.train --fold $f; done

■ 第 4 轮：整卷推理 + 3D 后处理 + 指标 + 评估
  交付：src/infer.py、src/postprocess.py、src/metrics.py、src/evaluate.py
  src/infer.py：
    @torch.no_grad()
    def predict_volume(model, case, cache_dir, cfg, device, pad_to_multiple=16, batch_slices=8)
        -> (prob[H,W,Z] float32, pred[H,W,Z] uint8, meta)
      # 逐层前向 → 新建 np.zeros((H,W,Z)) 按 z 索引赋值拼回整卷 → 还原原始面内尺寸
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

■ 第 5 轮：文档与运行手册
  交付：更新 docs/baseline.md（补第 2–4 轮的真实命令与期望输出）、按需补 README
    - 每步写清"期望输出"与"出错时该贴回哪几行"；
    - 标注本版不含：2.5D 三层输入、肝脏通道/三分类、期相分层报告、ImageNet 预训练对照。

■ 第 6 轮：按首次训练结果微调（预留）
    - 依据 --debug 的实测显存定稿 batch_size / num_workers；
    - 依据首折曲线决定是否调 pos_ratio_target（现基线 12.81% → 目标 30% 约 2.3 倍过采样）、
      lr、早停耐心；
    - 若 512×512 下 batch 上不去：退路是梯度累积（accumulate_grad）或按面内裁剪前景窗口，
      绝不改 spacing、绝不做 resize。

--------------------------------------------------------
跨轮约束（不可省）
--------------------------------------------------------
1. 划分按病人，不按切片；data/splits.json 入库。
2. 5 例仅肝脏只进训练集，用于约束假阳性。
3. 重采样到 1mm 后不再对整图 resize；尺寸差异只靠分桶 + padding 到 16 的倍数处理。
4. 每折划分、随机种子、指标口径写入产物，保证跨轮可复现。
5. 代码只写、不本地执行；需要远程执行时先提交再给命令；产物路径一律相对仓库根。