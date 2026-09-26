========================================================
CT 肝脏肿瘤分割 · 待办与约束
（已完成轮次的交付细节不在这里——口径见 docs/preprocess_notes.md，命令与判读见 docs/baseline.md）
========================================================

【已完成（一句话速览，细节看代码与上面两份文档）】
- 第 1 轮：预处理 + 缓存 + 5 折划分（按病人、按肿瘤体积分层）→ `cache/`、`data/splits.json`。
- 第 2 轮：切片数据集（统一补边到 512×512）+ 自实现 2D 增强 + 采样器。
- 第 3 轮：手搓 2D U-Net + DiceCELoss + 训练闭环（整卷验证 / 早停 / 续跑 / metrics.csv / TensorBoard）。
- 第 4 轮：修「塌缩到全预测背景」→ 平衡采样（每批 8 正 8 阴）+ `dice_positive_only` + **2.5D 三层输入**
  + 验证侧补 IoU/精确率/召回率/塌缩指标。远程 12 轮短跑 best 整卷 Dice **0.1983**
  （`runs/smoke_fold0_fixed`；case 33 0.76，小病灶 57/59 仍为 0）。

--------------------------------------------------------
■ 第 5 轮（下一步）：整卷推理 → 3D 后处理 → 指标 → 评估报告
--------------------------------------------------------
交付：`src/postprocess.py`、`src/metrics.py`、`src/evaluate.py`
（`src/infer.py` 已落地：`predict_volume` / `seg_prob_to_label` / `load_label_volume`）

  src/postprocess.py：
    def remove_small_lesions(binary3d, min_voxels, spacing) -> (cleaned, n_removed, removed_mm3)
    def label_lesions(binary3d) -> (int_labels, n_lesions, per_lesion_mm3)
      # 连通域统一用 scipy.ndimage.label（6 邻域，与预处理统计口径一致；别用 SimpleITK 的 ConnectedComponent）
  src/metrics.py：
    def dice(pred, gt, eps=1e-6) / iou(pred, gt, eps=1e-6) / precision / recall
      # 公式必须与 src/train.py 的 volume_metrics 完全一致（eps=1e-6、两边都空记 1.0）
    def lesion_detection(pred_bin, gt_bin, spacing, detect_min_mm3) -> dict
      # 判据（两条 OR）：与 GT 有任意重叠，或单病灶体积 ≥ 10 mm³ 即算检出
    def evaluate_case(pred_bin, gt_bin, spacing, cfg) -> dict
  src/evaluate.py：python -m src.evaluate --fold 0 | --all [--run-dir runs/smoke_fold0_fixed]
    - 载入 `runs/<run>/best.pt` → 该折 4 例验证病人：整卷推理 → 后处理（删 <50 mm³ 孤立块）→ 指标；
    - **开发阶段先用 `runs/smoke_fold0_fixed/best.pt`**（12 epoch 烟测权重，非结果）：
      case 33 有真实检出、57/59/60 为 0，正好能把后处理与病灶级检出的分支都走到；
    - 5 例仅肝脏病人（32/34/38/41/47）单独报告：FP 病例数/5 + 每例 FP 体积与最大连通块（后处理前后各一份）；
    - 输出 `reports/eval_fold<k>.json`、`reports/eval_summary.{json,md}`：
      **必须给 mean±std 与逐例值**（训练不稳定，只报均值会误导），另报病灶检出率与 5 例仅肝脏 FP 率；
    - `--save-pred` 时把预测卷存 `runs/<run>/pred/<case>.nii.gz`（保留 affine，便于叠图核对）。

--------------------------------------------------------
■ 第 6 轮：按评测结果调参（三件事，按优先级，别同时改多个变量）
--------------------------------------------------------
1. **训练不稳定**（同一配置两次 12 轮短跑 best 0.0048 vs 0.1983）：先固定评测口径
   （同 seed 多跑 2–3 次，报告 mean±std），再试更强增强 / EMA / 预训练编码器；
2. **小病灶长期为 0**（case 57 = 4 046 体素、59 = 652 体素）：
   `data.pos_ratio_train` 0.5 → 0.6、按病灶中心加权采样；
   损失形状候选（每次只试一个）：Tversky（`1 − TP/(TP + α·FP + β·FN)`，α=0.7 压假阳性）、
   逐样本 Dice（`loss.batch=false`）、把阴性样本重新纳入 Dice（`loss.dice_positive_only=false`）；
   注意 `ce_class_weights=[0.2,1.0]` 已试过一档，方向是错的（预测体积更大），已回退为 `null`；
3. lr / 调度器 / 早停耐心：等 1、2 有对照曲线再动。
已定稿不再讨论：`train.batch_size=16`（2.5D 实测峰值 10969 MB）、每批 8 正 8 阴、全阴性 batch 0。

--------------------------------------------------------
■ 第 7 轮：文档与运行手册收尾
--------------------------------------------------------
- 把 5 折真实数字（mean±std、逐例、FP 率）回填进 `docs/preprocess_notes.md` 与 `docs/baseline.md`；
- 标注本版不含：肝脏通道/三分类、期相分层报告、ImageNet 预训练对照。

--------------------------------------------------------
■ 进阶版（明确推迟到基础版闭环之后）
--------------------------------------------------------
迁移学习 / 预训练编码器：**ImageNet 预训练的 resnet18/34 当编码器**（torchvision 官方 45 MB 权重），
接回现有解码器微调 —— d2l 里"从零训练 vs 预训练微调"的对照。
  - 2.5D 三层输入**已在第 4 轮落地**，ImageNet 首层可原样用（不必压通道）；
  - resnet stem 先降 4 倍 ⇒ 整体下采样变 32 倍：`model.pad_to_multiple` 改 32、解码器改 5 级上采样；
  - 权重**只从本地路径读**（`model.pretrained_weights_path` 或 `~/.cache/torch/hub/checkpoints/`），代码绝不走网络；
  - 显存/耗时重新标定（编码器通道 64/128/256/512，bs=16 大概保不住），用 `--debug` 测；
  - 对照方式：独立 run 目录（如 `runs/fold0_imagenet`）+ 独立配置，报告里"基线 / 预训练"两栏并列。
**【禁忌】不要使用公开的肝脏/肝肿瘤分割权重**（LiTS、MSD Task03 等）：本数据集的编号（31–60）、
标签口径（1=肝脏、2=肿瘤且与肝脏互斥挖空）、48–52 几何错位被剔除，都与 LiTS 家族高度吻合、很可能同源
⇒ 用同类任务的公开权重等于把测试病人的标注喂进模型，5 折交叉验证直接失效。**只有 ImageNet 这类无关数据安全。**

--------------------------------------------------------
跨轮约束（不可省）
--------------------------------------------------------
1. 划分按病人，不按切片；`data/splits.json` 入库、不做本地改动。
2. 5 例仅肝脏（32/34/38/41/47）只进训练集，用于约束假阳性；验证集分母恒为 4 例含肿瘤病人。
3. 重采样到 1mm 后**不对整图 resize**；尺寸差异只靠统一补边到 `data.target_hw`（512×512）处理，
   补边偏移 `pad_offset` 一路带到推理，裁回原始尺寸必须用它。
4. **病人级隔离**：划分按病人；采样器只在本折 train 的切片索引内重排/重复；2.5D 三层窗只在本病例内索引
   （越界用端点复制）。任何一条被破坏都会让 Dice 虚高 —— `src/selfcheck_data.py` 里有硬断言。
5. 每折划分、随机种子、指标口径写入产物，保证跨轮可复现；报告一律给 mean±std 与逐例值。
6. 代码只写、不本地执行；需要远程执行时先提交再给命令；产物路径一律相对仓库根。
7. **文档保持精简**：新增信息写进现有文档的对应位置，不要新开归档文件、不要复述代码。
