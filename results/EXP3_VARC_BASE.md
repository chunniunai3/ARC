# EXP3: VARC Baseline (per-pixel, 12-class, fixed placement)

## 目标
实现 VARC 论文（Visual ARC）的逐像素 Transformer 基线，在 ARC-AGI 训练集上训练并评估。

## 模型架构
- **编码器**: PixelEmbed (nn.Embedding, dim=128) + 2D 可学习位置编码 + ViT encoder (6 层, 8 头, MLP 512)
- **解码器**: ViT decoder (6 层, 8 头, MLP 512) + linear head → 12 类
- **类别**: 0-9 颜色, BG=10 (背景/忽略), BD=11 (边界标记)
- **Task token**: 可学习嵌入，每任务 400 维，拼接在序列首
- **参数量**: 16.29M

## 关键改动（与初版对比）

| 改动 | 之前 | 之后 |
|------|------|------|
| 像素编码 | one-hot 拼接 | `nn.Embedding(12, 128)` |
| 位置编码 | 1D 可学习 | 2D 可学习 |
| MLP 维度 | 1024 | 512 |
| 类别数 | 12 (含 BG=10, BD=11) | 12 (含 BG=10, BD=11) |
| 数据增强 | 随机缩放+平移 | 固定居中放置（max scale） |
| 训练超参 | Adam, 无 warmup | Adam, 10 epoch warmup, cosine LR |
| 测试裁剪 | BD 边界检测 | 已知输出尺寸直接裁剪 |

## 训练日志

```
Epoch   1/100  lr=3.00e-05  train_loss=1.7241, val_loss=1.3858
Epoch  10/100  lr=3.00e-04  train_loss=1.1282, val_loss=1.3501
Epoch  20/100  lr=2.91e-04  train_loss=0.8718, val_loss=0.9292
Epoch  30/100  lr=2.65e-04  train_loss=0.7945, val_loss=0.7409
Epoch  40/100  lr=2.25e-04  train_loss=0.6439, val_loss=0.6475
Epoch  50/100  lr=1.76e-04  train_loss=0.5806, val_loss=0.5711
Epoch  60/100  lr=1.24e-04  train_loss=0.5338, val_loss=0.5479
```

继续训练 (epoch 61-76):
```
Epoch  61/100  lr=1.24e-04  train_loss=0.5283  val_loss=0.5319
Epoch  70/100  lr=7.96e-05  train_loss=0.4924  val_loss=0.4996
Epoch  76/100  lr=5.36e-05  train_loss=0.4712  val_loss=0.4843
```

## 评估结果 (50 个训练任务, ep76)

**准确率: 2% (1/50 正确)**

### 正确任务
- **0d3d703e**: 3x3 → 3x3, `[[9,5,4],[5,9,9],[4,9,9]]`

### 非常接近的任务 (第一行完全匹配但全网格有差异)
- **06df4c85**: 26x26, 36/676 像素错误 (边界颜色偏移 + BD 边界)
- **0dfd9992**: 10x10, 1 像素错误 (右下角颜色)
- **1f642eb9**: 6x6, 2/36 错误
- **2204b7a8**: 10x10, 4/100 错误
- **2281f1f4**: 10x10, 第一行 6/6 正确
- **228f6490**: 10x10, 第一行 6/6 正确
- **1b60fb0c**: 6x6, 第一行 6/6 正确
- **1e32b0e9**: 6x6, 第一行 6/6 正确

### 部分学习 (正确的颜色但错误的图案)
大多数任务模型预测到了正确的颜色集和大致结构，但缺少精细的像素级模式复制。

## 关键发现

1. **VARC 架构有效** — 模型学习了从输入到输出网格的逐像素映射，loss 从 2.40 (随机) 降至 0.47
2. **随机缩放/平移破坏训练** — 改为固定居中后 loss 大幅下降 (之前 1.73→0.61, 之后 1.72→0.47)
3. **BD 边界不可靠** — 模型不始终在右/下边界预测 BD，应使用已知输出尺寸直接裁剪
4. **数据不足** — 400 个任务 (~1600 训练对) 远少于论文的 RE-ARC (40 万样本)
5. **多视角推理缺失** — 论文用 510 个随机视角投票，我们只做单次推理

## 文件清单

```
src/varc/
  model.py      — VARC 模型定义 (ViT encoder-decoder, 16.29M)
  data.py       — 数据加载、画布放置、增强
  train.py      — 训练、TTT、checkpoint
  evaluate.py   — 推理、网格提取、评估
train_full.py   — 训练入口 (100 epoch)
final_eval.py   — 评估入口 (50 任务)
results/checkpoints/varc_exp3_base.pt  — 训练权重 (65 MB)
```

## 与论文差距
| 指标 | 论文 (VARC-Base) | 本实验 |
|------|-------------------|--------|
| 参数量 | 18M | 16.29M |
| 训练数据 | RE-ARC (400k) | ARC-400 (1.6k samples) |
| 推理方式 | 510 随机视角投票 | 单视角 |
| 训练 epoch | 100 | 76 (中断) |
| 准确率 (训练集) | 54.5% | 2% |

## 下一步改进方向
1. **RE-ARC 数据生成** — 实现原始变换 (对象移动、颜色替换、缩放等) 生成 400k 样本
2. **多视角推理** — 随机缩放/平移 + majority vote
3. **TTT** — 测试时在训练对微调 (当前 100 step/任务 超时)
4. **BD 可微分** — 让 BD 边界参与损失计算来稳定预测
