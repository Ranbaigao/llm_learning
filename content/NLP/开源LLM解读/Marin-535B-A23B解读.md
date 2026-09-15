# Marin 535B-A23B：一场完全公开的前沿 MoE 训练直播

> 截至日期：2026-09-15  
> 资料范围：W&B 报告《535B-A23B-18T-Token-Hero-Run-Scaling-Ladder》、GitHub issue #8435、Marin 公开 spec、Harrier K40 数据构成页、多家媒体复盘（AI Primer / LamjinLab / InfoQ / BestHub / QuidProQuo）。  
> 运行编号：hero-12d8b6f0-dee637；启动日期 2026-08-19；按 4,400 steps/day 速率推算 ETA 2026-11-18。截至 9 月初已完成约 13-15%。
>
> **结论先行**：Marin 535B-A23B 的核心不是"又一个 MoE"，而是把通常被模型公司严密保护的训练过程变成一个**可以实时观察、审计、干预**的公开研究对象。技术选择上，它走的是**细粒度 MoE + 潜在空间压缩 + 三 wave 固定 buffer 的专家并行**这条不太常见的路线，配合 4 步 scaling ladder（仅占 1% 算力预算）做事前稳定性验证。**真正值得关注的是"为什么回归 4K 起步"、"为什么 logit z-loss 几乎成为标配"、"为什么手写 EP 而不是用现成框架"这三个工程决定**，以及与之配套的开放开发流程（GitHub issue 写假设、W&B 流式 telemetry、负结果一并公开）。

## 1. 项目定位：这不是模型发布，是开放开发实验

Stanford CRFM 的 Marin 项目（Percy Liang 团队，2025 年由 Open Athena 接手）在 2026-08-19 启动了其迄今最大的训练任务：

- **模型**：535B 总参数 / 23B 激活参数的 MoE（fine-grained + 共享专家）
- **数据**：18.75T tokens（80% 预训练 + 20% 中期训练）
- **算力**：约 $2.7 \times 10^{24}$ FLOPs，11 套 GB200 NVL72（约 792 颗 Blackwell GPU）
- **时长**：约 100 天
- **经费**：Jen-Hsun & Lori Huang Foundation（CoreWeave）

与一般意义上的"模型发布"不同，Marin 在训练**进行中**就公开了：

- 数据组成（Harrier K40：23.11T tokens、40 个语义领域、5 个质量桶）
- 训练代码（Levanter/JAX）
- 实时 telemetry（W&B：loss、grad norm、drop fraction、MFU）
- 工程负结果（XLA 显存估算失败、CUDA OOM 等一并记录）
- 应急计划（前 25% token 预算内若出问题，会调整 LR 衰减与数据配比）

**这条时间线本身的稀有性**：截至 2026-09，公开报道过的最大规模"完全可观察"训练来自 OLMo / Pythia / Marin 自己 32B 这一档，DeepSeek / Llama / Qwen 等前沿模型即便事后开源，数据配方与训练决策的过程仍是黑箱。Marin 的 535B 是**同尺度下首次全过程直播**。

## 2. 模型架构

### 2.1 总体规格

| 项 | 值 |
| --- | --- |
| 总参数 | 535.3B |
| 激活参数 | 22.76B |
| Transformer 层数 | 48 |
| 隐藏维度 | 6,144 |
| 注意力头数 | 48 |
| 词表 | 128,256 |
| 序列长度（起步） | 4,096 |
| 目标最大序列长度 | 262,144 |
| 训练 tokens | 18.75T |
| 训练步数 | 390,139 |
| 总 FLOPs | $2.7 \times 10^{24}$ |

### 2.2 MoE 详解：fine-grained + 共享专家 + 潜在空间压缩

每一层 MoE 由三个部分组成：

```
输入 (d=6144)
   │
   ├─→ 注意力分支（3/4 层用 sliding-window，每 4 层与最后一层用 global full-causal）
   │
   └─→ MoE 分支
         │
         ├─→ 潜在压缩 latent_proj: 6144 → 3072 （2× 压缩）
         │
         ├─→ 直通路径：2 个共享专家（dense SwiGLU，宽 3072）
         │
         └─→ 路由路径：384 个路由专家，每个宽 3072
                 │
                 └─→ top-8 路由（histogram-based quantile router，
                               下一步零均值 bias，不改 router 梯度）
```

**几个关键设计点**：

1. **共享专家始终处理所有 token**：占激活参数约 1/3。即使路由专家超容量发生 token dropping，共享专家仍提供稠密 backbone，不会丢失全部信息。
2. **潜在空间压缩**：路由 token 在 All-to-All 之前从 6,144 压缩到 3,072，专家计算后解压回去。所有 to-all 通信量直接减半。
3. **激活维度等价 4× 隐藏**：$8 \times 3072 = 24{,}576 = 4 \times 6144$。即从 FLOPs 角度看，等价于一个标准 MLP 扩展 4 倍的 dense 模型——但只用了 8/384 个专家。
4. **router 不更新梯度**：quantile router 用下一步零均值 bias 平衡专家选择，bias 是 heuristic、不参与反传。

### 2.3 注意力：3/4 局部 + 1/4 全局

每 4 层中：

- 3 层用 local sliding-window attention
- 1 层用 global full-causal attention
- 最后一层固定为 global

这是当前长上下文模型常用配比，本质上是用全局层承担"长程检索"角色，用局部层降低 KV cache 与计算。

## 3. 专家并行（EP）：手写三 wave 固定 buffer

Marin 没有找到在 JAX/XLA on GPU 上性能足够好的现成 EP 实现，于是手写了一套基于 **pooled-wave 固定 all-to-all** 的方案，由 @ravwojdyla 主笔。

### 3.1 拓扑

| 项 | 值 |
| --- | --- |
| 集群 | 11 套 GB200 NVL72 |
| 每套 EP 组 | 64 GPU（16 节点 × 4 GPU/节点） |
| EP 度数 | 64 |
| 数据并行度数 | 11（11 套机架间 DP） |
| 每机架 batch | 1,024 sequences |
| 全局 batch | 11,264 sequences × 4,096 tokens = 46,137,344 tokens/step |

**为什么 EP64 而不是更大**：单个 NVL72（64 GPU、12.3 TB 统一内存）可以放下整个 535B 模型 + optimizer state，所以一个机架就是一个完整的模型副本，11 个机架间做数据并行。

### 3.2 三 wave 固定 buffer

EP collective 的核心约束是通信与计算的 overlap。三 wave 设计把每个训练步切成三段静态 buffer：

- **wave 1**：token 从源设备 all-to-all 派发到目标专家
- **wave 2**：专家计算
- **wave 3**：结果 all-to-all 收回到原设备

容量因子（capacity factor）= 1.15，每个专家的接收 buffer 是该 wave 内最大期望 token 数 × 1.15。超出部分直接丢弃（token dropping）。

**专家 ID 通过 activation collective 一起传输**，不再走单独的 metadata collective——节省一次同步开销。

### 3.3 Token dropping：开放开发的核心监控指标

Token dropping 是 MoE 训练的关键健康指标，过高意味着大量 token 没有完整经过路由专家，模型退化成"几乎只用共享专家"的稠密模型：

| 上下文长度 | 历史测试 dropping | pooled-wave EP dropping |
| --- | --- | --- |
| 4K | ~7% | ~3% |
| 65K | ~40% | 待实测 |

**为什么回归 4K 起步**：4K batch 比 8K 多容纳约 2 倍独立序列数，token 在专家间分布更均匀，dropping 更低。

Marin 列出三套应急方案：

1. 切换到 dropless ragged all-to-all
2. 增大 capacity factor（显存代价）
3. 序列级负载均衡（专家可能被迫重训，可能引发训练不稳定——备胎）

W&B 报告流式追踪 sender / receiver 两端的 drop fraction，作为主要健康指标之一。

## 4. Scaling Ladder：1% 算力预算的事前验证

启动 535B 之前，团队按 hero recipe 跑了 5 档 scaling ladder（公开脚本 `experiments/grug/moe_hero_ep/launch_scaling_ladder.py`）：

| 档位（宽度） | 机架 | batch | 激活参数 | FLOPs |
| --- | --- | --- | --- | --- |
| d768 | 1 | 1,024 | 61M | $5.5 \times 10^{19}$ |
| d2048 | 11 | 11,264 | 1.2B | $9.2 \times 10^{21}$ |
| d3072 | — | — | — | — |
| d4096 | — | — | — | — |
| **d6144（hero）** | 11 | 11,264 | 23B | $2.7 \times 10^{24}$ |

注意 d3072 与 d4096 两档在公开报告中未给具体数字，仅 d768/d2048/d6144 三档明确。整个 ladder 的总成本约为主训练算力的 1%。

### 4.1 三重价值

1. **Bug 捕获**：上一轮 ladder 发现 gradient norm 随 token horizon 扩大涨到 4 以上 → 团队加了 **logit z-loss** 修复 → 后续消融证明高 batch 配置不加 z-loss 会中途发散
2. **外推预测**：用小档训练曲线拟合 loss / grad norm / dropping 的 scale law → 预测 hero run 的演化轨迹
3. **决策参考**：hero run 中若 grad norm 在前 30% 持续上涨 → 与 ladder 模式一致则不干预；偏离模式则需要排查

### 4.2 logit z-loss：几乎成为大模型标配

logit z-loss（logits 经过平方抑制项）是 DeepSeek-V3、Llama 3、Kimi K2 等近期模型普遍采用的一项稳定化技巧，其形式通常为：

$$\mathcal{L}_{\text{z}} = \frac{1}{B} \sum_{i=1}^{B} \log^2 Z_i$$

其中 $Z_i = \sum_j \exp(\ell_{ij})$ 是 logsumexp。Marin 在发现梯度范数发散后引入该 loss 并证明其在高 batch 下不可或缺——这也是 ladder 价值的实证：避免了一次昂贵的中途崩盘。

## 5. 训练配置与时间表

### 5.1 训练 step 概览

| 项 | 值 |
| --- | --- |
| 总步数 | 390,139 |
| batch 序列数 | 11,264 |
| 序列长度（起步） | 4,096 |
| tokens/step | 46,137,344 |
| 速率 | ~4,400 steps/day |
| 启动 | 2026-08-19 |
| ETA | 2026-11-18 |

每个 transformer block 前向激活约 567 GB（不含 checkpointing），48 层合计 ~27 TB，远超 GPU 单卡显存——必须依赖 EP + activation checkpointing。

### 5.2 上下文扩展时间表（条件性）

| 阶段 | 触发点 | 目标序列长度 |
| --- | --- | --- |
| 起步 | 0% | 4K |
| 第一扩展 | ~50% | 8K |
| 第二扩展 | ~95% | 65K |
| 长上下文 | 接近结束 | 262K |
| 早期 cooldown | 第 10-20 天（1-2 天） | 不变 |

注：早期 cooldown 是个测试分支——用来给一个全规模 checkpoint 跑 RL 实验，并测试更长上下文对 routing 的影响，不改变主训练轨迹。

### 5.3 阶段性分配

| 阶段 | 比例 | tokens |
| --- | --- | --- |
| 预训练 | 80% | 15T |
| 中期训练 | 20% | 3.75T |
| **合计** | 100% | 18.75T |

### 5.4 中途可调整点

公开 issue 明确：在前 25% token 预算内若出现以下情况，会启动应急调整而非死守计划：

- 基础设施延迟
- 模型 FLOP 利用率低于预期

调整方式：

- 缩短总 token 范围
- 调整数据配比
- 重排线性 LR 衰减，确保终点仍是 peak LR 的 5%

**这意味着公布的 18.75T 是个工作目标，不是不可改的规格**。

## 6. 数据：Harrier K40 候选存储

Marin 在训练开始前先开源了约 **23 万亿 tokens** 的预训练候选池（Harrier K40）：

| 字段 | 值 |
| --- | --- |
| 原始规模 | 25.6T tokens |
| 过滤后 | 23.11T tokens（40 个语义领域 × 5 个质量桶） |
| 主要领域例 | Council 7.52%（1.74T）、Infra 7.04%（1.63T）、Web Code 6.73%（1.56T） |
| 公开方式 | S3 bucket 可下载；每个领域给出采样比例与解码样本 |

两阶段采样设计（"uniform-sampling" 与后续可能的质量加权采样）使数据配比的迭代可在不重做 tokenization 的前提下快速完成。

## 7. 实时训练指标（截至 2026-09-09）

公开 W&B 报告 `hero-12d8b6f0-dee637` 流式追踪以下关键指标：

| 指标 | 训练初期 | 训练中期（约 step 12k） | 健康参考 |
| --- | --- | --- | --- |
| loss | 11.801 | 1.321 | 持续下降 |
| gradient norm | peak 1.518 → 后稳定 | 0.205 | ladder 预测前 30% 单调上升、后下降 |
| drop fraction | 10.43% | 3.06% | <8% 为健康 |
| MFU | warmup 后 | ~21% | — |
| Paloma macro | — | 2.577（3% 阶段） | ladder 预测 3% 时约 2.04 |

**截至 2026-09-09 已完成一个重要里程碑**：step 81,716 时成功热切换到 ragged all-to-all 路由（之前的 fallback 方案）。这次切换的实际表现是观察后续是否如预测的关键节点之一。

**公开 issue 中标注的健康检查清单**：

- ✅ 前 3% 阶段：clear
- ⏳ 下一观察窗口：约 30% 时的真正 grad norm 峰值

## 8. 与 R1 / DeepSeek-V4 等前沿 MoE 的对比

| 项 | Marin 535B-A23B | DeepSeek-V3.2 / V4 | R1 (671B-A37B) |
| --- | --- | --- | --- |
| 总 / 激活 | 535B / 22.76B | V3.2 671B / 37B | 671B / 37B |
| 路由 | top-8 / 384 routed + 2 shared | top-8 / 256 routed + 1 shared | top-8 / 256 routed |
| 共享专家 | 有（dense SwiGLU） | 有 | 有 |
| 潜在压缩 | 有（6144 → 3072） | 部分 | — |
| logit z-loss | 有（ladder 验证） | 有 | 有 |
| 训练 tokens | 18.75T | — | — |
| 训练框架 | JAX/Levanter + 自研 EP | DeepSeek 内部 | DeepSeek 内部 |
| 训练时长 | ~100 天（直播） | 未公开 | 未公开 |
| 数据透明度 | 完全公开（Harrier K40） | 未公开 | 未公开 |

**两条公开与闭源路线的方法论分野**清晰可见：Marin 选了"细粒度 + 潜在压缩 + 共享专家"的组合，加上三 wave 固定 buffer 来控通信复杂度。这套组合在 DeepSeek 系里也有类似设计，但 Marin 把它做到了 100% 可审计。

## 9. 开放开发流程：这套方法学本身才是核心贡献

Marin 团队的"开放开发"不只是"事后开源权重"，而是一套流程：

1. **每个实验先开 GitHub issue**：写清楚假设、目标、风险、应急计划
2. **实现以可 review 代码形式提交**
3. **运行挂在公共 telemetry**（W&B、S3 指标）
4. **分析回归 issue**：包括失败尝试
5. **scaling ladder** 在每个新规模前重跑一遍

具体到 535B 这一跑，公开的工程记录已经覆盖了：

- hero-run 主 issue（运行计划 + 风险 + 上下文扩展 + 应急）
- W&B 实时报告（loss / grad / drop / MFU / eval）
- 完整代码仓库（Apache-2.0）
- Harrier 数据 dashboard
- EP 工程详述报告（区分实测结果与工程判断）
- **负结果**：fixed expert-cell XLA 估算 192.65 GiB 在 123.49 GiB CUDA 上失败、6-expert receiver bank OOM 等

这种"把过程当产品"的做法使得**外部研究者可以判断训练过程中每项干预的依据是否站得住脚**，而不仅凭一个最终模型卡做猜测。

## 10. 关键观察点（截至 2026-09-15）

训练仍在进行中，以下几项是后续值得跟踪的关键信号：

1. **30% 阶段的真正 grad norm 峰值**：ladder 预测应在前 40% 单调上升、后随 LR 衰减下降。Hero 跑是否遵循该曲线是 scaling law 假设是否成立的硬验证
2. **8K 上下文扩展（~50% 阶段）的 drop fraction**：pooled-wave EP 在 4K 表现良好（3%），但 65K 是否可控仍未实测
3. **ragged all-to-all 切换后的稳定性**：已在 step 81,716 完成热切换，后续表现需观察
4. **Paloma macro 实际轨迹与 ladder 预测的偏差**：当前 3% 阶段实测 2.577 vs 预测 2.04，已偏高 0.5 个 nats——这是一个偏负的早期信号，需在后续观察是否收敛
5. **应急计划是否触发**：前 25% 内是否需要调整 token 范围、LR 衰减或数据配比

## 11. 延伸阅读

- W&B 报告：[535B-A23B-18T-Token-Hero-Run-Scaling-Ladder](https://wandb.ai/marin-community/marin_moe/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng)
- GitHub：[Hero Run] 535B-A23B on 18T tokens（issue #8435）
- 数据 dashboard：[Harrier K40 cluster overview](https://storage.googleapis.com/marin-public/held/harrier-k40-cluster-overview/2026.08.18/index.html?revision=uniform-sampling)
- EP 工程报告：[MoE Fixed Wave All-to-All EP64](https://storage.googleapis.com/marin-public/rav/moe-fixed-wave-a2a-384/2026.08.17/index.html)
- 代码仓库：[marin-community/marin](https://github.com/marin-community/marin)（`experiments/grug/moe_hero_ep/launch_scaling_ladder.py`）
- Percy Liang 公告 X：训练启动说明
- 配套课程：Stanford CS336《Language Models From Scratch》

## 12. 一句话总结

**Marin 535B-A23B 的真正贡献不是 535B 这个数字，而是把一次约 100 天、$2.7 \times 10^{24}$ FLOPs、11 套 GB200 规模的前沿 MoE 训练的全部假设、配置、决策依据、负结果与实时 telemetry 公开化——使得"开放开发"从一个口号变成一组可被独立审计的工程工件。**
