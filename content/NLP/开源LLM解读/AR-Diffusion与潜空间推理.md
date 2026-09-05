# AR-Diffusion 与潜空间推理路线综述

> 主题：从显式 CoT 到 DiffCoT、LoopLM、Nemotron-Labs-Diffusion 与 TiDAR  
> 关注问题：LLM 如何在保持生成质量的同时，获得更强推理能力、更高推理吞吐和更低错误累积。  
> 阅读结论：这些论文共同指向一个趋势：未来高效推理模型不会只依赖单纯的自回归生成，而会把“思考”“草稿”“修正”“表达”拆成不同机制，再通过同一个模型或同一套训练目标统一起来。

## 1. 总览：这几篇论文在回答什么问题

传统 decoder-only AR LLM 的基本范式是：

```text
prefix -> next token -> append -> next token -> append -> ...
```

它的优势是质量高、训练稳定、KV cache 友好，但问题也很明显：

- 推理过程严格串行，一次 forward 通常只产出一个 token；
- 复杂问题需要长 CoT，导致 token 成本和上下文占用上升；
- 早期 reasoning step 一旦出错，后续步骤容易沿着错误前缀继续滚动；
- 低 batch size 解码时常处于 memory-bound，GPU 计算密度没有被充分利用。

这几篇论文分别从不同角度突破这个范式：

| 论文 | 主要路线 | 核心问题 |
| --- | --- | --- |
| DiffCoT | 显式 CoT 的 diffusion-style 修正 | 如何缓解 CoT 的 exposure bias 和错误累积 |
| Scaling Latent Reasoning via Looped LMs | 潜空间循环推理 | 如何在固定参数量下增加内部计算深度和知识操作能力 |
| Nemotron-Labs-Diffusion | AR、diffusion、self-speculation 三模式统一 | 如何让一个模型同时支持高质量生成和并行解码 |
| TiDAR | diffusion draft + AR talk 的单 forward 混合架构 | 如何利用 free token slots 提升实际 tokens/sec |

它们背后的共同趋势是：

```text
单一 AR 解码
  -> 显式 CoT 增加输出侧思考
  -> DiffCoT 允许显式推理步骤被修正
  -> LoopLM 把思考内化到 hidden states
  -> NLD / TiDAR 用 diffusion 并行 draft, 用 AR 保证最终表达质量
```

## 2. 四条技术路线

### 2.1 普通 CoT：把思考写成更多 token

普通 Chain-of-Thought 的做法是让模型先生成自然语言中间推理，再输出答案。

```text
Question -> step 1 -> step 2 -> step 3 -> answer
```

它的优点是工程简单、人类可读、兼容所有 AR 模型；缺点是推理 token 成本高，并且每一步都强依赖前面已经生成的内容。早期步骤如果出现语义偏移或计算错误，后面步骤通常无法主动回头修正。

因此，普通 CoT 的本质是：**用更长输出换取更多 test-time compute**。

### 2.2 DiffCoT：让显式 CoT 变成可修正轨迹

DiffCoT 没有放弃显式 CoT，而是把 CoT reasoning steps 看成一个可以被 iterative denoising 的轨迹。

它的关键设计包括：

- step-level forward noising：用 MCTS 或 reward-ranked candidates 为每个 reasoning step 构造从高质量到低质量的“噪声阶梯”；
- sliding-window denoising：模型在生成新步骤时，也能修正窗口内已经生成的历史步骤；
- causal diffusion noise：早期步骤噪声更弱、后期步骤噪声更强，用来保留推理链的因果结构；
- DPO-style preference optimization：把低噪声轨迹作为 winning sequence，把高噪声轨迹作为 losing sequence。

普通 CoT 是：

```text
写一步 -> 固定一步 -> 继续写下一步
```

DiffCoT 是：

```text
写一步 -> 在滑动窗口内修正旧步骤 -> 继续写下一步
```

DiffCoT 的价值在于，它直接针对 CoT 的错误累积问题。它仍保留自然语言推理过程，因此可读性强，也容易基于现有 AR 模型微调。但它的数据构造和训练更复杂，而且仍然保留长 CoT 的 token 成本。

### 2.3 LoopLM / Ouro：把思考放进 hidden states

LoopLM 的路线更激进：不再主要依赖输出更多 CoT token，而是在模型内部重复应用共享参数块，让 hidden states 经历多轮迭代更新。

形式上，普通 Transformer 是：

```text
Block1 -> Block2 -> ... -> BlockL
```

LoopLM 是：

```text
Shared Block Stack -> loop 1 -> loop 2 -> loop 3 -> loop 4
```

这相当于在固定参数量下增加有效计算深度。论文中的 Ouro 1.4B 和 2.6B 通过 7.7T tokens 预训练，在不少任务上接近或超过更大 dense Transformer，体现出明显参数效率优势。

更重要的是，论文通过 controlled experiments 说明：LoopLM 的收益不是来自“存储了更多知识”。在 synthetic biography 任务中，looped 和 non-looped 模型的知识容量都接近每参数 2 bits。LoopLM 更强的地方在于 **knowledge manipulation**：组合、检索和多跳使用已有知识。

LoopLM 的本质是：**用更多内部循环换取更强 latent reasoning，而不是用更多输出 token 换取显式思考**。

### 2.4 Nemotron-Labs-Diffusion：一个模型统一三种生成模式

Nemotron-Labs-Diffusion 提出 tri-mode LM，让同一个模型支持：

- AR decoding：标准左到右生成；
- diffusion decoding：block 内并行去噪；
- self-speculation：diffusion draft，AR verify。

训练目标是联合 AR loss 和 diffusion loss：

```text
L = L_AR + alpha * L_diff
```

作者强调 AR 和 diffusion 不是竞争关系，而是互补关系：

- AR 提供左到右语言先验和高质量 next-token modeling；
- diffusion 迫使模型学习未来 token 规划和块内并行预测；
- self-speculation 把 diffusion 的并行 draft 能力转化成实际推理加速。

这篇论文的关键思想是：**diffusion 不必替代 AR，它可以成为 AR 模型内部的高质量并行 draft 能力**。

### 2.5 TiDAR：在一个 forward 中 diffusion draft 与 AR talk

TiDAR 和 Nemotron-Labs-Diffusion 的 self-speculation 思想很接近，但 TiDAR 更强调 serving 和系统效率。

TiDAR 的核心观察是：低 batch 解码时，AR 模型通常 memory-bound，forward latency 主要花在加载权重和 KV cache 上。此时在同一次 forward 中多放几个 token slot，延迟可能几乎不变。论文称这些位置为 free token slots。

TiDAR 利用这些 free token slots 做两件事：

```text
验证上一轮 diffusion draft
+
预生成下一轮 diffusion draft
```

它在同一个模型、同一次 forward 中完成：

- diffusion-style parallel drafting；
- AR-style rejection sampling / verification；
- exact KV cache 支持；
- pre-draft 下一轮所有可能接受长度对应的候选。

因此 TiDAR 的标题可以理解为：

```text
Think in Diffusion: 用 diffusion 并行起草 token
Talk in Autoregression: 用 AR 决定最终输出 token
```

它不是主要提升 reasoning benchmark 的 CoT 方法，而是面向低延迟场景的生成加速架构。

## 3. 横向对比

| 维度 | 普通 CoT | DiffCoT | LoopLM | Nemotron-Labs-Diffusion | TiDAR |
| --- | --- | --- | --- | --- | --- |
| 推理载体 | 显式文本 | 显式文本步骤 | hidden states | token draft / decode modes | token draft / verify pipeline |
| 是否改架构 | 否 | 通常不改 backbone | 是 | 是 | 是 |
| 是否保留 CoT 可读性 | 是 | 是 | 否或弱 | 不以 CoT 为核心 | 不以 CoT 为核心 |
| 主要目标 | 提升复杂推理 | 修正 CoT 错误累积 | 参数效率和 latent reasoning | 三模式统一与推理加速 | 实际低延迟吞吐 |
| 计算扩展方式 | 输出更多 token | 步骤级 denoising refinement | 重复共享层 | diffusion draft + AR / diffusion decode | 单 forward draft + verify |
| 质量保障 | AR 生成本身 | diffusion-style 轨迹修正 | latent refinement | AR mode / AR verification | AR rejection sampling |
| 主要优势 | 简单、可解释 | 抗错误累积 | 参数效率高 | 灵活服务多场景 | tokens/sec 提升明显 |
| 主要短板 | 长、慢、容易滚错 | 数据和训练复杂 | 训练部署生态不成熟 | sampler 和基础设施仍有空间 | mask、cache、kernel 复杂 |

## 4. 实验评估效果对比

这几篇论文的实验不能直接混成一个总排行榜，因为它们优化的目标不同：DiffCoT 主要看数学推理准确率和纠错鲁棒性，LoopLM 主要看参数效率和 latent reasoning scaling，Nemotron-Labs-Diffusion 与 TiDAR 主要看质量-吞吐权衡。因此更合理的比较方式是把指标分成四类：

| 论文 | 主要评估对象 | 典型 benchmark | 核心指标 | 实验结论一句话 |
| --- | --- | --- | --- | --- |
| DiffCoT | 显式 CoT 推理修正 | GSM8K、SVAMP、MATH-L1 到 L5 | Accuracy、correction success rate | 相比 CoT / Step-DPO / Full-Step-DPO，在困难数学题上提升更明显 |
| LoopLM / Ouro | 潜空间循环推理与参数效率 | MMLU-Pro、BBH、GSM8K、MATH500、AIME、OlympiadBench、GPQA | Accuracy、pass@k、参数效率 | 1.4B 接近 4B，2.6B 在多项 reasoning benchmark 上追平或超过 8B dense 模型 |
| Nemotron-Labs-Diffusion | 三模式统一与 self-speculation | HumanEval、MBPP、GSM8K、Math500、AIME、MMLU、SPEED-Bench | Accuracy、TPF、throughput、SOL | 8B 模型在相近质量下实现更高 tokens per forward 和 SPEED-Bench 吞吐 |
| TiDAR | serving-oriented hybrid decoding | HumanEval、MBPP、GSM8K、Minerva Math、MMLU、ARC、HellaSwag | T/NFE、tokens/sec、accuracy | 1.5B / 8B 分别达到 4.71x / 5.91x tokens/sec 提升，同时保持接近 AR 的质量 |

### 4.1 质量-效率坐标图

可以把四条路线放到一个二维图里理解。横轴是推理吞吐或并行度，纵轴是任务质量或推理准确率。

```text
任务质量 / 推理准确率
^
|                         LoopLM
|                         参数更少，但 reasoning 能力接近更大模型
|
|             DiffCoT                      Nemotron-Labs-Diffusion / TiDAR
|             CoT 更稳                     质量接近 AR，同时显著提升吞吐
|
|  普通 CoT
|  可解释但慢
|
|        纯 diffusion / naive parallel decoding
|        并行强，但容易掉质量
+----------------------------------------------------------------> 吞吐 / 并行度
```

这个图的含义是：

- DiffCoT 主要把点往上推：质量更稳，但没有根本降低输出 token 成本；
- LoopLM 主要把点往左上推：同样或更少参数下获得更强 reasoning；
- Nemotron-Labs-Diffusion 和 TiDAR 主要把点往右上推：尽量不牺牲质量，同时提高并行解码效率；
- 纯 diffusion 并行解码如果没有 AR 兜底，容易向右下角滑，即更快但质量下降。

### 4.2 DiffCoT：数学推理准确率对比

DiffCoT 的核心实验是比较它和 CoT、TSFT、CPO、ToT、Step-DPO、Full-Step-DPO 在数学推理上的准确率。最直观的现象是：越难的题，DiffCoT 相比普通 CoT 的收益越明显。

| Backbone | Benchmark | CoT | Full-Step-DPO | DiffCoT | DiffCoT 相比 CoT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | GSM8K | 84.7 | 87.2 | 88.5 | +3.8 |
| Qwen3-4B | SVAMP | 86.8 | 89.7 | 90.2 | +3.4 |
| Qwen3-4B | MATH-L5 | 3.9 | 5.1 | 13.2 | +9.3 |
| Qwen3-8B | GSM8K | 87.3 | 88.7 | 91.5 | +4.2 |
| Qwen3-8B | MATH-L4 | 14.0 | 18.8 | 26.6 | +12.6 |
| Qwen3-8B | MATH-L5 | 4.9 | 7.1 | 14.9 | +10.0 |

用条形图看 MATH-L5 会更明显：

```text
Qwen3-4B / MATH-L5
CoT       3.9  | ██
FStep     5.1  | ███
DiffCoT  13.2  | █████████

Qwen3-8B / MATH-L5
CoT       4.9  | ███
FStep     7.1  | █████
DiffCoT  14.9  | ██████████
```

DiffCoT 的消融实验也说明，收益不是简单来自“多训练一点”，而是来自 sliding-window denoising 和 causal noise 的组合。

| Backbone | 设置 | GSM8K | SVAMP | 说明 |
| --- | --- | ---: | ---: | --- |
| Llama3-8B | Full DiffCoT | 64.4 | 76.9 | 完整方法 |
| Llama3-8B | window / stride = 1 | 62.9 | 75.8 | 退化成更接近 AR，纠错能力变弱 |
| Llama3-8B | window / stride = K | 55.4 | 68.5 | 过度 diffusion，因果结构被破坏 |
| Llama3-8B | 去掉 causal noise | 62.6 | 75.5 | 噪声不再尊重推理步骤顺序 |
| Qwen3-4B | Full DiffCoT | 88.5 | 90.2 | 完整方法 |
| Qwen3-4B | window / stride = K | 80.0 | 82.5 | 纯 diffusion 式窗口显著掉点 |

结论是：DiffCoT 的有效区间在 AR 和 full diffusion 中间。完全 AR 会缺少历史步骤修正，完全 diffusion 又会伤害 CoT 的因果顺序。

### 4.3 LoopLM：参数效率与循环深度对比

LoopLM 的实验最适合看“同等或更小参数量下，reasoning benchmark 是否变强”。论文中的 Ouro-1.4B-R4 可以接近或超过 Qwen3-4B；Ouro-2.6B-R4 在多个 reasoning-intensive benchmark 上超过 Qwen3-8B。

| 对比 | Benchmark | Dense baseline | Ouro LoopLM | 直观结论 |
| --- | --- | ---: | ---: | --- |
| Ouro-1.4B-R4 vs Qwen3-4B | BBH | 70.95 | 71.02 | 1.4B 接近 4B |
| Ouro-1.4B-R4 vs Qwen3-4B | GSM8K | 72.86 | 78.92 | 小模型反超 |
| Ouro-1.4B-R4 vs Qwen3-4B | MATH500 | 59.60 | 82.40 | 数学推理收益很大 |
| Ouro-2.6B-R4 vs Qwen3-8B | MMLU-Pro | 53.72 | 55.73 | 2.6B 超过 8B |
| Ouro-2.6B-R4 vs Qwen3-8B | BBH | 77.65 | 80.46 | 多步推理更强 |
| Ouro-2.6B-R4 vs Qwen3-8B | MATH500 | 62.30 | 90.85 | hard math 差距最大 |

把 MATH500 单独画出来：

```text
MATH500 accuracy
Qwen3-4B        59.60 | ████████████
Ouro-1.4B-R4    82.40 | ████████████████

Qwen3-8B        62.30 | ████████████
Ouro-2.6B-R4    90.85 | ██████████████████
```

但 LoopLM 的循环深度不是越多越好。Ouro-Thinking 在训练时使用 T=4，实验显示性能通常在 T=3 到 T=5 附近达到峰值，继续增加 recurrent steps 反而可能退化。

| 模型 | Benchmark | T=1 | T=2 | T=3 | T=4 | T=5 | T=8 | 观察 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ouro-1.4B-Thinking | AIME 2024 | 0.00 | 37.33 | 62.33 | 65.00 | 60.67 | 38.67 | T=4 左右最好 |
| Ouro-1.4B-Thinking | SuperGPQA | 2.03 | 33.07 | 44.50 | 47.37 | 48.73 | 42.88 | T=4/5 附近最好 |
| Ouro-2.6B-Thinking | AIME 2024 | 3.00 | 52.00 | 70.33 | 64.70 | 57.00 | 39.00 | T=3 最好 |
| Ouro-2.6B-Thinking | OlympiadBench | 18.96 | 68.59 | 75.56 | 76.44 | 71.85 | 39.26 | T=4 最好 |

这个结果说明：latent reasoning 的关键不是盲目“多想几轮”，而是让模型学会在合适深度停止。

### 4.4 Nemotron-Labs-Diffusion：三模式与 SOL 上限

Nemotron-Labs-Diffusion 的实验重点有两个：第一，联合 AR-diffusion training 是否稳定；第二，diffusion/self-speculation 是否真的能带来吞吐优势。

训练技术消融显示，完整训练流程把 diffusion-mode 平均准确率从 54.23 提升到 70.28，其中 two-stage training 和 AR loss 是最关键的两步。

```text
NLD diffusion-mode training ablation, Avg accuracy
Block-wise attention                 54.23 | ███████████
+ Global loss avg                    56.35 | ███████████
+ DP-rank varying masking ratios     57.06 | ███████████
+ Two-stage training                 62.80 | █████████████
+ AR loss                            70.28 | ███████████████
```

| 训练设置 | Avg accuracy | 增量解释 |
| --- | ---: | --- |
| Block-wise attention | 54.23 | 基础 diffusion LM 能跑起来 |
| + Global loss avg | 56.35 | 降低 variable masking 带来的梯度方差 |
| + DP-rank varying masking ratios | 57.06 | 更丰富的噪声比例 |
| + Two-stage training | 62.80 | 先强化 AR priors，再转 joint training |
| + AR loss | 70.28 | AR 目标显著增强 diffusion decoding 质量 |

在速度侧，论文给出两个很有代表性的数字：

| 指标 | 结果 | 含义 |
| --- | ---: | --- |
| NLD-8B vs Qwen3-8B tokens per forward | 6x | 相近质量下，每次 forward 可产出更多 token |
| NLD-8B on SPEED-Bench with SGLang / GB200 | 4x throughput | 真实 serving benchmark 上的吞吐提升 |
| Diffusion SOL 相比 self-speculation | +76.5% TPF | 理想 sampler 下 diffusion 还有未释放的并行潜力 |

SOL 分析还能看出 block size 的速度-质量取舍：

| Block size | SOL TPF / acceptance | Benchmark Acc | 直观解释 |
| ---: | ---: | ---: | --- |
| 4 | 2.89x | 64.04 | 更保守，质量较稳 |
| 8 | 4.17x | 65.43 | 质量最高的折中点 |
| 16 | 5.68x | 63.18 | 吞吐继续提升，质量开始回落 |
| 32 | 7.60x | 61.81 | 并行潜力最大，但准确率压力更大 |

这说明 NLD 的核心不是单纯追求最大 TPF，而是在不同服务场景下切换 AR、diffusion、self-speculation 三种模式。

### 4.5 TiDAR：tokens/sec 与质量前沿

TiDAR 的实验更系统工程导向：它关心的是一个 forward 能产出多少 token，以及这些 token 能不能通过 AR sampling 保住质量。

| 模型 | 生成质量 Avg | 平均 T/NFE | tokens/sec 相对 AR | 质量解读 |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-1.5B AR | 41.64 | 1.00 | 1.00x | AR baseline |
| TiDAR-1.5B | 44.03 | 7.45 | 4.71x | 质量不低于同源 AR，吞吐明显提升 |
| Qwen3-8B AR | 68.09 | 1.00 | 1.00x | 8B AR baseline |
| TiDAR-8B Trust Diff | 65.31 | 8.25 | 5.91x | 质量略降，但吞吐收益很大 |

吞吐提升可以这样看：

```text
Relative tokens/sec, AR = 1.00x
Qwen2.5-1.5B AR       1.00x | █
TiDAR-1.5B            4.71x | █████

Qwen3-8B AR           1.00x | █
TiDAR-8B              5.91x | ██████
```

TiDAR 还有一个重要观察：在 likelihood 任务上，TiDAR-8B 由于保留 AR likelihood 计算方式，平均分达到 75.40，略高于 Qwen3-8B 的 74.25。这说明它不是只在生成任务上“跑得快”，也保留了和 AR 评估兼容的 likelihood 能力。

| 模型 | Knowledge / Commonsense Avg |
| --- | ---: |
| Qwen3-8B AR | 74.25 |
| Dream-7B | 71.86 |
| LLaDA-8B | 68.06 |
| TiDAR-8B | 75.40 |

TiDAR 的一句话实验结论是：**在 batch size = 1 的 latency-critical 场景，free token slots 可以被 diffusion draft 高效利用，再由 AR sampling 保住质量。**

### 4.6 综合对比矩阵

| 方法 | 准确率提升 | 吞吐提升 | 参数效率 | 错误修正 | 可解释性 | 最适合的评估视角 |
| --- | --- | --- | --- | --- | --- | --- |
| 普通 CoT | 中 | 低 | 低 | 低 | 高 | baseline reasoning |
| DiffCoT | 高 | 低到中 | 中 | 高 | 高 | 数学推理准确率与纠错鲁棒性 |
| LoopLM | 高 | 中 | 高 | 中 | 低 | 参数效率、latent reasoning scaling |
| Nemotron-Labs-Diffusion | 中到高 | 高 | 中 | 中 | 低 | 质量-吞吐 trade-off、三模式服务 |
| TiDAR | 中 | 很高 | 中 | 中 | 低 | batch size = 1 的低延迟 tokens/sec |

如果只看“实验效果像什么”，可以这样概括：

```text
DiffCoT: 让 CoT 少滚错，尤其 hard math 更明显。
LoopLM: 让小模型通过内部循环获得大模型级 reasoning。
Nemotron-Labs-Diffusion: 一个模型三种开法，按场景切换质量和吞吐。
TiDAR: 把 GPU 空着的 token slot 填起来，用 AR 兜底输出质量。
```

## 5. 一个统一视角：思考、草稿、修正、表达

这几篇论文可以放进一个更统一的抽象：

```text
思考 Thinking
草稿 Drafting
修正 Revision
表达 Talking
```

不同方法只是把这些环节放在不同空间中：

| 方法 | Thinking | Drafting | Revision | Talking |
| --- | --- | --- | --- | --- |
| 普通 CoT | 文本 CoT | 文本步骤 | 几乎没有 | AR 输出 |
| DiffCoT | 文本 CoT | reasoning steps | sliding-window denoising | AR token generation |
| LoopLM | latent states | 中间 hidden predictors | recurrent refinement | LM head 输出 |
| Nemotron-Labs-Diffusion | diffusion / AR joint representation | diffusion draft | AR verify 或 sampler | AR / diffusion / self-spec |
| TiDAR | diffusion token slots | one-step diffusion pre-draft | AR rejection sampling | AR sampled output |

这个视角下，AR 不一定等于“思考”，它更像最终表达和质量约束；diffusion 不一定等于“最终生成”，它更像并行草稿、全局修正或未来规划机制。

## 6. 为什么 diffusion 在这些论文里反复出现

Diffusion 的吸引力不只在于“并行生成多个 token”。更深层的原因有三个。

第一，diffusion 天然支持从不完整或有噪声的状态中恢复，这和 reasoning 中的错误修正很匹配。DiffCoT 就是把 noisy reasoning chain 逐步修正成更 clean 的 CoT trajectory。

第二，diffusion 可以并行产生多个候选位置，适合作为 high-capacity drafter。Nemotron-Labs-Diffusion 和 TiDAR 都把 diffusion 放在 draft 侧，再用 AR 做 verification 或 sampling。

第三，diffusion 的非严格左到右结构有助于 future planning。Nemotron-Labs-Diffusion 的联合训练结果说明，AR 和 diffusion objective 可以互补，diffusion loss 可能增强模型对未来 token 的预测和规划能力。

所以这里的 diffusion 更像一种“并行思考和修正机制”，不是简单的 AR 替代品。

## 7. 工程落地判断

从短期到长期，可以这样看这些路线的落地难度：

```text
普通 CoT / preference optimization
  -> DiffCoT
  -> Nemotron-Labs-Diffusion self-speculation
  -> TiDAR serving-oriented hybrid architecture
  -> LoopLM-style pretraining architecture shift
```

短期最容易应用的是 DiffCoT 这类方法，因为它主要围绕现有 AR 模型做 step-level 数据构造和偏好优化。

中期最有部署吸引力的是 Nemotron-Labs-Diffusion 和 TiDAR，因为它们直接针对低 batch、低延迟、tokens/sec 这类实际 serving 目标。

长期最值得关注的是 LoopLM，因为它把 reasoning 作为预训练架构能力内化，而不是依赖后训练或输出侧技巧。但它需要新的训练稳定性经验、动态深度推理基础设施和更成熟的可解释分析工具。

## 8. 关键限制与开放问题

### 8.1 DiffCoT 的限制

- 数据构造依赖 MCTS、reward model 或 rollout success rate；
- 训练是 off-policy，行为策略和优化策略不一致可能带来分布偏移；
- revising historical steps 会打破普通 prefix-conditioned generation 的局部 Markov 性；
- 仍然需要输出显式 CoT，token 成本和上下文占用没有根本消失。

### 8.2 LoopLM 的限制

- recurrent computation 训练更不稳定，容易出现 loss spike 和梯度震荡；
- 动态 early-exit 对 vLLM / SGLang 这类现有推理系统不够友好；
- RLVR 在论文中没有带来明显收益，动态深度 rollout 仍是难点；
- hidden-state reasoning 不像 CoT 那样自然可读。

### 8.3 Nemotron-Labs-Diffusion 的限制

- diffusion mode 的实际 sampler 距离 SOL 上限仍有明显差距；
- self-speculation 只能接受 prefix，不能充分利用 diffusion 的非连续 token commit 潜力；
- quadratic self-speculation 虽然 TPF 高，但当前 kernel 和 attention mask 支持仍影响真实效率；
- 多模式 serving 需要框架层面更好支持。

### 8.4 TiDAR 的限制

- 训练时 clean section 和 mask section 使序列长度近似翻倍，长上下文训练成本更高；
- 特殊 attention mask、KV eviction、pre-draft 选择都要求专门推理实现；
- 论文重点测试 batch size = 1，较高并发下收益曲线需要结合硬件重新评估；
- 要充分利用 free token slots，后续仍依赖 custom kernels 和调度优化。

## 9. 我的思考

这几篇论文共同说明，LLM 推理正在从“输出更多 token”转向“设计更好的内部计算和草稿机制”。

普通 CoT 把思考暴露在文本里，简单有效，但代价是长、慢、容易错误累积。DiffCoT 说明即使保留显式 CoT，也应该让推理轨迹具备可修正性。LoopLM 则进一步说明，很多推理不一定要写出来，可以在 hidden states 中通过 recurrent refinement 完成。

Nemotron-Labs-Diffusion 和 TiDAR 则把问题推向系统层面：模型不仅要会想，还要会高效服务。它们都在弱化“diffusion 作为最终生成器”的定位，而是把 diffusion 作为并行 draft / planning 模块，让 AR 负责最终质量。

我觉得未来一个很自然的方向是把这些思想合起来：

```text
latent recurrent reasoning
  + diffusion-style draft / revision
  + AR-style final answer / verification
```

也就是说，模型内部可以通过 LoopLM 式循环做深层思考；需要生成时，用 diffusion 并行产生多个候选；最终输出由 AR 保证语言质量和因果一致性。这样一来，CoT 可以从“推理本体”变成“可选解释层”：只有在用户需要审计、教学或调试时才输出完整推理，否则模型可以主要在 latent space 和 draft space 中完成思考。

## 10. 后续阅读建议

如果继续围绕这个主题读论文，建议按下面顺序整理：

1. 先读 CoT / Step-DPO / Full-Step-DPO，理解传统显式推理链优化；
2. 再读 DiffCoT，理解显式 CoT 如何从 forward-only 变成 revisable trajectory；
3. 再读 LoopLM / latent reasoning，理解推理如何从文本空间迁移到 hidden states；
4. 再读 Nemotron-Labs-Diffusion，理解 AR 和 diffusion 如何联合训练并支持 self-speculation；
5. 最后读 TiDAR，理解这些模型能力如何转化为真实 serving throughput。

## 11. 一句话总结

这几篇论文共同揭示了一条清晰路线：**AR 仍然是高质量表达的核心，但推理、草稿和修正正在被迁移到 diffusion、latent recurrence 和 hybrid attention 结构中；未来的高效 LLM 很可能是“内部多轮思考、并行起草、AR 表达”的组合体。**
