# Nemotron-Labs-Diffusion 技术报告：统一 AR、Diffusion 与 Self-Speculation 的三模式语言模型

> 论文：Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding  
> 核心关键词：Autoregressive LM、Diffusion LM、Block Diffusion、Self-Speculation、Speculative Decoding、TPF、SGLang  
> 结论先行：这篇报告的核心不是提出一个纯 diffusion LM 来替代 AR LM，而是把 diffusion 重新定位为 AR 模型内部的并行预测能力。Nemotron-Labs-Diffusion 通过联合 AR-diffusion 训练，让同一个模型同时支持 AR 解码、block-wise diffusion 解码和 self-speculation 解码，从而在准确率基本保持的同时提升推理吞吐。

## 1. 技术背景与问题意识

传统自回归语言模型按 token 从左到右逐步生成：

```text
x1 -> x2 -> x3 -> ... -> xn
```

这种方式具有很强的语言建模能力，也天然适合 KV cache，但它的推理过程严格串行，每次 forward 通常只产生一个新 token。在低 batch size 或低并发场景下，GPU 计算资源容易利用不足，单用户吞吐受限。

Diffusion language model 尝试通过并行去噪一次生成多个 token，从而突破 token-by-token decoding 的限制。但已有 diffusion LM 常见问题包括：

- 准确率和训练效率往往落后于强 AR 模型；
- 由于需要双向建模，和 KV cache 的天然配合不如 AR；
- 实际部署中的速度优势未必稳定；
- 与 speculative decoding / MTP 等加速方法相比，优势并不总是明显。

这篇报告围绕三个问题展开：

1. Diffusion LM 是否应该和 AR LM 竞争，还是可以与 AR 统一？
2. Diffusion 能否提供比 MTP 更强的加速机制？
3. Diffusion decoding 的长期上限是否足够值得继续探索？

作者给出的答案是：**AR 与 diffusion 可以互补，self-speculation 是当前最实用的落地形式，而 diffusion mode 本身仍有更高的理论吞吐上限。**

## 2. 核心工作概览

Nemotron-Labs-Diffusion 的核心设计可以概括为：

```text
联合训练 AR objective + diffusion objective
        ↓
同一个 backbone 学到左到右预测与块内并行预测
        ↓
推理时通过 attention mask / decoding strategy 切换三种模式
        ↓
在不同部署场景下选择 AR、diffusion 或 self-speculation
```

它支持三种推理模式：

| 模式 | 生成方式 | 优点 | 适用场景 |
| --- | --- | --- | --- |
| AR mode | 从左到右逐 token 生成 | 稳定、兼容现有 serving、KV cache 友好 | 高并发、标准部署 |
| Diffusion mode | block 内并行去噪 | 一次 forward 可提交多个 token，理论上限高 | 追求并行解码、未来 sampler 优化 |
| Self-speculation mode | diffusion draft + AR verify | 实际部署收益明显，不需要额外 draft model | 低并发、单用户吞吐优化 |

报告发布了 3B、8B、14B 多个规模的 base、instruct 和 vision-language 模型，并在多个 benchmark 上展示了准确率与效率收益。

## 3. 联合 AR-Diffusion 训练

### 3.1 AR 目标

AR 目标是标准 next-token prediction：

$$
\mathcal{L}_{\mathrm{AR}}(\theta)=\mathbb{E}_{x\sim\mathcal{D}}\left[-\sum_i \log p_\theta(x_i \mid x_{<i})\right]
$$

它提供强左到右语言先验，让模型保持自然语言生成能力和现有 LLM 部署兼容性。

### 3.2 Diffusion 目标

报告采用 block-wise diffusion。序列被切分成多个连续 block，每次只对当前 block 加噪并让模型去噪，前缀 block 保持 clean：

```text
clean prefix + noisy current block -> denoise current block
```

block 内允许双向注意力，因此可以并行预测多个位置；block 间保持 causal 关系，因此仍能复用前缀 KV cache。

### 3.3 联合目标

总 loss 为：

$$
\mathcal{L}(\theta)=\mathcal{L}_{\mathrm{AR}}(\theta)+\alpha\mathcal{L}_{\mathrm{diff}}(\theta)
$$

报告中默认设置 $\alpha=0.3$。消融结果显示，这个权重在 AR mode 与 diffusion mode 上都表现较好，说明两种目标不是简单的零和竞争。

## 4. 关键训练策略

### 4.1 两阶段训练

训练采用两阶段策略：

1. **Stage 1：纯 AR 训练**  
   先强化左到右语言建模能力，建立稳定的语言先验。

2. **Stage 2：AR + diffusion 联合训练**  
   在已有 AR 能力基础上加入 diffusion supervision，让模型获得块内并行预测能力。

消融结果显示，两阶段训练非常关键。强 AR 初始化能提升 diffusion 学习效率，也让 diffusion mode 的输出更加连贯。

### 4.2 Global Loss Averaging

Diffusion 训练中，不同样本被 mask 的 token 数量不同，且噪声水平不同，会导致 token loss 的贡献不均衡。

报告比较了两种 loss averaging：

- sequence-wise averaging：先对每条样本内部平均，再对 batch 平均；
- global averaging：把 batch 中所有 contributing tokens 统一平均。

作者发现 global averaging 更稳定，因为它避免了少量高权重 masked token 对整个 batch 梯度产生过大影响。

### 4.3 Clean Stream 使用严格 Causal Mask

模型训练时使用 noisy stream 和 clean stream 的 dual-stream 输入。关键差异在于 clean stream：

- 之前一些 block diffusion 设计允许 clean stream 看未来 token；
- Nemotron-Labs-Diffusion 在 clean stream 中使用严格 causal mask；
- 这样可以在同一次 forward/backward 中同时计算 AR loss 和 diffusion loss，且不会发生 label leakage。

这是联合 AR-diffusion 训练成立的关键结构条件之一。

## 5. 三种推理模式

### 5.1 Mode 1：AR Decoding

AR mode 保留标准左到右生成：

```text
p(x_i | x_<i)
```

它的意义是让 Nemotron-Labs-Diffusion 可以作为常规 AR 模型的 drop-in replacement。高并发场景下，AR mode 仍然可能是最合适的部署方式。

### 5.2 Mode 2：Block-wise Diffusion Decoding

Diffusion mode 以 block 为单位生成。当前 block 初始化为 mask tokens，然后模型多轮去噪：

```text
[M] [M] [M] [M]
  ↓
commit high-confidence positions
  ↓
继续去噪剩余 mask positions
```

论文讨论了两种采样方式：

- confidence-based sampling：根据置信度阈值提交 token；
- trained sampler：训练一个轻量 sampler 来判断每个位置的 top-1 prediction 是否可以安全提交。

实验显示 trained sampler 能改善 accuracy-TPF trade-off，说明 diffusion decoding 的瓶颈不仅在模型本身，也在 sampler 是否能判断哪些 token 可以并行提交。

### 5.3 Mode 3：Self-Speculation Decoding

Self-speculation 是本文最有实际部署价值的推理模式。

流程如下：

```text
已验证前缀
   ↓
diffusion pathway 并行 draft k 个 token
   ↓
AR pathway 验证 draft tokens
   ↓
接受最长正确前缀，遇到第一个 mismatch 停止
```

它和传统 speculative decoding 的区别是：

| 方法 | Drafter | Verifier |
| --- | --- | --- |
| Eagle / speculative decoding | 小模型或辅助 head | 大 AR 模型 |
| Nemotron-Labs-Diffusion self-speculation | 同一个模型的 diffusion pathway | 同一个模型的 AR pathway |

因此它不需要额外训练一个小 draft model，也不需要模型外部的独立 verifier。

## 6. LoRA 增强的 Self-Speculation

作者进一步给 diffusion drafter 加 LoRA，使其 draft 更贴近 AR verifier。

设计要点：

- LoRA 只作用于 diffusion drafter 路径；
- AR verifier 固定不变；
- loss 包括 top-K distribution matching 和 token-level cross entropy；
- 只对“accepted prefix + 第一个 rejected position”计算 loss。

最后一点很重要。因为推理时一旦某个位置被拒绝，后面的 speculative continuation 就不会被真实采用。如果训练后续位置，会让 drafter 学到部署时不会出现的反事实上下文。

实验结果显示，LoRA 能显著提升 self-speculation 的平均 TPF，同时几乎不损失准确率。

## 7. Speed-of-Light 分析

报告提出 Speed-of-Light 分析来估计 diffusion mode 的理论上限。

核心问题是：

> 如果我们有一个近似 oracle 的策略，最多可以每次 forward 安全提交多少 token？

作者先通过 serial denoising 得到 diffusion 模型最终会收敛到的目标序列，然后分析哪些位置可以并行提交且最终仍能得到同样结果。

结论很重要：

- diffusion mode 在 block length 32 时，SOL acceptance rate 平均约为 7.60x；
- linear self-speculation 的 acceptance rate 接近这个上限，但真实 TPF 因为需要 draft + verify 两次 forward，大约只有一半；
- SOL 真实 TPF 约 6.02x，而 linear self-speculation 约 3.41x；
- 这意味着 diffusion mode 理论上仍有显著 headroom。

作者据此认为，未来更强的 diffusion sampler 可能比 prefix-only 的 AR verification 更有潜力，因为 diffusion mode 可以接受非连续位置，而 self-speculation 通常只能接受最长前缀。

## 8. 实验结论

### 8.1 Instruct 模型

以 Nemotron-Labs-Diffusion-8B instruct 为例：

- AR mode 平均准确率超过 Qwen3-8B；
- diffusion mode 在保持相近或更高准确率的同时提升 TPF；
- linear self-speculation 可进一步提升 TPF；
- quadratic self-speculation 的 TPF 更高，但受当前 kernel / attention mask 实现影响，真实设备效率不一定优于 linear self-speculation。

报告中特别强调，当前基础设施下 **linear self-speculation 是最实用的模式**。

### 8.2 Base 模型

在 base model benchmark 中，Nemotron-Labs-Diffusion-8B 同样表现出：

- AR mode 不弱于强 AR baseline；
- diffusion mode 明显强于已有 diffusion LM；
- self-speculation 在准确率保持的同时显著提高 TPF。

这说明三模式能力不是只在 instruction tuning 后出现，而是已经内化到 base model 能力中。

### 8.3 多尺度模型

报告评估了 3B、8B、14B。总体趋势是：

- 模型越大，越容易解锁更强的 parallel prediction 能力；
- self-speculation 的平均 TPF 随模型规模增大而提升；
- 更大的模型通常具备更强未来 token 预测能力，因此 draft acceptance length 更长。

### 8.4 Vision-Language Model

作者还把 Nemotron-Labs-Diffusion 扩展到 VLM：

- 加入 vision encoder 和 multimodal projector；
- LM backbone 继承 joint AR-diffusion 训练能力；
- 使用 asymmetric dual-stream layout，避免在 noisy half 中重复携带永不被 mask 的 vision tokens。

这个设计降低了视觉 token 带来的额外 attention FLOPs，同时保留完整视觉上下文。

## 9. 与 MTP / Eagle3 的对比

MTP 和 Eagle3 类 speculative decoding 依赖额外 draft module 或 draft model。它们的问题是：

- 小 drafter 能力有限，长 horizon draft 不稳定；
- 递归生成 draft 仍有额外开销；
- acceptance length 较短。

Nemotron-Labs-Diffusion 的优势在于：

- diffusion pathway 本身来自同一个强 backbone；
- draft 多个 token 时可以并行生成；
- AR verifier 与 diffusion drafter 共享模型能力；
- 平均 acceptance length 明显高于 Eagle3 / MTP。

在 SPEED-Bench 上，报告显示 LoRA-enhanced self-speculation 的平均 acceptance length 高于 Eagle3 和 MTP，尤其在 coding、math、reasoning、multilingual 等结构性较强的任务上优势更明显。

## 10. 关键洞察

这篇报告最值得记住的不是某个单点指标，而是以下几个判断：

1. **AR 与 diffusion 不是互斥路线**  
   AR 提供语言先验，diffusion 提供并行预测能力，联合训练可以互补。

2. **Diffusion 可以成为同一个 AR 模型内部的 draft 能力**  
   Self-speculation 不需要额外小模型，而是让模型自己 draft、自己 verify。

3. **当前最实用的是 linear self-speculation**  
   它在现有 serving infrastructure 下更容易获得真实吞吐收益。

4. **Diffusion decoding 的长期上限更高**  
   SOL 分析显示，非 prefix 的并行提交还有明显潜力，未来关键在更强 sampler。

5. **训练稳定性很重要**  
   两阶段训练、global loss averaging、合适的 AR-diffusion loss 权重，都是让 diffusion 能力不破坏 AR 能力的关键。

## 11. 对学习者的技术解读

如果把这篇报告放进 LLM 推理加速脉络中，它的位置大致是：

```text
标准 AR 解码
  ↓
Speculative Decoding / MTP：用外部或辅助模块 draft
  ↓
Self-Speculation：同一个模型内部 diffusion draft + AR verify
  ↓
更强 Diffusion Sampler：未来可能直接用 diffusion mode 高效提交非连续 token
```

它提供了一种很有启发性的模型设计方向：**不要把 diffusion LM 只看作 AR LM 的替代品，而要把 diffusion objective 看成增强 AR 模型未来预测和并行解码能力的一种训练信号。**

从工程角度看，Nemotron-Labs-Diffusion 的价值在于它没有要求部署系统完全抛弃 AR serving。模型可以在高并发时走 AR，在低并发时走 self-speculation，在未来 sampler 更强时走 diffusion mode。这种三模式设计比单一路线更灵活。

## 12. 一句话总结

Nemotron-Labs-Diffusion 通过联合 AR-diffusion 训练，把传统 AR 语言模型、扩散式并行生成和自投机解码统一到同一个模型中，证明了 diffusion 不必替代 AR，而可以作为 AR 模型内部的并行预测能力，在保持准确率的同时显著提升推理效率。
