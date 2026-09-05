# DeepSeek-V4 技术报告：面向百万上下文的高效长文本模型

> 截至日期：2026-06-25  
> 资料范围：DeepSeek 官方技术报告《Towards Highly Efficient Million-Token Context Intelligence》。  
> 结论先行：DeepSeek-V4 的核心不是单纯“把模型做大”，而是围绕百万 token 上下文，把注意力、KV cache、训练框架、推理框架和后训练链路一起重写。它最关键的技术主线是混合压缩注意力（CSA + HCA）、面向稳定训练的 mHC、面向 shared-prefix reuse 的异构 KV cache，以及把专家模型合并进统一学生模型的 OPD 后训练范式。

## 1. 这篇报告在解决什么问题

DeepSeek-V4 试图回答一个非常具体的问题：

- 当上下文长度从常规的 32K、128K 扩展到 1M token 时，Transformer 的注意力复杂度和 KV cache 体积会迅速成为瓶颈；
- 即使模型本身能力足够强，如果推理成本太高，长上下文、多轮 agent、长链推理这些场景仍然无法大规模落地；
- 因此，下一代长上下文模型不能只靠“继续堆参数”，而必须在注意力机制、缓存组织、内核实现和服务框架上同时做系统级优化。

报告给出的答案是：

```text
百万上下文可行性 = 压缩注意力 + 混合稀疏/稠密设计 + 更小 KV cache + 面向复用的推理框架 + 可承载这些结构的训练基础设施
```

## 2. 模型定位与规模

DeepSeek-V4 系列包含两个主要版本：

- **DeepSeek-V4-Pro**：1.6T 总参数，49B activated parameters。
- **DeepSeek-V4-Flash**：284B 总参数，13B activated parameters。

两者都支持 **1M token context**，但定位不同：

- Pro 更偏向高能力与高上限；
- Flash 更偏向更低激活参数下的高性价比推理。

报告最突出的效率结论是，在 1M 上下文场景下：

- DeepSeek-V4-Pro 的 single-token inference FLOPs 约为 DeepSeek-V3.2 的 **27%**，KV cache 约为 **10%**；
- DeepSeek-V4-Flash 的 single-token inference FLOPs 约为 DeepSeek-V3.2 的 **10%**，KV cache 约为 **7%**。

这说明 DeepSeek-V4 的重点不是单次算子的小修小补，而是把长上下文推理的主成本项整体压下来了。

## 3. 总体架构：继承了什么，又新增了什么

DeepSeek-V4 并不是完全推翻 DeepSeek-V3，而是在保留一些已验证组件的前提下，替换掉长上下文最关键的部分。

### 3.1 继承的部分

- 仍然使用 Transformer 主体。
- FFN 仍然采用 DeepSeekMoE 路线。
- 继续保留 MTP（Multi-Token Prediction）模块和目标。

### 3.2 新增的关键部分

- **混合注意力结构**：CSA + HCA。
- **mHC**：Manifold-Constrained Hyper-Connections，用来增强残差连接稳定性。
- **Muon 优化器**：替代大部分模块上的 AdamW。
- **异构 KV cache 推理框架**：专门适配压缩注意力和 shared-prefix reuse。
- **OPD 后训练范式**：把多个领域专家蒸馏回统一模型。

如果用一句话描述 DeepSeek-V4 的设计哲学，就是：

> 用压缩注意力解决长上下文的计算与显存问题，用工程化推理框架把这种结构真正服务化，再用后训练体系把能力补回来。

## 4. 核心创新一：混合注意力 CSA + HCA

这是整篇报告的中心。

### 4.1 为什么要做混合注意力

标准注意力在超长序列上有两个问题：

- attention FLOPs 太高；
- KV cache 随上下文长度线性膨胀，服务成本极高。

DeepSeek-V4 不是简单做稀疏注意力，而是把不同层设计成两类压缩注意力交替出现：

- **CSA（Compressed Sparse Attention）**：先压缩 KV，再做稀疏选择。
- **HCA（Heavily Compressed Attention）**：压得更狠，但保留 dense attention。

### 4.2 CSA 在做什么

CSA 的思路可以概括为：

1. 把连续多 token 的 KV 压缩成更少的 compressed KV entries；
2. 用一个 lightning indexer 为每个 query 选择 top-k 个最相关的 compressed KV blocks；
3. 再在这些选中的 compressed entries 上做核心注意力。

本质上，CSA 同时利用了：

- **序列维度压缩**；
- **选择性稀疏检索**；
- **低成本索引器**。

这比传统“全量 dense attention”便宜很多，也比完全静态稀疏更灵活，因为它仍然保留了 query-dependent 的动态选择能力。

### 4.3 HCA 在做什么

HCA 的思路比 CSA 更激进：

- 用更大的压缩率把更长范围的 KV 压成单个条目；
- 不再做 CSA 那种稀疏 top-k 选择，而是在重压缩后的条目上做 dense attention。

因此，HCA 更像一种“极端远程信息摘要器”。

可以把两者理解为：

- CSA 负责“压缩后再精挑细选”；
- HCA 负责“更粗粒度地保留超长距离上下文”。

### 4.4 为什么两者要交替混用

单独使用 CSA 或单独使用 HCA 都会有偏科：

- 只有 CSA，索引器和 sparse selection 的维护成本会更高；
- 只有 HCA，压缩过重可能损失细粒度信息。

交替使用后：

- CSA 层保留较强的信息选择能力；
- HCA 层提供超长程压缩能力；
- 整体上更适合 1M 上下文。

### 4.5 它不是简单“少看点 token”

报告里额外加了几项补偿机制，说明 DeepSeek-V4 很清楚压缩注意力会带来的表达损失：

- **Sliding Window Branch**：额外给最近若干 token 保留未压缩窗口，弥补局部细节。
- **Partial RoPE**：只在部分维度上施加旋转位置编码。
- **Attention Sink**：允许注意力头调节总注意力质量，不强迫每个头都把概率全部分掉。

这些设计说明它不是粗暴压缩，而是在努力保住局部依赖和训练稳定性。

## 5. 核心创新二：mHC 让深层训练更稳

DeepSeek-V4 另一个很有辨识度的创新是 **mHC（Manifold-Constrained Hyper-Connections）**。

### 5.1 它想解决什么问题

普通残差连接在超深模型里会遇到稳定性问题，而 Hyper-Connections 虽然能提升表达能力，但继续堆深时也容易出现数值不稳定。

mHC 的核心想法是：

- 把残差映射矩阵约束到 **双随机矩阵流形** 上；
- 让残差变换保持 non-expansive，也就是谱范数不超过 1；
- 这样前向传播和反向传播都更稳定。

### 5.2 关键直觉

报告的技术重点不是“多加一条残差分支”，而是给残差映射增加了几何约束：

- 输入映射和输出映射受 Sigmoid 约束，避免失控；
- 残差矩阵通过 Sinkhorn-Knopp 迭代投影到双随机集合；
- 这样即使层数很深，信号传播也不容易爆炸。

这个设计很像把“深层稳定性”从经验技巧提升到结构约束。

## 6. 核心创新三：KV cache 与 shared-prefix reuse

如果只看论文标题，容易以为 DeepSeek-V4 只是“新注意力结构”。但从工程价值看，它对推理框架的设计同样关键。

### 6.1 为什么传统 KV cache 不够用

DeepSeek-V4 的注意力层并不统一：

- CSA 和 HCA 的 KV 条目长度不同；
- SWA 层又有不同缓存策略；
- 压缩分支还会存在尚未凑满压缩块的 tail tokens。

这意味着传统统一 block 视角的 PagedAttention 假设被破坏了。报告直接指出，混合注意力会挑战原有 KV cache 管理框架。

### 6.2 他们怎么组织 KV cache

DeepSeek-V4 把 KV cache 分成两大类：

- **classical KV cache**：给 CSA/HCA 的压缩条目使用；
- **state cache**：给 SWA 和尚未凑齐压缩条件的尾部未压缩 token 使用。

这是一种典型的“异构缓存分层”思路：

- 主缓存负责真正适合长期复用的压缩前缀；
- 状态缓存负责短期、位置相关、更新更频繁的局部状态。

### 6.3 为了提高前缀复用，他们做了什么

这一部分最值得和 vLLM 的 prefix caching 对照理解。

DeepSeek-V4 在共享前缀复用上做了几件事：

1. **压缩后的 CSA/HCA KV 直接落盘**，作为主要可复用对象。
2. **命中 shared prefix 时，直接读取并复用压缩 KV**，直到最后一个完整压缩块。
3. **尾部不完整压缩块只做局部重算**，而不是整段 prefix 重做。
4. **SWA 单独设计三种恢复策略**，避免窗口态拖垮前缀缓存体系。

这说明 DeepSeek-V4 的目标不是只让“缓存更小”，而是让：

```text
共享前缀更容易保存下来
共享前缀命中后更容易恢复
命中后只对最后一小段做必要重算
```

### 6.4 SWA 的三种缓存策略

报告给了三种 on-disk SWA KV 管理方案：

- **Full SWA Caching**：把 SWA KV 全存下来，命中后几乎零重算，但写入很重。
- **Periodic Checkpointing**：每隔一段 token 存一次 SWA 状态，命中后从最近 checkpoint 恢复，再补尾巴。
- **Zero SWA Caching**：完全不存 SWA，只借助已缓存的 CSA/HCA 条目重算最后一段窗口态。

这里体现出的工程思维很务实：

- 不是追求单一最优策略；
- 而是根据 SSD 写放大、存储预算、恢复时延做部署期权衡。

### 6.5 这对“缓存命中率”的真正意义

报告没有公布一个显式 cache hit rate 百分比，但它对命中效果的努力非常明确：

- 更小的 KV cache 意味着同等资源下可保留更多历史 prefix；
- 压缩块作为主缓存对象，让大前缀更适合持久化；
- SWA 从主缓存中拆出去，减少对 prefix reuse 的干扰；
- 落盘设计让共享前缀不只在显存里有效，也能跨请求、跨时段复用。

所以它优化的不是单纯“哈希命中规则”，而是**共享前缀缓存体系的整体可用性**。

## 7. 训练与推理基础设施

DeepSeek-V4 的报告有很大篇幅不是在讲模型，而是在讲基础设施。这意味着它们认为新模型结构必须和系统实现协同设计。

### 7.1 MoE 侧的基础设施

- 单个 fused kernel 同时重叠计算、通信和访存。
- 细粒度 expert waves 调度，用来提升 EP 场景的 overlap。
- 在推理和 RL rollout 等小 batch、长尾场景下也有明显收益。

### 7.2 TileLang 与可复现内核

报告强调两点：

- 用 TileLang 提高复杂 kernel 的开发效率；
- 用 batch-invariant、deterministic kernels 维持训练、后训练和推理的一致性。

这部分很重要，因为如果长上下文模型在不同 batch、不同 kernel path 下行为不一致，那么后训练和服务部署都会变得难以调试。

### 7.3 面向长上下文的训练框架改造

- 为 Muon 设计 hybrid ZeRO 策略；
- 用 tensor-level checkpointing 做更细粒度重算控制；
- 为压缩注意力设计 two-stage contextual parallelism。

这些改造说明 DeepSeek-V4 不是“先想出结构，再硬塞进旧框架”，而是训练框架也随结构同步演进。

## 8. 后训练：从多个领域专家回蒸馏到统一模型

DeepSeek-V4 的 post-training 也很有代表性。

### 8.1 两阶段后训练范式

它采用：

1. **先训练多个 domain specialists**，例如数学、代码、agent、通用指令跟随；
2. **再通过 OPD（On-Policy Distillation）合并回统一模型**。

这个思路的优点是：

- 专家模型可以在各自任务上做更激进的 RL 或偏好优化；
- 最终部署不需要维护一堆独立专家权重；
- 统一学生模型仍然保留多域能力。

### 8.2 它为什么强调 OPD

报告明确表示，它用 OPD 取代了以前的 mixed RL 阶段。原因很现实：

- 多专家能力直接混在一个 RL 过程中，优化目标容易打架；
- 先训专家，再统一蒸馏，更容易保持每个方向的上限；
- full-vocabulary logit distillation 比 token-level 近似更稳。

### 8.3 Generative Reward Model

对 hard-to-verify 任务，DeepSeek-V4 不再依赖传统标量 reward model，而是使用 **Generative Reward Model**：

- 模型本身生成判语和评价；
- actor 同时兼任 GRM；
- 把“会做题”和“会评题”尽量统一在一个推理系统里。

这和近年把评估能力内生化的趋势是一致的。

## 9. Quick Instruction：直接复用已有 KV cache

这是一个很工程化但很实用的点。

在聊天系统里，搜索决策、query 生成、authority/domain 分类这些辅助任务，传统上会交给一个小模型做。问题在于：

- 小模型要重新 prefill；
- 这会造成冗余时延；
- 同时还增加系统复杂度。

DeepSeek-V4 的做法是：

- 在原输入序列后面直接附加 special tokens；
- 利用已经算好的 KV cache 并行完成这些辅助判断。

这件事本身不提高模型能力上限，但它非常符合现代产品化思路：

> 只要同一段上下文已经算过，就尽量不要让另一个小模型再重复算一遍。

## 10. DeepSeek-V4 的关键数字

| 项目 | DeepSeek-V4-Flash | DeepSeek-V4-Pro |
| --- | --- | --- |
| 总参数量 | 284B | 1.6T |
| Activated Params | 13B | 49B |
| 上下文长度 | 1M | 1M |
| 训练 token 数 | 32T | 33T |
| CSA 压缩率 | 4 | 4 |
| HCA 压缩率 | 128 | 128 |
| SWA 窗口大小 | 128 | 128 |
| 1M 上下文下相对 V3.2 的 KV cache | 7% | 10% |
| 1M 上下文下相对 V3.2 的单 token FLOPs | 10% | 27% |

这些数字反映出一个非常清晰的设计选择：

- 远程上下文被大幅压缩；
- 局部窗口被保留；
- 长距离信息访问被稀疏化；
- 模型能力则通过更复杂的训练和后训练管线补回。

## 11. 这篇报告最值得学的技术思想

### 11.1 长上下文不能只靠一个 trick

DeepSeek-V4 的启发是：

- 只做稀疏注意力不够；
- 只做 KV 压缩不够；
- 只做 PagedAttention 级缓存复用也不够；
- 需要把注意力结构、缓存布局、落盘策略、kernel 和训练框架一起改。

### 11.2 “压缩”不等于“粗暴丢信息”

CSA、HCA、Sliding Window、Indexer、Partial RoPE、Attention Sink 共同说明，它想做的是“分层保留信息”，而不是把所有远程 token 一刀切成摘要。

### 11.3 服务框架正在反向影响模型结构

DeepSeek-V4 非常典型地体现出：

- 模型结构不再只由算法论文决定；
- 它会被推理成本、缓存复用、SSD 读写模式、kernel 对齐约束反向塑形。

这也是为什么它的 KV cache 管理和 on-disk storage 值得单独当作架构创新来看。

## 12. 局限与开放问题

即使报告很完整，仍然有一些问题没有完全解决：

- CSA/HCA 的压缩信息损失到底如何随任务类型变化，报告没有给出足够细粒度的消融。
- 压缩注意力在极端知识检索、代码精细依赖、复杂长文事实追踪上的失败案例并未系统展开。
- shared-prefix reuse 的命中率指标、真实线上 workload 分布、SSD 成本模型也没有完整公开。
- OPD 虽然能统一专家能力，但不同领域之间是否出现能力冲突，报告没有给出更多长期分析。

这些空白说明 DeepSeek-V4 更像一篇“方向已经非常明确的系统报告”，而不是彻底封闭所有疑问的定论。

## 13. 对学习者的技术解读

如果从学习路线看，这篇报告最值得拆成四层理解：

1. **模型层**：CSA/HCA 是如何把 attention 变成“压缩 + 选择 + 局部保真”的。
2. **稳定性层**：mHC 和 Muon 如何支撑更复杂结构稳定训练。
3. **系统层**：异构 KV cache、落盘 prefix reuse、deterministic kernels 如何把结构真正服务化。
4. **产品层**：Quick Instruction、Think modes、OPD specialists 如何把模型能力组织成可发布的产品系统。

如果只学第一层，会把 DeepSeek-V4 看成“又一个长上下文注意力变体”；但如果把四层一起看，会更接近这篇报告真正的价值。

## 14. 一句话总结

DeepSeek-V4 的真正创新不是单一算法点，而是围绕“百万 token 上下文能否以可接受成本稳定训练、稳定推理并可服务化部署”这个问题，给出了一套从注意力、缓存、内核到后训练的系统级答案。

## 15. 参考资料

- [DeepSeek 技术报告原文](https://huggingface.co/collections/deepseek-ai/deepseek-v4)
- 研究整理稿：Towards Highly Efficient Million-Token Context Intelligence (请参考本地项目根目录下的 research_files/papers_md 目录)
