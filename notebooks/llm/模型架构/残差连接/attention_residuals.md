这是一份关于 Kimi 团队技术报告《ATTENTION RESIDUALS》的详细阅读笔记，涵盖了论文的主要创新点、技术细节、公式推导、工程实现以及实验结果五大核心模块。

---

# 论文阅读笔记：《Attention Residuals》

## 1. 主要创新点
*   **深度维度的注意力机制 (Attention Residuals, AttnRes)**：打破了现代 LLM 中标准残差连接（固定权重累加）的范式，提出利用**动态的、数据依赖的 Softmax 注意力机制**来聚合历史层的特征。这类似于序列模型中用 Transformer 替代 RNN（将序列维度的状态压缩替换为全局注意力），本文在“网络深度”维度上实现了同样的跨越。
*   **兼顾效率的 Block AttnRes**：为了解决 Full AttnRes 在大规模模型训练中带来的庞大内存和流水线通信开销（$O(Ld)$），提出将模型层划分为 $N$ 个块（Block）。块内进行标准残差累加，块间使用注意力聚合，将显存和通信复杂度降低到 $O(Nd)$，逼近标准残差的效率。
*   **系统级的工程突破**：设计了一套专为大语言模型规模化训练和推理的基础设施，包括“跨阶段缓存（Cross-stage caching）”以消减流水线并行通信，以及“两阶段计算策略（Two-phase computation）”结合 Online Softmax，将推理延迟增加控制在 2% 以内。

## 2. 技术细节
*   **Time and Depth 的对偶性**：标准的残差网络在深度上累加信息，这与 RNN 在时间维度上压缩状态本质相同。标准残差无法做到“差异化检索”，且会导致 PreNorm 架构下深层 Hidden State 范数不受控制地线性增长（即 PreNorm 稀释问题）。AttnRes 通过学习一个逐层独立的伪查询向量（Pseudo-query），在深度维度进行注意力聚合。
*   **Key 和 Value 的设计**：每一层的输入向量除了上一层的输出，还包括了最初的 Token Embedding。为了防止某些输出数值过大的层主导 Softmax 权重，所有的 Key 在进入内积计算前必须先通过 **RMSNorm** 进行归一化。
*   **查询向量（Query）解耦**：AttnRes 的 Query 不是由当前隐藏状态映射得来的，而是**为每一层单独初始化的可学习参数（Learned Parameter）**。这种解耦非常关键，它意味着查询不需要等待前序层的前向计算完成即可并发执行，为后续的推理优化（如批处理检索）奠定了基础。

## 3. 公式推导（模型演进的数学表达）

**1) 标准残差的局限（深度上的循环网络）：**
标准残差公式：

$$ \boldsymbol{h}_l = \boldsymbol{h}_{l-1} + f_{l-1}(\boldsymbol{h}_{l-1}) $$

展开后为无权重的累加（等价于全部设为 1）：

$$ \boldsymbol{h}_l = \boldsymbol{h}_1 + \sum_{i=1}^{l-1} f_i(\boldsymbol{h}_i) $$


**2) Full Attention Residuals 的推导：**
将固定的累加权重替换为动态的注意力权重 $\alpha_{i \to l}$：

<span class="arithmatex">\(\boldsymbol{h}_l = \alpha_{0 \to l} \cdot \boldsymbol{h}_1 + \sum_{i=1}^{l-1} \alpha_{i \to l} \cdot f_i(\boldsymbol{h}_i)\)</span>

其中，注意力权重通过 Softmax 计算：

$$ \alpha_{i \to l} = \frac{\phi(\boldsymbol{q}_l, \boldsymbol{k}_i)}{\sum_{j=0}^{l-1} \phi(\boldsymbol{q}_l, \boldsymbol{k}_j)} $$

核函数定义为：$\phi(\boldsymbol{q}, \boldsymbol{k}) = \exp(\boldsymbol{q}^\top \text{RMSNorm}(\boldsymbol{k}))$
在具体映射上：
*   Query: $\boldsymbol{q}_l = \boldsymbol{w}_l$ （每一层初始化的维度为 $d$ 的可学习向量）。
*   Key / Value: 对初始 Embedding，$\boldsymbol{k}_0 = \boldsymbol{v}_0 = \boldsymbol{h}_1$；对于后续层 $\boldsymbol{k}_i = \boldsymbol{v}_i = f_i(\boldsymbol{h}_i)$。

**3) Block Attention Residuals 的化简：**
为了减少历史缓存，将 $L$ 个层平均分为 $N$ 个块。
首先对第 $n$ 块内的层输出进行局部求和（聚合成 Block 表征）：
$$ \boldsymbol{b}_n = \sum_{j \in \mathcal{B}_n} f_j(\boldsymbol{h}_j) $$
当运行到第 $n$ 个块的中间层（设为该块内的第 $i$ 层）时，其 Value 矩阵集合变为：
前置所有 Block 的表征 + 当前 Block 的部分和：
$$ \mathbf{V} = [\boldsymbol{b}_0, \boldsymbol{b}_1, \ldots, \boldsymbol{b}_{n-1}, \boldsymbol{b}_n^{i-1}]^\top $$
这样，注意力计算的历史状态数从 $L$ 锐减到了 $N$ 左右。

## 4. 工程实现步骤

为了将 AttnRes 扩展到千亿参数的大模型训练与推理中，论文实施了以下关键工程步骤：

### 4.1 训练端 (Training)
*   **跨阶段缓存 (Cross-stage caching)**：在流水线并行（Pipeline Parallelism）中，如果使用 Interleaved 调度，相邻阶段直接传输全部历史 Block 会产生严重的通信冗余 $O(P \times V)$。工程上，让每个物理设备在本地**缓存**之前的虚拟阶段接收到的 Block，每次 Pipeline 切换时，仅传输**新增的** Block（增量传输），使得峰值通信成本大幅降低，完全可以掩盖在计算流中。

### 4.2 推理端 (Inference)
*   **两阶段计算策略 (Two-phase computation)**：
    *   *Phase 1（块间并行注意力）*：将一个 Block 内所有层的伪查询向量（Pseudo-queries）组成一个矩阵，**一次性并发地**和历史缓存的 Block 表征做 Attention，返回结果和 Softmax 的统计量（Max 和 LogSumExp）。这一步将多次显存读取平摊为一次。
    *   *Phase 2（块内顺序注意力与合并）*：逐层顺序进行正常的网络前向计算，利用在线 Softmax (Online Softmax) 技术，将 Phase 1 预计算好的块间结果与当前块内的“局部增量注意力”进行 $O(1)$ 复杂度的算子融合（Kernel Fusion）。这使得 I/O 开销降到最低，推理延迟增加低于 2%。
*   **内存友好的 Prefilling (Sequence Sharding)**：对于几十万 token 的超长上下文，为了避免缓存历史 Block 占用过大显存，工程上沿着 Sequence 维度对表示进行张量并行切片（TP-sharding）。在计算 Phase 1 时各显卡独立处理自己负责的序列分片，然后在 Phase 2 利用标准的 TP All-Reduce 通信进行归约聚合，大幅降低单卡峰值显存占用。

## 5. 实验结果

*   **缩放定律 (Scaling Laws)**：
    *   在包含 5 个尺寸梯度的模型实验中，Block/Full AttnRes 的拟合曲线始终在 Baseline (PreNorm) 之下。
    *   在最大的算力档位上（5.6 PFLOP/s-days），使用 **Block AttnRes ($N \approx 8$) 能够达到 Baseline 使用 1.25 倍计算量**才能达到的 Loss。Block 版本几乎恢复了 Full 版本的全部收益。
*   **百亿级模型全量训练 (Kimi Linear 48B)**：
    *   训练数据达到 1.4 Trillion Tokens。
    *   **内部动态改善**：彻底缓解了 PreNorm 导致的状态范数无限膨胀问题（稀释现象）。模型每经过一个 Block 的边界，特征大小都会自动回归周期性的有界状态；网络在深度方向上的梯度分布也变得极其均匀，避免了早期层梯度过大的问题。
*   **下游评估全面提升**：
    *   在相同预训练策略下，Block AttnRes 相比 Baseline 提升显著，尤其在多步逻辑推理（GPQA-Diamond 提升 7.5 个点）、数学（Minerva Math 提升 3.6）、代码（HumanEval 提升 3.1）等需要复杂组合能力的复杂任务上取得跨越式进步。
*   **网络架构审美偏移**：固定算力和参数预算下进行的架构搜索表明，相比标准 Transformer 倾向于 $d_{model}/L_b \approx 60$（宽而浅），AttnRes 将最优点推向了 $\approx 45$。即 **AttnRes 机制下的模型更偏爱“更深、更窄”的网络结构**，它极大增强了网络利用额外深度的能力。注意力可视化也证明，各层不仅关注紧邻的上一层，更自发学习到了远距离的跳跃连接 (skip connections)。