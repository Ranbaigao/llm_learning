# APEX-Agents 与 397B 知识工作 Agent RL 训练配方

> 资料范围：[Mercor 博客原文](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/)，微信转载《大模型前沿应用》专栏解读  
> 调研日期：2026-09-16  
> 作者背景：Edward Hu（LoRA 一作、OpenAI o1 核心成员）现任 Mercor Head of AI Modeling，与 SkyRL 团队合作发布  
>
> **结论先行**：本工作最重要的发现不是某个新算法，而是 **Agent RL 真正的瓶颈不在 Policy Loss**——单纯修好 Harness 就能把未训练模型的平均奖励从 22.74% 提升到 28.69%（+5.95pp），几乎相当于在旧 Harness 上完成一个 epoch 训练。最终 397B 在 APEX-Agents 上 Pass@1 从 16.11% → 27.29%（相对 +69%）。**最有杠杆的工程顺序**是：环境 → Token 对齐 → 异步系统 → 过拟合验证 → 算法消融 → 大模型训练 → 跨 Harness 泛化检验。Prompt Mean（+3.85pp）和 Context Nudge（+2.95pp）是最有效的算法级改动；DPPO 在奖励上不显著优于 GLM-5 Loss，但显著改变 Agent 行为模式（更多、更短的交互），更适合复杂 Agent 任务。

## 1. 背景：APEX-Agents 是什么

APEX-Agents 是一个面向**真实知识工作**的长程 Agent 基准，包含 **480 个公开评测任务**，覆盖三个专业领域：

- Corporate Law（公司法务）
- Management Consulting（管理咨询）
- Investment Banking（投资银行）

每个任务存在于一个**模拟企业环境**中，可能包含几十份 PDF、Excel、PowerPoint、邮件和聊天记录。Agent 通过 MCP 工具或代码执行进入环境，完成最终交付物。

为支持 RL 训练，Mercor 额外准备了 **1,928 个专家创建的私有训练任务**，分布在 112 个模拟企业环境中，结构与评测任务一致但 Prompt 与企业环境不重合（降低数据污染）。

## 2. 训练模型与基线设置

| 模型 | 总参数 | 激活参数 | 作用 |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | 35B | 3B | 系统调试 + 算法消融 |
| Qwen3.5-397B-A17B | 397B | 17B | Hero Run 大规模训练 |

**关键决策**：直接对基础模型开展 RL，**不做 SFT Warm-up**。理由是希望集中研究"最困难、最容易失败"的 RL 部分。

**最终成绩**：397B 在 APEX-Agents 上的 Pass@1 从 16.11% 提升至 27.29%（绝对 +11.18pp，相对 +69%）；35B 训练后在同基准上**超过 Claude Opus 4.5**。

## 3. 第一步：Harness 环境优化（最大单项杠杆）

正式训练前的第一项工作不是调算法，而是把 Harness 修好。三个子任务：

### 3.1 系统组件与运行位置

整套系统：SkyRL + Harbor + Ray + vLLM + Megatron + Modal

```
[训练数据：Harbor Task Directory]
       ↓
[GPU 集群：Ray 调度]
       - Megatron Training Nodes（更新参数）
       - vLLM Inference Nodes（生成 rollout）
       - SkyRL（全异步训练循环）
       - Harbor Trial（管理轨迹生命周期）
       - ArchipelagoAgent（执行多轮 MCP 工具调用）
       - NCCL（权重同步）
       ↓
[独立沙箱：Modal Sandbox + Docker 镜像]
       - PDF MCP / Excel MCP / PowerPoint MCP / Email MCP / Chat MCP
       - 搜索 + 代码执行
       - 文件系统 + Verifier
```

每条轨迹消耗一个 Harbor Task 目录，启动独立沙箱，最后销毁。

### 3.2 环境稳定性提升

长程 Agent 任务可能运行几十分钟、生成几万乃至十几万 Token。任意外部操作没有 timeout，最终都会永久挂起。

具体措施：

- **所有外部操作加 timeout**：文件下载、MCP 调用、容器启动/销毁、LLM Judge 请求
- **处理 Judge API 限流**：多 API Key 轮询 + 失败重试 + 指数退避
- **每条 Agent Loop 用独立进程**：早期共享 Python 进程导致大量 MCP 断连；改为每条轨迹放入独立 Ray Task
- **区分任务失败 vs 系统错误**：避免系统错误被记为模型零奖励污染训练信号

**经验**：开始 RL 前，先按正式训练时的并发规模对完整训练集跑一次 dry-run，把非模型错误率压到接近零。

### 3.3 Harness 行为优化

逐条检查未训练模型的失败轨迹，分类失败原因（模型能力 / 工具设计 / Harness 行为 / 环境依赖 / Verifier 错误）。发现并修复：

- 沙箱缺 Python 包 → 模型多回合探索环境限制
- PowerPoint 工具错误返回 None → 模型无法判断操作是否完成
- PDF MCP 把二维页面转一维文本 → 多栏文档内容错误拼接 → **提示优先使用 pdfplumber**
- 工具调用格式解析失败 → 重新生成而非终止轨迹
- 过长工具返回 → 截断避免耗尽上下文
- 上下文剩余不多 → 提醒模型开始收尾

**这一系列修复合起来，把未训练模型的平均奖励从 22.74% 提到 28.69%（+5.95pp）**——几乎等于在旧 Harness 上完成一个 epoch 训练。

**结论**：在 Agent RL 中，修好 Harness 本身可能比改一次 RL 算法更有效。

### 3.4 TITO：Token-in-Token-out

一个隐蔽但严重的 Off-policy 问题：$\text{encode}(\text{decode}(z)) \neq z$。

举例：

- 词表含 `<` (id=0), `search` (id=1), `<search` (id=2), `>` (id=3)
- 模型采样得到 [0, 1, 3] → 解码为 `<search>`
- Harness 保留文本后训练器重新编码 → [2, 3]
- 推理侧真实动作与训练器看到的动作不一致 → 破坏 PPO/GRPO/DPPO 的状态-动作-概率对应

**多轮 Agent 特别严重**：每轮"输出 → decode → 工具执行 → 拼接历史 → 重新 tokenize"都可能改变 token id。

**TITO 方案**：Agent 直接通过 `/completions` 接口处理原始 Token ID，保存：

- 输入 Token ID
- 输出 Token ID
- Rollout Logprob
- Loss Mask
- Reward

这样训练器使用的动作才与推理引擎实际采样的动作完全一致。

## 4. 第二步：RL 系统调优

### 4.1 异步训练

长程 Agent 任务完成时间差异巨大（简单任务快速结束，复杂任务生成 100K+ Token）。采用 **Fully Async RL**：

- Rollout 节点持续生成轨迹
- 完成轨迹立即进入缓冲区
- Trainer 持续从缓冲区取数据
- 不等同一批所有轨迹结束
- 更新权重通过 NCCL 同步到 vLLM 推理节点

### 4.2 Megatron 配置扫描

用独立脚本扫描 TP / EP / PP / CP / CPU Offloading / Micro-batch Size。**使用动态微批处理**：根据每条序列的 Token 数量动态组成 micro-batch，对长度差异极大的 Agent 轨迹至关重要。

### 4.3 Rollout vs Training 资源分配

**原则**：先固定训练 GPU 数量，再增加足够推理 GPU 直到 Trainer 不再等待。

| 模型 | 推理节点 | 训练节点 |
| --- | --- | --- |
| 35B | 12 | 4 |
| 397B | 12 | 8 |

### 4.4 并发数上限：两个约束

**系统上限（KV Cache）**：

$$C_{\text{system}} \approx \frac{\text{总 KV Cache 容量}}{\text{平均轨迹长度}}$$

**算法上限（staleness）**：

$$C_{\text{algorithm}} = (\text{max staleness steps} + 1) \times \text{mini batch size} \times n_{\text{samples}}$$

实际配置下为 $(3+1) \times 16 \times 16 = 1024$，但两次实验真正受限的都是 KV Cache：

| 模型 | 最大并发 |
| --- | --- |
| 35B | 550 |
| 397B | 300 |

模型越大、轨迹越长，可同时运行的 Agent 越少。

### 4.5 训练—推理概率偏差检查

用少量训练 step 比较 vLLM 推理端与 Megatron 训练端对相同 Token 的 Logprob。**平均 Logprob 差异 < 0.03 通常表示系统健康**。

这个检查还帮助发现了 vLLM CPU Offloading + GDN 模型 + In-flight 权重更新组合下的正确性问题。

## 5. 第三步：过拟合 Run（端到端单元测试）

即使环境与系统通过检查，团队**依然没有启动完整训练**，而是先做小规模过拟合实验：

- 从训练集选 32 个任务
- 每个任务必须存在**非零奖励方差**（GRPO 一类组内相对优化方法需要有效信号）
- Batch size = 32，每 prompt 采样 8 条轨迹
- 同步训练，每 step = 完整训练一个 epoch

平均奖励从 ~0.19 上升到 0.32，证明系统能产生明确学习信号。

**过拟合发现的真问题**："比较 rollout 前后文件差异"进行评分的任务明显难过拟合。排查发现：

- 文件内容提取不完整
- Diff 工具不能准确识别变化
- 正确修改也可能被误判

更换第三方文件 Diff 工具后任务才开始出现学习信号。

**小规模过拟合在 Agent RL 中相当于一次端到端单元测试**——检查任务可学习性、Reward 正确性、Verifier 可靠性、Token 对齐、梯度有效性。**如果 32 个任务都学不会，直接启动 397B 训练等于浪费 GPU**。

## 6. 第四步：35B 算法消融

前三步去风险后，在 Qwen3.6-35B-A3B 上做消融。每组实验取第一个 epoch 的 checkpoint，在 480 个 hold-out 任务上评测 3 次。

**注**：消融实验绝对分数低于最终成绩，因为后续又优化了 Harness；但所有消融用同一 Harness，**横向比较仍有效**。

### 6.1 Token Aggregation：Prompt Mean +3.9 个点

三种 Policy Loss 聚合方法：

| 方法 | 公式 | 特点 |
| --- | --- | --- |
| Token Mean | $L = \frac{\sum_i \sum_t L_{i,t}}{\sum_i T_i}$ | 长轨迹主导梯度（128K 轨迹影响是 2K 的几十倍） |
| Sequence Mean | $L = \frac{1}{N}\sum_i \frac{1}{T_i} \sum_t L_{i,t}$ | 每条序列权重相同，但同 prompt 多序列仍可能放大 |
| **Prompt Mean** | 先在 prompt 内聚合，再对 prompt 求平均 | **每个任务获得相同总体权重** |

**Prompt Mean 是文章最有效的单项算法改动**：28.69% → 32.54%（+3.85pp）。

### 6.2 DPPO vs GLM-5 Loss

Fully Async RL 同时存在两类偏差：

- vLLM 与 Megatron 之间的 train-inference mismatch
- Rollout Policy 与当前训练 Policy 之间的 staleness

传统 TIS 需要额外前向计算 Trainer Logprob，对 100K Token 轨迹代价巨大。

| 方法 | 机制 | 避免额外前向 |
| --- | --- | --- |
| **DPPO** | 训练/推理 Logprob 偏离过大时屏蔽对应 token | 是 |
| **GLM-5 Loss** | 截断 importance ratio 控制过大 off-policy 更新 | 是 |

奖励上看：**DPPO 相对 GLM-5 Loss 只高 +0.34pp，处于评测噪声范围内，不能认为显著更优**。

但 DPPO 显著改变 Agent 行为：

| 指标 | 训练前 | DPPO 训练后 |
| --- | --- | --- |
| 平均轮数 | 21.21 | 32.40 |
| 每轮 Assistant Token | 834 | 587.5 |

模型从"一次输出很长"转向"短思考 → 调工具 → 看结果 → 再思考 → 再调工具"。**作者最终选 DPPO 主要因为行为模式更适合复杂 Agent 工作，而非奖励优势**。

### 6.3 Context Nudge

当 Agent 消耗掉 80% 上下文预算时，Harness 插入提示要求模型尽快完成任务。**仅训练阶段使用，评测不加**。

效果：28.69% → 31.64%（+2.95pp）。

机理：减少上下文耗尽导致的零分轨迹，使每个 batch 包含更多有效奖励信号。

**附加优势**：训练上下文 160K，评测 256K。用相对比例而非固定 Token 数触发，可吸收训练/评测上下文差异。

### 6.4 没有帮助的方法

| 方法 | 结果 |
| --- | --- |
| Overlong Filtering（屏蔽超长轨迹） | **下降约 1.5pp** |
| Adaptive Length Penalty（分别惩罚模型生成 / 环境返回 token） | 多系数下中性或负面 |
| 重置 KV Cache | 无显著收益 |

可能原因：

- 训练/评测上下文差距不大
- 只训练一个 epoch，长度约束长期价值未体现
- 长轨迹不必然等于无效探索，部分复杂任务本身就需要大量操作

### 6.5 不能只看 Reward

480 个任务的单次评测有较大噪声（不同运行间可能波动 1-3pp）：

- 约 1pp 以内差距应视为持平
- 每组结果应多次评测
- 不仅看 Reward，还要观察行为：平均工具调用轮数、每轮 Assistant Token、工具调用成功率、代码执行比例、上下文耗尽比例

**最终选择 DPPO + Prompt Mean + Context Nudge** 作为 397B 训练配置。

## 7. 第五步：397B Hero Run

### 7.1 配置

- DPPO
- Prompt Mean
- Context Nudge
- 不使用长度惩罚
- 不使用 Curriculum Learning

**算法层面从 35B 到 397B 没有本质变化，工作量主要来自系统侧**：更多训练 GPU、更复杂并行策略、更低 rollout 并发（300 vs 550）、更大 KV Cache 压力、更高权重同步成本。

### 7.2 Pass@1 曲线

训练初期出现 Pass@1 下降。原因：**Fully Async RL 的动态偏差**——简单任务更快完成，训练缓冲区早期被简单任务占据，数据分布不均衡；随长任务完成，指标恢复并上升。

| 模型 | 最终 Pass@1 |
| --- | --- |
| 35B | 22.71% |
| 397B | 27.29% |

### 7.3 Pass@16 曲线

同一任务采样 16 次至少一次完全通过的比例：

- 35B 平滑曲线 ~0.56
- 397B 平滑曲线 ~0.65

**Pass@16 继续上升，说明模型保留了一定探索空间，未收敛到单一行为**。

### 7.4 Policy Entropy 曲线

两个模型的 policy entropy **没有持续坍缩，反而总体上升**：

- 397B 始终低于 35B → 397B 策略分布更集中
- 35B 保持更高不确定性 → 探索性更强

仅凭 entropy 无法判断好坏，结合 Pass@1/Pass@16 可见**训练未造成明显策略坍缩**。

## 8. 第六步：泛化检验

### 8.1 换 Harness：Archipelago（MCP）→ OpenCode（代码型）

训练用 Archipelago 基于 MCP 工具，评测完全移除 MCP Server 换为代码型 OpenCode（提供 bash、glob、read、grep、write、edit、todowrite）。任务仍是同一组 480 个 APEX-Agents。

| 模型 | 训练 Harness | OpenCode |
| --- | --- | --- |
| 35B | Mean +10.00pp, Pass@1 +8.74pp | Mean +11.71pp, Pass@1 +9.65pp |
| 397B | Mean +11.89pp, Pass@1 +11.18pp | Mean +8.70pp, Pass@1 +6.87pp |

**大部分 RL 收益能跨 Harness 迁移**。但 **35B 迁移明显优于 397B**。

### 8.2 为什么 35B 跨 Harness 迁移更好

分析代码工具使用比例：

- 35B 训练中：代码执行比例 0.5 → 0.75+（逐渐依赖代码执行）
- 397B 训练中：代码执行比例长期 ~0.5（更偏好 MCP 工具）

OpenCode 恰好是代码型 Harness，所以 35B 迁移更好。推测与 Qwen3.6-35B-A3B 基础训练阶段接受过更强的 Agentic Post-training 有关。

**结论**：Agent 跨 Harness 泛化能力不仅取决于模型规模，还取决于训练中形成的**工具偏好**。

### 8.3 同时换任务和 Harness：Terminal-Bench 2.1

数据集换为 Terminal-Bench 2.1，Harness 换为 Terminus，最多 1,000 步 / 沙箱最长 3 小时 / 每组执行 3 次。

| 模型 | 训练前 | 训练后 | 提升 |
| --- | --- | --- | --- |
| 35B | 44.57% | 50.94% | +6.37 |
| 397B | 50.56% | 55.43% | +4.87 |

**模型学到的不仅是 APEX-Agents 任务模式或 Archipelago 工具格式，而是获得了一定通用长程 Agent 能力**。

### 8.4 是否损害推理能力

在 HLE 与 GPQA 上测试：

- 35B：HLE +0.88pp，GPQA 无变化
- 397B：HLE +0.79pp，GPQA +1.01pp

差异都在误差范围内。正确解读不是"Agent RL 提高通用推理能力"，而是**没有观察到显著的非 Agent 推理能力退化**——专业知识工作 RL 至少没破坏模型原有知识和推理能力。

## 9. 关键结论：算法影响可能没有数据大

| 改动 | 收益 |
| --- | --- |
| 修复 Harness（不训练） | **+5.95pp** |
| Prompt Mean | +3.85pp |
| Context Nudge | +2.95pp |
| DPPO vs GLM-5 Loss | +0.34pp（不显著） |
| 完整 Post-training | +10~12pp |

**决定知识工作 Agent 表现的因素远不止 Policy Loss**：

- 任务是否来自真实专业工作
- 环境能否准确还原任务
- 工具是否可靠
- Verifier 能否正确判断结果
- Reward 是否稳定
- 失败轨迹究竟来自模型还是系统
- 数据是否覆盖真正的能力缺口

Coding Agent 较早通过 RL 取得明显进展，主要因为**代码天然具备可执行、可验证的奖励**。咨询、金融、法律任务没有天然单元测试——真正的困难是把专业工作转化成：

```
环境 + 任务 + 工具 + 轨迹 + 可验证结果 + 可靠 Reward
```

**高质量专业数据不仅是"收集更多 prompt"，而是在构造可供强化学习交互和验证的完整世界**。

## 10. 七步实施顺序（最重要的工程经验）

| 步骤 | 任务 | 关键输出 |
| --- | --- | --- |
| 1 | 把环境和 Harness 修好 | 消除 MCP 断连、PDF 解析、工具返回、沙箱超时、文件 Diff 等污染训练信号的因素 |
| 2 | 保证 Token 完全对齐（TITO） | 避免 decode 后重新 tokenize 导致隐蔽的 off-policy 错位 |
| 3 | 调优异步训练系统 | Megatron 并行策略、动态 micro-batch、rollout/training 比例、KV Cache、最大并发 |
| 4 | 先证明少量任务可过拟合 | 32 个任务学不会就不要启动 397B 训练 |
| 5 | 在小模型上完成算法消融 | Prompt Mean 最有效；DPPO 主要改变 Agent 行为；Context Nudge 显著减少无效轨迹 |
| 6 | 启动大模型正式训练 | 397B Pass@1: 16.11% → 27.29% |
| 7 | 必须检查跨 Harness 与跨任务泛化 | 只有离开训练 Harness 仍有效，才能说明模型获得了可复用 Agent 能力 |

## 11. 关键数字速查

| 指标 | 值 |
| --- | --- |
| APEX-Agents 评测任务数 | 480 |
| 训练任务数 | 1,928（112 个企业环境） |
| 397B Pass@1 提升 | 16.11% → 27.29%（+11.18pp，相对 +69%） |
| 仅 Harness 优化提升 | 22.74% → 28.69%（+5.95pp） |
| Prompt Mean 提升 | +3.85pp（最佳算法改动） |
| Context Nudge 提升 | +2.95pp |
| DPPO vs GLM-5 Loss 差距 | +0.34pp（不显著） |
| DPPO 改变行为 | 平均轮数 21.21 → 32.40，每轮 Assistant Token 834 → 587.5 |
| 并发数（35B / 397B） | 550 / 300 |
| Logprob 偏差健康阈值 | < 0.03 |
| 过拟合任务数 | 32（每 prompt 8 条轨迹） |
| 训练上下文 / 评测上下文 | 160K / 256K |
| Context Nudge 触发阈值 | 80% 上下文预算 |

## 12. 一句话总结

**Agent RL 的瓶颈不在 Policy Loss——修 Harness + 保证 Token 对齐 + 过拟合验证这三步的工程杠杆，远大于任何 RL 算法变体。** Mercor 与 SkyRL 给出的最重要经验不是某个新公式，而是一套**先修环境、再保证 Token 正确；先证明可过拟合、再比较算法；最后才值得把 397B 放进训练集群**的优先级。
