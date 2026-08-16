> **当 base model 已经足够强之后，能力增长的主要瓶颈，可能从 Pre-training Scaling 转移到 Post-training Environment Scaling。**

GLM-5.3 官方明确说明：**它和 GLM-5.2 使用同一个 base model，所有提升都来自后训练。** 但 Terminal-Bench 3.0 从 **4.6 → 28.3**，DeepSWE v1.1 从 **46.2 → 66.9**，Agents’ Last Exam CLI 从 **23.8 → 28.5**；官方内部 Z.ai Code Bench 则宣称总体 coding 提升约 50%。([Overview - Z.AI DEVELOPER DOCUMENT][1])

所以我会把 GLM-5.3 的核心创新总结成 **4 个层次**。

---

# 1. 最大的变化：从「训练答案」变成「训练工作过程」

这是我认为 GLM-5.3 最重要的地方。

传统 coding SFT / RL 数据通常是：

```text
Problem
   ↓
Model
   ↓
Code / Answer
   ↓
Unit Test
   ↓
Reward
```

哪怕升级成 Agent RL，很多任务实际上仍然是：

```text
issue
→ 看代码
→ 改几个文件
→ pytest
→ pass/fail
```

GLM-5.3 往前走了一步。

官方描述它的训练任务已经不再局限于 isolated programming problems，而是覆盖：

```text
Identify Problem
      ↓
Analyze
      ↓
Explore Environment
      ↓
Plan
      ↓
Implement
      ↓
Run / Debug
      ↓
Revise Strategy
      ↓
Verify
      ↓
Deliver
```

甚至某些训练 task 的工作量，官方称相当于**高级工程师连续几天的工作**。模型不是在一个简单 Docker sandbox 里做 LeetCode，而需要接触：

* 真实计算集群
* storage system
* internal documents
* code repositories
* 实验结果
* 多个相互依赖的系统

([docs.z.ai][1])

这实际上是在改变训练数据的基本单位：

**以前：**

[
\text{training sample}= (x,y)
]

**现在更接近：**

[
\text{training sample}
======================

(E, G, \tau, V)
]

其中：

* (E)：Environment
* (G)：Goal
* (\tau)：几百甚至几千步的 Agent trajectory
* (V)：Verifier

这就是我认为 5.3 最核心的思想：

> **训练 Agent 时，真正需要 scale 的不只是 token，而是 Environment × Task × Trajectory。**

---

# 2. 更关键的创新：自动生成「可验证环境」

这比简单的“多做 RL”重要得多。

Agent RL 最大的问题其实一直不是 PPO、GRPO、DAPO 哪个算法更漂亮，而是：

> **哪来那么多真实、复杂、同时又能自动判断对错的任务？**

GLM-5.3 在这里给出了一个很值得关注的 pipeline。

根据官方博客披露，它们会构建 task environment，然后让 **judge agent 实际执行任务，检查这个任务是不是可解的**；同时，Verifier 的构造**不直接接触 reference solution**，并利用 solver trajectories 去发现和堵住 reward shortcut。([Z.ai][2])

大概可以理解成：

```text
真实工作模式 / 工程任务
            ↓
      Task Synthesis
            ↓
  构造 Executable Environment
            ↓
       Judge Agent
            ↓
     任务是否真的可解？
            ↓
     Synthesized Verifier
            ↓
┌─────────────────────────┐
│ Oracle Check            │
│ No-op Check             │
│ Unsolved-state Check    │
│ Reward Hacking Check    │
└─────────────────────────┘
            ↓
      Reliable Reward
            ↓
         Agent RL
```

其中官方特别提到：

> 当 verifier 能通过 **oracle / no-op / unsolved-state** 等检查之后，就可以产生足够可靠的 **binary reward**，直接拿来训练。([Z.ai][2])

这个设计非常重要。

### 为什么？

比如训练一个 Agent：

> “修复这个项目的 memory leak。”

最简单 verifier：

```bash
pytest
```

模型很容易 reward hack：

```text
删掉失败测试
把异常 catch 掉
hardcode 返回值
关闭相关模块
修改 benchmark
```

结果：

```text
reward = 1
```

但问题根本没解决。

所以 GLM-5.3 强调的不是单纯 **Outcome Reward**，而是：

> **先把 Environment + Verifier 做到可信，再 scale RL。**

这其实是 Agent RL 非常核心、但经常被忽略的一点。

---

# 3. 从「RL Scaling」进一步变成「Environment Scaling」

因此我认为 GLM-5.3 最值得抽象出来的一句话，不是：

> GLM-5.3 使用了更多 RL。

而是：

> **它开始探索 Agentic Post-training 的 Scaling Law。**

官方描述过去一个月主要持续扩大：

**more environments + more diverse tasks + longer / richer workflows**。([Z.ai][2])

这和传统 scaling law 有明显区别。

以前是：

```text
Pre-training Scaling

Parameters ↑
Tokens ↑
Compute ↑
        ↓
Capability ↑
```

GLM-5.3 展示的是：

```text
Post-training Scaling

Executable Environments ↑
Task Diversity ↑
Trajectory Horizon ↑
Verifier Quality ↑
RL Rollouts ↑
        ↓
Agent Capability ↑
```

而且最有说服力的是：

**base model 没换。**

也就是说：

```text
GLM-5.2 Base
     │
     ├── old post-training ──→ GLM-5.2
     │
     └── scaled environment RL ──→ GLM-5.3
```

却出现了：

| Benchmark             | GLM-5.2 |  GLM-5.3 |
| --------------------- | ------: | -------: |
| Terminal-Bench 3.0    |     4.6 | **28.3** |
| DeepSWE v1.1          |    46.2 | **66.9** |
| Agents' Last Exam CLI |    23.8 | **28.5** |
| ExploitBench          |    24.4 | **54.4** |

([docs.z.ai][1])

这里尤其值得看 Terminal-Bench。

4.6 → 28.3 不是普通的：

> “代码生成准确率提高几个点”。

更像是：

> **原来模型无法稳定完成的长程任务类别，现在开始真正学会完成。**

所以它强化了一个观点：

**Agent 能力不是单纯由“知识”决定，而很大程度取决于模型是否在训练阶段真正经历过完整的行动—反馈—纠错循环。**

---

# 4. Cyber 能力其实是一个很有意思的「Post-training Emergence」案例

官方最意外的一部分反而是 cybersecurity。

他们把 vulnerability discovery 相关的数据和环境放进后训练，原本预期应该只是：

```text
读代码
→ 找漏洞
→ 判断漏洞
```

结果随着 long-horizon environment / RL scaling，能力开始延伸到更复杂的漏洞验证和 exploitation chain。

官方报告：

* CyberGym：**84.5%**
* ExploitBench：**24.4% → 54.4%**

并称 CyberGym 达到其测试中的 SOTA；不过这些比较目前主要还是 Z.ai 自己报告的结果，应当等公开权重和第三方复现再判断。([Overview - Z.AI DEVELOPER DOCUMENT][1])

这个现象从研究角度其实比 benchmark 本身更有意思。

可以把它理解成：

```text
训练目标：
Bug Discovery
    ↓
Code Understanding
    ↓
Program Execution
    ↓
Environment Interaction
    ↓
Hypothesis Testing
    ↓
Multi-step Planning
    ↓
──────────────
能力组合
──────────────
    ↓
Exploit Chain
```

也就是说它未必是：

> “模型突然凭空学会 hacking。”

更可能是几个已经存在的 primitive skill：

```text
代码理解
+
长期规划
+
工具调用
+
执行反馈
+
状态保持
+
反复试错
```

在足够复杂的 environment RL 中产生了 **composition**。

这才是所谓 emergent capability 更合理的解释。

---

# 5. slime 很重要，但它并不是 GLM-5.3 的新算法

这一点需要特别区分。

看到博客里：

> slime: Built for Long-Horizon RL Scaling

很容易误以为这是 GLM-5.3 的创新。

实际上不是。

GLM-5 时代，Z.ai 就已经把 **slime** 作为异步 RL infrastructure 引入，其核心是把：

```text
Rollout / Generation
        ↕
Training
```

解耦，从而解决 Agent trajectory 特别长、environment latency 长尾导致 GPU 等数据的问题。GLM-5 的技术报告也明确把 asynchronous RL infrastructure 和 asynchronous agent RL 作为核心后训练组件。([arXiv][3])

到 GLM-5.2，又进一步让 slime 支持：

* white-box rollout
* black-box rollout
* compact trajectory
* sub-agent workflow
* 多种 Agent RL workload
* OPD expert merging

并用它并行融合十多个 expert model。([Z.ai][4])

GLM-5.3 继续建立在这套基础设施之上，官方明确提到 **Megatron 负责 training、SGLang 负责 rollout**。([Z.ai][2])

所以关系更准确地说是：

```text
GLM-5
│
├─ DSA
├─ slime async RL infra
└─ Agent RL
        ↓
GLM-5.1
│
└─ 更长 horizon / multi-turn RL
        ↓
GLM-5.2
│
├─ IndexShare
├─ solid 1M context
├─ improved MTP
├─ Agentic RL scaling
├─ OPD
└─ long-horizon RL
        ↓
GLM-5.3
│
├─ SAME BASE
│
├─ 大规模 executable environments
├─ real-world expert workflows
├─ automated task/environment synthesis
├─ stronger verifier construction
└─ 更长、更复杂 Agent RL
```

所以 **5.3 的创新重点不在 architecture，而在 training recipe。**

---

# 6. IndexShare、1M Context 也不是 5.3 创新

这一点也要剥离掉。

GLM-5.2 就已经：

* 1M context
* DSA + IndexShare
* 每 4 个 Transformer layer 共享 indexer
* 1M context 下 indexer 相关 FLOPs 显著下降
* MTP + IndexShare + KV Share
* speculative decoding acceptance length 提高约 20%

([Z.ai][4])

GLM-5.3 官方仍然是：

* **1M context**
* **128K max output**
* Function Calling
* MCP
* Structured Output
* Context Caching

([docs.z.ai][1])

因此如果问：

> **GLM-5.3 架构上有什么创新？**

答案其实应该是：

**几乎没有公开的新 backbone architecture innovation。**

这反而正是这次工作的意义。

---

# 7. 它对 SFT / RL 的启发非常大

这也是我觉得这篇最值得你关注的地方。

GLM-5.3 并不能简单总结成：

> “证明 RL 比 SFT 重要。”

更准确的是：

### SFT 更适合建立基础行为先验

例如：

```text
怎么使用 shell
怎么读 repo
怎么调用 tool
怎么写 patch
怎么按照 SOP 做任务
怎么返回 structured output
```

即：

[
\pi_{\text{base}}
\rightarrow
\pi_{\text{SFT}}
]

让模型首先进入一个合理的 behavior manifold。

---

### RL 更适合学习「执行策略」

比如：

> 修复一个复杂系统。

不存在唯一标准 trajectory：

```text
方案 A：
读日志 → grep → 看代码 → 修改 → test

方案 B：
先跑 profiler → 定位瓶颈 → 查历史 commit → 修改

方案 C：
先构造 reproduction → 再定位
```

不能说：

```text
A trajectory = 正确
B trajectory = 错误
```

你真正关心的是：

```text
最终问题解决了吗？
测试通过了吗？
有没有作弊？
有没有破坏其他功能？
资源消耗合理吗？
```

这天然就是：

[
\max_\pi
E_{\tau\sim \pi}
[R(E,\tau)]
]

而不是：

[
\max_\theta
\log P_\theta(y^*|x)
]

所以 GLM-5.3 其实很好地展示了：

> **越接近真正 Agent 的任务，训练范式就越应该从“模仿标准答案”转向“在环境中优化策略”。**

---

# 8. 但有一点现在还不能从博客得出结论：它到底用了什么 RL 算法

这里我反而建议谨慎。

目前 GLM-5.3 的公开材料讲得很多的是：

* Environment
* Verifier
* Binary reward
* long-horizon rollout
* slime
* post-training scaling

但**没有完整披露 GLM-5.3 最终 optimizer / objective 的所有细节**。

所以现在不能看到 binary reward 就直接说：

> “GLM-5.3 就是用了 GRPO。”

也不能确定：

```text
SFT 占多少
OPD 占多少
on-policy RL 占多少
off-policy data 占多少
reward normalization 怎么做
group sampling 怎么做
credit assignment 怎么做
KL 是否保留
long trajectory 怎么切
不同 environment 怎么 batch
```

这些目前公开信息都不足。

GLM-5 技术报告披露了 asynchronous Agent RL 的整体方向，5.2 又进一步披露了 long-horizon RL / slime / OPD，但 **5.3 目前更像 release blog，而不是完整 technical report**。([arXiv][3])

所以现阶段最值得学习的不是“它用了哪个 PPO 变种”，而是：

> **它把 RL 的瓶颈从 optimizer 转向了 Environment Engineering。**

---

# 最后，我会怎么评价 GLM-5.3

如果把最近几代模型后训练的发展抽象一下，我觉得正在发生这样的变化：

```text
2023
RLHF
“让回答更符合人类偏好”
        ↓
2024
Reasoning RL
“让最终答案可验证”
        ↓
2025
Tool / Coding RL
“让模型学会调用工具解决任务”
        ↓
2026
Long-Horizon Agent RL
“让模型在真实环境里完成工作”
        ↓
GLM-5.3
Environment Scaling
“规模化生成整个可执行、可验证的工作世界”
```

所以我认为 **GLM-5.3 最核心的三个创新，按重要程度排序**：

**① Executable & Verifiable Environment Scaling**

不是继续造 prompt-answer，而是规模化造真实工作环境。

**② 从 Problem-level RL → Workflow-level / Project-level RL**

训练单位从一道题变成一个持续数小时甚至数天的完整工程任务。

**③ Verifier Engineering 成为 Agent RL 的核心基础设施**

自动合成 verifier，并用 oracle / no-op / unsolved-state / reward-hacking 检查确保 reward 可信。

至于 **slime、IndexShare、DSA、1M context**，它们非常重要，但更多属于 GLM-5～5.2 已经打好的地基，而不是 5.3 本身最大的创新。([Z.ai][4])

**从后训练研究的角度，我甚至觉得 GLM-5.3 的价值比它那些 benchmark 分数大得多：它给出了一个比较强的证据——base model 到了一定能力之后，“训练环境的数据规模”可能正在成为和 pretraining tokens 同等级的重要 scaling axis。**

而这恰好也会改变 SFT / GRPO 的设计思路：以后真正困难的可能不是“GRPO loss 怎么写”，而是**如何批量生产几十万/几百万个 executable + verifiable + non-hackable 的长程任务环境**。这才可能是下一阶段 Agent 后训练最贵的数据资产。

## 参考文献
[1]: https://docs.z.ai/guides/llm/glm-5.3 "GLM-5.3 - Overview - Z.AI DEVELOPER DOCUMENT"
[2]: https://z.ai/blog/glm-5.3?utm_source=chatgpt.com "GLM-5.3: Frontier Coding with Emergent Cyber Capabilities"
[3]: https://arxiv.org/abs/2602.15763?utm_source=chatgpt.com "GLM-5: from Vibe Coding to Agentic Engineering"
[4]: https://z.ai/blog/glm-5.2?utm_source=chatgpt.com "GLM-5.2: Built for Long-Horizon Tasks"
