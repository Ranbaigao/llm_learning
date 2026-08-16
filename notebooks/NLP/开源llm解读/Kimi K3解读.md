可以。Kimi K3 和刚才的 GLM-5.3 放在一起看特别有意思，因为两者代表了两种不同的 frontier scaling 路线。

**GLM-5.3 更像“底座不变，把后训练和 Agent 环境做到极致”；Kimi K3 则是“底座、架构、MoE、长上下文、多模态、Agent RL、推理基础设施一起 scale”。**

K3 官方的基本配置已经很激进：**2.78T 总参数、104.2B 激活参数、93 层、896 个 routed experts / 每 token 激活 16 个、1M context，并且从预训练阶段就是原生多模态。**相比 K2 的 1.04T / 32.6B activated，规模不是小升级。([arXiv][1])

我认为 Kimi K3 真正值得看的创新，可以归纳成下面 **7 个部分**。

---

# 1. K3 的总体思路：同时 Scale 三个信息流维度

K3 的架构设计其实可以用一个非常漂亮的三维框架理解：

```text
                    Kimi K3
                       │
        ┌──────────────┼──────────────┐
        │              │              │
 Sequence Mixing   Depth Mixing   Channel Mixing
        │              │              │
   KDA + MLA         AttnRes       LatentMoE
        │              │              │
 长序列信息流       跨层信息流       专家/参数容量
```

Kimi 官方也明确把 K3 描述为沿着：

* sequence length
* network depth
* model width

三个方向扩展 information flow。([arXiv][1])

这是理解整个 K3 架构最好的入口。

---

# 2. 第一大创新：KDA + MLA，把 Attention 变成「局部递归记忆 + 周期性全局检索」

K2 基本上还是 MLA。

K3 变成：

```text
KDA
KDA
KDA
MLA
│
KDA
KDA
KDA
MLA
...
```

也就是 **3:1 的 KDA : MLA hybrid attention**，整个模型最终是：

* 69 层 KDA
* 24 层 MLA

共 93 层。([arXiv][1])

### 为什么这么做？

Softmax Attention 的问题大家都知道：

[
KV\ Cache \propto L
]

context 越长，KV Cache 越大。

而 KDA 属于一种 recurrent / linear attention：

```text
token1
  ↓
state S1
  ↓
token2
  ↓
state S2
  ↓
token3
  ↓
...
```

它把历史压缩到固定大小的状态 (S_t) 里。

所以长上下文下：

```text
Traditional Attention

token ────────────────────► all previous KV
KV Cache ↑ with context


KDA

token → recurrent state S → token
          fixed size
```

K3 的 KDA state 不随着序列无限增长，因此特别适合 1M context。([arXiv][1])

但是纯 linear/recurrent attention 又有问题：

> 历史信息被压进 state 后，精确检索某个很久以前 token 的能力会弱于 full attention。

所以 K3 没有完全干掉 Softmax Attention，而是：

> **3 层 KDA 做廉价的信息传播 + 1 层 MLA 做全局精确 retrieval。**

这其实是一种非常合理的 hybrid architecture。

---

## K3 对 KDA 还有两个重要修改

KDA 本身不是 K3 才第一次提出，来自 Moonshot 此前的 Kimi Linear。

K3 真正改了两个工程上很关键的地方。

### ① Lower-bounded decay

以前 decay 可以无限接近 0。

那么 chunkwise parallel 时：

[
1/\Gamma
]

可能变得巨大，产生 numerical overflow。

K3 强制：

[
g_{\min}=-5
]

使 retention factor：

[
\alpha > e^{-5}
]

这样一个 16-token tile 内累计缩放仍然落在 BF16 能表达的范围里。

结果是：

> **连 diagonal tile 都可以直接 Tensor Core GEMM。**

不再需要特殊 position-pair kernel。([arXiv][1])

这属于非常典型的：

**algorithm ↔ hardware co-design。**

不是单纯数学上变漂亮，而是：

> 我重新设计 recurrence 参数化，让它恰好适合 GPU Tensor Core。

这类创新我认为实际价值很高。

---

# 3. 第二个很值得注意的东西：Attention Residuals

这个东西我觉得甚至比 KDA 更有研究味道。

传统 Transformer：

```text
Layer 1
  ↓ residual add
Layer 2
  ↓ residual add
Layer 3
  ↓
Layer 4
```

隐藏状态实际上是：

[
h_l=h_{l-1}+f_l(h_{l-1})
]

于是第 80 层想使用第 5 层的信息，只能依赖：

> 第 5 层的信息在前面 75 次 residual accumulation 之后还没有丢掉。

Moonshot 的一个观点是：

> 这其实和 Transformer 出现前的 RNN 有点像。

**sequence dimension 上我们已经从 RNN → Attention 了，为什么 depth dimension 还在做 sequential accumulation？**

于是 AttnRes 做了一件很直接的事：

```text
Layer 1 ───────────────┐
Layer 2 ────────────┐  │
Layer 3 ─────────┐  │  │
                 ▼  ▼  ▼
               Attention
                   ↓
                Layer N
```

换句话说：

> **让 Layer 对 Layer 也做 Attention。**

而不是：

```text
全部历史层
   ↓
不断相加
   ↓
一个 hidden state
```

K3 使用 learned pseudo-query，根据前面不同 block 的 representation 动态计算权重，再组合出当前层输入。([arXiv][1])

我认为这个思想特别值得关注：

### Transformer 传统上只有

[
Attention_{token}
]

现在变成：

[
Attention_{token}
+
Attention_{layer}
]

也就是：

```text
横向：
哪个历史 token 重要？

纵向：
哪个历史 layer representation 重要？
```

---

## 为什么最后不是每层互相 Attention？

因为那样需要保存所有层 activation。

所以 K3 实际用了 **Block AttnRes**：

93 层被压成大约 8 个主要 block，每个 block 约 12 层，然后对 block-level representation 做 attention。

于是 storage：

[
O(Ld)
]

下降到：

[
O(Nd)
]

官方说经验上约 8 个 block 已经可以恢复 Full AttnRes 的大部分收益。([arXiv][1])

这应该算 K3 最值得关注的架构创新之一。

---

# 4. Stable LatentMoE：真正把 MoE 做到了「近千专家」

K3 的 MoE 数字很夸张：

```text
K2:
384 experts
Top-8

K3:
896 experts
Top-16
```

但是这里不能简单理解为：

> 把专家数量翻倍。

因为 K3 又引入了 **LatentMoE**。

传统 MoE：

```text
hidden dimension = 7168

token
 ↓
7168-d
 ↓
Expert
 ↓
7168-d
```

16 个专家都处理 full-width representation，通信和 expert traffic 会非常大。

LatentMoE 变成：

```text
        full hidden
           7168
             ↓
        W_down
             ↓
          3584
             ↓
     ┌───────┼───────┐
     E1      E2 ...  E16
     └───────┼───────┘
             ↓
          latent
             ↓
         W_up
             ↓
          7168
```

也就是 routed expert 只在：

[
3584 = 0.5\times7168
]

的 latent space 里面工作。

与此同时还有 **2 个 full-width shared experts** 负责通用 transformation。([arXiv][1])

所以本质是：

> **Common knowledge → full-width shared expert**
>
> **Specialized knowledge → narrow latent experts**

这样才能负担得起：

[
896 \text{ experts},\ Top16
]

这个设计对 MoE scaling 很有意义。

---

# 5. 更有意思的是它怎么解决 896 专家的负载均衡

这块和你之前问 Qwen MoE load balancing 很相关。

K3 没采用经典 auxiliary load-balancing loss。

而是：

> **Auxiliary-loss-free routing + Quantile Balancing。**

以前 DeepSeek 那类 auxiliary-loss-free balancing 大致是维护 expert bias：

[
s'_j=s_j+b_j
]

哪个专家过载：

```text
bias ↓
```

哪个专家冷门：

```text
bias ↑
```

问题在于更新通常是：

[
b_j \leftarrow b_j+\gamma \cdot sign(target-load_j)
]

这里存在一个敏感超参数：

[
\gamma
]

太小：

> 调不过来。

太大：

> expert load 来回震荡。

对于：

[
896 experts
]

这个问题就很明显了。([arXiv][1])

---

## K3 的 Quantile Balancing

它不再慢慢“调 bias”。

而是直接问：

> **如果 expert j 应该获得 q 个 token，那么它的 score threshold 应该是多少？**

然后直接计算 router score margin 的 quantile。

简化理解：

假设：

```text
1000 tokens
100 experts
Top-10
```

总 assignment：

[
1000\times10=10000
]

理想每 expert：

[
q=100
]

那么 K3 就直接找：

> 让 expert j 恰好落在 top-k 的第 100 个 token 对应的 score quantile。

然后一次性设置 bias。

所以：

```text
旧方案：

过载 → bias -0.01
还是过载 → -0.01
还是过载 → -0.01
...

QB：

应该拿100个token？
→ 算第100名 threshold
→ 直接设置 bias
```

这对于 **近 1000 experts 的 MoE** 是非常漂亮的设计。([arXiv][1])

---

# 6. SiTU-GLU：一个看似小，但很实用的稳定性改动

LatentMoE 带来一个问题。

routing branch 实际上会连续经过很多 matrix multiplication：

```text
W_down
 ↓
expert W_gate
 ↓
expert W_up
 ↓
W_up to hidden
```

2.8T 规模以后容易出现 activation explosion。

传统 SwiGLU：

[
x\sigma(x)\cdot x
]

正方向是没有上界的。

K3 提出了 **SiTU-GLU**：

核心就是把两个 branch 都 soft-cap：

[
\beta\tanh(x/\beta)
]

于是：

```text
SwiGLU
activation
    ↑
    │          /
    │        /
    │      /
    │    /
────┴────────────→ x

SiTU
activation
    ↑
    │       ─────── cap
    │     /
    │   /
    │ /
────┴────────────→ x
```

小数值区域保持接近 SwiGLU，大 activation 时自动饱和。([arXiv][1])

这类设计并不性感，但对训练 **2.8T MoE** 很现实。

---

# 7. Native Multimodal：不是后面外挂一个 ViT

K3 与很多 VLM 一个非常重要的区别是：

> **vision 从 pre-training 一开始就参与训练。**

不是：

```text
先训练 LLM
      ↓
冻结 / partially freeze
      ↓
接 ViT
      ↓
alignment
```

而是：

```text
         Pre-training
             │
     ┌───────┴────────┐
     │                │
   text             vision
     │                │
     └───────┬────────┘
             ↓
      next-token loss
```

K3 使用约 **401M 参数、27 层的 MoonViT-V2**，图像和视频共享参数。([arXiv][1])

而且有一个数据细节非常重要：

他们大量增加：

```text
code
  ↕
rendered visual
```

这样的 programmatic multimodal data：

* SVG
* Webpage
* Game
* 3D assets
* CAD

([arXiv][1])

这很好解释了为什么 K3 在：

```text
看 screenshot
→ 修改代码
→ render
→ 再看 screenshot
→ 再修改
```

这类 **vision-in-the-loop coding** 上特别强。

这不是普通的“看图回答问题”。

而是在训练：

> **视觉 observation → action → 新视觉 observation**

---

# 8. 1M Context 这里的设计也比“把 RoPE 拉长”有意思

K3 使用 **NoPE**。

MLA 没有显式 positional embedding，KDA 本身的 recurrent decay 提供顺序信息。

因此从理论设计上：

> context extension 不需要调整 RoPE base / YaRN。

([arXiv][1])

训练则走：

```text
8K
 ↓
64K
 ↓
256K
 ↓
1M
```

的 progressive curriculum。([arXiv][1])

而且 Moonshot 特别指出：

> 有一百万 token 长的数据 ≠ 模型真的会使用一百万 token。

所以他们专门合成：

```text
information A ──────────────────────────────┐
                                           │
                     information B ────────┤
                                           ▼
                                    final question
```

只有把分散在整个 1M context 的信息组合起来才能完成任务。([arXiv][1])

这个数据设计比单纯 long-document continual pretraining 更合理。

---

# 9. 对你来说最值得看的其实是 K3 的 Post-training

这里我觉得 **K3 比 GLM-5.3 公布得详细很多。**

它明确写出了：

```text
                K3 Base
                   ↓
                  SFT
                   ↓
            Cold-start Policy
                   ↓
                  RL
        ┌──────────┼──────────┐
        │          │          │
     General   General Agent Coding Agent
        │          │          │
    Low/High/Max Low/High/Max Low/High/Max
        │          │          │
        └──────────┼──────────┘
                   ↓
             9 RL Experts
                   ↓
        Multi-Teacher On-Policy
             Distillation
                   ↓
               Kimi K3
```

这是我认为 K3 后训练最重要的框架。([arXiv][1])

---

## 第一阶段：SFT 只负责 Cold Start

这里和我们之前讨论的逻辑几乎一致。

Kimi 明确说：

> SFT establishes a high-quality cold-start policy for subsequent RL.

它通过之前 Kimi 系列的 domain expert 生成 agent trajectory，再经过：

* 多阶段 verification
* human-in-the-loop annotation

建立：

```text
基础 reasoning
工具调用
Agent protocol
长程任务行为
```

然后再进 RL。([arXiv][1])

所以它不是想靠 SFT 把最终 Agent 能力全部训出来。

---

# 10. RL 不是一个模型一起乱训，而是 3 Domain × 3 Effort = 9 个 Expert

这个设计我非常喜欢。

K3 把 RL 分成三个 domain：

### General

```text
reasoning
knowledge
vision
faithfulness
search
knowledge work
```

### General Agent

```text
long-horizon assistant
deep research
writing
```

### Coding Agent

```text
SWE
coding
kernel
web development
```

每一个又训练：

```text
low
high
max
```

三个 reasoning effort。

所以：

[
3\times3=9
]

个 RL expert。([arXiv][1])

这解决一个经典问题：

> **如果所有任务、所有 inference budget 都放到一个 RL policy 里直接优化，很容易产生梯度冲突。**

所以它先：

```text
specialize
```

再：

```text
merge
```

---

# 11. Reasoning Effort RL：不是 prompt 里说一句“少想一点”

它给每个问题建立初始 budget：

[
b_0(x)
]

然后设置：

[
T(y)\le\tau b_0(x)
]

超过预算直接：

[
reward=-1
]

于是：

### Max model

[
\tau = large
]

允许多思考。

### High

降低 (\tau)。

### Low

继续降低 (\tau)。

([arXiv][1])

对于 agent task：

[
T(y)
]

甚至不只是 thinking tokens，而包含：

> **reasoning + tool-call arguments 的累计输出 token。**

这其实是在训练：

[
\text{Capability}/\text{Compute}
]

而不仅仅：

[
\text{Capability}
]

也就是说模型真正学习：

> “这个问题值不值得花 20k token？”

这是 test-time scaling 下一阶段非常关键的东西。

---

# 12. 最精彩的部分：9 个 RL expert 最后不是普通 SFT merge

如果最后把九个 expert 的 trajectory 全部丢回 SFT：

```text
Expert trajectory
        ↓
       SFT
        ↓
Unified model
```

会有明显 distillation loss。

K3 使用：

# Multi-Teacher On-Policy Distillation

也就是 **MOPD**。

学生自己 rollout：

[
y_t\sim \pi_\theta
]

然后对应 domain/effort 的 teacher 给每个 token 一个 dense reward：

[
r_t
===

\log
\frac{
\pi_{\text{teacher}}(y_t|x,y_{<t})
}{
\pi_{\text{student}}(y_t|x,y_{<t})
}
]

再 clip 控制极端 reward。([arXiv][1])

所以不是：

> Teacher 给答案 → Student 模仿答案

而是：

> Student 自己探索 → Teacher 在 Student 访问到的 state distribution 上纠正它。

这就是 **on-policy distillation** 的意义。

---

## 和 SFT distillation 的差别非常大

普通：

```text
Teacher trajectory:

S0 → A → B → C → D
     ↑
 student learns this
```

问题是 Student 真正执行时可能：

```text
S0 → A → X
```

然后：

> 它从来没学过 X 后面怎么办。

MOPD：

```text
Student:

S0 → A → X → Y
          ↑
     Teacher evaluates
     token by token
```

所以可以减少：

[
distribution\ shift
]

对于长程 Agent 尤其重要。

**这是 K3 后训练里我认为最值得借鉴的设计之一。**

---

# 13. K3 与 GLM-5.3 出现了明显的共识：真正应该 Scale 的是 Environment

你刚刚问 GLM-5.3 时，我们总结的是：

> Environment Scaling。

K3 完全走到了同一个方向，而且披露得更具体。

K3 做了一个 **Unified White-Box RL Environment**。

把 Agent Harness 拆成：

```text
Agent Harness
│
├── System Prompt
├── Tools
├── Context Management
├── Skills
├── Memory
├── Subagents
└── Interaction Protocol
```

然后随机组合。

甚至可以模拟：

* Kimi Code
* Claude Code
* Codex
* OpenClaw
* Hermes

([arXiv][1])

这个设计的核心不是“支持很多工具”。

而是防止模型：

> **overfit 某一种 Agent scaffold。**

这是非常重要的。

例如只用：

```text
<tool_call>
...
</tool_call>
```

训练出来的 Agent，很可能其实学的是：

> “Claude Code harness policy”

而不是：

> “general tool-use policy”。

K3 在 RL 过程中动态换：

```text
tool schema
system prompt
memory
context management
sub-agent architecture
```

实际是在主动做：

# Agent Scaffold Domain Randomization

这个思路和 robotics 里的 domain randomization 很类似。

---

# 14. Knowledge Graph Guided Task Synthesis

这块也非常值得你后训练的数据章节借鉴。

普通 synthetic data：

```text
随机 topic
 ↓
LLM generate question
```

覆盖率很难控制。

K3 建一个会自动扩张的 hierarchical knowledge graph：

```text
Computer Science
      │
      ├── Distributed System
      │       ├── Consensus
      │       │     ├── Raft
      │       │     └── Paxos
      │       └── ...
      │
      └── Compiler
              ├── IR
              ├── SSA
              └── ...
```

Agent 自动：

```text
Web Search
→ 找新 concept
→ 去重
→ 判断父子关系
→ 加入 graph
→ 继续递归
```

直到 concept 足够 atomic。([arXiv][1])

训练时再按目标 domain distribution：

```text
sample graph node
       ↓
retrieve real material
       ↓
task synthesis
       ↓
RL environment
```

所以它解决了 synthetic data 两个很重要的问题：

[
Coverage
]

和：

[
Granularity
]

这比单纯 random prompt evolution 成熟很多。

---

# 15. K3 已经真正开始做「Living Environment RL」

这个我认为比 benchmark 更值得关注。

它自己构造 mock：

* Gmail
* Notion
* Slack
* Canvas

但是环境是 **persistent** 的。

例如：

```text
Day 1:
收到客户邮件
     ↓
修改 Notion

Day 2:
Slack 出现新信息
     ↓
发现 Day1 判断错误
     ↓
修改 Excel

Day 3:
收到审批
     ↓
再发邮件
```

一个 rollout 最长：

> **几千次 tool calls + millions of context tokens。**

([arXiv][1])

这意味着训练样本已经不再是：

[
(x,y)
]

甚至也不只是：

[
Environment + Trajectory
]

而开始接近：

[
WorldState_t
\rightarrow Action_t
\rightarrow WorldState_{t+1}
]

也就是：

# Agent RL 正在越来越接近真正的 sequential decision making。

---

# 16. 为此他们甚至重写了 RL Infrastructure

如果 trajectory 有：

```text
1000 tool calls
1M context
```

一个 rollout 不可能每一轮训练都从头跑。

于是 K3 使用 **Partial Rollout**：

```text
Iteration 1

trajectory A ───────────done
trajectory B ─────done
trajectory C ────────────────────── still running
trajectory D ─────────done

达到 λ 比例
       ↓
开始 training

C:
pause + save
       ↓
Iteration 2
resume
```

([arXiv][1])

这会产生一个很重要的问题：

> trajectory C 开始时用的是 policy θ₁，
> 后半部分已经训练到 θ₃ 了。

也就是说是极端：

[
off-policy / stale trajectory
]

Kimi 表示其 policy optimization 使用 per-token regularization，使其能容忍这种 stale data。([arXiv][1])

这一点很值得继续深入看 K2.5 的 RL objective。

---

# 17. 甚至 KV Cache 都必须变成 RL 基础设施的一部分

1M agent trajectory 最大的问题之一：

> pause 后 resume，如果 prefix KV 丢掉了怎么办？

重新 prefill 800K token 显然不能接受。

于是：

```text
GPU
│
├── Active KV
│
└── idle KV
       ↓ eviction
CPU DRAM KV Pool
       ↓
resume
       ↓
prefetch back GPU
```

训练权重 / optimizer state甚至会阶段性 offload 到：

```text
NVMe
```

给 CPU DRAM 腾空间存 rollout KV。([arXiv][1])

这已经非常清楚地说明：

> **Agent RL 的 Scaling 已经不是单纯 PPO/GRPO 算法问题了。**

而是：

```text
RL Algorithm
+
Inference Engine
+
KV Cache System
+
Sandbox
+
Environment
+
Scheduler
```

必须一起设计。

---

# 18. 还有一个非常容易被忽略，但我很喜欢的设计：Deployment-Aware Post-training

K3 从 **SFT 阶段开始就做 QAT**。

Expert weight：

[
MXFP4
]

Activation：

[
MXFP8
]

而且：

```text
SFT
↓
RL
↓
Rollout
↓
Deployment
```

全部维持相同 quantization setting。([arXiv][1])

这解决一个很现实的问题：

```text
BF16 model RL:
reward = 0.9

部署 FP4:
behavior changed
reward = ?
```

K3 相当于直接优化：

> **最终真正上线的 quantized policy。**

对于 2.8T model，这实际上不是可选优化，而是必要条件。

---

# 19. 所以 K3 真正的新东西应该怎么排序？

如果让我按研究价值排序，我会这样排：

| 层级    | K3 创新                                     | 我的评价                      |
| ----- | ----------------------------------------- | ------------------------- |
| ⭐⭐⭐⭐⭐ | **AttnRes**                               | 非常值得关注的架构思想               |
| ⭐⭐⭐⭐⭐ | **3×3 RL Experts + MOPD**                 | 后训练最值得研究                  |
| ⭐⭐⭐⭐⭐ | **1M Long-Horizon Agent RL**              | Agent scaling 的关键方向       |
| ⭐⭐⭐⭐½ | **Stable LatentMoE + Quantile Balancing** | 大规模 MoE 非常实用              |
| ⭐⭐⭐⭐½ | **Unified White-Box Environment**         | 防 scaffold overfit 很重要    |
| ⭐⭐⭐⭐  | **KDA + MLA Hybrid**                      | 长上下文架构的重要路线               |
| ⭐⭐⭐⭐  | **Knowledge-Graph Task Synthesis**        | 后训练数据工程很有价值               |
| ⭐⭐⭐½  | Per-Head Muon                             | 很好的 optimizer engineering |
| ⭐⭐⭐½  | SiTU-GLU                                  | 主要是训练稳定性                  |
| ⭐⭐⭐⭐  | SFT/RL 全程 MXFP4 QAT                       | deployment-oriented，很实用   |

其中需要注意：

**KDA、LatentMoE、Muon 本身都不是 K3 凭空第一次提出。**K3 真正厉害的是把这些东西进一步修改并组合起来，真正 scale 到 **2.8T + 896 expert + 1M context + agentic RL**。([arXiv][1])

---

# 20. Kimi K3 和 GLM-5.3，本质上已经在指向同一个结论

这是我觉得最有意思的地方。

### GLM-5.3：

```text
Strong Base
   ↓
Environment Scaling
   ↓
Long-Horizon RL
   ↓
Agent Capability
```

### Kimi K3：

```text
         Pre-training Scaling
                ↓
       2.8T / KDA / AttnRes
                +
         Post-training Scaling
                ↓
 Environment + Long-Horizon RL
                +
        Test-Time Scaling
                ↓
       Low / High / Max
```

Kimi 自己其实把这个讲得很明确：

以前：

[
Pretraining\ Compute \rightarrow Intelligence
]

后来 reasoning model 引入第二根轴：

[
Test\ Time\ Compute \rightarrow Intelligence
]

而 K3 的目标是：

# 两条 Scaling Axis 同时推进。

([arXiv][1])

这也是 K3 和 GLM-5.3 最大的哲学差异。

---

## 对你现在研究后训练，我认为最值得吸收的是这条 pipeline

不是 KDA，也不是 2.8T 参数，而是：

```text
                    Base
                      ↓
        ┌─────────── SFT ───────────┐
        │      建立行为先验          │
        │   Tool / Agent Cold Start │
        └────────────┬──────────────┘
                     ↓
              Specialized RL
                     ↓
      ┌──────────────┼──────────────┐
      │              │              │
   General        Agent           Coding
      │              │              │
 Low High Max   Low High Max   Low High Max
      └──────────────┼──────────────┘
                     ↓
             9 Specialized Policy
                     ↓
              On-Policy Distill
                     ↓
              Unified Model
```

与此同时，**RL 数据生产系统本身**又是：

```text
Knowledge Graph
       ↓
Real Material Retrieval
       ↓
Task Synthesis
       ↓
Executable Environment
       ↓
Different Agent Harness
       ↓
Rollout
       ↓
Verifier / GRM
       ↓
Reward
       ↓
RL
```

([arXiv][1])

我认为这一套比简单写成：

> SFT → GRPO → RLHF

成熟了一个层级。

**因为 K3 和 GLM-5.3 都在说明：2026 年真正的后训练竞争，已经逐渐从“GRPO 公式怎么改”转向“谁能构建规模更大、覆盖更广、更接近真实工作、同时可验证的 Agent Environment”。**

而 Kimi K3 又进一步补了一块：**不同 domain / reasoning budget 不必硬塞到一个 RL policy 里，可以先训练多个专门 policy，再通过 on-policy distillation 合回来。**

这一点其实非常适合进一步映射到你前面讨论的 **政务 MoE / ESFT / 通用 replay / Function Calling / GRPO** 方案里：K3 的做法能给出一个比“所有能力都在一个 SFT + 一个 GRPO 阶段混训”更清晰的训练分工。

## 参考文献
[1]: https://arxiv.org/html/2607.24653v2 "Kimi K3: Open Frontier Intelligence"
[2]: [Attention Residuals](../模型架构/残差连接/Attention Residuals.md)
[3]: [GLM-5.3 解读](GLM-5.3解读.md)