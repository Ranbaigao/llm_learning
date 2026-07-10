# 李宏毅DRL笔记
**强化学习进阶：从 Policy Gradient 到 PPO 与 Actor-Critic**

## 1. 轨迹 (Trajectory) 与 期望回报 (Expected Reward)

### 1.1 轨迹的概率分布

在强化学习中，一个回合（Episode）从开始到结束的状态-动作序列称为**轨迹 (Trajectory)**，记为 $\tau$：

$$\tau = \{s_1, a_1, s_2, a_2, \cdots, s_T, a_T\} $$

给定当前策略 $\theta$，生成这条特定轨迹的概率 $p_{\theta}(\tau)$，由**初始状态分布**、**策略函数的动作概率**以及**环境的状态转移概率**共同决定。利用马尔可夫性质，可以将其展开为连乘形式：

$$\begin{align*}
p_{\theta}(\tau) &= p(s_1) p_{\theta}(a_1 \mid s_1) p(s_2 \mid s_1, a_1) p_{\theta}(a_2 \mid s_2) p(s_3 \mid s_2, a_2) \cdots \\
&= p(s_1) \prod_{t=1}^{T} p_{\theta}(a_t \mid s_t) p(s_{t+1} \mid s_t, a_t)
\end{align*} $$

其中各符号的含义如下：

* $p_{\theta}(a_t \mid s_t)$：策略函数（Policy），表示在状态 $s_t$ 下，智能体选择动作 $a_t$ 的概率分布，由参数 $\theta$ 决定。
* $p(s_1)$：初始状态分布（Initial State Distribution），表示一局开始时初始状态 $s_1$ 出现的概率。
* $p(s_{t+1} \mid s_t, a_t)$：环境状态转移概率（Transition Probability），表示在状态 $s_t$ 采取动作 $a_t$ 后转移到下一状态 $s_{t+1}$ 的概率。

> **原理补充**：在上述公式中，$p_{\theta}(a_t \mid s_t)$ 是由我们的智能体（Actor）决定的，而 $p(s_1)$ 和 $p(s_{t+1} \mid s_t, a_t)$ 是由环境（Environment）决定的。

### 1.2 期望回报

一条轨迹的累积回报（Return）是路径上所有奖励的总和：

$$R(\tau) = \sum_{t=1}^{T} r_t$$

由于环境的随机性和策略的随机性，我们无法最大化单次轨迹的回报，而是要最大化在策略 $\theta$ 下的**期望回报 (Expected Reward)**：

$$\overline{R}_{\theta} = \sum_{\tau} R(\tau) p_{\theta}(\tau) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)} [R(\tau)]$$

---

## 2. 策略梯度定理 (Policy Gradient)

为了最大化期望回报 $\overline{R}_{\theta}$，我们需要对其关于参数 $\theta$ 求梯度，并使用梯度上升来更新网络。

### 2.1 引入 Log-Derivative Trick

$$\begin{align*}
\nabla \overline{R}_{\theta} &= \sum_{\tau} R(\tau) \nabla p_{\theta}(\tau) \\
&= \sum_{\tau} R(\tau) p_{\theta}(\tau) \frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)} \quad \text{(引入构造项)} \\
&= \sum_{\tau} R(\tau) p_{\theta}(\tau) \nabla \log p_{\theta}(\tau) \quad (\because \nabla \log x = \frac{\nabla x}{x}) \\
&=\mathbb{E}_{\tau \sim p_{\theta}(\tau)} [R(\tau) \nabla \log p_{\theta}(\tau)]
\end{align*} $$

> **从求和到期望：这一步推导的直观理解**
> 
> 这一步的推导完全基于概率论中**数学期望（Expected Value）的基本定义**。
> 
> 对于离散随机变量 $x$ 和概率分布 $P(x)$，函数 $f(x)$ 的期望定义为：
> $\mathbb{E}_{x \sim P(x)}[f(x)] = \sum_{x} P(x) f(x)$
> 
> 在上述公式中，我们只需要做一个"一一映射"：
> - **随机变量** $x$ → **轨迹 $\tau$**
> - **概率分布** $P(x)$ → **$p_{\theta}(\tau)$**
> - **目标函数** $f(x)$ → **$R(\tau) \nabla \log p_{\theta}(\tau)$**
> 
> 因此：
> $\sum_{\tau} \underbrace{p_{\theta}(\tau)}_{\text{概率 } P(x)} \underbrace{\Big( R(\tau) \nabla \log p_{\theta}(\tau) \Big)}_{\text{目标函数 } f(x)} = \mathbb{E}_{\tau \sim p_{\theta}(\tau)} [R(\tau) \nabla \log p_{\theta}(\tau)]$
> 
> **为何如此重要？** 这一步没有涉及任何复杂的微积分运算，仅仅是将"概率 × 某项求和"这种数学形式，简写成"期望"的标准符号表达。在强化学习中，将公式转化为期望形式至关重要，因为这意味着在实际写代码时，我们无需（也无法）遍历所有可能的轨迹求和，而是可以通过让智能体与环境交互进行**采样（Sampling），用样本均值来近似计算这个梯度。

注意到这里 $R(\tau)$ **不需要是可微的**，它甚至可以是一个完全的黑盒（Black Box）。我们求导的对象仅仅是策略网络输出的概率 $\log p_{\theta}(\tau)$。$R(\tau)$ 充当了一个标量权重：如果一条轨迹的回报高，我们就增加该轨迹发生的概率。

### 2.2 展开并用采样近似 (Sample Approximation)

由于我们无法穷举所有可能的轨迹，我们通过让智能体与环境交互，采样 $N$ 条轨迹来近似期望：

$$\nabla \overline{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log p_{\theta}(\tau^n)$$

将 $\log p_{\theta}(\tau^n)$ 展开（环境转移概率与 $\theta$ 无关，求导后为 0，被直接消除）：

$$\nabla \overline{R}_{\theta} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log p_{\theta}(a_t^n | s_t^n)$$

### 2.3 引入因果性 (Causality) 与 基线 (Baseline)

上述公式假设动作 $a_t$ 会受到整条轨迹总回报 $R(\tau^n)$ 的影响，这不符合因果律。**一个动作只能影响它之后的奖励，不能影响过去的奖励。** 此外，为了降低采样的方差，我们引入一个与动作无关的**基线 $b$**：

$$\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \left( \sum_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n - b \right) \nabla \log p_{\theta}(a_t^n | s_t^n)$$

* **$\gamma$ (Discount Factor)**：未来奖励需要打折，离得越远的奖励受当前动作的影响越小。
* **$b$ (Baseline)**：通常取状态的期望价值 $V(s_t)$。防止所有奖励均为正时，未被采样的好动作概率相对下降。

---

## 3. 优势函数 (Advantage Function)

我们将上述公式括号中的部分定义为**优势函数 (Advantage Function)** $A^{\theta}(s_t, a_t)$：

$$A^{\theta}(s_t, a_t)=\sum_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n - b $$

> **物理意义**：优势函数衡量的是**“在状态 $s_t$ 下采取动作 $a_t$，比该状态下的平均表现（Baseline）要好多少？”**
> * $A > 0$：该动作比平均水平好，应当增加其概率。
> * $A < 0$：该动作比平均水平差，应当降低其概率。
> 
> 

---

## 4. 近端策略优化 (PPO: Proximal Policy Optimization)

Policy Gradient 是 **On-policy（同策略）** 的，每次更新参数 $\theta$ 后，旧数据就作废了，样本效率极低。PPO 的目标是通过**重要性采样 (Importance Sampling)** 将其转变为 **Off-policy**，从而可以复用旧策略产生的数据。

### 4.1 重要性采样 (Importance Sampling)

如果我们想求函数在分布 $p$ 下的期望，但只能从分布 $q$ 中采样，可以通过乘上重要性权重 $\frac{p(x)}{q(x)}$ 来修正：

$$\begin{align*}
E_{x \sim p}[f(x)] &= \int f(x) p(x) dx \\
&= \int f(x) \frac{p(x)}{q(x)} q(x) dx \\
&= E_{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right]
\end{align*}$$

*注意：$p$ 和 $q$ 的分布不能差异过大，否则重要性权重 $\frac{p(x)}{q(x)}$ 会产生极大的方差。*

### 4.2 转化为 Off-Policy 目标函数

从轨迹级别的重要性采样公式出发：

$$\nabla \bar{R}_{\theta} = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[ \frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_{\theta}(\tau) \right]$$

要将上述整条轨迹（Trajectory）级别的期望，拆解为单步（Step）级别的期望，等号背后实际上经历了以下三个核心的数学与物理变换：

1. **消除环境动力学 (Environment Dynamics Cancellation)**：

   当我们展开轨迹的联合概率 $p_{\theta}(\tau)$ 时，它包含了初始状态概率、策略的动作概率以及环境的状态转移概率 $p(s_{t+1}|s_t, a_t)$。由于新旧策略在同一个环境中运行，分子分母中的环境转移概率和初始状态概率**完美抵消**，重要性权重只剩下了纯粹的策略动作概率之比的连乘。

2. **期望视角的转换 (Shifting the Expectation)**：

   我们不再以“一整条轨迹”为单位来计算期望，而是将其等价转换为在旧策略 $\pi_{\theta'}$ 下，对**每一个时间步所访问到的状态-动作对 $(s_t, a_t)$** 求期望。在这个视角转换中，自然引出了状态访问频率分布（即新旧策略访问某个具体状态 $s_t$ 的概率 $p_{\theta}(s_t)$ 与 $p_{\theta'}(s_t)$）。

3. **引入优势函数 (Replacing Return with Advantage)**：

   基于 Policy Gradient 中的因果律和基线（Baseline）技巧，我们将整条轨迹的总回报 $R(\tau)$，替换为旧策略视角下的单步优势函数 $A^{\theta'}(s_t, a_t)$，以降低采样的方差。

综合以上三步，我们得到了 Step 级别的精确展开式：

$$\begin{align*} \nabla \bar{R}_{\theta} &= \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} \frac{p_{\theta}(s_t)}{p_{\theta'}(s_t)} A^{\theta'}(s_t, a_t) \nabla \log p_{\theta}(a_t | s_t) \right] \end{align*} $$

**关键近似与目标构造**

在上式中，状态分布比值 $\frac{p_{\theta}(s_t)}{p_{\theta'}(s_t)}$ 很难计算。但由于我们在算法设计上会限制新策略 $\theta$ 不能偏离旧策略 $\theta'$ 太远，这意味着**新旧策略在环境中游走时，遇到各个状态的概率分布是极其相似的**。

因此，我们引入一个极其重要的近似假设：$\frac{p_{\theta}(s_t)}{p_{\theta'}(s_t)} \approx 1$。

约掉状态分布比值后，公式化简为：

$$\nabla \bar{R}_{\theta} \approx \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t) \nabla \log p_{\theta}(a_t | s_t) \right]$$

至此，我们得到了化简后的**期望更新梯度**。


这样一来，式子就只剩动作概率比值和优势函数了。接下来最自然的问题就是：**什么样的标量函数，求导以后会得到这个梯度？**

换句话说，我们现在是在寻找 $\nabla \bar{R}_{\theta}$ 的一个“原函数”。把梯度项放回标准的期望形式里，可以写成

$$\begin{align*}
\nabla \bar{R}_{\theta}
&\approx \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t) \nabla_{\theta} \log p_{\theta}(a_t | s_t) \right] \\
&= \nabla_{\theta} \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t) \right]
\end{align*} $$

因此，对应的 PPO 核心替代目标函数 (Surrogate Objective) 就是

$$J^{\theta'}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t) \right] $$

如果把这个目标函数对 $\theta$ 求导，就会回到上面的梯度形式。因为 $p_{\theta'}(a_t | s_t)$ 和 $A^{\theta'}(s_t, a_t)$ 都是由旧策略采样得到的常量，所以求导只作用在 $p_{\theta}(a_t | s_t)$ 上；再用 Log-Derivative Trick，$\nabla_{\theta} p_{\theta}(a_t | s_t) = p_{\theta}(a_t | s_t) \nabla_{\theta} \log p_{\theta}(a_t | s_t)$，就能重新得到


$$\nabla_{\theta} J^{\theta'}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t) \nabla_{\theta} \log p_{\theta}(a_t | s_t) \right] $$


这就说明，PPO 中“去掉 $\nabla$ 符号”本质上是在做**反向构造目标函数**：先把我们想要的更新方向写成梯度形式，再找出一个可以被框架直接最大化的标量目标，让自动求导机制替我们算出同样的梯度。

### 4.3 PPO1: KL 散度惩罚 (KL Penalty)

为了保证 $p$ 和 $q$（即新旧策略）不要相差太大（满足重要性采样的前提），PPO1 直接在目标函数中加入 KL 散度作为惩罚项：

$$J_{PPO}^{\theta^k}(\theta) = J^{\theta^k}(\theta) - \beta KL(\theta, \theta^k)$$

**动态调整 $ \beta $ 机制：**

* 如果 $ KL(\theta, \theta^k) > KL_{max} $：说明更新步子迈得太大了，**增加 $ \beta $** 以加大惩罚力度。
* 如果 $ KL(\theta, \theta^k) < KL_{min} $：说明更新太保守，**减少 $ \beta $** 以允许更大的策略更新。

### 4.4 PPO2: 截断机制 (Clip) - 业界主流

PPO2 摒弃了复杂的 KL 散度计算，直接通过硬截断 (Clipping) 来限制新旧策略的比值 $r_t(\theta) = \frac{p_{\theta}(a_t | s_t)}{p_{\theta^k}(a_t | s_t)}$：

$$J_{PPO2}^{\theta^k}(\theta) \approx \sum_{(s_t, a_t)} \min \left( r_t(\theta) A^{\theta^k}(s_t, a_t), \, \text{clip} \left( r_t(\theta), 1 - \epsilon, 1 + \epsilon \right) A^{\theta^k}(s_t, a_t) \right)$$

> **Clip 机制的直觉解析：**
> * **当 $A > 0$ 时（好动作）**：我们希望 $r_t(\theta)$ 越大越好。但如果 $r_t(\theta) > 1 + \epsilon$，说明新策略已经比旧策略生成该动作的概率大很多了，为了防止过度更新导致网络崩溃，我们将其截断在 $1+\epsilon$。
> * **当 $A < 0$ 时（坏动作）**：我们希望 $r_t(\theta)$ 越小越好。但如果 $r_t(\theta) < 1 - \epsilon$，说明新策略已经极大地降低了该动作的概率，同样为了稳定性，我们将其截断在 $1-\epsilon$。
> 
> 
> **总结**：Clip 操作就像给策略更新加了限速器，保证每次参数更新都在一个安全的“信任区域（Trust Region）”内。

---

## 5. 演员-评论家算法 (Actor-Critic)

单独的 Policy Gradient 存在方差大的问题（因为依赖完整轨迹的蒙特卡洛采样）。Actor-Critic 引入了一个评论家（Critic网络）来实时评估当前状态的价值，从而代替蒙特卡洛采样的真实回报。

### 5.1 状态价值函数 (State Value Function) 与 TD Error

根据贝尔曼期望方程，状态价值 $V^{\theta}(S_t)$ 可以递归表示：

$$\begin{aligned}
V^{\theta}\left(S_{t}\right) &=r_{t}+\gamma r_{t+1}+\gamma^{2} r_{t+2} ... \\
V^{\theta}\left(S_{t+1}\right) &=r_{t+1}+\gamma r_{t+2}+... \\
V^{\theta}\left(S_{t}\right) &=\gamma V^{\theta}\left(S_{t+1}\right)+r_{t}  \quad \text{(这里假设是确定的单步奖励)}
\end{aligned}$$

在实际中，等号两边会有误差，这个误差被称为 **时间差分误差 (TD Error)**：

$$\text{TD Error} = r_{t} + \gamma V^{\theta}(S_{t+1}) - V^{\theta}(S_{t}) $$

### 5.2 用 TD Error 替代优势函数

回忆之前的优势函数 $A_t = \text{实际回报} - \text{基线}$。在 Actor-Critic 中，我们用单步的 TD 目标代替完整的轨迹回报：

$$A_{t} \approx r_{t} + \gamma V^{\theta}(S_{t+1}) - V^{\theta}(S_{t}) $$

**$G_t'$ 与 $V^{\theta}(S_t)$ 的本质区别：**

* **$G_t'$ (Return, 回报)**: 蒙特卡洛概念。指的是在一个轨迹中，从时间步 $t$ 开始到结束，**实际经历并累加**的奖励总和。方差大（每次跑的结果都不一样），但无偏差。
* **$V^{\theta}(S_t)$ (状态价值)**: 神经网络预测概念。指从状态 $S_t$ 开始，根据策略 $\theta$，**预测未来预期**能获得的奖励总和。方差小（网络输出是稳定的），但可能存在偏差（网络没训练好时预测不准）。

### 5.3 算法运行逻辑

1. **Actor (演员)**：即策略网络 $\pi_\theta(a|s)$，负责根据当前状态做出动作，并通过最大化 Advantage 来更新参数。
2. **Critic (评论家)**：即价值网络 $V^\phi(s)$，负责计算状态的价值，计算出 TD Error（即 Advantage），以此来“评价” Actor 的动作，并通过最小化 TD Error 的均方差来更新自身的参数 $\phi$。