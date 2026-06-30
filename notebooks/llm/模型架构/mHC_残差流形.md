# mHC: Manifold-Constrained Hyper-Connections 笔记

## 1. 先看宏观：mHC 到底改了 Transformer 的哪里？

标准 Transformer 里，每个 Attention 或 FFN/MLP 子层外面都有一个经典残差连接：

$$
x_{l+1}=x_l+\mathcal{F}(x_l)
$$

**mHC 可以理解成：用一套可学习、受约束的“多车道残差路由系统”，替换这个简单加号。**

它不改变 Attention 怎么算，也不改变 FFN 怎么算。Attention/FFN 仍然是原来的主干计算 $\mathcal{F}$。mHC 改的是主干层外面的残差流拓扑：原来只有一条残差流，现在扩成 $n$ 条并行残差流，并且让这些流之间可以稳定通信。

## 2. mHC 作用于谁：残差流从 1 条变成 n 条

传统模型里，残差流是一条维度为 $C$ 的“单行道”。每一层做的事情大致是：保留旧状态 $x_l$，再加上 $\mathcal{F}(x_l)$ 产生的新信息。

mHC 把这条单行道扩成 $n$ 条并行车道，论文里常见设置是 $n=4$：

$$
x_l \in \mathbb{R}^{B \times T \times n \times C}
$$

这里 $B$ 是 batch，$T$ 是序列长度，$C$ 是单条流的 hidden size。直觉上，模型拥有了多条可以持续向后传递的“记忆通道”；但主干计算层依然只接收一条 $C$ 维输入，所以 mHC 必须负责在多条流和主干层之间做路由。

## 3. mHC 的四步宏观流程

下面这个流程图把 mHC 放回 LLM 的整体网络拓扑里看：它包裹在每一个 Attention/FFN 子层外面，负责读、算、混、写。

<iframe src="../../../.assets/mhc_topology_flow.html" width="100%" height="980px" style="border:1px solid #cbd5e1; border-radius: 8px;"></iframe>

### 3.1 Read / Pre-routing：用 $H_{pre}$ 聚合输入

主干层 $\mathcal{F}$ 只吃一条 $C$ 维输入，但 mHC 当前有 $n$ 条残差流。于是它用 $H_{pre}$ 做加权读取，把多条流聚合成一条主干输入：

$$
x_{\text{main}} = H_{pre} \cdot x_l
$$

可以把 $H_{pre}$ 理解成“读门控”：每一层、每个 token 都可以动态决定从哪几条残差流里多读一点。

### 3.2 Compute：Attention / FFN 正常计算

主干层拿到聚合后的 $x_{\text{main}}$，照常执行 Attention 或 FFN：

$$
y = \mathcal{F}(x_{\text{main}})
$$

这一步是 mHC 很重要的工程取舍：它没有把 Attention/FFN 的宽度也扩成 $nC$，因此主要 FLOPs 仍然接近原模型。

### 3.3 Mix：用 $H_{res}$ 让残差流互相通信

在主干层计算的同时，外面的 $n$ 条残差流也会通过 $H_{res}$ 互相混合：

$$
x_{\text{res}} = H_{res} \cdot x_l
$$

这里的 $H_{res}$ 是 mHC 的数学核心。它被约束成双随机矩阵：

- 元素非负；
- 每一行和为 1；
- 每一列和为 1。

也就是说，$H_{res}$ 位于 **Birkhoff Polytope** 内。这样做的目的不是为了“好看”，而是为了让流与流之间的信息交换保持非扩张性质：信息可以重新分配，但整体不容易被层层放大到失控。

### 3.4 Write / Post-routing：用 $H_{post}$ 分发主干输出

主干层输出 $y$ 后，mHC 用 $H_{post}$ 把这份新信息广播并写回到 $n$ 条残差流：

$$
x_{l+1} = H_{res} \cdot x_l + H_{post} \odot \mathcal{F}(H_{pre} \cdot x_l)
$$

所以 mHC 的一层不是“旧状态 + 新状态”这么简单，而是：

- 从多条残差流读取；
- 用原来的 Attention/FFN 计算；
- 让残差流之间稳定混合；
- 把新信息按比例写回多条残差流。

## 4. 为什么要引入这套机制？

传统扩展模型能力通常有两条路：

- **做深**：增加层数；
- **做宽**：增大 hidden size $C$。

但这两条路都会显著增加计算量，尤其是把主干层直接做宽时，矩阵乘法 FLOPs 会快速上升。Hyper-Connections 的思路是：**尽量保持主干计算层不变，只拓宽残差状态本身。**

这带来一个很诱人的收益：模型拥有更多并行残差记忆通道，但 Attention/FFN 的主要计算宽度仍然是 $C$。

问题是，如果多条残差流随便混合，深层堆叠后很容易破坏残差网络原本依赖的恒等映射性质，导致训练不稳定、梯度爆炸或数值漂移。mHC 的核心贡献就是给这套多车道残差系统加上约束：

- **数学交规**：用 Birkhoff Polytope / 双随机矩阵约束 $H_{res}$，保证残差混合更稳定；
- **工程优化**：通过算子融合和重计算降低多条残差流带来的显存带宽与显存占用压力；
- **软启动**：用很小的 $\alpha$ 初始化路由强度，让模型初始行为接近标准 ResNet/Transformer 残差连接。

一句话总结：

> mHC 不是改变模型“怎么思考”，而是改变模型内部“记忆如何并行保存、通信和传递”。

## 5. 与 Birkhoff Polytope 的联系

Birkhoff Polytope 是所有双随机矩阵构成的凸多面体，它的顶点是所有置换矩阵。

在 mHC 中，把 $H_{res}$ 约束到 Birkhoff Polytope 内，有一个非常直接的残差流解释：

- 置换矩阵只重排残差流，不改变整体大小；
- 双随机矩阵是置换矩阵的凸组合，可以看成“软重排”；
- 因此 $H_{res}$ 可以让多条残差流互相通信，同时不轻易放大信号。

Sinkhorn-Knopp 算法就是把一个普通的可学习矩阵反复做行归一化、列归一化，最终拉回到双随机矩阵集合附近：

```python
H = torch.exp(M / tau)
for _ in range(n_iters):
    H = H / (H.sum(dim=-1, keepdim=True) + 1e-8)
    H = H / (H.sum(dim=-2, keepdim=True) + 1e-8)
```

这就是前面流程图里 $H_{res}$ 那个“稳定混合矩阵”的来源。

## 6. 代码形状对照

如果把上面的宏观流程翻译成 PyTorch 张量形状，大致是：

```python
# x: [B, T, n, C]

H_pre = sigmoid(alpha_pre * proj_pre(x_global))      # [B, T, n, 1]
x_main = (x * H_pre).sum(dim=2)                      # [B, T, C]

y = layer_func(x_main)                               # [B, T, C]

H_res_raw = proj_res(x_global).view(B, T, n, n)
H_res = sinkhorn_knopp(alpha_res * H_res_raw)         # [B, T, n, n]
x_res = torch.matmul(H_res, x)                        # [B, T, n, C]

H_post = 2 * sigmoid(alpha_post * proj_post(x_global))# [B, T, n, 1]
x_next = x_res + H_post * y.unsqueeze(2)              # [B, T, n, C]
```

这一段代码对应的就是：

$$
x_{l+1}=H_{res}x_l+H_{post}\odot \mathcal{F}(H_{pre}x_l)
$$

## 7. 几何可视化

上面的流程图解决的是“mHC 在整台 Transformer 发动机里装在哪里”。下面两个图则继续深入那个齿轮本身：为什么 $H_{res}$ 要被约束到 Birkhoff Polytope，以及这个几何对象长什么样。

### Birkhoff Polytope 交互演示

<iframe src="../../../.assets/birkhoff_polytope.html" width="100%" height="720px" style="border:1px solid #cbd5e1; border-radius: 8px;"></iframe>

### 3D Birkhoff 投影演示

<iframe src="../../../.assets/3d_birkhoff.html" width="100%" height="680px" style="border:1px solid #cbd5e1; border-radius: 8px;"></iframe>
