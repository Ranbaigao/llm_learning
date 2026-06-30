<h1 id="bAAXt"></h1>
<h1 id="IZZb6">MoeRouter</h1>

```python
class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k
    
    def forward(self, hidden_states):
        # 计算路由logits
        router_logits = self.gate(hidden_states)  # shape is (b * s, expert_number)
        
        # 计算专家经过softmax之后的概率
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        # 计算topk的专家的输出
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )  # shape都是 (b * s, top_k)
        
        # 专家权重归一化
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)
        
        # 生成专家掩码
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )  # shape是 (b * s, top_k, expert_number)
        expert_mask = expert_mask.permute(2, 1, 0)  # (expert_number, top_k, b * s)
        
        return router_logits, router_weights, selected_experts, expert_mask
```

<h2 id="oulKs">permute</h2>

在PyTorch中，`permute`方法用于重新排列张量的维度，是一种常见的操作，用于调整张量的形状以适应后续计算的需求。`permute(2, 1, 0)`是`MOERouter`中对`expert_mask`张量进行维度变换的操作，下面我将详细解释其原理，结合`MOERouter`的上下文和示例，用中文清晰说明。

---

<h3 id="zcl9A">1. `permute`的基本原理</h3>

`permute`方法的作用是按照指定的维度顺序重新排列张量的维度。PyTorch中的张量是多维数组，每个维度用一个整数索引表示（从0开始）。`permute`的参数是一个整数序列，表示新的维度顺序。例如：

+ 假设一个张量的形状是`(d0, d1, d2)`（三维张量）。
+ 调用`tensor.permute(2, 1, 0)`后，张量的维度将被重新排列为`(d2, d1, d0)`。

具体来说：

+ 原始维度的索引`0, 1, 2`分别对应新的维度位置`2, 1, 0`。
+ 即：
  - 原始维度0（大小为`d0`）移动到新张量的维度2。
  - 原始维度1（大小为`d1`）移动到新张量的维度1。
  - 原始维度2（大小为`d2`）移动到新张量的维度0。

在数据层面，`permute`不会改变张量的元素总数，只是重新组织数据的访问顺序，改变维度的大小和含义。

---

<h3 id="JQNbb">2. `MOERouter`中的`permute(2, 1, 0)`</h3>

在`MOERouter`的代码中，`permute(2, 1, 0)`是对`expert_mask`张量进行的操作。我们先来看代码上下文：

```python
expert_mask = F.one_hot(
    selected_experts,
    num_classes=self.expert_number
)  # shape是 (b * s, top_k, expert_number)
expert_mask = expert_mask.permute(2, 1, 0)  # (expert_number, top_k, b * s)
```

<h4 id="tEkPD">上下文说明</h4>

+ `expert_mask`**的初始形状**：`F.one_hot`生成一个one-hot编码的张量，形状为`(b * s, top_k, expert_number)`。

  - `b * s`：batch大小（`b`）和序列长度（`s`）的乘积，表示展平后的token总数。

  - `top_k`：每个token选择的专家数量。

  - `expert_number`：总的专家数量。

  - ###### `**<font style="color:#DF2A3F;">expert_mask[i,j]</font>**`**<font style="color:#DF2A3F;">表示一个one-hot向量，</font>**`**<font style="color:#DF2A3F;">expert_mask[i,j,k]</font>**`**<font style="color:#DF2A3F;"> 用于指示第 i 个 token 的第 j 个选择的专家是否是第 k 个专家。如果是，则该位置为 1；如果不是，则为 0。。</font>**

+ `permute(2, 1, 0)`**的作用**：将`expert_mask`的形状从`(b * s, top_k, expert_number)`变换为`(expert_number, top_k, b * s)`。

<h4 id="uq7F9">为什么需要`permute`？</h4>

在`SparseMOE`中，`expert_mask`用于为每个专家选择需要处理的token。变换维度后，`expert_mask`的形状更适合后续的计算：

+ 新形状`(expert_number, top_k, b * s)`表示：
  - 第一个维度（`expert_number`）：每个专家。
  - 第二个维度（`top_k`）：每个专家的top-1或top-2（或其他top-k）分配。
  - 第三个维度（`b * s`）：所有token。
+ 这种形状便于通过索引（如`expert_mask[expert_idx]`）快速获取某个专家处理的所有token的掩码，进而提取对应的`hidden_states`。

---

<h3 id="QlyeC">3. 逐步分解`permute(2, 1, 0)`</h3>

假设`expert_mask`的初始形状是`(b * s, top_k, expert_number)`，我们用一个具体示例来说明。

<h4 id="BM32x">示例</h4>

假设：

+ `b = 1`（batch大小为1）。
+ `s = 2`（序列长度为2）。
+ `top_k = 2`（每个token选择2个专家）。
+ `expert_number = 3`（总共有3个专家）。
+ `selected_experts`是一个形状为`(b * s, top_k) = (2, 2)`的张量，假设为：

```python
selected_experts = [[2, 1], [2, 1]]
```

表示两个token都选择了专家2（top-1）和专家1（top-2）。

<h5 id="BrC4A">步骤1：生成`expert_mask`</h5>

调用`F.one_hot(selected_experts, num_classes=3)`，生成one-hot编码：

+ 第一个token的`selected_experts = [2, 1]`：
  - 专家2的one-hot编码：`[0, 0, 1]`。
  - 专家1的one-hot编码：`[0, 1, 0]`。
+ 第二个token同理。
+ 结果张量形状为`(b * s, top_k, expert_number) = (2, 2, 3)`：

```python
expert_mask = [
    [[0, 0, 1], [0, 1, 0]],  # 第一个token
    [[0, 0, 1], [0, 1, 0]]   # 第二个token
]
```

用维度索引表示：

+ 维度0：`b * s = 2`（token索引）。
+ 维度1：`top_k = 2`（top-1或top-2）。
+ 维度2：`expert_number = 3`（专家索引）。

<h5 id="hazr9">步骤2：应用`permute(2, 1, 0)`</h5>

调用`expert_mask.permute(2, 1, 0)`，将维度从`(0, 1, 2)`重新排列为`(2, 1, 0)`，即：

+ 原始维度2（`expert_number`）→ 新维度0。
+ 原始维度1（`top_k`）→ 新维度1。
+ 原始维度0（`b * s`）→ 新维度2。

新形状为`(expert_number, top_k, b * s) = (3, 2, 2)`。我们来看数据如何重新组织：

+ **原始张量**（形状`(2, 2, 3)`）：

```python
[
    [[0, 0, 1], [0, 1, 0]],  # token 0
    [[0, 0, 1], [0, 1, 0]]   # token 1
]
```

索引`[i, j, k]`表示：

    - `i`：token索引（0或1）。
    - `j`：top-k索引（0或1）。
    - `k`：专家索引（0, 1, 2）。

+ **新张量**（形状`(3, 2, 2)`）：

```python
[
    [[0, 0], [0, 0]],  # 专家0
    [[0, 0], [1, 1]],  # 专家1
    [[1, 1], [0, 0]]   # 专家2
]
```

索引`[k, j, i]`表示：

    - `k`：专家索引（0, 1, 2）。
    - `j`：top-k索引（0或1）。
    - `i`：token索引（0或1）。

<h4 id="qpWmW">数据变换的细节</h4>

+ **专家0**（`k=0`）：
  - 原始`expert_mask[:, :, 0] = [[0, 0], [0, 0]]`（因为没有token选择专家0）。
  - 新张量`[0, :, :] = [[0, 0], [0, 0]]`。
+ **专家1**（`k=1`）：
  - 原始`expert_mask[:, :, 1] = [[0, 1], [0, 1]]`（两个token的top-2是专家1）。
  - 新张量`[1, :, :] = [[0, 0], [1, 1]]`。
+ **专家2**（`k=2`）：
  - 原始`expert_mask[:, :, 2] = [[1, 0], [1, 0]]`（两个token的top-1是专家2）。
  - 新张量`[2, :, :] = [[1, 1], [0, 0]]`。

<h4 id="JHc0N">为什么这样变换？</h4>

+ 原始形状`(b * s, top_k, expert_number)`以token为中心，便于计算每个token的专家分配。
+ 新形状`(expert_number, top_k, b * s)`以专家为中心，便于后续在`SparseMOE`中按专家索引提取token。例如，`expert_mask[expert_idx]`返回形状为`(top_k, b * s)`的掩码，直接指示哪些token需要由该专家处理。

---

<h3 id="OzcUF">4. 在`SparseMOE`中的作用</h3>

在`SparseMOE`的`forward`方法中，`expert_mask`的形状`(expert_number, top_k, b * s)`用于：

+ 通过`torch.where(expert_mask[expert_idx])`获取哪些token选择了某个专家，以及这些token是作为top-1还是top-2。
+ 示例中：
  - `expert_mask[1] = [[0, 0], [1, 1]]`表示专家1被两个token选为top-2。
  - `torch.where(expert_mask[1])`返回：
    * `idx = [1, 1]`（top-2）。
    * `top_x = [0, 1]`（token 0和token 1）。
  - 这允许`SparseMOE`高效地为专家1提取对应的`hidden_states`并进行处理。

---

<h3 id="w145S">5. 总结</h3>

`permute(2, 1, 0)`的原理是将`expert_mask`的维度从`(b * s, top_k, expert_number)`重新排列为`(expert_number, top_k, b * s)`，以便后续按专家索引组织数据。这种变换：

+ 改变维度顺序但不改变数据内容。
+ 使`expert_mask`适合`SparseMOE`的计算逻辑，方便为每个专家提取需要处理的token。
+ 在示例中，变换后的张量以专家为中心，便于高效索引和计算。

如果需要进一步澄清或更复杂的示例，请告诉我！





<h1 id="w3DKs">SparseMOE</h1>

```python
class SparseMOE(nn.Module):
    # 稀疏 MOE 模型，这里每一个 token 都会过 topk 个专家，得到对应token 的 hidden_embeddings
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.expert_number = config.expert_number
        self.top_k = config.top_k

        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_number)
            ]
        )

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)
    
    def forward(self, x):
        # x shape is (b, s, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        # 合并前两个维度，因为不是 Sample 维度了，而是 token 维度
        hidden_states = x.view(-1, hidden_dim) # shape is(b * s, hidden_dim)

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        # 其中 selected_experts_indices shape 是 (b * s, top_k)
        # 其中 expert_mask shape 是 (expert_number, top_k, b * s)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape 是 (top_k, b * s)
            idx, top_x = torch.where(expert_mask[expert_idx]) 
            # idx 和 top_x 都是一维 tensor
            # idx 的值是 0 或 1, 表示这个 token 是作为当前专家的 top1 还是 top2
            # top_x 的值是 token 在 batch*seq_len 中的位置索引
            # 例如对于 batch_size=2, seq_len=4 的输入:
            # top_x 的值范围是 0-7, 表示在展平后的 8 个 token 中的位置
            # idx 的值是 0/1, 表示这个 token 把当前专家作为其 top1/top2 专家

            # hidden_states 的 shape 是 (b * s, hidden_dim)
            # 需要取到 top_x 对应的 hidden_states
            current_state = hidden_states.unsqueeze(
                0
            )[:, top_x, :].reshape(-1, hidden_dim) # （selected_token_number, hidden_dim）

            # router_weight 的 shape 是 (b * s, top_k)
            current_hidden_states = expert_layer(
                current_state
            ) * router_weights[top_x, idx].unsqueeze(-1)  # （selected_token_number, 1） 这里有广播

            # 把当前专家的输出加到 final_hidden_states 中
            # 方式1 的写法性能更好，并且方式1容易出现
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            # 方式2
            # final_hidden_states[top_x] += current_hidden_states.to(hidden_states.dtype)
            # 方式1 的写法性能更差，并且方式1容易出现错误，+= 操作在处理重复索引时需要多次读写内存，可能会导致竞争条件

        # 把 final_hidden_states 还原到原来的 shape
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        return final_hidden_states, router_logits # shape 是 (b * s, expert_number)

```

<h2 id="HhJgU">torch.where 函数的作用</h2>

在这段 `SparseMOE` 模型的前向传播代码中，`idx, top_x = torch.where(expert_mask[expert_idx])` 是用来从 `expert_mask` 中提取哪些 token 选择了当前专家（`expert_idx`）以及它们的选择顺序（top1 或 top2）的关键步骤。下面我将详细解释 `idx` 和 `top_x` 的作用，并通过一个例子来说明。

---

<h3 id="grnLs">背景知识</h3>

`SparseMOE` 是一个稀疏 Mixture of Experts（MOE）模型，每个 token 会通过一个路由器（`router`）选择 `top_k` 个专家来处理。在代码中：

+ `expert_mask` 是一个三维张量，形状为 `(expert_number, top_k, b * s)`，其中：
  - `expert_number` 是专家数量；
  - `top_k` 是每个 token 选择的专家数量（这里假设 `top_k=2`）；
  - `b * s` 是 batch size（`b`）和 sequence length（`s`）的乘积，表示所有 token 的总数。
+ `expert_mask[expert_idx]` 是一个二维张量，形状为 `(top_k, b * s)`，表示当前专家 `expert_idx` 被哪些 token 选择，以及这些 token 是将它作为第几个专家（top1, top2, ...）。

`torch.where(expert_mask[expert_idx])` 会返回一个元组 `(idx, top_x)`，其中：

+ `idx`：一维张量，表示 `expert_mask[expert_idx]` 中值为 `True` 的元素的行索引，范围是 `[0, top_k-1]`。它告诉我们某个 token 将当前专家 `expert_idx` 作为它的第 `idx+1` 个专家。
+ `top_x`：一维张量，表示 `expert_mask[expert_idx]` 中值为 `True` 的元素的列索引，范围是 `[0, b * s - 1]`。它对应于展平后的 token 索引（即 `batch_idx * seq_len + seq_idx`），告诉我们是哪个 token 选择了当前专家。

---

<h3 id="E4Hwi">`idx` 和 `top_x` 的作用</h3>

1. `top_x`** 的作用**：
   - `top_x` 是一个 token 的位置索引，用于从 `hidden_states` 中提取选择了当前专家的 token 的隐藏状态。
   - `hidden_states` 的形状是 `(b * s, hidden_dim)`，通过 `hidden_states[top_x]` 可以取出这些 token 的隐藏状态，供当前专家处理。
2. `idx`** 的作用**：
   - `idx` 表示某个 token 将当前专家 `expert_idx` 作为它的第几个专家（top1, top2, ...）。
   - 它用于从 `router_weights`（形状为 `(b * s, top_k)`）中提取对应的路由权重。例如，`router_weights[top_x, idx]` 会取出每个 token 在选择当前专家时的权重，用于加权专家的输出。
3. **协同作用**：
   - `top_x` 确定了哪些 token 选择了当前专家；
   - `idx` 确定了这些 token 是将当前专家作为第几个专家；
   - 两者结合，用于正确地将专家的输出加权并累加到 `final_hidden_states` 中。

---

<h3 id="Q5DAk">举例说明</h3>

假设我们有以下参数：

+ `batch_size = 2`
+ `seq_len = 4`
+ `expert_number = 3`
+ `top_k = 2`
+ `hidden_dim = 512`

那么：

+ `hidden_states` 的形状是 `(8, 512)`，因为 `b * s = 2 * 4 = 8`。
+ `expert_mask` 的形状是 `(3, 2, 8)`。
+ `router_weights` 的形状是 `(8, 2)`。

<h3 id="eYhtD">假设数据</h3>

对于某个专家 `expert_idx = 0`，`expert_mask[0]` 是一个 `(2, 8)` 的张量，假设其值为：

```plain
[[True, False, True, False, False, False, False, False],
 [False, True, False, True, False, False, False, False]]
```

+ 行索引 `0` 表示 top1 专家，行索引 `1` 表示 top2 专家。
+ 列索引 `0-7` 表示展平后的 8 个 token。

<h4 id="EmDqE">解释 `expert_mask[0]` 的含义</h4>

+ 第一行（top1）：`[True, False, True, False, False, False, False, False]`
  - token 0 和 token 2 选择了专家 0 作为它们的 top1 专家。
+ 第二行（top2）：`[False, True, False, True, False, False, False, False]`
  - token 1 和 token 3 选择了专家 0 作为它们的 top2 专家。

<h5 id="v0v9k">执行 `torch.where(expert_mask[0])`</h5>

结果为：

+ `idx = [0, 0, 1, 1]`
+ `top_x = [0, 2, 1, 3]`

**逐项解释**：

+ `(idx=0, top_x=0)`：token 0 选择了专家 0 作为 top1。
+ `(idx=0, top_x=2)`：token 2 选择了专家 0 作为 top1。
+ `(idx=1, top_x=1)`：token 1 选择了专家 0 作为 top2。
+ `(idx=1, top_x=3)`：token 3 选择了专家 0 作为 top2。

<h4 id="NolFO">后续操作</h4>

1. **提取隐藏状态**：
   - `current_state = hidden_states.unsqueeze(0)[:, top_x, :].reshape(-1, hidden_dim)`
   - `top_x = [0, 2, 1, 3]`，从 `hidden_states` 中取出 token 0、2、1、3 的隐藏状态，形状变为 `(4, 512)`。
2. **专家处理**：
   - `current_hidden_states = expert_layer(current_state)`，专家 0 处理这 4 个 token 的隐藏状态。
3. **加权输出**：
   - `router_weights[top_x, idx]` 取出对应权重：
     * token 0：`router_weights[0, 0]`（top1 权重）
     * token 2：`router_weights[2, 0]`（top1 权重）
     * token 1：`router_weights[1, 1]`（top2 权重）
     * token 3：`router_weights[3, 1]`（top2 权重）
   - 将专家输出乘以这些权重。
4. **累加到结果**：
   - `final_hidden_states.index_add_(0, top_x, current_hidden_states)`，将加权后的输出加到 `final_hidden_states` 的 `top_x` 位置（即 token 0、2、1、3）。

---

<h3 id="VEFqx">torch.where 函数的作用</h3>

`torch.where` 是 PyTorch 中一个非常有用的函数，主要用于根据给定的条件从两个张量中选择元素，生成一个新的张量。它的功能类似于条件选择操作，在数据处理、模型实现和自定义逻辑中应用广泛。

---

<h4 id="DYmbf">基本用法</h4>

`torch.where(condition, x, y)` 函数接受三个参数：

+ `condition`：一个布尔张量，用于指定选择条件。
+ `x`：第一个输入张量，提供条件为 `True` 时的元素。
+ `y`：第二个输入张量，提供条件为 `False` 时的元素。

函数会返回一个新的张量，其形状与 `condition` 相同。返回张量的每个元素根据 `condition` 对应位置的值，从 `x` 或 `y` 中选择：

+ 如果 `condition[i] == True`，则选择 `x[i]`。
+ 如果 `condition[i] == False`，则选择 `y[i]`。

**示例**：

```python
import torch

condition = torch.tensor([True, False, True])
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
result = torch.where(condition, x, y)
print(result)  # 输出: tensor([1, 5, 3])
```

+ 第一个元素：`condition[0] == True`，选择 `x[0] = 1`。
+ 第二个元素：`condition[1] == False`，选择 `y[1] = 5`。
+ 第三个元素：`condition[2] == True`，选择 `x[2] = 3`。

---

<h4 id="JDE0f">广播机制</h4>

`torch.where` 支持 PyTorch 的广播机制，意味着 `condition`、`x` 和 `y` 的形状不必完全相同，只要它们能广播到相同的形状即可。这增加了函数的灵活性。

**示例**：

```python
condition = torch.tensor([[True, False], [False, True]])
x = torch.tensor([1, 2])  # 形状 (2,)，广播后为 [[1, 2], [1, 2]]
y = torch.tensor([[3, 4], [5, 6]])  # 形状 (2, 2)
result = torch.where(condition, x, y)
print(result)
# 输出:
# tensor([[1, 4],
#         [5, 2]])
```

+ `x` 被广播到 `(2, 2)` 的形状。
+ 根据 `condition` 的值，从广播后的 `x` 或 `y` 中选择元素。

---

<h4 id="go0z2">常见应用</h4>

1. **条件修改张量**  
   `torch.where` 可以用来根据条件创建或修改张量，例如将满足条件的元素设置为特定值。**示例**：

```python
a = torch.tensor([1, 2, 3, 4, 5])
result = torch.where(a > 3, torch.tensor(0), a)
print(result)  # 输出: tensor([1, 2, 3, 0, 0])
```

    - 大于 3 的元素被替换为 0，其余保持不变。

2. **模型中的自定义逻辑**  
   在深度学习中，`torch.where` 可用于实现条件操作，例如自定义激活函数、选择计算路径或调整损失函数。

---

在代码 `idx, top_x = torch.where(expert_mask[expert_idx])` 中，`torch.where` 的作用是**找到 **`expert_mask[expert_idx]`** 这个二维张量中值为 **`True`** 的元素的行和列索引**，从而帮助模型识别出哪些 token 选择了特定的专家以及它们的选择顺序。这在稀疏 Mixture of Experts (MoE) 模型中尤为重要，用于高效地分配和处理数据。

<h4 id="a5uUa">具体解释</h4>

让我们一步步拆解代码并说明 `torch.where` 的功能：

1. **输入的含义**：
   - `expert_mask[expert_idx]` 是一个二维布尔张量（矩阵），通常与 MoE 模型相关。
   - 它的形状可能是 `(top_k, b * s)`：
     * `top_k` 表示每个 token 最多选择的专家数量（例如 top-1 或 top-2）。
     * `b * s` 是 batch size（`b`）和 sequence length（`s`）的乘积，表示 token 的总数。
   - 张量中的每个元素是 `True` 或 `False`，表示某个 token 是否选择了当前专家（由 `expert_idx` 指定）。
2. `torch.where`** 的功能**：
   - 对于二维张量，`torch.where(condition)` 会返回两个一维张量，分别对应满足条件的元素的行索引和列索引。
   - 在这里，条件是 `expert_mask[expert_idx]` 中的值为 `True`。
   - 输出的 `idx` 是行索引，`top_x` 是列索引。
3. **输出含义**：
   - `idx`：包含值为 `True` 的元素的行索引，范围在 `[0, top_k-1]` 内。它表示某个 token 将当前专家作为第几个选择（例如 top-1 对应行 0，top-2 对应行 1）。
   - `top_x`：包含值为 `True` 的元素的列索引，范围在 `[0, b * s - 1]` 内。它表示选择了当前专家的 token 在展平后的位置索引。
   - 每一对 `(idx[i], top_x[i])` 对应于 `expert_mask[expert_idx]` 中一个值为 `True` 的位置。
4. **实际作用**：
   - 在 MoE 模型中，`expert_mask` 通常用于记录哪些 token 被分配给哪些专家。
   - 通过 `torch.where`，模型能够快速提取出选择了当前专家（`expert_idx`）的所有 token 及其选择顺序（例如 top-1 或 top-2）。
   - 这些索引可以用于后续操作，比如从输入张量中提取对应的隐藏状态，只对相关 token 进行专家处理，从而减少计算开销。

<h4 id="TXYzz">示例说明</h4>

假设 `expert_mask[expert_idx]` 是一个 `(2, 4)` 的布尔张量（`top_k = 2`，`b * s = 4`）：

```plain
expert_mask[expert_idx] = [[True, False, True, False],
                           [False, True, False, True]]
```

运行 `idx, top_x = torch.where(expert_mask[expert_idx])`，结果为：

+ `idx = [0, 0, 1, 1]`（行索引）
+ `top_x = [0, 2, 1, 3]`（列索引）

解释：

+ `(idx[0], top_x[0]) = (0, 0)`：第 0 个 token 将当前专家作为 top-1。
+ `(idx[1], top_x[1]) = (0, 2)`：第 2 个 token 将当前专家作为 top-1。
+ `(idx[2], top_x[2]) = (1, 1)`：第 1 个 token 将当前专家作为 top-2。
+ `(idx[3], top_x[3]) = (1, 3)`：第 3 个 token 将当前专家作为 top-2。

<h4 id="kh2JH">总结</h4>

`torch.where` 在 `idx, top_x = torch.where(expert_mask[expert_idx])` 中的作用是**提取布尔张量中 **`True`** 值的行和列索引**，具体来说：

+ 它帮助模型确定哪些 token 选择了当前专家（通过 `top_x`）以及它们的选择顺序（通过 `idx`）。
+ 这种操作是 MoE 模型实现稀疏计算的关键步骤，使得只有被选中的 token 需要被当前专家处理，从而提高效率。

