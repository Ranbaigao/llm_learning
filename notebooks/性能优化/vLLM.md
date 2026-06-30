# vLLM

vLLM 是面向大模型推理服务的高吞吐框架，核心优化点包括 PagedAttention、连续批处理、KV cache 管理、调度抢占、前缀缓存和分布式执行。它的关键思想不是让单个请求的每一步都更快，而是减少显存浪费、提高并发下的有效吞吐。

## 参考资料

[1] 知乎专栏. [《图解大模型计算加速系列之：vLLM核心技术PagedAttention原理》](https://zhuanlan.zhihu.com/p/691038809). 访问日期：2026年6月19日。

[2] vLLM 官方文档. [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).

[3] vLLM 官方设计文档. [Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/).

## 核心问题

vLLM 的核心目标不是让单个 token 的 Transformer 计算本身发生根本变化，而是在推理服务场景下更高效地管理请求、批处理和 KV cache。传统推理框架常把每条序列的 KV cache 预留成一块连续大矩形，例如按照 `(batch_size, max_seq_len)` 预分配显存。这样实现简单，但会带来明显浪费：

- 预留碎片：decode 阶段还没有生成到的位置被提前占住，未来可能用到，也可能用不到。
- 内部碎片：序列提前结束后，预留区域中永远不会被使用的位置。
- 外部碎片：显存总剩余量足够，但剩余空间不连续，无法满足新的连续分配请求。

PagedAttention 的核心思路是把 KV cache 从“连续大数组”改成“按块分页管理”：逻辑上每个请求仍然看到连续的 token 序列，物理上 KV cache 可以分散存放在不同显存块中，由 block table 负责映射。

## Prefill 与 Decode

一次 LLM 推理通常分成两个阶段：

- Prefill：模型一次性处理完整 prompt，计算 prompt 中所有 token 的 K/V，并写入 KV cache。这个阶段可以充分并行，但 prompt 很长时会占用大量显存和计算。严格来说，prefill forward 的最后一个位置已经能给出第一个输出 token 的 logits。
- Decode：模型基于已有 KV cache 逐 token 生成输出。每生成一个 token，就把该 token 的 K/V 追加进 KV cache。这个阶段串行性更强，长输出时会成为主要耗时。

KV cache 的作用是避免每次生成新 token 时重新计算全部历史 token 的 K/V。代价是显存占用会随请求数、上下文长度、输出长度、层数、hidden size 增长而迅速扩大。

## PagedAttention

PagedAttention 借鉴操作系统分页管理：

| 操作系统分页 | vLLM PagedAttention |
| --- | --- |
| 进程 | request / sequence |
| 虚拟内存 | logical KV blocks |
| 页表 | block table |
| 物理页 | physical KV blocks |
| 页大小 | block size，常见默认值是 16 tokens |

在 vLLM 中，一个 block 会容纳固定数量 token 的 K/V。请求到来后，vLLM 按 token 序列切分逻辑块，再为这些逻辑块分配物理块。注意力计算仍然按逻辑序列进行，底层 attention kernel 通过 block table 找到对应物理块。

这样做的收益：

- 不需要为每条请求提前分配最大长度的连续 KV cache。
- 新 token 只在需要时追加块，显存分配更接近“用多少占多少”。
- 物理显存块可以不连续，从而减少外部碎片。
- 多请求混合时，调度器更容易把显存填满，提高吞吐。

### 工作流程

1. Prefill 分块：vLLM 将 prompt token 按 block size 划分为多个逻辑块，然后从物理 KV block 池中分配实际显存块。
2. 写入 KV cache：模型计算出的每层 K/V 被写入对应物理块。block table 保存逻辑块号到物理块号的映射。
3. Decode 追加：生成新 token 后，把它的 K/V 写入当前最后一个 block 的空 slot；当前 block 写满后，再分配新 block。
4. 逻辑连续，物理离散：请求不需要关心物理块是否连续，减少了外部碎片，也更容易动态容纳不同长度的请求。

## 多请求与共享

PagedAttention 的分页结构让多个请求可以更灵活地共享或释放 KV block。

典型场景：

- Parallel Sampling：同一个 prompt 生成多条候选输出。prompt 部分完全相同，可以共享已有物理 KV block；当不同候选开始生成不同 token 时，再触发写时复制。
- Beam Search：不同 beam 分支拥有大量共同前缀。仍被保留的 beam 可以共享前缀块，被淘汰的 beam 对应块可以释放。
- Shared Prefix：许多请求共享 system prompt、工具说明、长文档上下文或固定模板时，共享前缀的 KV cache 可以被复用。

这里要区分两个概念：PagedAttention 是 KV cache 的分页式显存管理机制；Automatic Prefix Caching 是在分页结构上进一步复用已经算过的完整前缀块。

## 调度与抢占

vLLM 的调度目标是在有限显存下让更多请求持续推进。常见原则可以概括为：

- 先来的请求优先服务，即 FCFS。
- 显存不足时，较晚到达或优先级较低的请求更容易被抢占。

被抢占的请求需要暂停执行，并释放或转移其 KV cache。常见处理方式有：

- Swapping：把被抢占请求的 KV block 从 GPU 换出到 CPU，后续资源充足时再换回。优点是不用重算，缺点是依赖 CPU 内存和 PCIe/NVLink 传输带宽。
- Recomputation：直接丢弃部分 KV cache，之后从 prompt 重新 prefill。对某些请求来说，重算可能比换出换入更划算。

这两种策略本质上是在 GPU 显存、CPU 内存、PCIe/NVLink 带宽和重算开销之间做权衡。直觉上，长公共前缀更值得保留或交换，短请求更容易选择重算。

## 分布式场景

在多 GPU 推理中，vLLM 通常由调度器统一维护请求状态和 block table，再把映射信息广播给各个 worker。每张卡上的 cache engine 管理本卡的物理 KV block。

在张量并行中，各卡处理相同 token 序列但负责不同 attention head 或模型分片。因此逻辑块到物理块的映射关系可以一致，但每张卡物理块里保存的是该卡负责的那部分 K/V 数据。

## 缓存命中：Automatic Prefix Caching

Automatic Prefix Caching，简称 APC，是 vLLM 针对 prefill 阶段的缓存复用机制。它缓存已经计算过的 KV cache block；当新请求和历史请求拥有相同前缀时，新请求可以跳过这部分 prefill 计算，直接复用已有 KV block。

APC 主要减少 prefill 计算，不会减少新 token 的 decode 时间。因此如果请求主要耗时来自超长输出，或者请求之间没有公共前缀，APC 的收益会很有限。

### 命中条件

APC 命中的关键是“相同前缀”，不是“任意位置出现相同片段”。如果两个请求只有中间或结尾某段文本一样，但它们之前的 token 不同，则这些 token 在 Transformer 高层中的隐状态已经包含不同上下文信息，不能安全复用 KV cache。

vLLM 的前缀缓存按 block 粒度工作，并且只缓存完整 block。假设 block size 为 16：

- 两个请求共享 15 个前缀 token：不能命中完整 block。
- 两个请求共享 16 到 31 个前缀 token：能命中 1 个 block。
- 两个请求共享 32 到 47 个前缀 token：能命中 2 个 block。

可复用 token 数可以近似写成：

```text
hit_tokens = floor(shared_prefix_tokens / block_size) * block_size
```

不完整的最后一个 block 仍需要重新 prefill。官方示例中，如果第三个 block 只匹配了前 2 个 slot，即使 token 相同，也不会把这个部分块作为缓存命中。

### Block Hash

vLLM 用 hash 来标识可复用的 KV block。一个 block 的 hash 不只看当前 block 内的 token，还会包含：

- parent hash：前一个 block 的 hash，用来表达“此前前缀也一致”。
- block tokens：当前 block 内的 token ID。
- extra hashes：影响 KV 结果的额外条件，例如 LoRA ID、多模态输入 hash、cache salt 等。

因此，缓存命中本质上是沿着请求开头逐 block 查找最长可复用前缀。只比较当前 block 文本是不够的，因为同一段 token 在不同前文下对应的 K/V 语义不同。

```text
block_hash = hash(parent_block_hash, block_tokens, extra_hashes)
```

这就是为什么 vLLM 复用的是 prefix，而不是任意相同 suffix。即使当前 block token 完全一样，只要父前缀不同，hash 就不同，KV 也不能共用。

### 生命周期

KV cache manager 会维护 block pool、free queue、cache blocks 和 request blocks：

- block pool：启动时预先创建全部 KVCacheBlock，降低运行期对象创建开销。
- cache blocks：从 block hash 映射到可复用的 block ID。
- request blocks：记录每个请求当前使用的 block ID。
- free queue：管理可被再次分配或被淘汰的 block。
- ref_cnt：记录一个 block 当前被多少请求引用。

新请求命中缓存时，KV cache manager 会 touch 命中的 block，增加引用计数，并把它从 free queue 中移走，避免马上被淘汰。未命中的部分再分配新 block 并执行 prefill。请求结束后，如果某些 block 的引用计数归零，它们会回到 free queue；当需要新块而队列头部是已缓存块时，会按 LRU 思路淘汰旧缓存。

### 适合命中的场景

- 多轮对话：同一 session 的历史消息在每轮请求中保持完全相同的前缀。
- RAG 长文档问答：系统提示词和检索文档很长，用户问题放在后面。
- 固定 system prompt：大量请求共享同一段系统提示词、工具说明或策略说明。
- parallel sampling / beam search：多个分支共享相同 prompt 前缀。
- 批量评测：多个样本共享相同 few-shot 示例，只替换最后输入。

### 容易降低命中的情况

- 每次请求前缀里加入动态时间戳、随机 ID、计数器等变化内容。
- chat template、空格、换行或特殊 token 不一致，导致 token IDs 不同。
- 共享内容放在用户问题后面，而不是放在 prompt 前缀。
- 输出很长，瓶颈主要在 decode，而不是 prefill。
- 只有相同后缀或局部片段，没有相同前缀。

### 提高命中率的实践

- 把稳定的 system prompt、工具说明、RAG 文档放在最前面，把变化的问题放在后面。
- 保证 chat template 和 tokenizer 完全一致，不要在共享前缀里混入动态字段。
- 对需要复用的长文档，尽量复用同一份规范化文本，避免无意义空格或换行差异。
- 多租户或隐私敏感场景使用 cache salt 隔离缓存，避免跨用户共享带来的侧信道风险。
- 在线服务中可以显式开启 APC。不同 vLLM 版本默认值可能不同，以 `vllm serve --help` 为准。

```shell
vllm serve /path/to/model \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

如果需要跨环境可复现的 hash，可以关注官方文档中的 `sha256_cbor`；如果在可信单租户环境追求更快 hash，也可以评估非加密 hash 的性能和碰撞风险。

### 两种 KV 缓存：Active KV Cache vs Prefix Cache

讨论 vLLM 的缓存命中行为时，必须区分两个不同的对象，因为它们的生命周期、淘汰策略和命中语义都不同：

- Active KV Cache：正在运行请求的 KV block。这些 block 在请求 prefill/decode 期间被持有，ref_cnt > 0，受 KV cache manager 直接保护。
- Prefix Cache：请求结束后留下来的、供 APC 复用的 cached block。这些 block ref_cnt = 0，挂在 cache blocks 映射上，随时可以被新请求命中并 touch 走，也可能被 LRU 策略淘汰。

这个区分对应到 vLLM 内部数据结构上：

| 概念 | 在 vLLM 中的体现 | 引用计数 | 可见范围 | 典型淘汰时机 |
| --- | --- | --- | --- | --- |
| Active KV Cache | request blocks | ref_cnt > 0 | 单个请求独占或与其他 active 请求共享前缀 | 显存紧张时调度器通过 swap / recompute 主动抢占 |
| Prefix Cache | cache blocks（hash → block_id 映射） | ref_cnt = 0 | 任何后续请求只要前缀 hash 匹配即可命中 | 新分配 block 时按 LRU 从 free queue 头部淘汰旧缓存 |

只有请求正在生成 token 时，它的 KV block 才是 active 的；请求结束后，如果 block 命中过且仍有复用价值，会留在 cache blocks 里转成 prefix cache。

#### 单人连续追问：并不“永远”命中

直觉上，同一个用户连续多轮对话，前缀应当可以一直复用。但更准确的表述是：

- 如果每轮请求都把完整历史对话作为 prompt 发出，且 prompt 前缀 token 完全一致，且启用了 APC，那么前面已经算过的完整 block 在新一轮中通常会命中。
- 这种命中主要省的是 prefill 的重复计算，不省新生成 token 的 decode。
- 命中按 block 粒度发生，未填满的最后一个 block 仍需重新 prefill。

但"永远命中"并不成立。下面这些情况会破坏命中：

- APC 没有开启，或当前 vLLM 版本默认行为不同。
- prompt 中混入了动态时间戳、随机 ID、变化的工具 schema、变化的 system message 等。
- chat template、空格、换行、特殊 token 在不同轮次不一致，导致 token IDs 错位。
- 共享前缀不足一个完整 block。
- vLLM 服务重启，block pool 整体清空。
- 显存压力下旧的 cached blocks 被淘汰。
- 多租户 cache salt、LoRA ID、多模态 extra hash 等发生变化，使 block hash 不一致。

#### 并发请求下的淘汰行为

多个用户同时在线时，显存压力会让缓存行为更复杂。需要分开看两种对象：

##### Active KV Cache 的抢占

- 正在生成中的请求，其 KV block 是带引用计数的，不会因为新请求到来就被随意清掉。
- 显存不够时，调度器会触发 preemption，对被抢占的请求做 swap 或 recompute。
- 策略上更偏向保护先到的请求，后到的请求或低优先级请求更容易被抢占，而不是"最先问的人最先被清"。

##### Prefix Cache 的淘汰

- 请求结束后，ref_cnt 归零的 block 会回到 free queue，作为 prefix cache 留存。
- 这些 cached block 并不是永久的。当需要分配新 block 而 free queue 头部是已缓存的 block 时，vLLM 会按 LRU 思路淘汰旧缓存。
- 如果第一个用户暂停很久，他之前留下的 prefix cache 很可能被其他并发请求挤占掉；等他再追问时，就无法命中，需要重新 prefill。

所以更准确的整体表述是：

- 单人连续对话时，只要 prompt 前缀稳定、APC 开启、缓存未被淘汰，通常会持续命中已有完整前缀块。
- 并发场景下，其他请求会消耗 KV block，显存紧张时会导致旧的、未被引用的 cached prefix blocks 被淘汰，从而降低后续命中率。
- 但 vLLM 一般不会优先清掉最早正在运行的用户的 active KV cache；更常见的是淘汰已经空闲的旧 prefix cache，或抢占较晚或低优先级的请求。

把"并发会降低缓存命中"这个直觉保留下来，但把"先问的人最先被清"改成"未被引用、较久未被 touch 的缓存更容易被淘汰"，会更贴近 vLLM 实际行为。

## 常见问题

**逻辑块存在哪里？**

逻辑块和 block table 主要是调度侧的元数据，可理解为 host-side 对象；真正占显存的是物理 KV block。发生 swap/offload 时，被搬运的是物理 KV 数据，而不是一个额外的“逻辑块副本”。

**一个 token 是一个 block 吗？**

不是。一个 block 通常包含多个 slot，一个 slot 存一个 token 对应的 K/V。文章或图示中有时会为了讲解把 block 画得很细，但实现上 block size 通常大于 1。

**不同请求能不能映射到同一个物理 block？**

可以，但前提是它们命中同一个完整前缀 block，也就是 block hash 一致。只要前缀不同，即使局部 token 相同，也不能共用同一个物理 KV block。

## 实操

linux

```shell
CUDA_VISIBLE_DEVICES=0 nohup vllm serve E:\Models\Qwen3.5-2B-AWQ-4bit \
  --port 8009 \
  --tensor-parallel-size 1 \
  --max-model-len 32000 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --served-model-name qwen3.5-2b \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --gpu-memory-utilization 0.5
```

powershell:

```shell
$env:CUDA_VISIBLE_DEVICES="0"
vllm serve "E:\Models\Qwen3.5-2B-AWQ-4bit" `
  --port 8009 `
  --tensor-parallel-size 1 `
  --max-model-len 32000 `
  --speculative-config "{\`"method\`":\`"qwen3_next_mtp\`",\`"num_speculative_tokens\`":2}" `
  --served-model-name qwen3.5-2b `
  --reasoning-parser qwen3 `
  --gpu-memory-utilization 0.5
```

cmd:

```shell
vllm serve "E:\Models\Qwen3.5-2B-AWQ-4bit" ^
  --port 8009 ^
  --tensor-parallel-size 1 ^
  --max-model-len 10000 ^
  --served-model-name qwen3.5-2b ^
  --reasoning-parser qwen3 ^
  --default-chat-template-kwargs '{"enable_thinking": true}' ^
  --gpu-memory-utilization 0.7
```

如果当前 vLLM 版本需要显式开启前缀缓存，可以在启动参数中加入：

```shell
--enable-prefix-caching
```

如果需要控制前缀缓存 hash 算法，可参考官方参数：

```shell
--prefix-caching-hash-algo sha256
```
