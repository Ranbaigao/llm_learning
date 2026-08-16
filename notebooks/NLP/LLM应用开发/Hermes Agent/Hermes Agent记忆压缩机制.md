# 内存压缩机制

> **受众：** Hermes Agent 贡献者、网关开发人员以及调优长时间运行会话的高级用户。
> **源文件：** `hermes_state.py`、`agent/context_compressor.py`、`agent/conversation_compression.py`、`agent/conversation_loop.py`、`agent/turn_context.py`、`gateway/run.py`、`tools/session_search_tool.py`
> **最后更新：** 2026-08-16

## 概述

Hermes Agent 通过**摘要**（压缩）对话记录中间部分而非直接截断，使长时间对话保持可用。本文档端到端地描述其工作原理：磁盘上的表示形式是什么、什么触发压缩、压缩输出是什么样子，以及压缩失败时会发生什么。

系统包含三个松散耦合的部分，容易混淆：

1. **压缩引擎**——`hermes_state.py` 中的 `archive_and_compact()` 以及 `agent/context_compressor.py` 中的 `ContextCompressor`。这部分实际重写对话记录。
2. **触发系统**——大约十五个调用点，分布在 `agent/conversation_loop.py`、`agent/turn_context.py`、`gateway/run.py` 以及三个手动入口点（CLI、网关、ACP）。这些决定*何时*压缩。
3. **跨会话回忆层**——`hermes_state_search.py` + `tools/session_search_tool.py`，它是对原始对话记录（而非摘要）进行关键词搜索。它回答的问题是“用户搜索聊天历史时会看到什么？”。

这三个部分以令人惊讶的方式交互：压缩**不会**删除旧消息，这意味着即使会话已被压缩，搜索仍基于原始文本工作。这是设计中最重要且非显而易见的特性，本文其余部分将展开说明。

---

## 1. 磁盘上的表示

每个已压缩会话在 `messages` 表中包含三种行：

| 标志值 | 含义 | 对实时上下文可见？ | 在 FTS 中索引？ |
|---|---|---|---|
| `active=1, compacted=0` | 实时消息，原始文本 | 是 | 是 |
| `active=0, compacted=1` | 被压缩软归档，**原始文本保留** | 否 | 是 |
| `active=0, compacted=0` | 被用户回退（例如 `/rewind`） | 否 | 否（默认排除） |

关键点是第二行。`archive_and_compact()` **不会**删除它摘要的行——它将这些行翻转为 `active=0, compacted=1`，并在保留的尾部前插入新的摘要行：

```python
# hermes_state.py — archive_and_compact()
conn.execute(
    "UPDATE messages SET active = 0, compacted = 1 "
    "WHERE session_id = ? AND active = 1",
    (session_id,),
)
inserted, tool_calls_total = self._insert_message_rows(
    conn, session_id, compacted_messages
)
```

FTS 触发器（`messages_fts`、`messages_fts_trigram`、`messages_fts_cjk`）在 `INSERT` 时索引，在 `DELETE` 时删除，但它们**不**根据 `active`/`compacted` 进行过滤。翻转标志是一个保留内容的 `UPDATE`，因此原始文本仍保留在索引中。这就是为什么即使在压缩之后，跨会话回忆仍能基于原始对话记录工作。

---

## 2. 跨会话回忆——对原始文本的关键词搜索

回忆层是 **SQLite FTS5**（[hermes_state_search.py](hermes_state_search.py)）。搜索路径中没有 RAG、没有嵌入、没有 LLM。`SessionSearchMixin` 对 `messages_fts`（unicode61 分词器）运行查询，并具有可选的 trigram（`messages_fts_trigram`）和 CJK-二元分词（`messages_fts_cjk`）后备索引。

`search_messages()` 默认返回压缩前和压缩后的行：

```python
where_clauses.append("(m.active = 1 OR m.compacted = 1)")
```

因此，即使会话已被压缩，搜索一个只出现在原始对话记录中的短语仍然能找到它。工具层（[tools/session_search_tool.py](tools/session_search_tool.py)）为每个命中返回三样东西：

- `snippet`——原始匹配消息的 FTS5 高亮摘录
- `messages`——匹配消息加上 ±5 条消息的窗口，锚点被标记
- `bookend_start` / `bookend_end`——周围上下文的简短尾部，用于定位

唯一的后搜索过滤是**压缩摘要消息会从书签中剥离**（以避免将大型摘要负载重新注入新会话），但 FTS 命中本身原样保留。

### 设计说明

> *“任何地方都没有 LLM 调用——每种形式都返回数据库中的实际消息。”*
> ——`tools/session_search_tool.py` docstring

整个回忆路径是确定性的。摘要与原始消息一起存储在数据库中，但除非关键词恰好出现在摘要文本中，否则摘要**永远不会**是用户从关键词搜索中得到的返回结果。

---

## 3. 压缩输出的样子

压缩器将新对话记录组装为**头部 + 1 条摘要消息 + 尾部**——而不是多次重写的“缩小版对话记录”。头部和尾部保持原样。

```python
# agent/context_compressor.py — Phase 4: Assemble compressed message list
compressed = []
compressed.extend(messages[:compress_start])          # head, unchanged
compressed.append({                                    # ONE summary message
    "role": summary_role,                              # user or assistant (alternation-safe)
    "content": self._with_summary_prefix(summary),
    COMPRESSED_SUMMARY_METADATA_KEY: True,
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: bool(self._summary_has_user_turn),
})
compressed.extend(messages[compress_end:])             # tail, unchanged
```

默认保护如下：

| 区域 | 数量 | 来源 |
|---|---|---|
| `protect_first_n` | 头部 3 条消息 | `compression.protect_first_n` |
| `protect_last_n` | 尾部 20 条消息 | `compression.protect_last_n` |
| `min_tail_user_messages` | 尾部至少 1 条真实用户消息 | `compression.min_tail_user_messages` |

只有中间部分可被摘要。摘要消息携带 `SUMMARY_PREFIX` 头部和 `_SUMMARY_END_MARKER` 尾部标记，以便后续解析时边界明确。

---

## 4. 什么触发压缩

大约有**十五个**触发点，分为自动、手动和特殊路径。

### 4.1 自动——token 阈值（主路径）

决策函数是 `ContextCompressor.should_compress_info()`：

```python
# agent/context_compressor.py
def should_compress_info(self, prompt_tokens=None):
    tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
    if tokens < self.threshold_tokens:
        return False, None
    if self._automatic_compression_blocked():
        return False, self._compression_block_reason() or "blocked"
    return True, None
```

`threshold_tokens` 由 `_compute_threshold_tokens()` 计算：

```python
effective_window = context_length - (max_tokens or 0)
pct_value = int(effective_window * threshold_percent)     # default 0.50
floored = max(pct_value, MINIMUM_CONTEXT_LENGTH)           # 64K floor
```

| 模型 / 上下文窗口 | 触发阈值 |
|---|---|
| 默认（≥512K 上下文） | `context_length − max_tokens` 的 50%，下限为 **64K** |
| 上下文 < 512K | 75%（`_SMALL_CTX_THRESHOLD_PERCENT`） |
| 退化的极小窗口 | 有效窗口的 85%（`_MIN_CTX_TRIGGER_RATIO`） |
| Arcee Trinity Thinking | 按模型覆盖为 75% |
| Codex OAuth、gpt-5.4/5.5/5.6 | 85%（由 `compression.codex_gpt55_autoraise` 控制） |
| Codex、gpt-5.3-codex-spark | 70% |
| 按模型覆盖 | `compression.model_thresholds`（最长子串匹配胜出） |
| 硬上限 | `compression.threshold_tokens`（取 `min(ratio, cap)`） |

每次 API 往返都会进行检查：

- **API 前**在 `agent/conversation_loop.py:2376`——使用 `request_pressure_tokens`
- **API 后**在 `agent/conversation_loop.py:7025`——使用来自提供商响应的 `_real_tokens`

### 4.2 自动——消息数量硬限制（网关卫生）

网关在每一轮代理运行前执行卫生检查。它使用两个独立维度：

```python
# gateway/run.py
_needs_compress = (
    _approx_tokens >= _compress_token_threshold   # 85% by default
    or _msg_count >= _HARD_MSG_LIMIT               # 5000 by default
)
```

5000 条消息限制是**死亡螺旋安全阀**——见 §6。硬限制可通过 `compression.hygiene_hard_message_limit` 配置。

### 4.3 自动——空闲时间（可选启用）

```python
# agent/turn_context.py
_idle_after = getattr(agent, "compression_idle_compact_after_seconds", 0)
if agent.compression_enabled and _idle_after > 0 and messages:
    _idle_gap = time.time() - getattr(agent, "_last_activity_ts", time.time())
    if _idle_gap >= _idle_after:
        if _should_idle_compact(messages, floor_tokens=...):
            messages, active_system_prompt = agent._compress_context(...)
```

默认为 `0`（禁用）。该触发经过一个 **token 下限**（§4.4）——即使会话已经空闲了一个小时，如果其 token 数低于下限，也不会被压缩。

### 4.4 token 下限

阈值计算有一个下界：

```python
floored = max(pct_value, MINIMUM_CONTEXT_LENGTH)  # 64K
```

这就是空闲路径中“仍受 token 下限限制”的含义：**仅时间是不够的**——在任何基于时间的触发触发之前，会话必须至少有 `min(64K, computed_threshold)` 个 token。原因是压缩并非免费的：它需要一次 LLM 调用、破坏提示缓存并重写数据库行。压缩一个很小的会话是净损失。

### 4.5 自动——错误响应

提供商错误也会触发压缩：

- **HTTP 413 Payload Too Large**——`agent/conversation_loop.py:5165`
- **HTTP 429 + long-context-tier**——`agent/conversation_loop.py:4904`（同时将 `context_length` 降至 200K）
- **输出超过 `max_tokens`**——`agent/conversation_loop.py:5312`
- **通用上下文溢出**——`agent/conversation_loop.py:5466`

### 4.6 自动——预检

多轮循环在轮次前运行：

- **轮次序言预检**——`agent/turn_context.py:985`。token 估算值高于阈值**或**消息数量高于 `protect_first_n + protect_last_n + 1`。
- **引擎驱动预检**——`agent/turn_context.py:1118`。插件引擎（例如 LCM）可以通过 `should_compress_preflight()` 请求低于阈值的压缩。

### 4.7 手动触发（`force=True`）

| 入口 | 位置 | 绕过冷却/防抖动？ |
|---|---|---|
| CLI `/compress` | `cli.py:11845` | 是 |
| CLI `/compress here [N]` | `cli.py:11807` | 是（部分） |
| 网关 `/compress` | `gateway/slash_commands.py:4206` | 是 |
| ACP `/compress` | `acp_adapter/server.py:2353` | 是（同时清除 `_session_db` 以避免 ID 轮换） |

这四个入口都跳过摘要失败冷却和防抖动保护。

### 4.8 特殊路径（绕过主压缩器）

- **Codex 应用服务器**——当 `agent.api_mode == "codex_app_server"` 时，`compression.codex_app_server_auto` 选择 `"native"`（codex 压缩自己的线程）、`"hermes"`（Hermes 调用 `codex_session.compact_thread()`）或 `"off"`。
- **OpenAI 原生服务器端压缩**——通过 `compression.codex_responses_native: False` 选择加入。仅限 gpt-5.6，在 `api.openai.com` / ChatGPT Codex 后端。本地触发被钳制在本地压缩器触发阈值以下 **8,192 个 token**，因此服务器总是先触发。
- **微压缩**——`compression.micro_compact: False`（默认）。每个成功轮次后，一次交换折叠进滚动摘要。代价：每轮一次提示缓存破坏。
- **主动工具结果修剪**——当主压缩因冷却/防抖动而被**阻止**但 token 仍超过阈值时，在每次工具调用后运行。确定性的，无 LLM，对工具结果进行确定性去重 + 摘要。

### 4.9 总开关

`compression.enabled: False` 禁用所有自动路径。手动 `/compress` 仍然有效。

---

## 5. 网关卫生检查

网关卫生检查是“廉价的本地检查”，在每条用户消息时、代理看到会话之前运行。它执行粗略的 token 估算（无 API 调用）并统计消息数量。

### 5.1 它做什么

```python
# gateway/run.py:18142-18345
if history and len(history) >= 4:
    _msg_count = len(history)
    _approx_tokens = _rough_token_estimate(history)
    _compress_token_threshold = int(_hyg_context_length * _hyg_threshold_pct)  # 85% default
    _HARD_MSG_LIMIT = _hyg_hard_msg_limit  # 5000 default
    _needs_compress = (
        _approx_tokens >= _compress_token_threshold
        or _msg_count >= _HARD_MSG_LIMIT
    )
```

### 5.2 它防止的死亡螺旋

基于 token 的触发是单点故障：它依赖于提供商返回带有 `last_prompt_tokens` 的响应。如果 API 调用失败——网络抖动、提供商中断、超时——响应永远不会到达，触发永远看不到新的 token，压缩也永远不会触发。会话持续增长。

这产生了一个正反馈循环：

```
会话增长 → 下一个负载更大 → 更可能失败
      ↑                                          ↓
      └──── 无压缩（无 token 数据） ←────────────┘
```

如果没有后备，会话可以无限增长，使每次后续启动变慢，并最终使提示构建器 OOM。5000 条消息硬限制**不依赖提供商响应**——它统计数据库行数，这是一个纯本地查询。因此，当 token 路径失明时，计数路径仍然有效。

代码注释直接写道：

> *“硬安全阀：当消息数量极端时强制压缩……防止 API 断开导致 token 数据收集缺失的死亡螺旋。”*

### 5.3 “网关轮次”的含义

“网关轮次”是一个完整的用户消息 → 代理响应 → 用户消息循环。网关是 Telegram、Discord、TUI、ACP 等的统一入口点。卫生检查在代理加载会话**之前**运行——因此代理永远不必处理已经危险地过大的会话。

| 触发 | 位置 | 节奏 | 依赖 |
|---|---|---|---|
| API 前压力检查 | `conversation_loop.py` | 每次 API 往返 | 提供商响应 |
| API 后真实 token 检查 | `conversation_loop.py` | 每次 API 往返 | 提供商响应 |
| **网关卫生** | `gateway/run.py` | **每一轮** | **无**（仅本地） |
| 手动 `/compress` | CLI / 网关 / ACP | 用户发起 | 无 |

卫生检查是最粗的筛子——它运行频率最低，但依赖最少。

---

## 6. 压缩输出的长度限制

摘要消息有一个 **[2,000, 10,000] 个 token** 的软预算，仅通过提示指导来执行。

### 6.1 预算计算

```python
# agent/context_compressor.py
_MIN_SUMMARY_TOKENS = 2_000
_SUMMARY_RATIO = 0.20
_SUMMARY_TOKENS_CEILING = 10_000

def _compute_summary_budget(self, turns_to_summarize):
    content_tokens = estimate_messages_tokens_rough(turns_to_summarize)
    budget = int(content_tokens * _SUMMARY_RATIO)
    return max(_MIN_SUMMARY_TOKENS, min(budget, self.max_summary_tokens))

@property
def max_summary_tokens(self) -> int:
    if self._max_summary_tokens is None:
        self._max_summary_tokens = min(
            int(self.context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )
    return self._max_summary_tokens
```

因此有效预算为：

```
max(2000, min(被压缩内容 × 0.20, min(上下文 × 0.05, 10000)))
```

### 6.2 为什么 `max_tokens` 故意不发送给 API

```python
# agent/context_compressor.py:4183-4204
call_kwargs = {
    "task": "compression",
    "messages": [{"role": "user", "content": prompt}],
    # NO max_tokens: the output cap must never truncate a summary.
    # ``summary_budget`` is prompt-level guidance only ("Target ~N tokens" above).
    # Most OpenAI-compatible wires already omit the param (see _build_call_kwargs),
    # but the Anthropic Messages wire and NVIDIA NIM forward it — a hard cap
    # there cut summaries mid-section (thinking models burn the cap on
    # reasoning first), producing truncated/thinking-only summaries and
    # compaction loops. Omitting lets the adapter fall back to the model's
    # native output ceiling.
}
```

压缩器明确从线路调用中省略 `max_tokens`。历史证明，在此处设置硬上限会导致具备思考能力的模型将整个输出预算花在推理上，然后返回空/完全截断——这产生了不完整的摘要和重复压缩循环。让适配器回退到模型的原生输出上限是更安全的选择。

### 6.3 输入侧限制

提供给摘要器的提示也是有限的：

| 常量 | 值 | 用途 |
|---|---|---|
| `_SUMMARY_INPUT_MAX_CHARS` | 160,000 字符（约 40K token） | 整个提示的上限 |
| `_CONTENT_MAX` | 6,000 字符 | 每条消息正文的上限 |
| `_CONTENT_HEAD` | 4,000 字符 | 截断时从开头保留的部分 |
| `_CONTENT_TAIL` | 1,500 字符 | 截断时从结尾保留的部分 |
| `_TOOL_ARGS_MAX` | 1,500 字符 | 工具调用参数的上限 |

当输入超过 160K 字符时，`_bound_summary_input()` 保留头部 45% + 尾部 55%，并带有明确的 `[summary input truncated: omitted N chars from the middle to keep compression prompt bounded]` 标记。

### 6.4 为什么没有调用后截断

压缩器在成功 LLM 调用后**不会**检查 `finish_reason` 或摘要长度。如果模型返回 15K token 的摘要，则原样存储。如果模型达到其原生输出上限并返回 `finish_reason="length"`，该截断响应也会被原样存储。这是一个有意的权衡——另一种选择是损坏的摘要。

这种权衡的代价是，某些会话会积累很大的摘要。下一次压缩会将那个大摘要视为 `_previous_summary` 并重新摘要，最终收敛。

### 6.5 回退（LLM 调用失败）

当 LLM 调用失败（429、超时、网络）时，压缩器回退到本地生成的确定性存根：

```python
_FALLBACK_SUMMARY_MAX_CHARS = 8_000         # 8K chars
_FALLBACK_PREVIOUS_SUMMARY_MAX_CHARS = 3_000  # 3K chars
```

回退在 8K 字符处**截断**，并带有 `\n...[fallback summary truncated]`。与 LLM 生成的摘要不同，回退确实有一个硬上限。

### 6.6 摘要预算总结

| 方面 | 值 | 执行位置 |
|---|---|---|
| 线路上的 `max_tokens` | **未设置** | 故意省略 |
| 软预算（仅提示） | 2,000 – 10,000 个 token | 提示文本 |
| 每条消息输入上限 | 6,000 字符（4K 头部 + 1.5K 尾部） | 输入序列化器 |
| 整个提示输入上限 | 160,000 字符 | 输入序列化器 |
| LLM 成功时的输出截断 | **无** | 压缩器不检查 `finish_reason` |
| LLM 失败时的输出截断 | 8,000 字符（前一个为 3,000） | 回退构建器 |

---

## 7. 失败处理——三层防御

当压缩失败（摘要 LLM 不可达、429、瞬时错误）时，三层防御保护会话免于失控循环。

### 7.1 第一层——冷却

```python
# agent/context_compressor.py:2955
_cooldown_remaining = self._summary_failure_cooldown_until - time.monotonic()
if _cooldown_remaining > 0:
    return f"cooldown:{_cooldown_remaining:.0f}s"
```

在 429 / 瞬时 aux-LLM 失败之后，`_summary_failure_cooldown_until` 阻止进一步自动压缩 600 秒（`_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600`）。冷却**持久化到数据库**，因此同一会话上的兄弟代理也遵守它。原始 bug (#11529) 是每个后续轮次都重新触发无操作，使 CLI 看起来像冻结了。

### 7.2 第二层——防抖动

```python
# agent/context_compressor.py:3015
_ANTI_THRASH_RECOVERY_SECONDS = 300.0  # 5 minutes
if self._ineffective_compression_count >= 2 or self._fallback_compression_streak >= 2:
    if _now >= self._anti_thrash_recovery_deadline:
        # probation probe
```

如果最近两次压缩每次节省的不足 10%，则退避 5 分钟。连续阻塞 5 分钟后，允许一次试探——如果试探也没有用，则重新触发。该状态通过持久化行保存，因此重启不会解除它 (#54923)。

### 7.3 第三层——硬上限 → 会话重置

`compression_exhausted` 标志在 `max_compression_attempts` 超出后在 `agent/conversation_loop.py:5155, 5300, 5455, 5518` 设置。网关读取此标志并执行会话的**自动重置**。当冷却和防抖动都失败时，逃生舱是最后的停止点。

### 7.4 不同路径失败方式不同

- **主动工具结果修剪**——静默失败，保留原始对话记录，无缓存破坏。
- **微压缩**——静默失败，内存中的拼接仍会发生，恢复时原始行会在下次批量压缩清理之前以 `active=1` 重新出现，与摘要并存。
- **主 `compress_context()`**——异常传播；工具后调用点将其检测为 `compression_skipped_due_to_lock` 并返还尝试预算。在 `max_compression_attempts` 耗尽后，网关自动重置。

主触发器没有“会话持续增长”的优雅路径——只有主动修剪和微压缩是完全失败无操作的。对于主路径，反复失败会触发自动重置逃生舱。

---

## 8. 配置参考

所有压缩设置都位于 `config.yaml` 的 `compression:` 下。默认值在 `hermes_cli/config_defaults.py:620-823` 中。

### 8.1 阈值

| 键 | 默认值 | 效果 |
|---|---|---|
| `enabled` | `True` | 总开关——禁用所有自动压缩 |
| `threshold` | `0.50` | 触发比例（<512K 模型的小窗口下限为 0.75） |
| `threshold_tokens` | `None` | 绝对 token 上限；取 `min(ratio, cap)` |
| `model_thresholds` | `{}` | 按模型比例覆盖（最长子串匹配胜出） |

### 8.2 行为

| 键 | 默认值 | 效果 |
|---|---|---|
| `target_ratio` | `0.20` | 压缩后目标 = `threshold × target_ratio` |
| `protect_first_n` | `3` | 头部消息永不被压缩 |
| `protect_last_n` | `20` | 尾部消息永不被压缩 |
| `min_tail_user_messages` | `1` | 尾部保证的真实用户消息 |
| `max_attempts` | `3` | 每轮重试的硬上限（1–10） |
| `in_place` | `True` | 软归档 + 插入与会话轮换 |
| `abort_on_summary_failure` | `False` | 失败关闭与插入占位符 |

### 8.3 主动修剪

| 键 | 默认值 | 效果 |
|---|---|---|
| `proactive_prune_tokens` | `0` | 确定性工具结果修剪的较低阈值 |
| `proactive_prune_min_reclaim_tokens` | `4096` | 提交修剪所需的最小回收量 |
| `proactive_prune_min_result_chars` | `8000` | 有资格摘要的工具结果大小下限 |

### 8.4 微压缩

| 键 | 默认值 | 效果 |
|---|---|---|
| `micro_compact` | `False` | 总开关（每次传递 = 1 次提示缓存破坏） |
| `micro_compact_every_n_turns` | `1` | 节奏：每 N 轮传递一次 |
| `micro_compact_defrag_threshold_tokens` | `2000` | 何时对滚动摘要进行碎片整理 |

### 8.5 空闲时间

| 键 | 默认值 | 效果 |
|---|---|---|
| `idle_compact_after_seconds` | `0` | 非活动这么多秒后压缩 |

### 8.6 网关卫生

| 键 | 默认值 | 效果 |
|---|---|---|
| `hygiene_hard_message_limit` | `5000` | 消息数量安全网 |
| `hygiene_timeout_seconds` | `30` | 卫生压缩的非活动预算 |
| `hygiene_total_ceiling_seconds` | `600` | 卫生压缩等待的绝对上限 |
| `hygiene_failure_cooldown_seconds` | `300` | 跳过重复失败的卫生尝试 |

### 8.7 提供商特定

| 键 | 默认值 | 效果 |
|---|---|---|
| `codex_gpt55_autoraise` | `True` | 在 Codex OAuth 上将 gpt-5.4/5.5/5.6 的阈值提高到 0.85 |
| `codex_app_server_auto` | `"native"` | `"native"` / `"hermes"` / `"off"` |
| `codex_responses_native` | `False` | 选择加入 OpenAI 服务器端压缩（仅 gpt-5.6） |
| `codex_responses_compact_threshold` | `200000` | 服务器端触发 token |

### 8.8 超时

| 键 | 默认值 | 效果 |
|---|---|---|
| `context_timeout_seconds` | `120` | 代理内 `compress_context` 的空闲预算 |
| `context_total_ceiling_seconds` | `600` | 摘要等待的绝对上限 |

没有直接控制阈值的环境变量——配置仅通过 `config.yaml` 进行。

---

## 9. 端到端走查

一次压缩事件的完整跟踪：

```
1. 用户在 Telegram 上发送一条消息。
2. 网关接收它；卫生检查运行。
   - msg_count = 800, approx_tokens = 120K → 低于两个阈值 → 无操作。
3. 代理加载会话，调用 API。
4. API 返回带有 last_prompt_tokens = 110K 的响应。
5. should_compress_info(110K) → 阈值为 100K，110K > 100K → True。
6. 调用 _compress_context()。
7. protect_first_n = 3 → 头部 = msgs[0:3]
   protect_last_n = 20 → 尾部 = msgs[-20:]
   中间 = msgs[3:-20] → 待摘要
8. _compute_summary_budget(middle) → 比如 6500 个 token。
9. prompt = 目标约 6500 个 token 的结构化摘要模板。
10. LLM 调用返回 5800 个 token 的摘要。
11. 新消息列表 = 头部 + 1 条摘要消息 + 尾部。
12. archive_and_compact() 执行：
    - UPDATE messages SET active=0, compacted=1 WHERE session_id=? AND active=1
    - INSERT 新的压缩消息（头部 + 摘要 + 尾部）
13. FTS 触发器在新的 INSERT 上触发；旧行保留在索引中，
    因为它们的标志通过 UPDATE（而非 DELETE）翻转。
14. 代理看到精简后的会话，继续。
15. 数据库现在包含：
    - 800 条实时行（active=1）
    - 一些压缩行（active=0, compacted=1）——原始文本
    - 一些摘要行（active=1）——新摘要
16. 搜索“出现在消息 47 中的关键词”仍然能找到它，
    因为消息 47 仍在 FTS 索引中。
```

---

## 10. 关键不变量

1. **压缩是非破坏性的。** 原始文本保留在数据库和 FTS 索引中。压缩后，基于原始对话记录的搜索仍然有效。
2. **头部和尾部永不被压缩。** 只有中间 3–20 区域可被摘要。
3. **摘要长度受提示指导软约束，而非线路硬上限。** 压缩器故意省略 `max_tokens`，以避免截断思考模型的输出。
4. **消息数量硬限制是 token 路径的备份。** 当提供商不可达时，卫生检查仍通过计数行来触发压缩。
5. **三层失败防护防止失控循环。** 冷却（10 分钟）→ 防抖动（5 分钟）→ 会话重置。
6. **手动 `/compress` 绕过冷却和防抖动。** 当用户需要立即压缩时，这是用户的逃生舱。
7. **摘要消息是唯一的新内容。** 压缩后的对话记录是*头部 + 1 条摘要 + 尾部*，而不是多条缩短的消息。