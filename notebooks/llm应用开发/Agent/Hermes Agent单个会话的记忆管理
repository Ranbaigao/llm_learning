# Hermes Agent 单会话记忆管理策略


## 1. 总览：双层架构

Hermes 的会话记忆由两个层次构成，二者严格分离：

```
┌────────────────────────────────────────────────────────────┐
│ 活动 Context（每轮发给模型的 prompt）                        │
│  = system prompt + 当前会话窗口（消息序列）+ 当轮显式注入      │
│  特征：受 token 预算约束，会被压缩                            │
├────────────────────────────────────────────────────────────┤
│ 持久存储（模型不可见，用于恢复 / 搜索 / 审计）                  │
│  = state.db（SQLite + FTS5，WAL 模式，schema v24）           │
│  特征：完整保留一切，压缩不删除历史                           │
└────────────────────────────────────────────────────────────┘
```

官方文档的核心原则（`sessions.md` 第 31-34 行）：

> "Hermes stores session history so it can resume conversations, but it does **not** keep re-sending every byte it has ever handled. On each turn, the model sees the selected system prompt, the current conversation window, and any content Hermes explicitly injects for that turn."

**存下来的 ≠ 每轮都发给模型。**

### 两条硬不变式

来自 `hermes-agent` skill 的 Hard Invariants（在任何配置下都成立）：

1. **Never break prompt caching** — 会话中途不得修改过去的 context、工具集或 system prompt。**唯一例外是 context compression**。
2. **Message role alternation** — 绝不出现两个连续 assistant 或 user 消息；只有 `tool` 结果可以连续重复。

---

## 2. 核心概念与术语

| 术语                                 | 定义                                     | 源码锚点                             |
| :----------------------------------- | :--------------------------------------- | :----------------------------------- |
| **Context window（会话窗口）**       | 当前发送给模型的消息序列                 | —                                    |
| **Compaction / Compression（压缩）** | 用 LLM 摘要替换早期消息、缩小窗口的过程  | `context_compressor.py`              |
| **Protected head（头部保护）**       | system prompt + 前 N 条消息，永不被摘要  | `_protect_head_size()` 行 4682       |
| **Protected tail（尾部保护）**       | 按 token 预算保留的最近消息，原样保留    | `_find_tail_cut_by_tokens()` 行 6059 |
| **Summary（摘要）**                  | 压缩产物：结构化 markdown 块，替代中间段 | `_generate_summary()` 行 3469        |
| **Rotate（轮换）**                   | 压缩后创建子会话、切换 session id        | `compress_context()` 行 3254         |
| **In-place（原地）**                 | 压缩后旧行软归档、同 id 继续             | 行 1852-1870                         |
| **Lineage（谱系）**                  | 由 `parent_session_id` 链接的压缩会话链  | schema 行 204                        |
| **Anti-thrash（防抖）**              | 连续低效压缩后的熔断                     | 行 2602-2606                         |
| **Cooldown（冷却）**                 | 摘要 LLM 失败后的触发抑制                | 行 2599-2601                         |

---

## 3. 每轮的消息流：context 如何组装

每个 turn 发给模型的内容由三部分组成：

| 组成          | 内容                                                         | 约束                                      |
| :------------ | :----------------------------------------------------------- | :---------------------------------------- |
| system prompt | 人格规则、memory 注入、skills 索引、项目 context 文件        | 会话中途**不可变**（保护 prompt caching） |
| 会话窗口      | 当前 session 的 user / assistant / tool 消息序列             | 角色严格交替                              |
| 当轮注入      | 附件提取文本、`@` context references、out-of-band 消息、todo 注入 | turn-scoped，用完即弃                     |

### 3.1 附件的 turn-scoped 处理

媒体文件**不会**反复进入未来 prompt（`sessions.md` 第 36-52 行）：

- **图片**：仅附在下一次模型调用；模型不支持原生视觉时，先由辅助视觉模型预分析成文字描述
- **音频**：STT 转写为文本
- **文档**：抽取文本，或仅保留本地路径 + 短注
- 原始字节（JPEG / 音频 / 二进制）**绝不重复拷贝**进后续 prompt

官方明确指出：context 膨胀的头号元凶**不是媒体**，而是冗长文本——粘贴的 transcript、完整日志、大工具输出、长 diff、重复状态报告。

### 3.2 图片超宽恢复

`conversation_compression.py` 的 `try_shrink_image_parts_in_messages()`（文件头 docstring 行 20-22）：当 provider 报图片过大时，把 `data:image/...;base64` 部件重编码为更小尺寸，使重试能塞进 provider 上限（如 Anthropic 的 5 MB）。

---

## 4. ContextEngine：插件化的生命周期

`agent/context_engine.py` 定义抽象基类，压缩只是其中一个钩子。四个钩子职责严格正交：

| 钩子                        | 触发时机       | 职责                                               | 基类行号 |
| :-------------------------- | :------------- | :------------------------------------------------- | :------- |
| `should_compress()`         | **每轮结束后** | 判定本轮后是否需要压缩                             | 行 146   |
| `compress()`                | 判定为真时     | 缩短已有 context（摘要/截断/任意手段）             | 行 162   |
| `select_context()`          | **每轮发送前** | 为本轮选择用哪些 context（检索/话题路由/分支切换） | 行 215   |
| `prune_tool_results_only()` | 低成本独立触发 | 不调 LLM，确定性裁剪旧工具输出                     | 行 194   |

源码原话（`context_engine.py` 行 22-23）：

```
4. should_compress() checked after each turn
5. compress() called when should_compress() returns True
```

`compress()` 与 `select_context()` 的正交性（行 230-234）：

> - `compress()`：context 太长 → 把它变短
> - `select_context()`：本轮属于另一个 context → 换那个用

默认实现是 `ContextCompressor`（`context_compressor.py` 行 1317），插件可提供自己的引擎（如检索式引擎），接口契约不变。

---

## 5. 压缩触发判定

### 5.1 阈值

`should_compress_info()`（行 2554-2585）：

```python
tokens = prompt_tokens or self.last_prompt_tokens
if tokens < self.threshold_tokens:           # 未超阈值，不压
    return False, None
if self._automatic_compression_blocked():    # 防护闸拦截
    return False, reason                     # "cooldown:N" / "ineffective"
return True, None
```

**`threshold_tokens` 的推导**（`__init__` 行 2206+、属性行 1544-1559）：

```
threshold_tokens = min(
    context_length × threshold_percent,    # 比例阈值，默认 50%
    threshold_tokens_cap                   # 配置里的绝对上限（可空）
)
```

小窗口模型（< 512K）有地板规则（行 1500-1508）：触发点按 `_MIN_CTX_TRIGGER_RATIO` 下调，避免小窗口模型过半才压导致可用余量不足。`model_thresholds` 配置支持按模型名子串覆盖比例（最长匹配优先，`resolve_model_threshold()` 行 1291）。

### 5.2 默认参数表

`ContextCompressor.__init__`（行 2206-2224）：

| 参数                       | 默认值                   | 含义                                                        |
| :------------------------- | :----------------------- | :---------------------------------------------------------- |
| `threshold_percent`        | **0.50**                 | 窗口用量达 50% 触发压缩                                     |
| `protect_first_n`          | 3                        | system prompt 之外额外保护的头部消息数                      |
| `protect_last_n`           | 20                       | 尾部最少保护的消息条数（受 `_MAX_TAIL_MESSAGE_FLOOR` 封顶） |
| `summary_target_ratio`     | 0.20（钳制在 0.10~0.80） | 尾部 token 预算 = 阈值 × 该比例（默认约 20K tokens）        |
| `abort_on_summary_failure` | False                    | 摘要失败时是否中止（默认走降级路径）                        |
| `proactive_prune_tokens`   | 0（关闭）                | 主动工具输出剪枝的触发阈值                                  |
| `min_tail_user_messages`   | 1                        | 尾部至少保留的可行动用户消息数                              |

### 5.3 两道防护闸（持久化，跨进程生效）

**① Cooldown（冷却）**（行 2599-2601、2647-2658）：
摘要 LLM 遇 429 / 瞬时网络错误后进入冷却期。无此闸时，每轮都会重复触发压缩 → 重复插入降级标记 → CLI 假死（issue #11529）。手动 `/compress` 传 `force=True` 可清除冷却立即重试（行 6001-6005）。

**② Anti-thrash（防抖）**（行 2602-2606）：
满足任一条件即停止自动压缩：

- 连续 2 次压缩各省下 < 10%
- fallback 降级路径连续使用 2 次

防止"每次只删 1-2 条消息"的死循环。

**持久化**：冷却截止时间、fallback 连败、低效计数都写在 `sessions` 表（`compression_failure_cooldown_until` / `compression_fallback_streak` / `compression_ineffective_count`，schema 行 234-237）。防护闸判定前会 `_refresh_durable_guards()` 重读 DB（行 2609-2631），避免多进程下用过期的内存快照误判。

---

## 6. 压缩算法：五步流水线

`compress()` 主入口（行 5934），类 docstring（行 1317-1326）定义算法：

```
1. Prune old tool results   ← 无 LLM 的廉价预剪
2. Protect head             ← system prompt 永护 + 前 N 条
3. Find tail cut by tokens  ← 按 token 预算划尾界
4. Summarize middle         ← LLM 结构化摘要中间段
5. Iterative update         ← 二次压缩时增量更新旧摘要
```

### 第 1 步：工具输出预剪（无 LLM 成本）

`_prune_old_tool_results()`（行 2732-2761），三件事：

**a) 一行摘要替换旧工具输出**——不是泛化占位符，而是信息性摘要：

```
[terminal] ran `npm test` -> exit 0, 47 lines output
[read_file] read config.py from line 1 (3,400 chars)
```

**b) 相同工具结果去重**——读同一文件 5 次，只保留最新一份完整副本（去重地板 200 字符）。

**c) 截断尾部保护圈之外的大 `tool_call` 参数**（assistant 消息里的）。

**边界规则**：token 预算优先、条数地板兜底（行 2786-2815）；保护圈自身超过软预算 1.5 倍时，还有一道 pressure pass 降级圈内的大型已完成输出，仅保留最近一小段原文（#61932）。

### 第 2 步：头部保护

`_protect_head_size()`（行 4682-4705）：

- system prompt（若在 index 0）**永远隐含保护**——"load-bearing context that must never be summarised away"
- `protect_first_n` 是 system 之外**额外**保护的条数
- **关键设计**：`protect_first_n` 在首次压缩后**衰减**（`_effective_protect_first_n`），早期用户轮次不会在反复压缩中"化石化"（#11996）。首次压缩：system + 3 条；之后：仅 system。

### 第 3 步：尾部按 token 预算划界

`_find_tail_cut_by_tokens()`（行 6059）：从末尾向前累计 token，直到耗尽 `tail_token_budget`（= 阈值 × `summary_target_ratio` ≈ 20K tokens）。用预算而非固定条数，保证"最近 20K token 的原文不动"。

**边界对齐**（`_align_boundary_backward`，行 4707-4729）：若切分点落在 tool_call/tool_result 组中间，向后走出连续 tool 消息、把边界移到 parent assistant 之前——**绝不劈开一对工具调用**。劈开会导致 orphaned tool result 被 `_sanitize_tool_pairs` 清洗，造成静默数据丢失。

另有一条角色冲突规则（行 6061-6073）：当尾界恰好等于最新可行动用户消息时，保留更老的 assistant/tool 桥接消息，防止摘要与首条尾部消息合并成双角色碰撞。

### 第 4 步：LLM 结构化摘要

详见第 7 节（摘要 prompt 工程）。

**可行性跳过**（`_FEASIBILITY_SKIP_MIDDLE_FRACTION`，行 5948-5952）：当中间段低于阈值的一个零头、且此前有过真实低效记录时，跳过 LLM 调用，直接用确定性丢弃回收——微乎其微的收益不值得一次摘要调用。

### 第 5 步：迭代更新

二次及以后压缩时，旧摘要作为输入的一部分喂给摘要器（`_previous_summary`，行 3720-3729），并受聚合上限约束（防止病理级大手交接把迭代 prompt 撑爆）。摘要器在保留旧信息的基础上追加新进展，信息跨多次压缩得以传承。

### 压缩的兜底语义

**① 消息太少：no-op 但必须记账**（行 6006-6029）

压缩要求至少凑出"头部 + 3 条尾部 + 1 条中间"的结构，否则没有中间段可摘要。典型场景：system prompt + 工具 schema 本身就占了窗口 55%（已超 50% 阈值），但会话只有 system 和两三句对话——`should_compress()` 每轮都返回 True，`compress()` 却永远凑不齐可压结构。若此时只是原样返回，下一轮会再次触发、再次空转，用户看到的就是 CLI 反复"压缩中"的假死。源码对策：**no-op 返回的同时记一次低效判定**；计数到 2 后 anti-thrash 闸切断自动触发（见 5.3 节）——"这个会话压不动，别再试了"。

**② 空白平台回显清理：先于摘要、不可回滚**（行 6041-6051）

消息平台（Telegram/QQ 等）有时会向会话注入空白用户消息行（平台回显产物，无实际内容）。这些行在 Phase 1 与工具预剪同阶段删除——**早于任何摘要逻辑**。后果是：即使摘要随后失败、压缩整体中止，这些行也已经被删了。`compress()` 的返回值**不是"全改或全不改"的事务**——中止的压缩也可能带回一个修改过的列表（echo 已删、摘要未做）。

**③ 孤儿工具对清洗：配对完整性的最后防线**（行 5962-5963）

OpenAI 格式的工具调用必须成对出现：assistant 消息声明 `tool_calls`（id=abc），后续必须有 `role=tool`、`tool_call_id=abc` 的结果消息；API 收到缺半对的消息直接报 400。压缩整段删除中间段时可能恰好切开一对（调用声明留在尾部、结果被摘要，或反之）。收尾时的清洗把失配半对删除，保证发给 API 的序列永远配对完整。它与第 3 步的边界对齐是**预防-补救**关系：对齐（`_align_boundary_backward`）尽量不切开工具对；清洗是切开后的兜底——被清洗的半对内容静默丢失，所以预防优先。

### 多轮压缩的迭代差异：第一次 vs 第二次

第二次及以后的压缩与第一次有六项实质差异：

| 维度        | 第一次压缩                                            | 第二次及以后                                                 | 源码                      |
| :---------- | :---------------------------------------------------- | :----------------------------------------------------------- | :------------------------ |
| 头部保护    | system + 前 3 条非 system 消息（`protect_first_n=3`） | 仅 system（`protect_first_n` **衰减为 0**）                  | 行 4666-4667              |
| 摘要器输入  | 仅中间段（`TURNS TO SUMMARIZE`）                      | `PREVIOUS SUMMARY` + `NEW TURNS TO INCORPORATE`              | 行 3731-3743              |
| 摘要 prompt | "Create a structured checkpoint summary..."           | "You are **updating** a context compaction summary... PRESERVE still-relevant info / ADD new actions (**continue numbering**) / Move In Progress → Completed / Remove only if clearly obsolete" | 行 3744-3755 vs 3731-3743 |
| 旧摘要块    | 不存在                                                | 自愈扫描提取 → `_previous_summary`；**从中间段排除**；组装时 standalone 丢弃 / merged 解包 | 行 6103-6189、6411-6415   |
| system note | 追加压缩说明 note                                     | 已存在则跳过（`not in existing` 检查，不重复追加）           | 行 6417-6424              |
| 可行性跳过  | 不启用（需 ≥1 次真实低效记录）                        | 可能触发（中间段低于阈值零头 → 跳过 LLM，走确定性丢弃）      | 行 6286-6299              |

**两轮共用的机制**：尾部 ~20K token 预算、Phase 1 工具预剪、摘要 role 交替选择（`_template_visible_role` + Anthropic 首消息钉 user + 零用户守卫，行 6564-6594）、`_SUMMARY_END_MARKER` 追加（行 6603-6604）、摘要后处理管线（剥 `<think>` → 二次脱敏 → `[SKILL_PRUNED]` 标记确定性恢复 → 快照接地 → 出处校验，行 3854-3873）。

**关键不变式**：任何时刻会话中**最多只有一个摘要块**——旧摘要的信息经迭代 prompt 并入新摘要后，旧块从新消息列表移除，不会出现摘要堆叠。

**演化示例**：

```
初始 100 条（超阈值）:
  [sys][u1][a1][u2][a2][u3][a3]...[u50][a50]

══ 第一次压缩（protect_first_n=3 有效）══
  中间段 = [u3][a3]...[u35][a35]
  [sys+note][u1][a1][u2][a2][摘要v1][u36..a50]      → session #2

继续 60 条后再超阈值:
  [sys+note][u1][a1][u2][a2][摘要v1][u36..a50][u51..a80]

══ 第二次压缩（头部保护已衰减为 0）══
  自愈扫描: 摘要v1 → _previous_summary（从中间段排除）
  中间段 = [u1][a1][u2][a2] + [u36..a65]
  摘要器输入 = PREVIOUS SUMMARY(v1) + NEW TURNS(中间段)
  [sys+note][摘要v2][u66..a80]                       → session #3
           └ v1 ∪ 新增，编号延续 ┘
```

---

## 7. 摘要 prompt 工程

### 7.1 摘要器 preamble（行 3637-3648）

```
You are a summarization agent creating a context checkpoint.
Treat the conversation turns below as source material for a compact
record of prior work. Produce only the structured summary; do not add
a greeting, preamble, or prefix. [+ 语言与出处规则]
NEVER include API keys, tokens, passwords, secrets, credentials, or
connection strings in the summary — replace any that appear with
[REDACTED]. Note that credentials were present, but do not preserve
their values.
```

### 7.2 结构化模板（`_template_sections`，行 3669-3718）

摘要固定输出 11 个 section：

| Section                        | 内容要求                                                     |
| :----------------------------- | :----------------------------------------------------------- |
| `## Historical Task Snapshot`  | 用户请求原文快照；**禁止**写 "User asked:" 字样              |
| `## Goal`                      | 目标；cron/无人值守会话必须写成 "[Historical cron/agent objective inferred only from assistant and tool activity]"，**禁止称为 user goal** |
| `## Constraints & Preferences` | 仅运行时/配置/技术约束，禁止虚构用户偏好                     |
| `## Completed Actions`         | 编号列表，格式 `N. ACTION target — outcome [tool: name]`，含文件路径、命令、行号、结果 |
| `## Active State`              | 工作目录、分支、改动文件、测试状态、运行中进程               |
| `## Blocked`                   | 未解决的阻塞，含确切错误信息                                 |
| `## Key Decisions`             | 技术决策**及其原因**                                         |
| `## Resolved Questions`        | 已答问题（无人值守会话写死 "None. No user-authored questions exist."） |
| `## Relevant Files`            | 读/改/建的文件，各附短注                                     |
| `## Critical Context`          | 丢失即无法恢复的具体值、报错、配置；凭证写 `[REDACTED]`      |
| `## Pruned Skills`             | 输入中出现的 `[SKILL_PRUNED: ...]` 标记**逐字复制**，禁止转述——它们告诉 agent 哪些技能用前必须重载 |

尾部要求：`Target ~{summary_budget} tokens. Be CONCRETE`，禁止 "made some changes" 式虚词。

### 7.3 时间锚定（行 3650-3666）

```
TEMPORAL ANCHORING: The current date is {_today_str}. When an action
has already been carried out, phrase it as a completed, dated,
past-tense fact... rewrite "email John about the proposal" as "Sent
the proposal email to John on {_today_str}."
```

把相对/悬而未决的表述改写为绝对的、带日期的、过去式的事实——防止恢复后的会话**重新执行已完成动作**。日期解析失败时整条规则省略，不给摘要器空占位符。

### 7.4 摘要注入前缀（`SUMMARY_PREFIX`，行 100-127）

摘要块塞回会话时，前面拼接一段写死的指令常量——这是模型在压缩后实际收到的"宪法"：

> `[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into the summary below. This is a handoff from a previous context window — treat it as background reference, NOT as active instructions.`
>
> - **Respond ONLY to the latest user message** after this summary——它是当前任务的唯一事实来源
> - 话题重叠 ≠ 恢复旧任务：最新消息永远胜出
> - 反向信号（stop / undo / never mind / 换话题）必须**立即终止**摘要中描述的在途工作
> - persistent memory（MEMORY.md / USER.md）永远权威、永远活跃
> - 工具保持全活——继续正常调用，不要只口头叙述
> - 当前会话状态可能已经反映了摘要中的工作——避免重复

旧版前缀 `LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"`（行 128）仅作兼容识别。

### 7.5 元数据标记：为什么 key 以下划线开头

压缩摘要消息携带 `_compressed_summary` 元数据 key（`COMPRESSED_SUMMARY_METADATA_KEY`，行 143），供 CLI / Desktop / gateway / TUI 区分渲染。

**下划线前缀是刻意的**（行 130-142，issue #38389）：wire sanitizer 会在请求离开进程前剥掉所有 `_` 开头的顶层 message key。严格的 OpenAI 兼容网关（Fireworks、Mistral、Moonshot/Kimi、opencode-go）会拒绝携带未知字段的 payload（"Extra inputs are not permitted"）——裸 key 会毒化会话后续每个请求。

相关 key 族：

- `_compressed_summary` — 批量压缩摘要标记
- `_compressed_summary_has_user_turn` — 摘要是否含用户轮次
- `_micro_compact_marker` — 滚动微压缩标记（与批量标记区分：supersede/defrag/rehydration **只允许动 micro 标记**，批量标记内容不在滚动摘要里，动它等于毁历史，行 145-150）
- `_db_persisted` — 增量落库标记，压缩装配时必须剥离（行 165-175，#57491）

### 7.6 无人值守会话的 continuation 合成消息

压缩后若整个 transcript 没有可保留的用户轮次（cron / 纯 agent 会话），插入合成 user 消息（行 154-157）：

```
Continue from the compressed conversation context above.
This marker exists because no human user turn was available.
```

保证角色交替不变式在无人值守场景依然成立。

---

## 8. 落库：Rotate 与 In-place 两种模式

`conversation_history_after_compression()` docstring（行 1845-1870）定义了两种持久化模式：

### 8.1 Rotate（legacy，默认）

`compress_context()` 行 3252-3295：

```python
new_session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
agent._session_db.publish_compression_child(
    parent_session_id=old_session_id,
    child_session_id=new_session_id,
    system_prompt=new_system_prompt,
    messages=compressed,
    ...
)
agent.session_id = new_session_id   # 轮换
```

- **原子提交**：`publish_compression_child` 在单事务内创建子会话并写入压缩后消息
- **谱系**：子会话 `parent_session_id` 指向父会话（外键自引用，schema 行 204/242）
- **标题编号**：`my project` → `my project #2` → `#3`；按名恢复自动取谱系最新
- **goal 迁移**：持久 `/goal` 随压缩迁到子会话（行 3291-3293，#33618）
- **环境同步**：`HERMES_SESSION_ID`、日志 context 同步切换（行 3273-3284）
- 旧会话**完整保留**——官方原话："Compression reduces the active context; it is not a privacy delete."

### 8.2 In-place

`archive_and_compact()`：旧活跃行软归档（`active=0`），压缩后消息作为新活跃行插入，**session id 不变**。

**flush 基线陷阱**（行 1856-1864）：in-place 后若同一 turn 继续以 `conversation_history=None` 刷新，基于身份的增量 flush 会把已持久化的压缩消息**重复 append**——活跃 context 翻倍并重新触发压缩。因此 in-place 模式用浅拷贝捕获压缩后 dict 身份作为历史基线。

### 8.3 恢复时的行为

- Resume 会话时加载完整消息历史（`hermes --resume <id>` / `-c` / 按标题）
- CLI 显示 compact recap 面板：最近 10 轮、user 300 字符截断、assistant 200 字符/3 行、工具调用折叠为计数（`sessions.md` 行 187-207）
- Resume 同时 `cd` 回会话记录的工作目录（`--no-restore-cwd` 可禁）

---

## 9. SQLite 持久层（schema v24）

`hermes_state_common.py`，三张核心表 + 一张锁表：

### `sessions`（行 190-243）

| 列                                                           | 用途                                       |
| :----------------------------------------------------------- | :----------------------------------------- |
| `id` / `source` / `user_id` / `session_key` / `chat_id` / `chat_type` / `thread_id` | 身份与平台路由                             |
| `model` / `model_config` / `system_prompt`                   | **system prompt 快照**——恢复时还原当时配置 |
| `parent_session_id`                                          | 压缩谱系（外键自引用）                     |
| `input/output/cache_read/cache_write/reasoning_tokens`       | 五维 token 计数                            |
| `compression_failure_cooldown_until` / `compression_fallback_streak` / `compression_ineffective_count` | **持久化压缩防护状态**                     |
| `title`（唯一索引）/ `last_activity_*` / `archived` / `pinned` | 管理与展示                                 |
| `estimated_cost_usd` / `actual_cost_usd` / `billing_*`       | 成本核算                                   |

### `messages`（行 245-269）

| 列                                                           | 用途                                |
| :----------------------------------------------------------- | :---------------------------------- |
| `role` / `content` / `tool_calls` / `tool_name` / `tool_call_id` | OpenAI 消息格式                     |
| `token_count` / `timestamp` / `finish_reason`                | 计量                                |
| **`active`**                                                 | in-place 压缩的软归档标记（默认 1） |
| **`compacted`**                                              | 该消息是否已被压缩覆盖              |
| `api_content`                                                | 发给 API 的内容（与展示内容可不同） |
| `reasoning` / `reasoning_content` / `reasoning_details`      | 推理模型思维链                      |
| `platform_message_id` / `observed`                           | 平台回执与已读                      |

### `messages_fts` + `messages_fts_trigram`

FTS5 虚表（外部内容模式），`session_search` 工具直接查询，**零 LLM 调用**。trigram 索引排除 tool 行（通过 `messages_fts_trigram_src` 视图）。

### `compression_locks`（行 306-311）

```sql
session_id TEXT PRIMARY KEY,
holder TEXT NOT NULL,
acquired_at REAL NOT NULL,
expires_at REAL NOT NULL
```

durable per-session 压缩租约：同一会话绝不并发压缩；不同会话可在池上并发（引擎实例若跨会话共享须自行保证线程安全）。

---

## 10. 并发、超时与失败防护

### 10.1 锁竞争

拿不到压缩锁时，本轮记 `compression_deferred`（`conversation_loop.py` 行 880-924）——**不**计为压缩耗尽，提示用户"另一路径正在压缩本会话"。锁持有者可被识别，防止陈旧 contender 在谱系歧义时误操作（`recover_rotated_compression_session`，行 1291）。

### 10.2 超时栅栏（CommitFence）

整个压缩 pass——包括插件 context engine（`compress()` / 边界回调）和 memory provider（`on_pre_compress` / `on_session_switch`）——跑在**池化 daemon 线程**（`conversation_compression.py` 文件头 docstring 行 28-49）：

- 输入消息列表是 worker 私有的深快照，引擎可原地修改，**不 commit 就对活会话不可见**
- 状态发布只发生在被采纳的 commit（`CompressionCommitFence`）；host 超时后仍在运行的引擎工作**直接丢弃**
- 终态 provenance（`AGENT_COMPRESSION_TIMEOUT` / `AGENT_COMPRESSION_COOLDOWN`）一旦盖章，detached heartbeat worker 不得回写（行 79-89）

### 10.3 摘要失败的降级路径

- 瞬时错误（429/网络）：设冷却 → 返回原消息 → 插入静态 fallback 标记 → 防重触发
- 永久错误（401/402/403/quota 耗尽，`_is_summary_access_or_quota_error` 行 73-94）：不重试
- `_last_summary_auth_failure` / `_last_summary_network_failure` **不在 compress() 入口重置**（行 5988-5997，#29559）——否则冷却早退会被误判为"可以走降级"，造成数据丢失
- `abort_on_summary_failure=True` 时中止而非降级

### 10.4 UI 状态标记

```
COMPACTION_STATUS = "🗜️ Compacting context — summarizing earlier conversation..."
COMPACTION_DONE_STATUS = "✓ Context compaction complete — continuing turn..."
```

（行 96-101）gateway 匹配 `COMPACTION_STATUS_MARKER` 把状态标记为 `kind="compacting"`，桌面端据此显示 "Summarizing…" 指示器，避免用户看到 transcript"静默重置"。人-facing 平台（Telegram 等）会按噪音正则抑制常规压缩状态行。

---

## 11. 辅助机制

### 11.1 Proactive tool-result pruning

`proactive_prune_tokens`（默认 0 = 关闭）。独立于 `should_compress` 的低成本触发（`prune_tool_results_only()`，`context_engine.py` 行 194-211）：远在完整压缩触发之前，回收反复重发的大型工具输出（默认 ≥ 8000 字符、可回收 ≥ 4096 token 才动手）。定位是**省钱**，不是防爆窗。

### 11.2 Micro-compaction（滚动微压缩）

`compression.micro_compact: true` 显式开启（默认关，行 2300-2302）。

- 维护 `_micro_compact_rolling_summary` 滚动摘要，逐个 exchange 合并
- 每次合并一次 aux LLM 调用：`max_tokens=min(1500, max_summary_tokens)`，`temperature=0.1`（行 5388-5392）
- prompt 要求：保留结构、丢弃已解决的细节、追加新决策/文件路径/开放问题、凭证 `[REDACTED]`（行 5343-5359）
- 连续失败 3 次熔断（`_MICRO_COMPACT_MAX_CONSECUTIVE_FAILURES`，行 381）
- 与批量压缩的边界：见 7.5 节 `_micro_compact_marker` 规则

### 11.3 `select_context()`：正交的每轮选材

引擎可选实现。每轮消息组装后、发送前调用，做检索式选材 / 话题路由 / 角色分支切换——**不缩小** context，而是**替换**本轮用的 context。避免引擎为了拿到每轮回调而谎报 `should_compress()=True`（行 236-239）。

### 11.4 压缩前的 memory 保全

`memory_context` 参数（行 5975-5976）：memory provider 在压缩前返回的保全文本，非空即并入摘要 prompt。配合 gateway 会话自动重置策略——**重置前 agent 获得一个 turn 把重要内容存入 memory/skill**（`sessions.md` 行 683）。

---

## 12. 用户控制面

### 12.1 会话内 slash 命令

| 命令                         | 作用                                                         | 对应机制                              |
| :--------------------------- | :----------------------------------------------------------- | :------------------------------------ |
| `/compress`（`/compact`）    | 手动压缩；`here [N]` 保留 N 轮；`--preview` 预览；`<focus>` 引导式压缩（优先保留某话题，灵感来自 Claude Code `/compact`） | `force=True` 清冷却、跳过可行性跳过   |
| `/new [name]`（`/reset`）    | 全新会话（可命名）                                           | `on_session_reset()` 重置全部压缩状态 |
| `/undo [N]`                  | 回退 N 个用户轮次重新提问                                    | rewind（`rewind_count` 列计数）       |
| `/branch`（`/fork`）[name]   | 从当前点分叉会话                                             | 谱系分支                              |
| `/status`                    | 查看 session / model / token / **context 用量**              | —                                     |
| `/resume [name]` `/sessions` | 恢复 / 浏览历史会话                                          | 谱系感知                              |
| `/title [name]`              | 命名会话                                                     | 压缩时自动编号的基础                  |

### 12.2 配置项（config.yaml）

| 配置                                  | 默认   | 作用                                                         |
| :------------------------------------ | :----- | :----------------------------------------------------------- |
| `compression.threshold_percent`       | 0.50   | 触发阈值比例                                                 |
| `compression.threshold_tokens`        | 空     | 绝对 token 上限（与比例取小）                                |
| `compression.model_thresholds`        | 空     | 按模型名子串覆盖比例                                         |
| `compression.micro_compact`           | false  | 滚动微压缩开关                                               |
| `compression.context_timeout_seconds` | >0     | 压缩 pass 的 host 级超时栅栏                                 |
| `session_reset.mode`                  | `none` | gateway 会话自动重置策略（none/idle/daily/both）             |
| `group_sessions_per_user`             | true   | 群聊 per-user 隔离（他人长任务不污染你的窗口）               |
| `sessions.auto_prune`                 | false  | 已结束会话的自动清理（**默认关**，历史是 session_search 的燃料） |

---

## 13. 与其他记忆系统的边界

| 系统                    | 层次                           | 关系                                                         |
| :---------------------- | :----------------------------- | :----------------------------------------------------------- |
| **会话窗口**（本文）    | 单会话内，易失                 | 被压缩管理                                                   |
| **state.db 持久历史**   | 单/跨会话，持久                | 压缩**不删除**；`session_search` 的检索源                    |
| **MEMORY.md / USER.md** | 跨会话，每轮注入 system prompt | 压缩前缀中声明"永远权威、永远活跃"；gateway 重置前的保全出口 |
| **Skills**              | 跨会话，按需加载               | 压缩时正文可剪枝为 `[SKILL_PRUNED]` 标记，摘要有义务逐字传递标记 |
| **session_search**      | 跨会话检索工具                 | FTS5 直查 state.db，bookends + 窗口还原"目标→命中→结论"      |

---

## 14. 全流程总图

```
用户消息
   │
   ▼
┌─ 消息组装 ──────────────────────────────────────────┐
│ system prompt（不变）+ 会话窗口 + 当轮注入（附件/@）  │
│ select_context()?  → 本轮选材替换（可选引擎）         │
└────────────────────────────────────────────────────┘
   │ 发送给模型
   ▼
模型回复 + 可能的工具调用循环
   │
   ▼ 轮次结束
┌─ should_compress()? ───────────────────────────────┐
│ tokens < threshold（默认 50% 窗口）→ 否，直接落库    │
│ cooldown / anti-thrash 闸 → 阻塞并给出 reason        │
└────────────────────────────────────────────────────┘
   │ 是
   ▼
┌─ compress() 五步 ──────────────────────────────────┐
│ 1. 预剪旧工具输出（无 LLM）：一行摘要/去重/截断       │
│ 2. 保头：system prompt + 前 N 条（N 逐次压缩衰减）    │
│ 3. 保尾：按 ~20K token 预算划界，边界避开工具对       │
│ 4. LLM 摘要中间段（11-section 模板 + 时间锚定）       │
│ 5. 迭代：旧摘要作为输入增量更新                       │
└────────────────────────────────────────────────────┘
   │
   ▼
┌─ 落库（单事务）─────────────────────────────────────┐
│ Rotate: publish_compression_child → 新 session id    │
│ In-place: 旧行 active=0，压缩行插入，id 不变          │
│ 谱系：parent_session_id 链接；标题 #N 编号           │
│ goal 迁移；锁/冷却/失败计数持久化                    │
└────────────────────────────────────────────────────┘
   │
   ▼
继续对话：模型看到 [SUMMARY_PREFIX 指令 + 摘要 + 保护尾]
   │
   ▼ 会话结束
state.db 永久保存（auto_prune 默认关）→ session_search 可检索
```

