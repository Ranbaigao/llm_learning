# Dsh 会话记忆机制：the log IS the state

> **受众：** 关注 Agent 运行时、会话状态管理与上下文压缩实现的开发者。
> **最后更新：** 2026-08-23

## 0. 引言

DeepSeek Harness（dsh）的会话记忆建立在一个非常干净的原则上：**append-only Event Log**。

dsh 使用 `SessionEvent[]` 记录会话，事件只允许 append，不允许修改历史事件。模型真正看到的消息通过 `deriveMessages(events)` 投影得到。这一原则可以总结为一句话：

```text
the log IS the state
```

而不是“日志记录状态”。

当然，并非所有 Event 都进入模型。真正影响模型历史的核心 Surface Event 主要包括：

```text
user/message
assistant/message
tool/result
```

而：

```text
turn/start
turn/end
step/start
step/end
assistant/chunk
request/header
compaction/*
```

可以保留在 Event Log 中，但不直接成为 Message History。这就实现了：

```text
         Session Event Log
               │
      ┌────────┴────────┐
      ↓                 ↓
Model Projection    Observability
      ↓                 ↓
Messages          Trajectory
```

模型可见事件与结构/回放事件、log-only 事件被分层存储，从而同时获得完整可观测性和稳定模型上下文。

## 问题的提出

这套设计乍看很优雅，但稍加推敲就会冒出一个尖锐的矛盾：

> **如果对话超出模型上下文、触发了上下文压缩（compaction），模型看到的历史就被改变了；但事件又只允许 append、不允许修改——旧事件不能动，新上下文又对不上旧日志，那“the log IS the state”还成立吗？Log 和 Context 岂不就出现了不一致？**

答案是：如果 dsh 真的只是“所有事件 append，然后 `deriveMessages()` 简单把所有 `user/message` / `assistant/message` 拼起来”，这个矛盾确实会成立。

但 dsh 实际上多设计了一层叫 **Session Surface** 的结构。真正的结构不是：

```text
Append-only Event Log
        ↓
把所有消息直接拼起来
        ↓
LLM Context
```

而是：

```text
Append-only Event Log
        ↓
根据每个事件的 surfaceOp
        ↓
构造 Current Surface
        ↓
deriveMessages(surface)
        ↓
LLM Context
```

所以这里真正需要区分三个概念：

```text
Raw Event Log      永远 append-only
Surface            当前“有效历史”的投影
LLM Context         从 Surface 派生出的模型输入
```

**Compaction 不修改旧 Event Log；它 append 一个新的“替换事件”，告诉 Surface：旧的这一段以后不再直接出现在模型历史中，用这个摘要节点替代。**

这正是 dsh 为解决这个矛盾专门设计的机制。官方 Session Surface 设计文档甚至明确说，过去如果由 compaction 插件直接在 request hook 里改写 history，会导致改写行为没有被持久记录；因此才加入了一个“由 Event Log 中的 `surfaceOp` 驱动的有序投影 Surface”。([GitHub][1])

---

# 1. 先纠正一句容易产生误解的话

前面的表述：

> `the log IS the state`

容易让人理解成：

```text
Event Log 中的每一条消息
=
当前送给模型的消息
```

这其实不准确。

更准确的说法应该是：

> **Event Log 是唯一权威状态源（source of truth）；当前模型历史是 Event Log 的确定性投影。**

即：

```text
State = reduce(EventLog)
```

而不是：

```text
State = EventLog 最后长什么样
```

这其实就是 Event Sourcing 的典型思想。

例如银行账户也可以是 append-only：

```text
Event 1: +100
Event 2: -30
Event 3: +50
```

你不能修改：

```text
Event 1
```

但当前余额依然会变化：

```text
100
↓
70
↓
120
```

因为：

```text
Balance = reduce(events)
```

同理，dsh 是：

```text
Surface = reduce(SessionEvents)
```

然后：

```text
Messages = deriveMessages(Surface)
```

所以：

> **append-only 不代表“当前状态不能变化”；它代表“状态变化只能通过追加新的事实来表达，而不能篡改历史事实”。**

---

# 2. dsh 实际有一个 Surface 层

官方现在明确维护了一个：

```text
Session Surface
```

官方定义它是：

> “an ordered projection over the event log”

也就是：

> **Event Log 上的一个有序投影视图。**

官方源码设计里，`Session` 内部有一个 `SurfaceManager`，它维护一个：

```ts
number[]
```

里面存的不是消息本身，而是：

```text
当前有效 Surface Event 的 seq
```

例如原始 Event Log：

```text
seq=1  user/message
seq=2  assistant/message
seq=3  user/message
seq=4  assistant/message
```

Surface：

```text
[1, 2, 3, 4]
```

于是：

```text
deriveMessages()
```

按照 Surface 去取 Event：

```text
seq1 → user
seq2 → assistant
seq3 → user
seq4 → assistant
```

得到模型历史。官方文档明确写道，`deriveMessages()` 在存在 Surface 标记时会以 Surface 作为派生路径，而不是直接扫描所有历史消息。([GitHub][1])

所以结构其实是：

```text
                 SessionEvent[]
                      │
                      │ authoritative
                      ↓
               SurfaceManager
                      │
                [1,2,3,4]
                      │
                      ↓
              deriveMessages()
                      │
                      ↓
                 LLM Context
```

---

# 3. 那 Compaction 怎么做？

关键来了。

假设现在历史是：

```text
seq 1  user:      帮我分析项目
seq 2  assistant: 好，我先分析...
seq 3  tool:      ...
seq 4  assistant: ...
seq 5  user:      再看一下数据库
seq 6  assistant: ...
seq 7  tool:      ...
seq 8  assistant: ...
seq 9  user:      再继续...
seq10  assistant: ...
```

Surface：

```text
[1,2,3,4,5,6,7,8,9,10]
```

模型看到：

```text
U1
A1
Tool1
A2
U2
A3
Tool2
A4
U3
A5
```

随着任务越来越长：

```text
Context Token > Threshold
```

触发 Compaction。

比如决定把：

```text
seq 1 ~ seq 6
```

压缩成：

> 用户要求分析项目，Agent 已检查核心代码与数据库结构，发现……

---

# 4. 最重要的地方：它不删除 seq 1~6

dsh 不会：

```text
DELETE seq1
DELETE seq2
...
DELETE seq6
```

也不会：

```text
UPDATE seq1...
```

而是继续 append。

官方 Compaction 流程大致是：

```text
seq11 compaction/start
seq12 compaction/summary
seq13 user/message: "摘要..."
seq14 compaction/end
```

也就是说原始 Event Log 变成：

```text
1   old user
2   old assistant
3   old tool result
4   old assistant
5   old user
6   old assistant
7   ...
8   ...
9   ...
10  ...

11  compaction/start
12  compaction/summary
13  summary checkpoint
14  compaction/end
```

**所有原始事件仍然存在。**

这点官方写得非常明确：

> 被 shadow 的事件仍然保留在 log 中，只是不再处于当前 Surface 上。([GitHub][1])

---

# 5. 真正神奇的是 seq13 上有一个 `surfaceOp`

seq13 不是普通：

```text
user/message
```

而是类似：

```ts
{
  seq: 13,

  type: "user/message",

  data: {
    content: "此前用户要求分析项目……"
  },

  surfaceOp: {
    op: "replace",
    start: 1,
    end: 6
  },

  sourceEventSeqs: [
    1, 2, 3, 4, 5, 6
  ]
}
```

这里：

```text
surfaceOp:
{
    op: "replace",
    start: 1,
    end: 6
}
```

的意思不是：

> 修改 Event 1～6。

而是：

> **对当前 Surface 进行投影时，把 Surface 中 seq1～seq6 这一段 shadow 掉，然后把 seq13 插在它原来的位置。**

官方 `SurfaceOp` 就定义了两种操作：

```ts
type SurfaceOp =
  | 'append'
  | {
      op: 'replace'
      start: number
      end: number
    }
```

其中 `replace` 会把 Surface 中 `start` 到 `end` 的节点移除，并把新的 event seq 放进去；旧事件仍留在 raw log。([GitHub][1])

---

# 6. 所以 Compaction 前后其实是这样

### Compaction 前

Raw Log：

```text
1  U1
2  A1
3  T1
4  A2
5  U2
6  A3
7  T2
8  A4
9  U3
10 A5
```

Surface：

```text
[1,2,3,4,5,6,7,8,9,10]
```

模型 Context：

```text
U1
A1
T1
A2
U2
A3
T2
A4
U3
A5
```

---

### Compaction 发生

新增：

```text
11 compaction/start
12 compaction/summary

13 user/message
   content = SUMMARY

   surfaceOp = replace(1,6)

14 compaction/end
```

注意：

```text
1～10
```

一个都没改。

---

### Compaction 后 Raw Log

```text
[
  1  U1,
  2  A1,
  3  T1,
  4  A2,
  5  U2,
  6  A3,
  7  T2,
  8  A4,
  9  U3,
  10 A5,

  11 compaction/start,
  12 compaction/summary,
  13 SUMMARY replace(1,6),
  14 compaction/end
]
```

仍然 append-only。

但是 SurfaceManager 看到 seq13：

```text
replace(1,6)
```

所以原 Surface：

```text
[1,2,3,4,5,6,7,8,9,10]
```

变成：

```text
[13,7,8,9,10]
```

官方实现说明的就是：`replace` 根据 Surface 中 start/end 的位置，将这一段 splice 掉，再放入 replacement seq。([GitHub][1])

于是：

```text
deriveMessages()
```

现在不是读取：

```text
1,2,3,4,5,6,7,8,9,10
```

而是读取：

```text
13,7,8,9,10
```

得到：

```text
[此前历史摘要]

T2
A4
U3
A5
```

这才送给模型。

---

# 7. “上下文改变了”，确实改变了

但改变的是：

```text
Derived State / Surface
```

而不是：

```text
Event Log
```

可以把它画成：

```text
                    Event Log
                       │
                       │ append only
                       ↓
 ┌──────────────────────────────────────┐
 │ 1 U1                                 │
 │ 2 A1                                 │
 │ 3 T1                                 │
 │ 4 A2                                 │
 │ 5 U2                                 │
 │ 6 A3                                 │
 │ 7 T2                                 │
 │ 8 A4                                 │
 │ 9 U3                                 │
 │10 A5                                 │
 │11 compaction/start                   │
 │12 compaction/summary                 │
 │13 SUMMARY + replace(1,6)             │
 │14 compaction/end                     │
 └──────────────────────────────────────┘
                       │
                       │ fold surfaceOp
                       ↓
               Current Surface
                       │
                [13,7,8,9,10]
                       │
                       ↓
               deriveMessages()
                       │
                       ↓
 ┌──────────────────────────────────────┐
 │ Summary of seq 1-6                   │
 │ T2                                   │
 │ A4                                   │
 │ U3                                   │
 │ A5                                   │
 └──────────────────────────────────────┘
                       │
                       ↓
                      LLM
```

所以不存在：

```text
Log 表示旧上下文
Context 是新上下文
二者不一致
```

因为 Log 本身也记录了：

```text
“从现在开始，
在模型 Surface 中，
1~6 被 13 替代。”
```

---

# 8. 这反而是 `the log IS the state` 最漂亮的地方

假如 dsh 不是这么做，而是：

```text
原 Event Log 保留：

1 U1
2 A1
3 A2
...
```

然后某个内存变量直接变成：

```text
messages = [
   SUMMARY,
   recent messages...
]
```

那么这个矛盾就真的出现了。

此时：

```text
Event Log
≠
Actual Model Context
```

因为“压缩”这个动作只发生在内存里，没有记录。

机器一重启：

```text
读 Event Log
 ↓
不知道之前压缩过
 ↓
重新得到全部历史
```

或者 Replay：

```text
昨天模型看到 Summary
今天 replay 却看到原始历史
```

这就是不一致。

官方 Session Surface 设计说明恰恰说，他们引入 Surface 的原因就是：如果历史改写只是通过 request listener 临时完成，那么“改了什么”没有 durable record，replay 也无法可靠重建。([GitHub][1])

所以 dsh 的做法是：

```text
不要修改 State
不要偷偷修改 Projection

而是：

append 一个
“Projection 应如何改变”
的 Event
```

这就是 Event Sourcing 的核心精神。

---

# 9. `sourceEventSeqs` 又是干什么的？

你可能又会问：

```ts
sourceEventSeqs: [1,2,3,4,5,6]
```

不是已经有：

```ts
replace(1,6)
```

了吗？

为什么还需要它？

因为 dsh 想保证：

> 这个 Summary 到底是从哪些 Event 生成的？

例如：

```text
seq13 SUMMARY
```

声称：

```text
replace seq1 ~ seq6
```

那么必须证明它引用了：

```text
1
2
3
4
5
6
```

所以：

```ts
sourceEventSeqs
```

就是 lineage / provenance：

```text
SUMMARY seq13
       │
       ├── source: seq1
       ├── source: seq2
       ├── source: seq3
       ├── source: seq4
       ├── source: seq5
       └── source: seq6
```

官方 Session 在 append 边界还会校验：

* source seq 必须已经存在；
* 不能引用未来事件；
* 引用不能重复；
* replace 的 start/end 必须存在于当前 Surface；
* `sourceEventSeqs` 必须覆盖所有被 shadow 的 Surface node。([GitHub][1])

所以不是随便写：

```text
“我总结过了”
```

而是有完整溯源。

---

# 10. `compaction/summary` 和那个 `user/message Summary` 又有什么区别？

这个地方也容易混。

官方流程实际是：

```text
compaction/start
        ↓
调用模型生成 summary
        ↓
compaction/summary
        ↓
user/message + surfaceOp replace
        ↓
compaction/end
```

其中：

### `compaction/summary`

是：

```text
log-only
```

记录：

* 原始 summary；
* 使用哪个 provider；
* 哪个 model；
* token usage；
* shadow 哪个范围；
* 哪些 seq 被 shadow。

主要为了：

```text
Audit
Replay
Telemetry
Debug
```

但它**不进入模型上下文**。([GitHub][2])

---

### `user/message + surfaceOp replace`

才是真正：

```text
Surface Mutation
```

也就是说：

```text
Compaction Metadata
        ↓
compaction/summary

真正给模型看的摘要
        ↓
user/message
```

官方文档甚至特意解释：

> `compaction/*` 不允许成为 Surface Event；真正的摘要由一个 `user/message` 承载。([GitHub][3])

最终：

```text
deriveMessages()

→ [
    summary_as_user_message,
    ...retained_entries
  ]
```

---

# 11. 重启以后会不会丢失 Surface 状态？

不会，因为：

```text
Surface
```

不是唯一真源。

真正真源仍然是：

```text
Event Log
```

重启后可以重新 replay：

```text
seq1  append
Surface = [1]

seq2  append
Surface = [1,2]

seq3  append
Surface = [1,2,3]

...

seq13 replace(1,6)
Surface = [13,7,8,9,10]
```

最终一定恢复到：

```text
[13,7,8,9,10]
```

官方明确写：

> 被 shadow 的事件仍留在原始 log，因此 replay 是 deterministic 的。([GitHub][4])

所以：

```text
Event Log
      ↓ replay
Surface
      ↓
Messages
```

是一个确定性函数。

可以近似理解成：

```python
def replay(events):

    surface = []

    for event in events:

        if event.surfaceOp == "append":
            surface.append(event.seq)

        elif event.surfaceOp.op == "replace":

            start = surface.index(
                event.surfaceOp.start
            )

            end = surface.index(
                event.surfaceOp.end
            )

            surface[start:end+1] = [
                event.seq
            ]

    return surface
```

于是：

```text
同一 Event Log
        ↓
永远得到
        ↓
同一个 Surface
```

这才是 dsh 所谓可重放。

---

# 12. 你可以把它类比成 Git

这个类比其实非常好理解。

Event Log 类似：

```text
Git Commit History
```

你不会为了得到新的代码状态就去修改旧 commit。

比如：

```text
Commit A
Commit B
Commit C
Commit D
```

后来：

```text
Commit E:
“把前面那套实现替换掉”
```

旧 Commit 仍然存在。

但：

```text
git checkout HEAD
```

得到的是新的工作树状态。

所以：

```text
Git Commit History
        ↓
replay commits
        ↓
Current Working Tree
```

对应：

```text
Session Event Log
        ↓
replay surfaceOp
        ↓
Current Surface
        ↓
deriveMessages
        ↓
Current LLM Context
```

这里：

```text
Raw Log ≈ Commit History

Surface ≈ 当前逻辑历史视图

Messages ≈ checkout 后的工作树
```

这个理解会非常接近 dsh 的真实设计。

---

# 13. 更准确的表述

之前：

```text
SessionEvent[]
事件只允许 append
不允许修改历史事件。

模型真正看到的消息通过
deriveMessages(events)
投影得到。
```

这个表述**太简化了**，正因为少了 `Session Surface`，才会产生引言中提出的那个逻辑矛盾。

更准确应该写成：

```text
SessionEvent[]          ← append-only authoritative log
       ↓
SurfaceOp
       ↓
Session Surface         ← 当前有效消息节点序列
       ↓
deriveMessages()
       ↓
LLM Message History
```

其中普通消息：

```text
surfaceOp = append
```

上下文压缩则追加：

```text
user/message
+
surfaceOp = {
    op: replace,
    start,
    end
}
```

被替换的旧 Event：

```text
仍保留在 Event Log
```

但：

```text
不再存在于 Current Surface
```

因此：

```text
Raw history preserved
+
Current context compacted
+
Compaction decision persisted
```

三件事情可以同时成立。

---

## 总结

回到最初的问题：

> **“上下文压缩后上下文变了，但 Event 又不能修改，怎么还能说 the log IS the state？”**

答案是：

> **dsh 不把“压缩后的上下文”存成一份独立可变状态，而是把“用摘要替换旧 Surface 区间”这件事本身作为新的 Event append 进去。旧 Event 不改，新的 replace Event 改变的是对整个 Event Log 的投影结果。**

也就是：

```text
不是：

修改历史
      ↓
得到新状态
```

而是：

```text
历史1
历史2
历史3
...
“将历史1~3替换为摘要S” ← 新事件
              ↓
         replay/fold
              ↓
       当前状态 = S
```

因此更严格的公式是：

```text
Authoritative State = Event Log

Current Surface
    = fold(Event Log, surfaceOp)

LLM Context
    = deriveMessages(Current Surface)
```

而不是简单的：

```text
LLM Context = Event Log
```

这三个公式一分开，DeepSeek Harness 这套设计基本就完全说通了。([GitHub][1])

## 参考文献

1. [Session Surface 设计文档（deepseek-harness）][1]
2. [Compaction 子系统文档（deepseek-harness）][2]
3. [Compaction capability seam 设计笔记（deepseek-harness）][3]
4. [compaction 包 README（deepseek-harness）][4]

[1]: https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/implemented/architecture/2026-06-18-session-surface.md "Session Surface 设计文档"
[2]: https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/compaction.md "Compaction 子系统文档"
[3]: https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/implemented/feature/2026-06-18-compaction-capability-seam.md "Compaction capability seam 设计笔记"
[4]: https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/compaction/compaction/README.md "compaction 包 README"
