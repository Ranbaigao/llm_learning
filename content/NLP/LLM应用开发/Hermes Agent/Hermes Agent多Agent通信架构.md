# 多 Agent 通信架构

> **受众：** Hermes Agent 使用者与贡献者、需要设计 Multi-Agent 系统的开发者。
> **最后更新：** 2026-08-22

## 概述

多 Agent 之间的通信，本质上不是"Agent 彼此直接聊天"这么简单，而是：

> **Agent A 产生结构化状态/任务/结果 → 通过某种通信介质交给 Agent B → Agent B 更新自己的上下文和状态。**

工程上有五种主流模式——父子调用、共享状态、消息队列、RPC、发布订阅——真正的生产系统往往混用。但比"选哪种模式"更重要的是先想清楚**协议**：Agent 之间传什么、谁能跟谁通信、什么时候终止、失败怎么处理。传输层（Kafka 还是 Redis）是最后才需要回答的问题。

本篇先过一遍五种通信模式，再讨论通信内容的设计（结构化协议、三类消息、Context 披露、Artifact 分离），最后给出生产级参考架构和 Hermes / Coding Agent 场景下的落地建议。

---

## 1. 默认模式：父 Agent 直接调用子 Agent

这是最常见的 Supervisor / Worker 模式：

```text
User
  │
  ▼
Supervisor Agent
  │
  ├── call ResearchAgent(task)
  │
  ├── call CodeAgent(task)
  │
  └── call ReviewAgent(task)
           │
           ▼
        result
           │
           ▼
      Supervisor
```

通信其实就是一次函数调用：

```python
result = research_agent.run({
    "task": "查找 GRPO 的相关资料",
    "context": {...}
})
```

子 Agent 返回结构化结果：

```json
{
  "status": "success",
  "summary": "...",
  "sources": [...],
  "artifacts": [...]
}
```

注意这里的形态**不是**：

```text
Agent A: "你好 B，你帮我查一下"
```

而是：

```text
Agent A → Task Request → Agent B → Structured Result
```

这是大多数场景下最推荐的默认模式，优点是：

- 简单、好调试
- 因果关系清晰
- 容易限制权限
- 容易做 tracing
- 不容易出现 Agent 互相无限聊天

Hermes 的 subagent / delegate_task、很多 LangGraph supervisor 架构，本质上都可以归到这一类。([LangGraph 文档][1])

---

## 2. Shared State：通过共享状态通信

LangGraph 这类系统经常不让 Agent 互相直接发消息，而是所有 Agent 读写一个共享 State：

```python
state = {
    "user_request": "...",
    "plan": [],
    "research_result": None,
    "code_result": None,
    "review_result": None
}
```

架构上是一块"共享黑板"：

```text
                  Shared State
                ┌───────────────┐
                │ user_request  │
                │ plan          │
                │ evidence      │
                │ artifacts     │
                │ status        │
                └───────┬───────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
 ResearchAgent       CodeAgent      ReviewAgent
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
                    update state
```

ResearchAgent 写结果：

```python
state["research_result"] = research(...)
```

ReviewAgent 读结果：

```python
research = state["research_result"]
```

Agent A 和 Agent B 甚至没有直接通信——**通过共享黑板（Blackboard）通信**。这类架构叫 **Blackboard Architecture**，是 AI 领域的经典模式。([Wikipedia][4]) 它非常适合：

- 工作流 Agent
- 多阶段业务处理
- 审批流程
- 长任务

---

## 3. Message Queue：通过消息总线通信

如果 Agent 真正独立部署，就不能只靠内存里的 `state`：

```text
Research Agent 运行在机器 A
Code Agent     运行在机器 B
Review Agent   运行在机器 C
```

这时引入消息中间件：Kafka、RabbitMQ、Redis Streams、NATS、Pulsar。

```text
                 Message Bus
     ┌────────────────────────────────┐
     │ agent.task.research            │
     │ agent.task.code                │
     │ agent.result.review            │
     │ agent.event.completed          │
     └──────────────┬─────────────────┘
                    │
      ┌─────────────┼─────────────┐
      │             │             │
      ▼             ▼             ▼
 Research        Code          Review
 Agent           Agent         Agent
```

Supervisor 发布任务：

```json
{
  "type": "task",
  "task_id": "123",
  "target": "research-agent",
  "payload": {
    "query": "GRPO训练方法"
  }
}
```

Research Agent 处理后回发结果：

```json
{
  "type": "task_result",
  "task_id": "123",
  "status": "success",
  "result": {...}
}
```

这种模式最大的优势是 **Agent 可以真正分布式**，并且天然支持：重试、异步、削峰、故障恢复、消费者组、横向扩容。

---

## 4. RPC / API：Agent 直接调用 Agent

如果 Agent 是独立服务，可以直接同步调用：

```text
Planner Agent
     │
     │ HTTP / gRPC
     ▼
Research Agent Service
     │
     │ HTTP / gRPC
     ▼
Browser Agent
```

例如：

```http
POST /agents/research/run
```

```json
{
  "task_id": "abc",
  "query": "分析 FlashAttention 3",
  "context": {
    "project": "LLM知识库"
  }
}
```

返回：

```json
{
  "status": "completed",
  "output": "...",
  "citations": [...]
}
```

这就是 **Agent-as-a-Service**。它和消息队列最大的区别：

```text
RPC         = 同步请求/响应
Message Bus = 异步事件驱动
```

简单任务用 HTTP / gRPC；复杂长任务用 Queue + Event 更合适。跨组织、跨框架的 Agent 互联还可以参考 A2A 这类开放协议——它明确定位为"Agent 对 Agent"的通信层，与 MCP（Agent 对工具）互补。([A2A Protocol][3])

---

## 5. Pub/Sub：Agent 广播事件

有时候 Agent 并不知道"谁应该处理"。例如 DocumentAgent 发现一篇新文档，只发一个 `document.created` 事件，然后由感兴趣的订阅方各自消费：

```text
                 document.created
                        │
                        ▼
                   Event Bus
            ┌───────────┼───────────┐
            │           │           │
            ▼           ▼           ▼
        Embedding     Graph       Search
         Agent        Agent       Agent
```

EmbeddingAgent、GraphAgent、IndexerAgent、AuditAgent 都监听 `document.created`，而 DocumentAgent 根本不需要知道其他 Agent 存不存在。这种模式叫 **Event-driven Multi-Agent Architecture**，特别适合规模大的系统。

---

## 6. Agent 之间到底传什么？

这是多 Agent 系统里最关键的问题之一。不要只传自然语言：

```text
"我已经查好了，你看看吧"
```

生产系统更推荐结构化载荷：

```json
{
  "task_id": "task_123",
  "sender": "research_agent",
  "receiver": "planner_agent",
  "type": "research_result",
  "status": "success",
  "summary": "...",
  "data": {
    "papers": [],
    "claims": []
  },
  "artifacts": [
    {
      "type": "file",
      "uri": "..."
    }
  ],
  "provenance": {
    "sources": []
  }
}
```

也就是说：**Agent-to-Agent 通信最好是结构化协议，而不是纯自然语言**。自然语言只放在 `payload` / `summary` / `reasoning result` 里。Anthropic 在总结生产级 Agent 系统经验时也反复强调：成功的实现往往是简单、可组合的模式，而非复杂框架。([Anthropic][2])

---

## 7. 最好区分三种通信内容

一个很实用的设计是把消息分成三类，不要混成"一大坨 prompt"。

**第一类：Control Message**——控制信息，负责"做什么、谁来做、优先级、deadline、状态"：

```json
{
  "type": "task",
  "task_id": "123",
  "action": "research"
}
```

**第二类：Data Message**——真正的业务数据：

```json
{
  "query": "...",
  "documents": [...],
  "constraints": [...]
}
```

**第三类：Event**——告诉系统发生了什么：

```json
{
  "event": "research.completed",
  "task_id": "123"
}
```

---

## 8. Context 不应该在 Agent 间全量复制

这是很多 Multi-Agent 系统最容易犯的问题。错误做法：

```text
Supervisor 有 100k context
      ↓ 复制
Research Agent 100k
      ↓ 复制
Review Agent 100k
      ↓ 复制
Code Agent 100k
```

Token 成本直接爆炸。正确做法是共享 Context Store + 按需取：

```text
                  Shared Context Store
                       │
        ┌──────────────┼───────────────┐
        │              │               │
        ▼              ▼               ▼
     Agent A         Agent B         Agent C
       │               │               │
   只拿自己需要的    只拿相关内容     只拿相关内容
```

例如：

```text
Global task
      │
      ▼
Context Router
      │
      ├── Research context
      ├── Code context
      └── Review context
```

这实际上就是**渐进式上下文披露（progressive disclosure）**：Agent 之间不传整个 context，而传"task description + 必要摘要 + artifact references + retrievable state IDs"。

---

## 9. 大文件不要通过消息传

比如 CodeAgent 生成了 300 MB 数据，不要塞进消息体：

```json
{
  "result": "<300MB内容>"
}
```

而应该走对象存储/文件系统，消息只传引用：

```text
Agent A
  │
  ▼
Object Storage / File System
  │
  ├── artifact://xxx
  │
  ▼
Agent B
```

```json
{
  "artifact_id": "artifact_123",
  "uri": "s3://bucket/result.json"
}
```

本地 Agent 则传路径，如 `/workspace/artifacts/result.json`。这就是**控制面（Control Plane）和数据面（Data Plane）分离**。

---

## 10. Memory 是另一回事

多 Agent 通信不能和 Memory 混为一谈。Memory 可以作为"间接通信媒介"，但更适合存长期事实、用户偏好、历史任务、任务结果、经验——而不是作为实时消息队列。四者应该分开：

```text
Message Bus    = 实时通信
Shared State   = 当前任务状态
Memory         = 长期信息
Artifact Store = 大文件/结果
```

Hermes 自身的记忆体系（会话内记忆管理、压缩机制）就是独立的子系统，见站内笔记 ([记忆管理][5])、([记忆压缩][6])。

---

## 11. 生产级参考架构

一个拆得比较成熟的 Multi-Agent 系统：

```text
                         User
                           │
                           ▼
                     API Gateway
                           │
                           ▼
                     Orchestrator
                     / Supervisor
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
             Task Router         State Manager
                 │                   │
                 │             PostgreSQL/Redis
                 │
      ┌──────────┼────────────┐
      │          │            │
      ▼          ▼            ▼
 Research      Code        Review
 Agent         Agent       Agent
      │          │            │
      └──────────┼────────────┘
                 │
                 ▼
              Event Bus
          Kafka / NATS / Redis
                 │
       ┌─────────┼───────────┐
       │         │           │
       ▼         ▼           ▼
     Memory   Artifact     Audit
     Store     Store        Log
       │         │
    Vector    S3/File
     + SQL
```

每个组件职责非常清楚：Orchestrator 编排、Task Router 路由、State Manager 管状态、Event Bus 做异步通信、Memory / Artifact / Audit 分别管长期信息、大文件和审计。

---

## 12. Hermes / Coding Agent 场景的落地形态

对于 Hermes / Claude Code / Codex 这类 Coding Agent 架构，更可能的形态是：

```text
                   Parent Hermes
                        │
                        ▼
                  Task Planner
                        │
      ┌─────────────────┼─────────────────┐
      │                 │                 │
      ▼                 ▼                 ▼
 Research Agent      Code Agent       Review Agent
      │                 │                 │
      └─────────────────┼─────────────────┘
                        │
                        ▼
                 Shared Workspace
                  /workspace
                        │
                        ▼
                  Shared State
```

子 Agent 之间通常没必要 `A → B → C → A → B` 疯狂聊天，更合理的是：

```text
Parent
  ↓
A 完成
  ↓
写状态
  ↓
Parent 判断
  ↓
调用 B
```

即：**Supervisor 负责通信编排，Worker 尽量彼此解耦**。这比"Agent 群聊"稳定得多。Hermes 的子代理执行模型（子代理在独立 git worktree 中操作、委派上下文有专门标记、凭证和看板权限被裁剪）正是这个思路的工程实现，详见 ([安全与工作区隔离][7])。

---

## 13. 什么时候才需要 Agent-to-Agent 直接通信？

比如任务强耦合：实现 Agent 发现接口设计有问题，需要架构 Agent 修改协议后再继续：

```text
CodeAgent
   │
   │ request_revision
   ▼
ArchitectureAgent
```

但即便如此，也最好经过 Task Router / Orchestrator，而不是两个 Agent 无限自由聊天。否则非常容易出现：

```text
Agent A: 我觉得这样。
Agent B: 我不同意。
Agent A: 那这样呢？
Agent B: 可以，不过……
Agent A: ……
```

Token 疯狂消耗，而且没有 termination condition。

---

## 14. 最重要的不是"通信"，而是"协议"

把 Agent 消息统一成类似这样的协议：

```json
{
  "message_id": "msg_xxx",
  "task_id": "task_xxx",
  "parent_task_id": "task_parent",
  "sender": "planner",
  "receiver": "coder",
  "message_type": "TASK_REQUEST",
  "payload": {},
  "context_refs": [],
  "artifact_refs": [],
  "status": "PENDING",
  "priority": 5,
  "created_at": "..."
}
```

返回：

```json
{
  "message_type": "TASK_RESULT",
  "task_id": "task_xxx",
  "status": "SUCCESS",
  "summary": "...",
  "artifact_refs": [],
  "next_actions": []
}
```

一旦有了这个 protocol，底层用 Python function、HTTP、gRPC、Redis、Kafka、NATS，都只是 transport layer 的变化。

---

## 15. 四层模型

Multi-Agent 通信架构最清晰的理解方式是四层：

```text
        Agent Communication

             Protocol
                │
       "传什么、格式是什么"
                │
                ▼
             Routing
                │
         "谁应该收到"
                │
                ▼
            Transport
                │
   Function / HTTP / MQ / PubSub
                │
                ▼
             Storage
                │
   State / Memory / Artifact / Log
```

**Protocol > Routing > Transport > Storage**

很多系统一上来就纠结"Kafka 还是 Redis"，其实顺序反了。首先应该确定：Agent 之间传什么、谁能跟谁通信、什么时候终止、失败怎么处理——然后才是 Redis / Kafka / HTTP。

---

## 16. 核心设计原则

**第一，通信的本质是结构化状态交接，不是自然语言聊天**。Agent A 产出结构化任务/结果，经通信介质交给 Agent B，B 更新自己的上下文和状态。

**第二，默认用 Supervisor / Worker**。父子调用简单、可调试、因果清晰、权限好控。Hermes 的 subagent / delegate_task 就属于这一类。

**第三，Worker 之间尽量解耦，由 Supervisor 编排**。避免 Agent 群聊——没有终止条件的自由对话只会烧 token。

**第四，协议先于传输**。先定好消息格式（Control / Data / Event 三类分开），再选 Redis / Kafka / HTTP。

**第五，Context 按需披露，不全量复制**。传任务描述 + 摘要 + artifact 引用 + 可检索的 state ID，而不是 100k 全量上下文。

**第六，控制面与数据面分离**。大结果走 Artifact Store，消息里只放 URI。

**第七，通信、状态、记忆、存储是四件事**。Message Bus 管实时通信，Shared State 管当前任务，Memory 管长期信息，Artifact Store 管大文件——不要混用。

**第八，复杂度按需升级**。Hermes / Coding Agent 场景优先"Supervisor + 结构化任务协议 + 共享状态 + 共享工作区 + 子 Agent RPC"；只有当 Agent 真正跨机器、任务异步、并发量高时，才升级成"Message Bus + Event-driven Agents + Distributed State"。

## 参考文献

1. [LangGraph: Multi-agent Systems Concepts][1]
2. [Building Effective Agents (Anthropic)][2]
3. [Agent2Agent (A2A) Protocol][3]
4. [Blackboard system (Wikipedia)][4]
5. [Hermes Agent单个会话的记忆管理][5]
6. [Hermes Agent记忆压缩机制][6]
7. [Hermes Agent安全、工作区隔离][7]

[1]: https://langchain-ai.github.io/langgraph/concepts/multi_agent/ "LangGraph: Multi-agent Systems"
[2]: https://www.anthropic.com/research/building-effective-agents "Building Effective Agents"
[3]: https://google.github.io/A2A/ "Agent2Agent (A2A) Protocol"
[4]: https://en.wikipedia.org/wiki/Blackboard_system "Blackboard system"
[5]: Hermes%20Agent单个会话的记忆管理.md "Hermes Agent单个会话的记忆管理"
[6]: Hermes%20Agent记忆压缩机制.md "Hermes Agent记忆压缩机制"
[7]: Hermes%20Agent安全、工作区隔离.md "Hermes Agent安全、工作区隔离"
