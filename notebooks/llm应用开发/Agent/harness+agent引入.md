**AI 领域的“Agent Harness”（智能体基础设施/约束工程）**。

在 2026 年的 AI 工程界，行业达成了一个共识公式：**`Agent = Model + Harness`**。模型（Model）是大脑，而 Harness（约束壳/基础设施）是包裹在模型外围的一切——包括 Agent Loop（对话循环）、记忆系统、工具调用、沙箱环境、安全护栏等。正是 Harness 让 Claude Code、Cursor 这样的 AI 编程助手能够稳定地执行复杂长任务。

为了帮你系统学习 **Agent Harness (Harness Engineering)**，我为你整理了目前最新、最优质的学习资料和开源教程：

### 一、 核心概念与必读文章（理论篇）
如果你刚接触这个概念，建议先通过以下文章建立认知：
1. **《The Anatomy of an Agent Harness》**（LangChain 官方博客）
   * **简介**：由 LangChain 核心开发者 Vivek Trivedy 撰写的奠基性文章。详细拆解了 Agent Harness 的核心组件（文件系统、沙箱、记忆、编排等），解释了为什么仅靠模型不够，以及如何通过 Harness 提升任务成功率。
2. **《一文读懂 Harness Engineering》 / 《拆解 Agent Harness 的 11 大核心组件》**
   * **简介**：国内深度解析文章，追溯了 Harness 概念的发生史，并解释了为何单纯的“Prompt 工程”正在被“上下文工程与 Harness 工程”取代。
3. **《一文讲透如何构建 Harness——六大组件全解析》**
   * **简介**：腾讯云开发者社区的爆款文章，详细讲解了从工具系统到编排 + Hooks（让单兵作战变成集团军）的六大组件逻辑。

### 二、 开源教程与实战书籍（实战解析篇）
这是目前社区里最受推崇的几个从 0 到 1 搭建 Agent Harness 的开源项目资源：

1. **《御舆：解码 Agent Harness》 (Claude Code Book)**
   * **链接**：[lintsinghua/claude-code-book](https://github.com/lintsinghua/claude-code-book)
   * **简介**：一本 42 万字的开源神书，深度拆解了目前最成熟的 Agent Harness 框架（Anthropic 的 Claude Code）。全书 15 章，包含 139 张架构图，从对话循环、工具系统、四阶段权限管线讲到上下文压缩，教你如何自己构建一个生产级的 Agent Harness。
2. **Learn Claude Code (Bash is all you need)**
   * **链接**：[shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)
   * **简介**：一个循序渐进的 12 节课学习项目（Nano Claude Code）。从最基础的 Agent Loop 写起，一步步添加 Tool Use、TodoWrite（任务拆解）、Subagents（子代理调度）、技能树等，用极简代码还原了 Harness 的核心设计理念。
3. **Agent Harness Tutorials 实战代码库**
   * **链接**：[helloworldtang/harness-tutorials](https://github.com/helloworldtang/harness-tutorials)
   * **简介**：展示如何用 Python 代码实现 Agent Harness 的三大核心功能以解决 AI 幻觉问题：验证循环（让 AI 验证自己的输出）、防护栏机制（阻止危险操作）和上下文管理。
4. **Learn OpenHarness 交互式教程**
   * **链接**：[joyehuang/Learn-Open-Harness](https://github.com/joyehuang/Learn-Open-Harness)
   * **简介**：面向零基础的 AI Agent 交互式教程，将复杂的 1 万多行 OpenHarness 基础设施代码拆解为 12 个交互式章节，带你理解 MCP、工具注册表和权限分层等。

### 三、 视频教程（视听学习篇）
1. **YouTube / B站搜索推荐**：
   * 搜索 **“What Is an AI Agent Harness?”**：可以找到 KodeKloud 等博主关于 Claude Code 背后 Harness 机制的拆解，解释为何模型能在执行长达几小时的多步任务时不崩溃。
   * 搜索 **“Harness Engineering 保姆级教程”**：B站上有多位 UP 主（如“AI大模型小冉Agent”、“马克的技术工作坊”）系统讲解 Harness 的底层原理和代码实战（如串联 OpenClaw 等工具）。

### 💡 学习路径建议
* **第一周**：阅读《The Anatomy of an Agent Harness》等理论文章，理解 Harness 是解决多 Agent 编排、重试、错误回传和状态持久化的**底层操作系统**。
* **第二周**：阅读《御舆》的架构图（至少看懂 Agent Loop 循环机制、Context 怎么压缩、Guardrails 怎么拦截危险命令）。
* **第三周**：克隆 `learn-claude-code` 或 `harness-tutorials`，在本地环境（结合 DeepSeek / Claude / GPT API）跑通第一个带有“验证循环”和“工具权限控制”的自定义 Agent。

---

*(注：如果你实际上问的是 **CI/CD DevOps 领域的 Harness 公司** 及其执行节点 **Harness Agent (官方称为 Harness Delegate)**，那么它是用来执行持续集成/持续交付流水线的组件。你可以直接访问 [Harness Developer Hub 的 Delegate 概述](https://developer.harness.io/docs/platform/delegates/delegate-concepts/delegate-overview/) 或在 YouTube 搜索 `Harness Delegate Installation` 找到 5 分钟在 Kubernetes 部署 Delegate 的官方实操视频。)*