# Claude Mythos 5 技术报告：官方披露的训练方法与模型架构

> 截至日期：2026-06-12  
> 资料范围：Anthropic 官方发布页、产品页、API 文档、Project Glasswing 页面，以及 `Claude Fable 5 & Claude Mythos 5 System Card`、`Claude Mythos Preview System Card`。  
> 结论先行：Anthropic 已披露 Mythos 5 的训练数据来源、训练/后训练流程、安全治理、访问策略和 API 规格，但没有公开参数量、层数、注意力结构、MoE/稠密架构、训练算力、token 数等底层模型架构细节。

## 1. 定位与发布时间线

### 1.1 Mythos-class 的含义

Anthropic 将 Mythos-class 定义为能力位于 Opus class 之上的 Claude 模型层级。官方在 2026-06-09 发布 `Claude Fable 5 and Claude Mythos 5` 时说明：

- Claude Mythos Preview 是第一个 Mythos-class 模型，于 2026-04 通过 Project Glasswing 释放给少量网络防御和关键软件基础设施伙伴。
- Claude Fable 5 与 Claude Mythos 5 是后续发布的同一代 Mythos-class 能力模型。
- Fable 5 面向一般用户发布；Mythos 5 仅面向通过审核的可信访问伙伴。

### 1.2 Fable 5 与 Mythos 5 的关系

官方表述非常关键：Claude Mythos 5 与 Claude Fable 5 是同一个底层模型的两个配置。

- `Claude Fable 5`：同一底层模型，但加入更强的网络安全、生物、化学和模型蒸馏相关防护；在高风险请求上会回退到 Claude Opus 4.8。
- `Claude Mythos 5`：同一底层模型，但面向可信伙伴，在特定高风险能力上解除相应防护，例如 Project Glasswing 的网络安全可信访问。

因此，从模型技术角度看，Fable 5 与 Mythos 5 的差异不是官方披露的参数结构差异，而主要是部署防护、访问权限和路由策略差异。

## 2. 官方披露的训练方法

### 2.1 预训练数据来源

`Claude Fable 5 & Claude Mythos 5 System Card` 的训练章节披露，Mythos 5 与 Fable 5 使用以下数据来源训练：

- 来自互联网的公开可用信息；
- 公共数据集与私有数据集；
- 由其他模型生成的合成数据。

系统卡同时说明，训练过程中使用了多种数据清洗与过滤方法，包括去重与分类。

### 2.2 ClaudeBot 与网页数据采集

官方说明其使用名为 `ClaudeBot` 的通用网页爬虫获取公开网站数据，并遵循网站运营者通过 `robots.txt` 表达的抓取偏好。Anthropic 还声明：

- 不访问密码保护页面；
- 不访问需要登录的页面；
- 不访问需要 CAPTCHA 验证的页面；
- 对训练数据进行尽职调查；
- 网站运营者可以识别 ClaudeBot 抓取行为并表达偏好。

### 2.3 后训练与微调

系统卡明确写到：预训练之后，模型经历了 substantial post-training and fine-tuning，目标是使其成为行为符合 Claude 宪法价值观的助手。

这意味着官方披露的训练流水线可以概括为：

```text
混合数据收集 → 数据清洗/过滤/去重/分类 → 预训练 → 后训练与微调 → 安全评估与部署防护
```

但系统卡没有公开后训练算法的完整细节，例如是否使用 RLHF、RLAIF、DPO、PPO、拒绝采样、偏好模型训练，或各阶段比例。因此不能把这些方法写成 Mythos 5 的官方事实。

### 2.4 合成数据与宪法价值观

Anthropic 披露训练数据包括其他模型生成的合成数据，并在后训练阶段对齐 Claude 宪法价值观。结合 Anthropic 过往公开路线，这与其 Constitutional AI 传统一致；但在 Mythos 5 系统卡中，官方没有给出完整的合成数据生成配方、宪法条目权重、奖励模型结构或训练超参数。

## 3. 官方披露的模型架构

### 3.1 已披露内容

官方系统卡将 Mythos 5/Fable 5 称为 a new large language model from Anthropic，并披露它具有：

- 多语言能力，通常会用与用户输入相同的语言回复；
- 长上下文能力；
- 视觉、代码、知识工作、生命科学、网络安全等多领域能力；
- adaptive thinking 能力在 API 文档表中标注为 always on；
- Fable 5 与 Mythos 5 使用 Claude Opus 4.7 引入的 tokenizer，同一文本相较 Opus 4.7 之前模型大约会产生更多 token。

### 3.2 未披露内容

截至当前官方材料，Anthropic 没有公开以下底层架构信息：

- 参数量；
- 层数、隐藏维度、注意力头数；
- 是否为稠密 Transformer、Mixture-of-Experts，或其他架构；
- 上下文扩展机制，例如 RoPE 变体、稀疏注意力、滑窗注意力、检索增强或外部记忆机制；
- 训练 token 数、数据配比、训练算力与硬件集群；
- 预训练目标函数、优化器、学习率策略、batch size；
- 后训练中偏好数据、奖励模型、RL 或直接偏好优化细节。

因此，关于 Mythos 5 的“架构分析”只能停留在官方给出的产品级规格和行为评估层面，不能把业界对 Claude 系列的常见猜测当成官方披露。

## 4. API 与产品规格

官方 API 模型概览页面披露的关键规格如下：

| 项目 | Claude Fable 5 | Claude Mythos 5 |
| --- | --- | --- |
| Claude API ID | `claude-fable-5` | `claude-mythos-5` |
| 定位 | Anthropic 最强的广泛发布模型，面向高难推理与长程 agentic 工作 | 通过 Project Glasswing 提供，Claude Mythos Preview 的后继者 |
| AWS Bedrock | `anthropic.claude-fable-5` | Limited availability |
| Vertex AI | `claude-fable-5` | Limited availability |
| Extended thinking | No | No |
| Adaptive thinking | Yes, always on | Yes, always on |
| Context window | 1M tokens | 1M tokens |
| Max output | 128k tokens | 128k tokens |
| 价格 | $10 / input MTok, $50 / output MTok | $10 / input MTok, $50 / output MTok |

产品页还披露：使用 Mythos 5 需要接受 30 天数据保留政策，用于安全监控；Anthropic 表示这些数据不会用于训练新的 Claude 模型或非安全相关目的。

## 5. 能力侧披露

### 5.1 网络安全

Project Glasswing 是 Mythos Preview/5 最重要的官方披露场景。Anthropic 称 Mythos Preview 能显著提升发现和利用软件漏洞的能力，因此只向网络防御者、关键基础设施和开源安全相关伙伴开放。

官方披露的高层能力包括：

- 在重要软件中发现大量高危或关键漏洞；
- 能执行本地漏洞检测、黑盒测试、端点安全、渗透测试、漏洞复现与补丁辅助等防御任务；
- Mythos-class 模型的网络安全能力存在明显双用途风险，因此不做一般开放。

报告中不展开具体攻击步骤或漏洞利用细节，只记录官方对能力边界和访问策略的描述。

### 5.2 生命科学与生物医学

Anthropic 在发布页中说明，Mythos 5 在药物设计、蛋白设计、生物信息学工具使用、分子生物学假设生成、基因组研究等任务中表现突出。

官方示例包括：

- 内部蛋白设计专家使用 Mythos 5 将部分药物设计流程加速约 10 倍；
- Mythos 5 能在没有人工直接辅助的情况下选择结合位点、选择并运行蛋白设计工具、从失败中恢复；
- 在盲测中，科学家约 80% 时间更偏好 Mythos 的分子生物学假设；
- Mythos 5 可进行多日级、较自主的基因组研究工作，并设计训练自定义机器学习模型。

这些能力同样被 Anthropic 归为强双用途能力，因此 Mythos 5 的生物能力访问也走可信访问计划。

### 5.3 知识工作、代码与视觉

发布页主要把一般用户可见能力放在 Fable 5 上描述，因为 Fable 5 是一般可用形态。由于 Fable 5 与 Mythos 5 是同一底层模型，Anthropic 认为在未触发高风险防护的会话中，Fable 5 的表现基本等同于 Mythos 5。

官方披露的强项包括：

- 长程软件工程与代码迁移；
- 高难知识工作、金融推理、文档和图表理解；
- 视觉任务，例如从截图重建应用源码；
- 长上下文任务中保持关注并利用自身笔记改进输出。

## 6. 安全机制与发布策略

### 6.1 Fable 5 的防护与回退

Fable 5 是 Mythos 级能力的一般发布版本。Anthropic 为其加入分类器防护：当请求涉及网络安全、生物/化学、模型蒸馏等敏感领域时，系统会自动由 Claude Opus 4.8 处理，而不是让 Fable 5 直接回答。

官方说明这些分类器目标包括：

- 检测潜在滥用；
- 检测 jailbreak 尝试；
- 降低网络安全、生物化学和模型蒸馏相关滥用风险；
- 牺牲部分误报率以尽快安全发布。

### 6.2 Mythos 5 的可信访问

Mythos 5 不是一般可用模型。官方当前访问路径包括：

- Project Glasswing 伙伴，用于网络防御任务；
- 计划中的生物可信访问计划，用于生命科学研究；
- 通过 Anthropic、AWS 或 Google Cloud 账户团队申请的有限可用路径。

### 6.3 Constitutional Classifiers++ 与 ASL/RSP 背景

Anthropic 在相关研究中披露了下一代 Constitutional Classifiers++：

- 使用内部 activation probe 作为低成本第一阶段筛选；
- 对可疑 exchange 升级到更强分类器；
- 分类器会同时参考输入和输出，而不是只单独看输入或输出；
- 生产级系统的额外计算开销约 1%；
- 在其红队测试中尚未发现 universal jailbreak。

这些并不等同于 Mythos 5 模型架构本身，而是模型外部部署防护与安全监控体系的一部分。

## 7. 与 Mythos Preview 的关系

Claude Mythos Preview 是 2026-04 Project Glasswing 中释放的研究预览模型。官方称其是 Anthropic 训练的新前沿通用模型，具有强 agentic coding 与推理能力，尤其在网络安全漏洞发现与利用方面表现突出。

Mythos 5 是 Mythos Preview 的后继更新：

- 产品页称 Mythos 5 在网络安全、生物和医疗健康 benchmark 上相比 Mythos Preview 有提升；
- 发布页称 Mythos 5 在大多数场景中与 Mythos Preview 相当或更强，同时成本显著降低；
- Mythos Preview 价格为 $25 / input MTok 和 $125 / output MTok，Mythos 5/Fable 5 为 $10 / input MTok 和 $50 / output MTok。

## 8. 可确认事实与不可确认事实

### 8.1 可确认事实

- Mythos 5 与 Fable 5 是同一底层 LLM 的不同部署配置。
- Mythos-class 是 Opus 之上的能力层级。
- Mythos 5 的官方 API ID 是 `claude-mythos-5`。
- Mythos 5 具有 1M token 上下文窗口与 128k token 最大输出。
- Mythos 5/Fable 5 的训练数据包括互联网公开信息、公共/私有数据集、其他模型生成的合成数据。
- 训练流程包含数据清洗、过滤、去重、分类、预训练、后训练与微调。
- 后训练目标是使行为符合 Claude 宪法价值观。
- Mythos 5 当前不是一般可用模型，而是可信访问模型。

### 8.2 不可确认或未披露事实

- 不能确认 Mythos 5 是 MoE 还是稠密模型。
- 不能确认参数量、训练 token 数、算力规模或训练成本。
- 不能确认后训练具体算法，例如 PPO、DPO、RLHF、RLAIF 的使用与比例。
- 不能确认其长上下文实现机制。
- 不能确认是否使用特定注意力优化、位置编码或外部记忆架构。

## 9. 对学习者的技术解读

从公开资料看，Mythos 5 的技术重点不是 Anthropic 对外公开了一个可复现架构，而是展示了一个前沿闭源模型在以下方面的工程化组合：

1. **能力层级提升**：Mythos-class 被定义为 Opus 之上的能力层级，突出长程 agentic 工作、网络安全、生命科学和知识工作。
2. **同模双部署**：同一底层模型通过防护与访问策略分成 Fable 5 与 Mythos 5，体现“能力不等于开放面”的发布思想。
3. **模型外防护成为核心系统**：分类器、回退、数据保留、安全监控、可信访问计划构成了闭源前沿模型的重要组成部分。
4. **安全发布优先于透明架构披露**：Anthropic 披露了训练数据类别和安全评估，但没有披露底层结构细节，这与其对高能力模型的风险管理策略一致。

## 10. 参考资料

- [Claude Fable 5 and Claude Mythos 5](https://www.anthropic.com/news/claude-fable-5-mythos-5)
- [Claude Mythos 5 产品页](https://www.anthropic.com/claude/mythos)
- [Claude API Models Overview](https://platform.claude.com/docs/en/docs/about-claude/models/overview)
- [Claude Fable 5 & Claude Mythos 5 System Card](https://www.anthropic.com/claude-fable-5-mythos-5-system-card)
- [Project Glasswing](https://www.anthropic.com/glasswing)
- [Project Glasswing: An initial update](https://www.anthropic.com/research/glasswing-initial-update)
- [Expanding Project Glasswing](https://www.anthropic.com/news/expanding-project-glasswing)
- [Claude Mythos Preview System Card](https://anthropic.com/claude-mythos-preview-system-card)
- [Next-generation Constitutional Classifiers](https://www.anthropic.com/research/next-generation-constitutional-classifiers)
- [Activating AI Safety Level 3 protections](https://www.anthropic.com/news/activating-asl3-protections)
- [Announcing our updated Responsible Scaling Policy](https://www.anthropic.com/news/announcing-our-updated-responsible-scaling-policy)
