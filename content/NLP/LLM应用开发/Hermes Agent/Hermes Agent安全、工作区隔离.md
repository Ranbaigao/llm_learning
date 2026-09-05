# 安全隔离与工作区隔离

> **受众：** Hermes Agent 贡献者、网关部署人员、安全审计人员，以及需要运行多用户/多任务场景的高级用户。
> **最后更新：** 2026-08-17

## 概述

Hermes Agent 的隔离设计遵循一个明确的原则：**唯一真正安全的边界是操作系统级隔离**，进程内的所有检查都被视为启发式防御，不是真正的安全边界。这个原则贯穿整个系统：从工作区隔离到凭证管理，从通道授权到子代理执行，处处都是"用多层启发式叠加去模拟一个真边界"。

整个隔离体系按"从外到内"可以分成三层思考框架：

最外层是**操作系统级隔离**——终端命令在容器或远程沙箱里运行，文件系统、网络、进程能力都被限制在沙箱内。这是真边界。

中间层是**工作区隔离**——每个 profile 拥有独立的配置文件、独立的数据库、独立的凭证、独立的插件注册表。Profile 之间互相看不到对方的数据。

最内层是**进程内授权**——多重的用户 allowlist、配对审批、命令危险模式识别、输出脱敏。这些机制即使在沙箱逃逸的极端情况下也能提供"至少让用户知道发生了什么"的审计能力。

三层不是替代关系，而是叠加关系。一个稳态的生产部署通常同时激活这三层：Docker 沙箱 + 多 profile + 通道 allowlist。

---

## 1. 信任模型的根本原则

在深入具体机制之前，先明确四条贯穿全系统的原则。

**第一，进程内检查不是安全边界**。哪怕代码写得再严谨，进程内的检查都可以被恶意插件绕过——插件一旦加载到代理进程里，与代理同权限，能读取所有凭证、调用所有工具、注册所有钩子。第三方插件的真正边界是"安装前的人工审查"。

**第二，授权必须跨信任边界**。仅靠网络命名空间内的所有检查都视为启发式，不是真正的安全边界。这意味着任何对外暴露的接口都必须有 allowlist 配置，"默认放行"是设计缺陷。

**第三，失败时关闭**。所有配置缺失、检测失败、解析异常的情况，默认行为是关闭访问而不是放行。这条原则在第二章到第十章里反复出现。

**第四，权限和审计并存**。每个允许的操作都对应一个可追溯的审计记录；每个拒绝的操作都对应一个明确的拒绝原因。审计不是事后追溯，而是设计决策的一部分。

---

## 2. 工作区隔离——HERMES_HOME 为核心

工作区隔离的整个基础设施围绕一个核心概念：**HERMES_HOME**，即 Hermes Agent 的工作根目录。每个 profile 拥有独立的 HERMES_HOME，从而拥有独立的配置、独立的数据库、独立的凭证、独立的插件。

### 2.1 HERMES_HOME 的三种解析方式

系统在不同场景下用不同的方式解析 HERMES_HOME，确保请求看着一致，但实际隔离彻底。

**Context-local 方式**最高优先级。系统通过一个特殊的上下文变量存储当前请求对应的 HERMES_HOME 路径，调用 `get_hermes_home()` 优先返回这个值。这个机制让多路复用网关能在一个进程里同时服务多个 profile，每个请求自动看到自己的根目录。

**环境变量方式**次之。普通 CLI 进程启动时通过 `HERMES_HOME` 环境变量指定根目录。这种方式简单直接，但进程内是全局的——如果进程里同时有多个任务，它们会共享同一个 HERMES_HOME。

**平台默认方式**兜底。不指定时使用平台原生默认路径：Windows 是 `%LOCALAPPDATA%\hermes`，POSIX 是 `~/.hermes`。

还存在一个特殊的"进程级"解析方式，**故意忽略** Context-local 的覆盖。它返回进程启动时的 HERMES_HOME，用于少数需要在 profile 切换时依然保持可见的资产，比如主题文件、面板插件清单。这个对比说明 Context-local 不是一刀切——系统故意区分"请求相关数据"和"进程级数据"。

### 2.2 上下文局部覆盖——为什么不用环境变量

要理解这个机制，先想一个具体场景。

假设一个网关进程同时服务两个用户：用户 A 用 "coder" profile，用户 B 用 "writer" profile。两条消息几乎同时到达，进程需要同时处理。

如果用**环境变量**实现隔离，会发生什么？系统想给 A 处理时设置 `HERMES_HOME=/home/user/.hermes/profiles/coder`，想给 B 处理时设置 `HERMES_HOME=/home/user/.hermes/profiles/writer`。但环境变量是**整个进程共享的**——A 改完的瞬间，B 也读到了 coder 的路径。然后 B 把环境变量改成 writer，A 又读到了 writer 的路径。两边的处理都开始错乱，看到对方的数据。

这就是"并发读写冲突"——多个任务同时改同一个全局变量，后改的会覆盖先改的，谁也保证不了自己看到的值是不是自己设置的。

**上下文局部变量**（Python 里的术语叫 ContextVar）就是为了解决这种问题。每个任务在启动时拿到一份"自己的小账本"，所有变量读写都操作这个小账本，而不是进程公共的全局账本。A 改 A 的账本，B 改 B 的账本，互不干扰。

具体到 Hermes Agent，多路复用网关收到一条消息时，会按这条消息所属的 profile 临时设置上下文局部覆盖，让这条消息处理期间的所有"获取 HERMES_HOME"调用都返回该 profile 的根目录。消息处理完毕后，覆盖被解除，下一条消息可以切换到另一个 profile。

这个机制有一个副作用：子进程不会自动继承上下文局部覆盖。子进程是一个全新的进程，没有任何"上下文"概念。所以当需要启动子进程时，系统会显式把当前 HERMES_HOME 写入子进程的环境变量，让子进程用环境变量方式解析——这时子进程只看到当前 profile 的值，不会与其他 profile 冲突，因为子进程本身就是在为当前 profile 工作的。

### 2.3 多 Profile 架构

HERMES_HOME 的实际目录结构是分层的：

```
~/.hermes/                          # 根目录（共享）
├── config.yaml                     # 根级配置
├── .env                            # 根级凭证
├── state.db                        # 共享根级状态（如有）
├── profiles/                       # 子 profile 目录
│   ├── coder/                      # "coder" profile
│   │   ├── config.yaml             # coder 专属配置
│   │   ├── .env                    # coder 专属凭证
│   │   ├── state.db                # coder 专属会话数据库
│   │   ├── sessions/               # coder 专属会话
│   │   ├── skills/                 # coder 专属技能
│   │   ├── plugins/                # coder 专属插件
│   │   ├── cron/                   # coder 专属定时任务
│   │   └── logs/                   # coder 专属日志
│   └── writer/                     # "writer" profile
│       └── ...
└── kanban/                         # 共享看板（设计意图）
    └── boards/
```

每个 profile 拥有完全独立的配置、凭证、数据库、技能、插件、定时任务、日志。Profile 之间互相看不到对方的数据——一个 profile 的会话不会出现在另一个 profile 的搜索结果里。

**Profile 命名规则**严格：只允许小写字母、数字、下划线和连字符，长度不超过 64 个字符，且不能以下划线或连字符开头。同时拒绝保留名（如 `hermes`、`default`、`root`、`sudo`）和与 Hermes 子命令冲突的名（如 `chat`、`model`、`gateway`、`setup`）。这套规则防止路径遍历攻击——恶意名无法逃出 profiles 目录。

**Profile 包装脚本**是另一种使用方式。系统可以为每个 profile 在 `~/.local/bin/` 下生成一个同名的可执行脚本，运行它等价于 `hermes -p <profile> "$@"`。包装脚本名同样严格校验，防止形如 `../../.bashrc` 的路径遍历。

### 2.4 配置文件目录的权限

HERMES_HOME 目录及其所有文件都有严格的权限设置：

- 目录权限 `0o700`（仅 owner 可读写执行）
- 配置文件权限 `0o600`（仅 owner 可读写）
- 容器内自动跳过 chmod（Docker mount 需要更宽松权限）
- Managed 模式（包管理器安装）使用组可读 `0o640` 或 `0o2770`
- 可通过环境变量 `HERMES_HOME_MODE` 覆盖默认目录权限

这些权限设置通过专门的"安全父目录"检测防止意外破坏主机文件系统——对于根目录或路径段数小于 3 的目录，拒绝应用 chmod，避免破坏系统关键位置。

### 2.5 配置层叠（系统级 / 用户级 / 项目级）

配置文件按三个层级叠加，按优先级合并：

最上层是**系统级 Managed 配置**。系统管理员可以通过 `/etc/hermes/config.yaml`（路径可自定义）注入 IT 级别的强制配置。Managed 配置按叶子键深度合并到用户配置之上，且这些被锁定的键不能被用户修改——用户尝试修改时会得到硬拒绝（而不是静默丢失）。

中层是**用户 Profile 配置**。每个 profile 有自己的 `config.yaml`，是日常配置的主要载体。

底层是**内置默认值**。系统在代码里硬编码的默认配置，作为兜底。

每一层合并时，关键配置展开 `${VAR}` 引用——但**只从进程环境变量展开**，不从用户配置里递归引用，防止用户通过受控的 `${VAR}` 覆盖系统级字面值。

---

## 3. 注册表按 Profile 隔离

在解释"按 Profile 隔离"之前，先把术语对齐。

**什么是"插件"？** 插件是 Hermes 之外的扩展包，可以给 Hermes 添加新能力——比如新的工具、新的命令、新的认证方式。插件以单独的文件形式存在，运行时被加载进 Hermes 进程。

**什么是"工具"？** 工具是 Hermes 让代理可以使用的能力——读文件、写文件、执行命令、调用 API 等。系统内置一组工具，插件可以注册新工具。

**什么是"平台适配器"？** 适配器是 Hermes 与外部消息平台（Telegram、Discord、Slack、终端界面等）的连接层。不同平台的协议不一样，适配器负责把不同平台的协议统一成 Hermes 内部的消息格式。

**什么是"凭证源"？** 凭证是访问外部服务需要的钥匙——比如调用 OpenAI API 需要 API key，调用 GitHub 需要个人访问令牌。凭证源就是这些钥匙的存储位置——可以是 `.env` 文件、可以是 Bitwarden 或 1Password 这样的密码管理器、也可以是其他插件。

**什么是"注册表"？** 注册表就是一个登记表。插件、工具、适配器、凭证源这些"动态资源"在启动时按登记表加载，运行时按登记表查询。登记表是按 HERMES_HOME 键控的——这就是隔离的核心。

整个隔离体系的工作方式：每个 profile 拥有独立的登记表，profile 之间互不干扰。下面分别看四种注册表。

### 3.1 插件管理器按 Profile 隔离

插件管理器历史上是进程全局的单例——多个 profile 共享同一个管理器，导致 profile 切换时插件状态互相污染。修复后的设计用一个"按 Profile 键控的缓存"取代单例：

- 每个 HERMES_HOME 路径对应一个独立的插件管理器
- 当请求通过 Context-local 切换到另一个 profile 的根目录时，请求看到的是该 profile 专属的管理器
- 卸载时只清理该 profile 的插件注册，不影响其他 profile

更关键的是 Python 模块导入的隔离：插件按 `hermes_plugins.<slug>` 命名空间导入，模块本身缓存在 `sys.modules` 里。Profile 切换时如果只替换顶层模块，子模块（如 `hermes_plugins.<slug>.state`）依然留在 `sys.modules` 里，跨 profile 泄漏状态。修复方案是**驱逐整个命名空间**——加载新插件前清空 `hermes_plugins.<slug>` 及所有子模块，确保完整重载。

每个插件管理器还有一个独立的"所有权账本"——记录每个插件的注册信息。不同 profile 的账本互不影响，profile A 的卸载绝对不会清除 profile B 的注册。

### 3.2 工具注册表按 Profile 隔离

工具注册表用嵌套字典结构：外层按 HERMES_HOME 键控，内层按工具名索引。每个 profile 看到自己的工具集合加上全局内置工具。

插件可以用"覆盖策略"覆盖内置工具，但必须显式声明并获得操作员授权。这个授权按 scope 维度存储——也就是说，A profile 授权的覆盖策略对 B profile 不可见。

恶意插件即使能加载到进程内，也无法在不通过"显示声明 + 操作员授权"流程的情况下覆盖内置工具。每次工具集合变更都增加一个单调递增的代际号，工具定义查询可以基于这个代际号做缓存——确保变更后下游立即看到新版本。

### 3.3 平台适配器按 Profile 隔离

平台适配器（Discord、Telegram、Slack 等）也按 HERMES_HOME 隔离。每个 profile 加载自己的适配器版本，配置独立的允许用户列表和环境变量。

适配器支持延迟加载（不在启动时立即 import 重型 SDK），延迟加载器按 profile 键控——profile A 的加载器不会污染 profile B 的状态。

### 3.4 凭证源注册中心按 Profile 隔离

凭证源（如 Bitwarden、1Password 的同步器）也按 HERMES_HOME 隔离。第三方插件作为外部凭证源时，必须显式指定 scope，否则只在该 profile 的范围内生效。

---

## 4. 凭证隔离

在解释"凭证隔离"之前，先把术语对齐。

**什么是凭证？** 凭证是访问外部服务需要的"钥匙"，比如：

- API key：调用 OpenAI、Anthropic、GitHub 等 API 需要的字符串
- OAuth token：用户授权第三方应用访问自己账号的令牌
- 个人访问令牌（Personal Access Token）：开发者用来访问 GitHub 等平台的字符串
- 数据库连接字符串：访问数据库用的用户名、密码、地址
- 私钥：用于 SSH 登录或加签名的密钥

凭证的核心特征是：拿到凭证就等于拿到相应的访问权限。**泄漏一个凭证就意味着泄漏整个 profile 的访问权限**，凭证隔离必须做到极致。

**什么是"环境变量"？** 环境变量是操作系统层面的全局变量，可以被进程读取，用来传递配置和凭证。比如 `OPENAI_API_KEY=sk-xxx` 就是一个环境变量。

**什么是 `.env` 文件？** `.env` 文件是项目里常见的配置文件，里面按 `KEY=value` 格式写变量名和值。开发者用它来管理配置和凭证。Hermes 的每个 profile 都有自己的 `.env` 文件。

**什么是 OAuth？** OAuth 是一种授权协议，让用户可以让第三方应用"代替自己"访问某个服务，而不用把密码给出去。Hermes 支持的 MCP 协议常用 OAuth 拿 token。

**什么是"上下文变量"？** 上一节讲过，是每个任务私有的小账本，不会与其他任务冲突。

### 4.1 Secret Scope——上下文变量隔离

操作系统环境变量是进程全局的——profile A 的 API key 会泄漏到 profile B 的调用，再泄漏到所有用 `env = dict(os.environ)` 派生的子进程。这种"全进程共享"是隔离的噩梦。

解决方案是**Secret Scope 上下文变量**。系统用一个上下文变量存储当前请求的凭证映射，所有"获取凭证"调用都从这里读而不是从环境变量读。

具体来说有三个层次：

**全局始终可访问的变量**：如 `HERMES_HOME`、`PATH`、`API_SERVER_HOST`、`API_SERVER_PORT` 等部署配置。这些项跨 profile 共享是合理的设计——多个 profile 的 API 服务器应该共享同一个监听端口。

**已安装 scope 的变量**：当前 profile 的凭证。当请求进入某个 profile 时，系统会构建该 profile 的凭证映射（profile 的 `.env` 文件 + 外部凭证源），安装到上下文变量。多路复用模式下，scope 缺失时**严格不 fallback 到 `os.environ`**——直接抛错，强制关闭。

**未安装 scope 的变量**：非多路复用模式下，透明回退到 `os.environ`——保持向后兼容。

这种"多路复用 fail-closed"语义是设计精髓：单进程单 profile 时一切正常；多 profile 共存时，凭证绝不会跨 profile 泄漏。

### 4.2 .env 文件的权限与校验

每个 profile 创建时都会同步生成自己的 `.env` 文件，权限 `0o600`（仅 owner 可读写）。已存在的 profile 也会被"回填" `.env`——历史上它们从根 `.env` 继承凭证，现在每个 profile 拥有独立的 `.env`。

`.env` 加载时执行严格的清洗：

- **剥离非 ASCII 字符**：防止从 PDF 或网页复制粘贴时引入的 Unicode 同形异义字符（如 Cyrillic 'а' 假装成 Latin 'a'）
- **移除 NUL 字节**：防止字符串终止攻击
- **处理 UTF-16/UTF-32 BOM**：自动识别并转换
- **凭证来源溯源**：当凭证来自 Bitwarden 或 1Password 时显示 "(from Bitwarden)" 等标识，让用户清楚知道来源

加载时使用 `override=True`，确保 profile 的 `.env` 优先于 shell 环境——避免用户在一个 profile 里设置的值被另一个 profile 的 shell 环境覆盖。

### 4.3 跨 Profile 环境清理

某些"行为路由"环境变量（如 ACP 认证方式、Copilot 端点）如果在不同 profile 之间泄漏，会导致路由漂移——profile A 的请求被路由到 profile B 配置的端点。

解决方案是**显式清理这些路由键**。系统在 profile 启动时清理一组预定义的环境变量键（不是凭证，是行为路由），确保每个 profile 看到自己的配置。凭证类键不在此清理集合里——它们通过更严格的 Secret Scope 机制隔离。

### 4.4 凭证读取的拒绝清单

作为额外防御层，文件读取工具有一份"凭证 deny 列表"——如果代理试图读取这些文件，操作会被拒绝并提示错误：

- 内部 Hermes 缓存文件（技能中心目录、索引缓存）——它们是提示注入的潜在载体
- 凭证存储文件（认证文件、OAuth 令牌、`.env`、webhook 订阅、谷歌认证文件）
- 整个 MCP 令牌目录
- 项目本地 `.env`、`.env.local`、`.env.development`、`.env.test` 等变体
- 各种 `.envrc`、shell 配置

同样，写入工具有一份"写入 deny 列表"——代理不能写入 SSH 密钥、AWS 配置、GPG 密钥、Kubernetes 配置、sudoers 配置等。这份列表与读取 deny 列表同步维护，确保一致性。

需要明确的是：**这套 deny 列表不是安全边界**。终端工具以相同 OS 用户运行，可以绕过文件工具直接 `cat` 文件。这只是第一道防线——真正的边界是容器沙箱。

### 4.5 容器中的凭证挂载

远程后端（Docker、Modal、SSH）创建的是无宿主机文件的沙箱。系统需要把凭证文件、技能目录、缓存目录以 mount 或 sync 形式注入沙箱。

注入前执行严格的安全检查链：

- **拒绝绝对路径**：必须相对 HERMES_HOME
- **拒绝路径遍历**：防止 `..` 逃出 HERMES_HOME
- **拒绝凭证文件**：与读取 deny 列表同步，认证、`.env`、MCP 令牌等拒绝 mount
- **失败关闭**：导入异常时拒绝 mount 而不是放行

凭证文件注册存储在一个上下文变量里，防止跨会话数据泄漏。

---

## 5. 数据库隔离

每种数据存储都有明确的"共享 vs 隔离"决定。

### 5.1 会话数据库（per-Profile 隔离）

会话状态数据库（state.db）按 profile 隔离——每个 profile 拥有独立的 SQLite 数据库。默认路径是该 profile 的 `state.db`。

测试隔离尤其重要——历史问题显示 pytest 测试的临时数据曾泄漏到真实 `~/.hermes/state.db`，并破坏其模式。修复方案是**测试隔离守卫**：在 pytest 环境下检测 state.db 路径是否落在 production root，是则直接硬失败而不是污染真实数据。即使在 subprocess 里（未导入 conftest），`PYTEST_CURRENT_TEST` 和 `PYTEST_VERSION` 环境变量也能让守卫生效。

### 5.2 项目数据库（per-Profile 隔离）

项目数据库（projects.db）严格 per-Profile——与会话数据库、配置、定时任务一致。Profile 间的项目数据完全隔离。

### 5.3 看板数据库（Root-anchored 共享）

看板数据库（kanban.db）**故意共享**——所有 profile 看到同一个看板。设计意图是：profile 切换不能 fork 看板，否则调度器和工作线程之间的交接会断。

多个看板可以独立存在：`~/.hermes/kanban/boards/<slug>/` 各自有独立的 `kanban.db`、`workspaces/`、`logs/`。可通过 `HERMES_KANBAN_DB` 环境变量覆盖默认数据库路径。

看板工具本身有 profile-scoped 限制：被委派的子代理不能修改看板任务——必须返回发现给父代理，由父代理决定如何处理。

### 5.4 配对数据（per-Profile 隔离）

平台配对数据（已批准用户列表）按 profile 隔离。多路复用场景下，每个 profile 有独立的配对存储——profile A 的批准不会泄漏到 profile B。

---

## 6. 操作系统级隔离

当工作区隔离和凭证隔离都不够用时——比如要运行不受信任的代码、要限制网络访问、要隔离进程能力——就需要操作系统级隔离了。

### 6.1 终端后端的多种执行环境

终端工具支持多种执行后端，本地（默认）是直接执行 shell 命令。需要隔离时，可以切换到远程后端：

- **Docker**：在容器内执行命令
- **Singularity**：科学计算常用的容器运行时
- **Modal**：云端 serverless 容器
- **Daytona**：开发环境云平台
- **SSH**：远程主机
- **Vercel Sandbox**：云端沙箱

切换后端后，**所有文件工具**（读、写、修改）也通过 shell 契约实现，所以它们也就无法访问后端未暴露的路径。沙箱不仅隔离命令执行，还隔离文件访问。

某些强化学习环境（TerminalBench2、HermesSweEnv）会要求每个任务独立 Docker/Modal 镜像。系统支持任务级别的镜像覆盖：每个任务 ID 可以独立指定镜像，未指定时折叠为 "default" 共享同一容器。

### 6.2 Docker 容器的硬化

Docker 后端默认应用一系列安全参数：

- **能力丢弃**：默认 `--cap-drop ALL`，只保留代理必需的最小能力集（`DAC_OVERRIDE`、`CHOWN`、`FOWNER`）
- **禁止提权**：`--security-opt no-new-privileges`，禁止 SUID 提权
- **临时文件系统限制**：`/tmp` 限制为 512M、`/var/tmp` 限制为 256M 且禁止 SUID
- **进程数限制**：`--pids-limit 256`，防止 fork 炸弹
- **CPU 和内存限制**：通过 cgroup 控制器动态应用
- **网络隔离**：`--network=none` 默认完全隔离网络

每个 session 是否独立容器可配置——persistent 模式下多 session 共享容器，否则每个 session 分配独立容器。

### 6.3 网络出口隔离

即使有容器隔离，代理依然可能通过 HTTP 请求泄漏数据到任意主机。更严格的部署会用双 Docker 网络：

- **内部网络**：无默认路由、无互联网连接，仅内部服务可达
- **出口网络**：有互联网连接，但仅允许白名单上的目的主机

代理、面板、网关都跑在内部网络；一个独立的"出口代理"（基于 Squid 或 Envoy）跑在出口网络，仅允许 HTTPS CONNECT 到白名单主机（如 `api.openai.com`、`api.anthropic.com`、`api.telegram.org`）。代理通过标准的 `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` 环境变量走出口代理。

威胁驱动是：即便代理在容器内被提示注入攻击，它也无法连接到白名单之外的目的地。

### 6.4 整进程包裹

最高强度的隔离是"整进程包裹"——把整个 Hermes 进程跑在一个专门的沙箱里：

- **Hermes 自带的 Docker Compose**：轻量级，适合自托管部署
- **NVIDIA OpenShell**：per-session 沙箱，文件系统、网络、进程 syscall、推理路由四层都可热重载；凭证从 Provider 存储注入，永不进沙箱文件系统

---

## 7. URL / SSRF 隔离

即使有网络出口隔离，应用层还要防 SSRF（Server-Side Request Forgery）——代理被注入恶意 URL 后访问内部服务。

### 7.1 永远阻止的 IP 范围

URL 安全模块维护一份"永远阻止"清单，任何允许配置都覆盖不了：

- **云元数据 IP**：`169.254.169.254`（AWS/GCP/Azure）、`169.254.170.2`（ECS 任务元数据）、`169.254.169.253`（Azure IMDS）、`100.100.100.200`（阿里云）
- **云元数据 IPv6**：`fd00:ec2::254` 及其 IPv4-mapped 变体
- **整个 link-local 范围**：`169.254.0.0/16`
- **云元数据主机名**：`metadata.google.internal`、`metadata.goog`
- **私有 IP、loopback IP、保留 IP、多播 IP、未指定 IP**
- **CGNAT 范围**：`100.64.0.0/10`（RFC 6598，涵盖 Tailscale、WireGuard 等）

### 7.2 DNS Rebinding 攻击的防御

经典 SSRF 防护容易被 DNS rebinding 绕过：用户在请求时解析的 IP 和 TCP 连接时实际连接的 IP 不同。修复方案是**连接时重新解析 IP**：

- HTTP 客户端在建立 TCP 连接前再次解析目标 IP
- 如果解析结果与之前不同，连接被拒绝
- 即使有重定向，每个 redirect 目标也都重新校验

### 7.3 跨 Profile 解析保证

URL 解析的允许配置（`security.allow_private_urls`）也按 profile 隔离——多路复用模式下切换 profile 时，每次都重新解析配置，避免进程级缓存导致跨 profile 配置泄漏。

---

## 8. 平台和通道授权

网关是所有外部消息的入口——Telegram、Discord、Slack、Signal、终端界面等。通道授权是"谁能指挥代理"的第一道防线。

### 8.1 配对机制

对于不支持强大的内置访问控制的平台（如个人 Telegram 机器人），系统提供**配对机制**：用户首次发送消息时收到一个配对码，操作员通过 `hermes gateway pairing approve <code>` 批准。批准后该用户永久可访问。

配对机制有速率限制和锁定——防止暴力破解配对码。配对状态按 profile 隔离存储，每个 profile 有独立的 pending、approved、速率限制数据。

### 8.2 多层 Allowlist 协同

平台用户授权由多层 allowlist 联合把关：

- **平台特定允许所有**：`TELEGRAM_ALLOW_ALL_USERS=true`、`DISCORD_ALLOW_ALL_USERS=true` 等
- **平台特定用户列表**：`TELEGRAM_ALLOWED_USERS=123456,789012`（逗号分隔的用户 ID 列表）
- **群组允许**：`TELEGRAM_GROUP_ALLOWED_USERS`（发送者）、`TELEGRAM_GROUP_ALLOWED_CHATS`（群组 ID）
- **机器人允许**：`{PLATFORM}_ALLOW_BOTS=mentions/all`（仅允许被提及的机器人）
- **配对存储**：已配对用户自动授权（不需要在 allowlist 中）
- **全局用户列表**：`GATEWAY_ALLOWED_USERS`
- **全局允许所有**：`GATEWAY_ALLOW_ALL_USERS=true`
- **角色授权**：Discord 适配器支持"持有特定角色的用户自动授权"
- **信任上游**：Relay 适配器允许 Team Gateway connector 完成 owner-only 权限绑定

任何一层生效即视为授权。设计意图是支持"明确的小白名单"和"完全开放"两种简单场景，同时提供"配对流程"作为中间过渡。

### 8.3 适配器自身策略

某些平台（如 WeCom、Weixin、Yuanbao、QQBot、WhatsApp）有自己的访问控制机制——`dm_policy`、`group_policy`、`allow_from` 等。这些适配器在内部就完成授权检查，消息不会到达网关除非通过。

网关对这类适配器有限信任——只在"有效策略实际为 allowlist"时才信任其授权结果，避免 fail-open 漏洞。

### 8.4 Profile 路由

单个网关实例可以服务多个独立 profile，根据消息来源（platform、guild、chat、thread）路由到对应 profile。匹配优先级：

1. 精确 thread（platform + chat_id + thread_id）
2. channel（platform + chat_id）
3. server（platform + guild_id）
4. 默认 profile

这意味着一个 Telegram 群组可以路由到 "coder" profile，另一个 Telegram 群组路由到 "writer" profile——用户在自己的频道里使用自己的工作环境，完全隔离。

### 8.5 启动时合规检查

网关启动时执行严格的合规检查：

- 平台的 `dm_policy`/`group_policy` 为 `open` 时，必须有 `GATEWAY_ALLOW_ALL_USERS` 或对应平台特定 flag
- 否则启动失败，记录到 `gateway_state.json: startup_failed`

这条规则强制实现"如果有公开网络接口，就必须有 allowlist"的硬性要求。

---

## 9. Subagent 隔离

代理可以委派任务给子代理（subagent）。子代理隔离是另一个重要的边界——父代理不应该被恶意子代理拖垮，反之亦然。

### 9.1 Git Worktree 隔离

每个被委派的子代理可以分配独立的 git worktree：从父代理当前 commit 分支出，路径在 `<repo>/.worktrees/subagent-<id>`，分支命名 `hermes-subagent/<id>`。父代理可以审查子代理的改动并合并。

这个机制保护核心仓库不被子代理意外破坏——子代理在自己的 worktree 里操作，父代理可以挑选值得合并的改动。空 worktree（无改动）自动清理。

### 9.2 迭代预算

每个代理（父或子）有独立的迭代预算：父代理默认 500 次，子代理默认 50 次。迭代预算防止无限循环——单次会话的 API 调用次数有上限。

子代理自动拒绝危险命令（默认行为），避免子代理在父代理的 TUI 上调用 `input()` 制造死锁。子代理可以 opt-in YOLO 模式（`delegation.subagent_auto_approve=true`），但仅推荐用于定时任务和批处理——不允许交互的子代理。

### 9.3 委派上下文

子代理运行在一个特殊的委派上下文里——通过 ContextVar 标记 `delegated_child_context`，让框架知道当前执行的是被委派的任务。

这个标记有两个作用：

第一，**Kanban 权限隔离**。子代理不能修改看板任务——只能收集发现并返回给父代理。Kanban 环境变量在子代理启动时会被显式清理（仅保留必要的子代理身份标记），去掉 dispatcher-only 键。

第二，**身份判断统一入口**。所有 `HERMES_KANBAN_*` 身份判断都通过一个统一的 predicate 函数 `is_dispatcher_owned_worker_context()`，避免在多个地方重复实现判定逻辑。

### 9.4 运行时工作目录

每个会话有自己的逻辑工作目录（cwd），通过 ContextVar 存储。这让不同的会话可以有不同的 cwd，而不会互相干扰。

工作目录解析有"包根屏蔽"——避免 fallback 注入到 Hermes 包根目录（因为 Hermes 包根的源码不应该被作为代理的工作目录）。

---

## 10. Plugin / Tool 隔离

插件是 Hermes Agent 的扩展机制，但"插件一旦加载就与代理同权限"——这是必须明确的边界。

### 10.1 Capability 同意系统

为了控制插件的能力边界（虽然不能真正限制恶意插件），系统提供 capability 同意系统：

每种"特权操作"对应一个 capability ID：

- `tools.override`：覆盖内置工具
- `llm.provider_override`：覆盖 LLM 提供者
- `llm.model_override`：覆盖 LLM 模型
- `llm.profile_override`：覆盖 profile
- `gateway.platform_actions`：平台网关操作

插件首次声明这些 capability 时，用户必须**显式同意**。同意状态被持久化，包含：

- 声明的 capability 集合的哈希（让用户知道当时同意了什么）
- 同意时间戳
- 当前授权的 capability 集合

如果插件更新时声明的 capability 集合变化（哈希变化），用户必须重新同意。这给了用户一个"插件新增能力时会被询问"的保障。

需要再次强调：**这个系统不是沙箱**。恶意插件可以导入任何模块、monkey-patch 核心、绕过 capability 检查。真正能阻止恶意插件的是"安装前人工审查"+ 沙箱隔离。

### 10.2 Skill Guard——技能安装时的扫描

从外部安装技能（skill）时有严格的扫描流程：

**信任分级**：
- `builtin`：Hermes 自带技能，永不扫描
- `trusted`：`openai/skills`、`anthropics/skills`、`huggingface/skills`、`NVIDIA/skills` 严格匹配
- `community`：其他社区技能
- `agent-created`：代理自己创建的技能

**扫描检测**：
- 威胁模式（外泄、提示注入、破坏性、持久化、网络混淆、执行、遍历、挖矿、供应链、特权升级、代理配置持久化、硬编码密钥、上下文外泄）
- 不可见字符（零宽空格、RTL 覆盖等）
- 结构性限制（文件数 50、总大小 1MB、单个文件 256KB、无二进制）

**裁决**：
- `safe`：无严重/高危发现
- `caution`：有高危发现
- `dangerous`：有严重发现

**安装策略**（基于 trust + verdict）：
- builtin 全允许
- trusted caution 允许、dangerous 阻止
- community caution 阻止、dangerous 阻止
- agent-created dangerous 询问用户

强制参数 `--force` 不能覆盖 community/trusted 的 dangerous 裁决。

### 10.3 MCP 服务器安全验证

MCP（Model Context Protocol）服务器是另一种扩展机制。系统对 MCP 服务器配置做安全验证：

- IOC 黑名单（已知的恶意 SSH 公钥、源 IP）
- 出口模式检测（是否尝试外泄）
- 持久化模式检测（是否尝试自我持久化）

### 10.4 工具集边界

工具按"工具集"分组管理。内置工具集（`core`、`hermes-gateway`、`hermes-cli`）固定；插件工具集（`hermes-<platform>`）自动生成。

某些工具集依赖 core 工具集——禁用某工具集时，`bundle_non_core_tools` 机制只删除非 core 的 delta，避免误删核心工具。

---

## 11. 审批门与输出脱敏

这两层是"启发式防御"——不能阻止操作，但能给用户清晰的风险提示。

### 11.1 危险命令模式识别

终端工具内置一份"危险命令"模式清单，覆盖约 50 种常见破坏性模式：

- `rm -rf`、`chmod 777`、`chown -R root`
- `mkfs`、`dd if=`
- `SQL DROP`、`SQL DELETE FROM`
- `PowerShell -EncodedCommand`（常用于绕过审计）
- `curl | sh`、`wget | bash`（下载即执行）
- SSH 密钥读取
- Windows 特化：`taskkill /F`、`vssadmin delete shadows`、`reg delete`、`cipher /w`、`format.com D:`、`bcdedit /set`、`Stop-Service -Force`

检测到危险模式时，根据审批模式（`manual`/`smart`/`off`）采取不同行为：

- `manual`：总是询问用户
- `smart`：用辅助 LLM 决策是否需要询问
- `off`：不询问（YOLO 模式，但有冻结机制防止中途通过环境变量绕过）

审批模式是 **profile-scoped** 的持久设置，修改立即生效。

### 11.2 Tirith 预执行扫描

除了内置的危险模式识别，系统还集成 Tirith（一个外部预执行安全扫描工具）：

- 扫描命令的内容级威胁（同形 URL、管道到解释器、终端注入）
- 退出码 0=允许、1=阻止、2=警告
- JSON 输出丰富检测结果但不覆盖裁决
- 操作失败时遵循 `fail_open` 配置

Tirith 二进制自动安装到 `$HERMES_HOME/bin/tirith`，SHA-256 校验。Cosign 可用时验证 GitHub Actions workflow 签名。Circuit breaker 机制防止连续失败导致挂起。

### 11.3 审批 YOLO 模式的冻结

YOLO 模式（审批全关）容易通过运行时设置环境变量绕过——恶意插件可以在任何时候修改 `os.environ` 让审批失效。修复方案是**导入时冻结**：

系统在启动时解析审批模式配置并冻结，后续的运行时修改不影响。这个机制也在导入时完成，恶意插件无法在 Hermes 启动后修改。

### 11.4 审批状态的 ContextVar

同样的考虑适用于审批会话状态。使用 ContextVar 存储"哪个会话、哪个 turn、哪个工具调用"作为审计关联键，避免 `os.environ` 在并发 ACP session 共享 ThreadPoolExecutor 时的 race condition（这个 race 历史上曾导致 fail-open 漏洞）。

### 11.5 输出脱敏

输出里有敏感信息是另一个问题。日志或终端输出可能无意间包含 API 密钥、token、密码。系统提供脱敏机制：

- **已知前缀正则**：`sk-*`、`ghp_*`、`xoxb-*`、`AKIA*`、`sk_live_*`、`SG.*`、`hf_*`、`r8_*`、`npm_*`、`pypi-*`、`tvly-*`、`exa_*`、`gsk_*`、`ntn_*`、`fw-*`、`gAAAA` 等
- **敏感查询参数**：`access_token`、`api_key`、`jwt`、`password`、`session_id`、`signature`、`x_amz_signature` 等
- **敏感正文键**：精确匹配（不是子串），避免 `token_count` 误判
- **环境变量赋值**：`KEY=value` 形式
- **短 token 全部 mask**（< 18 字符）、**长 token 保留前 6 后 4**

脱敏默认开启（`HERMES_REDACT_SECRETS=true`），可在配置中关闭。同样在导入时冻结——运行时修改无效。

### 11.6 自仓库守卫

代理有一个特殊的守卫：阻止改写本地 Hermes 检出背后的解释器。这防止出现"混合模块版本"——不同的 import 路径加载不同版本的同一个模块。

远程后端自然接触不到本地检出，所以无需检查。

---

## 12. 审计与可观测性

隔离不是孤岛，审计与可观测性让操作可追溯。

### 12.1 供应链审计

系统提供 on-demand 供应链审计，扫描三个目标：

- **Hermes venv**（每个 PyPI dist）
- **插件依赖**（requirements.txt + pyproject.toml 的最佳努力版本固定）
- **MCP 服务器**（npx -y pkg@ver, uvx pkg==ver）

查询 OSV.dev（开源漏洞数据库）发现已知漏洞。这不是日常自动执行，而是按需触发。

### 12.2 已知漏洞检测

系统维护一份**已知被攻陷的包**目录（如 Shai-Hulud 蠕虫攻击的 mini Shai-Hulud、mistralai 2.4.6）。三个触发点：

- `hermes doctor` 命令
- CLI 启动横幅
- 网关启动

用户可以通过 `hermes doctor --ack <id>` 确认已知漏洞，持久化到配置中。

### 12.3 启动合规检查

网关启动时执行严格检查：

- 对外暴露的适配器必须有 allowlist
- 配置缺失时启动失败
- 失败信息写入 `gateway_state.json`

启动失败不是警告，是硬性失败——避免"忘记配置就上线"的事故。

---

## 13. 纵深防御的协同示例

各个隔离机制不是独立的——它们协同工作形成纵深防御。

### 13.1 多路复用网关同时服务两个 Profile

设想一个多路复用网关同时运行两个 profile（"coder" 和 "writer"）：

网关启动时加载 default profile（进程级）。一条来自 Telegram 的消息到达，触发以下流程：

1. 网关检查 `TELEGRAM_ALLOWED_USERS`——用户是否在允许列表中
2. 检查 `match_profile_route`——决定该消息路由到哪个 profile（假设是 "coder"）
3. `set_hermes_home_override(<coder_home>)` 安装 Context-local 覆盖
4. `set_secret_scope(build_profile_secret_scope(<coder_home>))` 安装 coder 的凭证 scope
5. `get_plugin_manager()` 返回 coder profile 的插件管理器
6. 工具注册表返回 coder 自己的工具集合
7. 平台注册表返回 coder 自己的平台适配器
8. State DB 指向 `<coder_home>/state.db`
9. Kanban DB 解析到根目录（所有 profile 共享看板）

接着另一条来自 Discord 的消息到达，路由到 "writer" profile——同样的流程，但是为 writer 独立设置。两者的状态完全隔离。

如果 coder 的处理 spawn 一个子代理：

1. 子代理在自己的 git worktree 里操作
2. 委派上下文标记为 `delegated_child_context`
3. Kanban 环境变量被清理（不能改看板）
4. 子代理试图读 `<home>/.env`，被文件安全 deny 列表拦截
5. 子代理试图跨 profile 写，被警告并要求 `cross_profile=True` 显式确认

### 13.2 Profile 创建流程

管理员执行 `hermes profile create coder --clone` 时：

1. 创建 `<root>/profiles/coder/` 目录
2. Bootstrap 子目录（memories、sessions、skills、plugins、cron、logs、workspace、home）
3. 克隆配置文件、`.env`、`SOUL.md`、技能
4. `.env` 文件 chmod 为 `0o600`
5. 在 managed scope 上下文中迁移配置 schema

整个流程在一个事务里完成，失败时回滚。

### 13.3 跨 Profile 技能安装

`hermes skills install anthropics/skills/some-skill -p coder` 时：

1. 启动新子进程（fresh import），显式传递 `-p coder`
2. 子进程中 `SKILLS_DIR` 模块全局在 coder profile 下解析
3. 下载技能到隔离区
4. 调用 `scan_skill_cached` 扫描（基于 bundle hash + 来源 + 扫描器版本缓存）
5. `should_allow_install` 裁决（trusted 来源 + safe 评分 → 允许）
6. 写入 `<coder_home>/skills/`
7. 追加审计日志到 coder profile 的审计日志

因为 `SKILLS_DIR` 在 import time 绑定，coder profile 的安装不会污染 default profile。

---

## 14. 核心设计原则

**第一，OS 边界是真边界，其余都是启发式**。容器沙箱是隔离的根本；进程内的检查都给攻击者留了路。生产部署必须激活容器隔离。

**第二，工作区隔离围绕 HERMES_HOME**。所有动态注册的资源都按 HERMES_HOME 键控，Profile 切换通过 ContextVar 隔离。这个机制是整个体系的核心。

**第三，凭证隔离用 Secret Scope 替代环境变量**。多路复用模式严格 fail-closed，确保凭证绝不跨 profile 泄漏。

**第四，授权必须跨信任边界**。任何对外暴露的接口都必须有 allowlist，"默认放行"是设计缺陷。启动时强制检查。

**第五，失败时关闭**。所有配置缺失、检测失败、解析异常都默认拒绝，放行是例外。

**第六，进程内状态用 ContextVar 不用环境变量**。并发场景下环境变量 race 会导致 fail-open，ContextVar 隔离 task 状态。

**第七，敏感决策在导入时冻结**。审批模式、脱敏开关、YOLO 模式等都在导入时确定，运行时修改无效，防止恶意插件绕过。

**第八，能力与审计并存**。每个允许的操作都有审计记录；每个拒绝的操作都有明确原因。审计不是事后追溯，是设计决策的一部分。

**第九，妥协是设计意图**。进程内检查不是真边界，所以用户必须理解："如果你运行不信任的第三方插件，那它能做任何 Hermes 能做的事"。沙箱是唯一的硬边界。

简单说：**隔离设计的本质是"用多层启发式去模拟一个真边界"**——OS 隔离是底层，profile 隔离是中间层，权限审批和审计是上层。三层叠加让攻击者即使逃过一层也会被下一层拦下来。
