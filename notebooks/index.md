---
hide:
  - navigation
  - toc
---

<style>
  .md-main__inner {
    max-width: min(1320px, calc(100vw - 72px));
    margin-left: auto;
    margin-right: auto;
  }

  .md-content {
    max-width: none;
  }

  .md-content__inner {
    max-width: none;
    margin: 0;
    padding-top: 12px;
  }

  .md-content__inner::before {
    display: none;
  }

  .knowledge-graph-frame {
    width: 100%;
    height: min(740px, calc(100vh - 140px));
    min-height: 560px;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
    margin: 0 0 40px;
    background: #f8fafc;
  }

  @media (max-width: 760px) {
    .md-main__inner {
      max-width: calc(100vw - 20px);
    }

    .knowledge-graph-frame {
      min-height: 560px;
      height: calc(100vh - 120px);
      border-radius: 10px;
    }
  }
</style>

<iframe
  class="knowledge-graph-frame"
  src=".assets/knowledge_graph.html"
  loading="lazy"
  title="LLM 笔记知识图谱"
></iframe>

<span id="kg-home"></span>

# 我的LLM笔记

这是一个用于自我学习的 LLM / VLM 学习仓库。

它是一个 AI 与我共创的仓库，需要 AI 与我一同维护仓库。

AI 的行动准则参考 [AGENT.md](https://github.com/Sui-Xing/llm_learning/blob/main/AGENT.md)。

<span id="kg-start"></span>

## 项目启动

```shell
mkdocs serve -a 127.0.0.1:6033
```

<span id="kg-nav"></span>

## 笔记导航

<span id="kg-llm"></span>

### 🚀 LLM 大语言模型

<span id="kg-llm-reports"></span>

#### 📘 LLM 技术报告

- [AR-Diffusion 与潜空间推理路线综述](llm/llm技术报告/AR-Diffusion与潜空间推理路线综述.md)
- [Claude Mythos 5 技术报告](llm/llm技术报告/Claude%20Mythos%205技术报告.md)
- [DeepSeek-V4 技术报告](llm/llm技术报告/DeepSeek-V4技术报告.md)
- [Nemotron-Labs-Diffusion 技术报告](llm/llm技术报告/Nemotron-Labs-Diffusion技术报告.md)
- [Qwen3.5 技术报告](llm/llm技术报告/qwen3.5.md)

<span id="kg-llm-architecture"></span>

#### 🏗️ 模型架构

- [mHC 残差流形](llm/模型架构/mHC_残差流形.md)
- [流形](llm/模型架构/流形.md)

**优化器**

- [Muon 优化器](llm/模型架构/优化器/Muon优化器.md)

**位置编码**

- [RoPe](llm/模型架构/位置编码/RoPe.md)

**归一化**

- [Pre-Norm vs Post-Norm](llm/模型架构/归一化/Pre-Norm%20vs%20Post-Norm.md)

**注意力机制**

- [DSA](llm/模型架构/注意力机制/DSA.md)
- [Gated Attention](llm/模型架构/注意力机制/GatedAttention.md)
- [Linear Attention](llm/模型架构/注意力机制/LinearAttention.md)
- [MLA](llm/模型架构/注意力机制/MLA.md)

<span id="kg-llm-training"></span>

#### 🎯 模型训练

**SFT**

- [LoRA](llm/模型训练/SFT/LoRA.md)

**强化学习**

- [DPO](llm/模型训练/强化学习/DPO.md)
- [GSPO](llm/模型训练/强化学习/GSPO.md)
- [PPO](llm/模型训练/强化学习/PPO.md)
- [总结](llm/模型训练/强化学习/总结.md)
- [李宏毅 DRL 笔记](llm/模型训练/强化学习/李宏毅DRL笔记.md)

<span id="kg-llm-source"></span>

#### 💻 源码解读

- [ChatGPT](llm/源码解读/ChatGPT.md)
- [Einsum](llm/源码解读/Einsum.md)
- [MOE](llm/源码解读/MOE.md)
- [核心要义](llm/源码解读/核心要义.md)

<span id="kg-llm-app"></span>

#### 🛠️ LLM 应用开发

_（整理中）_

<span id="kg-cv"></span>

### 👁️ 计算机视觉（CV / VLM）

<span id="kg-cv-basic"></span>

#### 📐 基础

- [Reparameterization Trick](computer_vision/基础/Reparameterization%20Trick.md)
- [SD 公式推导](computer_vision/基础/SD公式推导.md)

<span id="kg-cv-architecture"></span>

#### 🏗️ 模型架构

- [JEPA](computer_vision/模型架构/JEPA.md)
- [JiT](computer_vision/模型架构/JiT.md)
- [SAM](computer_vision/模型架构/SAM.md)
- [VAE](computer_vision/模型架构/VAE.md)

<span id="kg-performance"></span>

### ⚡ 性能优化

- [分布式](性能优化/分布式.md)
- [推理加速方法总结](性能优化/推理加速方法总结.md)
- [Deepspeed](性能优化/Deepspeed.md)
- [Flash-Attention](性能优化/Flash-Attention.md)
- [MOE 负载均衡](性能优化/MOE负载均衡.md)
- [SGLang](性能优化/SGLang.md)
- [vLLM](性能优化/vLLM.md)

<span id="kg-other"></span>

### 📚 其他

- [知识图谱](knowledge_graph.md)
