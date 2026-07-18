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

  /* 隐藏 MkDocs 右上角自带搜索 */
  .md-search, .md-header__source {
      display: none !important;
  }
</style>

<iframe
  class="knowledge-graph-frame"
  src=".assets/knowledge_graph.html"
  loading="lazy"
  title="LLM 笔记知识图谱"
></iframe>

<span id="kg-home"></span>

# Ranhao的LLM知识库

专注LLM/VLM算法，跟踪LLM最新技术

<span id="kg-start"></span>

<span id="kg-nav"></span>

## 笔记导航

<!-- AUTO-GENERATED-NOTE-NAV:START -->

<span id="kg-llm"></span>

### 🚀 LLM 大语言模型

<span id="kg-llm-开源llm解读"></span>

#### 开源llm解读

- [AR-Diffusion与潜空间推理](llm/开源llm解读/AR-Diffusion与潜空间推理.md)
- [Claude Mythos 5推测](llm/开源llm解读/Claude%20Mythos%205推测.md)
- [DeepSeek-V4](llm/开源llm解读/DeepSeek-V4.md)
- [Nemotron-Labs-Diffusion](llm/开源llm解读/Nemotron-Labs-Diffusion.md)
- [OpenMythos Recurrent](llm/开源llm解读/OpenMythos%20Recurrent.md)
- [qwen3.5](llm/开源llm解读/qwen3.5.md)

<span id="kg-llm-architecture"></span>

#### 🏗️ 模型架构

<span id="kg-llm-模型架构-muon优化器"></span>

##### Muon优化器

- [Muon优化器](llm/模型架构/Muon优化器/Muon优化器.md)
- [奇异值、特征值和正交矩阵](llm/模型架构/Muon优化器/奇异值、特征值和正交矩阵.md)

<span id="kg-llm-architecture-position"></span>

##### 位置编码

- [RoPe](llm/模型架构/位置编码/RoPe.md)

<span id="kg-llm-模型架构-残差连接"></span>

##### 残差连接

- [attention residuals](llm/模型架构/残差连接/attention_residuals.md)
- [mHC 残差流形](llm/模型架构/残差连接/mHC_残差流形.md)
- [Pre-Norm vs Post-Norm](llm/模型架构/残差连接/Pre-Norm%20vs%20Post-Norm.md)
- [流形比较](llm/模型架构/残差连接/流形比较.md)

<span id="kg-llm-architecture-attention"></span>

##### 注意力机制

- [DSA](llm/模型架构/注意力机制/DSA.md)
- [GatedAttention](llm/模型架构/注意力机制/GatedAttention.md)
- [LinearAttention](llm/模型架构/注意力机制/LinearAttention.md)
- [MLA](llm/模型架构/注意力机制/MLA.md)

<span id="kg-llm-training"></span>

#### 🎯 模型训练

<span id="kg-llm-training-sft"></span>

##### SFT

- [LoRA](llm/模型训练/SFT/LoRA.md)

<span id="kg-llm-training-rl"></span>

##### 强化学习

- [ClipPPO原理](llm/模型训练/强化学习/ClipPPO原理.md)
- [DPO](llm/模型训练/强化学习/DPO.md)
- [GSPO](llm/模型训练/强化学习/GSPO.md)
- [PPO](llm/模型训练/强化学习/PPO.md)
- [RL知识图谱](llm/模型训练/强化学习/RL知识图谱.md)
- [李宏毅DRL笔记](llm/模型训练/强化学习/李宏毅DRL笔记.md)

<span id="kg-llm-source"></span>

#### 💻 源码解读

- [ChatGPT](llm/源码解读/ChatGPT.md)
- [Einsum](llm/源码解读/Einsum.md)
- [MOE](llm/源码解读/MOE.md)
- [核心要义](llm/源码解读/核心要义.md)

<span id="kg-llm-源码解读-openmythos"></span>

##### OpenMythos

- [ACT](llm/源码解读/OpenMythos/ACT.md)
- [LTI Injection](llm/源码解读/OpenMythos/LTI%20Injection.md)
- [OpenMythos框架](llm/源码解读/OpenMythos/OpenMythos框架.md)

<span id="kg-llm-app"></span>

### 🛠️ LLM 应用开发

<span id="kg-llm应用开发-agent"></span>

#### Agent

- [harness+agent引入](llm应用开发/Agent/harness+agent引入.md)

<span id="kg-cv"></span>

### 👁️ 计算机视觉（CV / VLM）

<span id="kg-cv-basic"></span>

#### 📐 基础

- [Reparameterization Trick](computer_vision/基础/Reparameterization%20Trick.md)
- [SD公式推导](computer_vision/基础/SD公式推导.md)

<span id="kg-cv-architecture"></span>

#### 🏗️ 模型架构

- [JEPA](computer_vision/模型架构/JEPA.md)
- [JiT](computer_vision/模型架构/JiT.md)
- [SAM](computer_vision/模型架构/SAM.md)
- [VAE](computer_vision/模型架构/VAE.md)

<span id="kg-performance"></span>

### ⚡ 性能优化

- [Deepspeed](性能优化/Deepspeed.md)
- [Flash-Attention](性能优化/Flash-Attention.md)
- [MOE负载均衡](性能优化/MOE负载均衡.md)
- [SGLang](性能优化/SGLang.md)
- [vLLM](性能优化/vLLM.md)
- [分布式](性能优化/分布式.md)
- [推理加速方法总结](性能优化/推理加速方法总结.md)

<span id="kg-code-practice"></span>

### 💻 代码实践

- [gpt](code_practice/gpt.ipynb)
- [gqa](code_practice/gqa.ipynb)
- [mHC](code_practice/mhc.ipynb)
- [moe](code_practice/moe.ipynb)
- [vit](code_practice/vit.ipynb)

<span id="kg-other"></span>

### 📚 其他

- [知识图谱](knowledge_graph.md)

<!-- AUTO-GENERATED-NOTE-NAV:END -->
