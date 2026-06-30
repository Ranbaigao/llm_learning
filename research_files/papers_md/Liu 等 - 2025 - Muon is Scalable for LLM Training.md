# MUON IS SCALABLE FOR LLM TRAINING

TECHNICAL REPORT

Jingyuan Liu $^{1}$ Jianlin Su $^{1}$ Xingcheng Yao $^{2}$ Zhejun Jiang $^{1}$ Guokun Lai $^{1}$ Yulun Du $^{1}$ Yidao Qin $^{1}$ Weixin Xu $^{1}$ Enzhe Lu $^{1}$ Junjie Yan $^{1}$ Yanru Chen $^{1}$ Huabin Zheng $^{1}$ Yibo Liu $^{1}$ Shaowei Liu $^{1}$ Bohong Yin $^{1}$ Weiran He $^{1}$ Han Zhu $^{1}$ Yuzhi Wang $^{1}$ Jianzhou Wang $^{1}$ Mengnan Dong $^{1}$ Zheng Zhang $^{1}$ Yongsheng Kang $^{1}$ Hao Zhang $^{1}$ Xinran Xu $^{1}$ Yutao Zhang $^{1}$ Yuxin Wu $^{1}$ Xinyu Zhou $^{1*}$ Zhilin Yang $^{1}$

$^{1}$ Moonshot AI $^{2}$ UCLA

## ABSTRACT

Recently, the Muon optimizer (K. Jordan et al. 2024) based on matrix orthogonalization has demonstrated strong results in training small-scale language models, but the scalability to larger models has not been proven. We identify two crucial techniques for scaling up Muon: (1) adding weight decay and (2) carefully adjusting the per-parameter update scale. These techniques allow Muon to work out-of-the-box on large-scale training without the need of hyper-parameter tuning. Scaling law experiments indicate that Muon achieves $\sim 2\times$ computational efficiency compared to AdamW with compute optimal training. Based on these improvements, we introduce Moonlight, a 3B/16B-parameter Mixture-of-Expert (MoE) model trained with 5.7T tokens using Muon. Our model improves the current Pareto frontier, achieving better performance with much fewer training FLOPs compared to prior models. We open-source our distributed Muon implementation that is memory optimal and communication efficient. We also release the pretrained, instruction-tuned, and intermediate checkpoints to support future research.

![](images/a4d7e5cf2bddf8791ec480e4a58f749ff5cde4554c7549bff468a7fc9082421d.jpg)  
(a)

![](images/ab30366f27461c6e125db819fa950353564173fc1ac83e4929ab60bd673aac29.jpg)  
(b)  
Figure 1: Scaling up with Muon. (a) Scaling law experiments comparing Muon and Adam. Muon is $\sim 2\times$ more computational efficient than Adam with compute optimal training. (b) The MMLU performance of our Moonlight model optimized with Muon and other comparable models. Moonlight advances the Pareto frontier of performance vs training FLOPs.

## 1 Introduction

The rapid advancement of large language models (LLMs) (OpenAI et al. 2024; DeepSeek-AI et al. 2024; Grattafiori et al. 2024; Gemini Team et al. 2024) has significantly pushed forward the progress in artificial general intelligence. However, training capable LLMs remains a computationally intensive and resource-demanding process due to scaling laws (Kaplan et al. 2020; Hoffmann et al. 2022). Optimizers play a crucial role in efficiently and effectively training of LLMs, with Adam (Kingma et al. 2015) and its variant AdamW (Loshchilov et al. 2019) being the standard choice for most large-scale training.

Recent developments in optimization algorithms have shown potential to improve training efficiency beyond AdamW (Liu et al. 2024; K. Jordan et al. 2024; Yuan et al. 2024; Vyas et al. 2025; X.-L. Li 2018a; X.-L. Li 2018b; Pooladzandi et al. 2024; X. Li 2022; X.-L. Li 2024; Pethick et al. 2025). Among these, K. Jordan et al. 2024 proposed Muon, which updates matrix parameters with orthogonalized gradient momentum using Newton-Schulz iteration. Initial experiments with Muon have demonstrated promising results in small-scale language model training. However, as discussed in this blog (K. Jordan et al. 2024), several critical challenges remain unaddressed: (1) how to effectively scale optimizers based on matrix orthogonalization to larger models with billions of parameters trained with trillions of tokens, (2) how to compute approximate orthogonalization in a distributed setting, and (3) whether such optimizers can generalize across different training stages including pre-training and supervised finetuning (SFT).

In this technical report, we present a comprehensive study addressing these challenges. Our work builds upon Muon while systematically identifying and resolving its limitations in large-scale training scenarios. Our technical contributions include:

\- Analysis for Effective Scaling of Muon: Through extensive analysis, we identify that weight decay plays a crucial role in Muon's scalability. Besides, we propose scale adjustments to Muon's parameter-wise update rule. Such adjustments allow Muon to work out-of-the-box without hyper-parameter tuning, and also significantly improve training stability.

\- Efficient Distributed Implementation: We develop a distributed version of Muon with ZeRO-1 (Rajbhandari et al. 2020) style optimization, achieving optimal memory efficiency and reduced communication overhead while preserving the mathematical properties of the algorithm.

\- Scaling Law Validation: We performed scaling law research that compares Muon with strong AdamW baselines, and showed the superior performance of Muon (1a). Based on the scaling law results, Muon achieves comparable performance to AdamW trained counterparts while requiring only approximately 52% of the training FLOPs.

Our comprehensive experiments demonstrate that Muon can effectively replace AdamW as the de facto optimizer for large-scale LLM training, offering significant improvements in both training efficiency and model performance. As a result of this work, we release Moonlight, a 16B-parameter MoE model trained using Muon, along with our implementation and intermediate training checkpoints to facilitate further research in scalable optimization techniques for LLMs.

## 2 Methods

## 2.1 Background

The Muon Optimizer Muon (K. Jordan et al. 2024) has recently been proposed to optimize neural network weights representable as matrices. At iteration t, given current weight $W_{t-1}$ , momentum $\mu$ , learning rate $\eta_{t}$ and objective $L_{t}$ , the update rule of the Muon optimizer can be stated as follows:

$$
\begin{array}{c} \mathbf {M} _ {t} = \mu \mathbf {M} _ {t - 1} + \nabla \mathcal {L} _ {t} (\mathbf {W} _ {t - 1}) \\ \mathbf {O} _ {t} = \text { Newton - Schulz } (\mathbf {M} _ {t}) ^ {1} \\ \mathbf {W} _ {t} = \mathbf {W} _ {t - 1} - \eta_ {t} \mathbf {O} _ {t} \end{array}\tag{1}
$$

Here, $M_{t}$ is the momentum of gradient at iteration t, set as a zero matrix when t = 0. In Equation 1, a Newton-Schulz iteration process (Bernstein et al. 2024) is adopted to approximately solve $(\mathbf{M}_{t}\mathbf{M}_{t}^{\mathrm{T}})^{-1/2}\mathbf{M}_{t}$ . Let $U\Sigma V^{T} = M_{t}$ be the singular value decomposition (SVD) of $M_{t}$ , we will have $(\mathbf{M}_{t}\mathbf{M}_{t}^{\mathrm{T}})^{-1/2}\mathbf{M}_{t} = UV^{T}$ , which orthogonalizes $M_{t}$ . Intuitively, orthogonalization can ensure that the update matrices are isomorphic, preventing the weight from learning along a few dominant directions (K. Jordan et al. 2024).

Newton-Schulz Iterations for Matrix Orthogonalization Equation 1 is calculated in an iterative process. At the beginning, we set $\mathbf{X}_0 = \mathbf{M}_t / \| \mathbf{M}_t\|_{\mathrm{F}}$ . Then, at each iteration $k$ , we update $\mathbf{X}_k$ from $\mathbf{X}_{k - 1}$ as follows:

$$
\mathbf {X} _ {k} = a \mathbf {X} _ {k - 1} + b (\mathbf {X} _ {k - 1} \mathbf {X} _ {k - 1} ^ {\mathrm{T}}) \mathbf {X} _ {k - 1} + c (\mathbf {X} _ {k - 1} \mathbf {X} _ {k - 1} ^ {\mathrm{T}}) ^ {2} \mathbf {X} _ {k - 1}\tag{2}
$$

where $X_{N}$ is the result of such process after N iteration steps. Here a, b, c are coefficients. In order to ensure the correct convergence of Equation 2, we need to tune the coefficients so that the polynomial $f(x) = ax + bx^{3} + cx^{5}$ has a fixed point near 1. In the original design of K. Jordan et al. 2024, the coefficients are set to a = 3.4445, b = -4.7750, c = 2.0315 in order to make the iterative process converge faster for small initial singular values. In this work, we follow the same setting of coefficients.

Steepest Descent Under Norm Constraints Bernstein et al. 2024 proposed to view the optimization process in deep learning as steepest descent under norm constraints. From this perspective, we can view the difference between Muon and Adam (Kingma et al. 2015; Loshchilov et al. 2019) as the difference in norm constraints. Whereas Adam is a steepest descent under the a norm constraint dynamically adjusted from a Max-of-Max norm, Muon offers a norm constraint that lies in a static range of Schatten-p norm for some large p (Franz 2024). When equation 1 is accurately computed, the norm constraint offered by Muon will be the spectral norm. Weights of neural networks are used as operators on the input space or the hidden space, which are usually (locally) Euclidean (Cesista 2024), so the norm constraint on weights should be an induced operator norm (or spectral norm for weight matrices). In this sense, the norm constraint offered by Muon is more reasonable than that offered by Adam.

## 2.2 Scaling Up Muon

Weight Decay While Muon performs significantly better than AdamW on a small scale as shown by K. Jordan et al. 2024, we found the performance gains diminish when we scale up to train a larger model with more tokens. We observed that both the weight and the layer output's RMS keep growing to a large scale, exceeding the high-precision range of bf16, which might hurt the model's performance. To resolve this issue, we introduced the standard AdamW (Loshchilov et al. 2019) weight decay mechanism into Muon $^{2}$ .

$$
\mathbf {W} _ {t} = \mathbf {W} _ {t - 1} - \eta_ {t} (\mathbf {O} _ {t} + \lambda \mathbf {W} _ {t - 1})\tag{3}
$$

We experimented on Muon both with and without weight decay to understand its impact on the training dynamics of LLMs. Based on our scaling law research in Sec 3.2, we trained an 800M parameters model with 100B tokens ( $\sim 5\times$ optimal training tokens). Figure 2 shows validation loss curves of the model trained with AdamW, vanilla Muon (without weight decay), and Muon with weight decay. While vanilla Muon initially converges faster, we observed that some model weights grew too large over time, potentially limiting the model's long-term performances. Adding weight decay addressed this issue - the results demonstrate that Muon with weight decay outperforms both vanilla Muon and AdamW, achieving lower validation loss in the over-train regime. Therefore, we adjusted our update rule to equation 3, where $\lambda$ is the weight decay ratio.

Consistent update RMS An important property of Adam and AdamW (Kingma et al. 2015, Loshchilov et al. 2019) is that they maintain a theoretical update RMS around $1^{3}$ . However, we show that Muon's update RMS varies depending on the shape of the parameters, according to the following lemma:

Lemma 1. For a full-rank matrix parameter of shape $[A, B]$ , its theoretical Muon update RMS is $\sqrt{1 / \max(A, B)}$ .

The proof can be found in the Appendix A. We monitored Muon's update RMS during training and found it typically close to the theoretical value given above. We note that such inconsistency can be problematic when scaling up the model size:

\- When $\max(A, B)$ is too large, e.g. the dense MLP matrix, the updates become too small, thus limiting the model's representational capacity and leading to suboptimal performances;

\- When $\max(A, B)$ is too small, e.g. treating each KV head in GQA (Shazeer 2019) or MLA (DeepSeek-AI et al. 2024) as a separate parameter, the updates become too large, thus causing training instabilities and leading to suboptimal performances as well.

![](images/be9525d3c54d796061178d9e8af8028cf0376d8be1846377a0e385bbc9baec7e.jpg)  
Figure 2: Validation loss curves for AdamW (green), Muon without weight decay (red), and Muon with weight decay (blue).

In order to maintain consistent update RMS among matrices of different shapes, we propose to scale the Muon update for each matrix by its $\sqrt{\max(A,B)}$ to cancel the effect of Lemma 1 $^{4}$ . Experiments in Sec 3.1 show that this strategy is beneficial for optimization.

Matching update RMS of AdamW Muon is designed to update matrix-based parameters. In practice, AdamW is used in couple with Muon to handle non-matrix based parameters, like RMSNorm, LM head, and embedding parameters. We would like the optimizer hyper-parameters (learning rate $\eta$ , weight decay $\lambda$ ) to be shared among matrix and non-matrix parameters.

We propose to match Muon's update RMS to be similar to that of AdamW. From empirical observations, AdamW's update RMS is usually around 0.2 to 0.4. Therefore, we scale Muon's update RMS to this range by the following adjustment:

$$
\mathbf {W} _ {t} = \mathbf {W} _ {t - 1} - \eta_ {t} (0. 2 \cdot \mathbf {O} _ {t} \cdot \sqrt {\max (A , B)} + \lambda \mathbf {W} _ {t - 1})\tag{4}
$$

We validated this choice with empirical results (see Appendix A for details). Moreover, we highlighted that with this adjustment, Muon can directly reuse the learning rate and weight decay tuned for AdamW.

Other Hyper-parameters Muon contains two other tunnable hyper-parameters: Newton-Schulz iteration steps and momentum $\mu$ . We empirically observe that when setting $N$ to 10, the iterative process will yield a more accurate orthogonalization result than $N = 5$ , but it won't lead to better performances. Hence we set $N = 5$ in this work for the sake of efficiency. We do not see a consistent performance gain in tuning momentum, so we chose 0.95, same as K. Jordan et al. 2024.

## 2.3 Distributed Muon

ZeRO-1 and Megatron-LM Rajbhandari et al. 2020 introduced the ZeRO-1 technique that partitions the expensive optimizer states (e.g. master weights, momentum) all over the cluster. Megatron-LM (Shoeybi et al. 2020) integrated ZeRO-1 into its native parallel designs. Based on Megatron-LM's sophisticated parallel strategies, e.g. Tensor-Parallel (TP), Pipeline Parallel (PP), Expert Parallel (EP) and Data Parallel (DP), the communication workload of ZeRO-1 can be reduced from gathering all over the distributed world to only gathering over the data parallel group.

Method ZeRO-1 is efficient for AdamW because it calculates updates in an element-wise fashion. However, Muon requires the full gradient matrix to calculate the updates. Therefore, vanilla ZeRO-1 is not directly applicable to Muon.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Distributed Muon

Require: Full Gradients G, DP partitioned Momentum m, DP partitioned parameters p, momentum  $\mu$ .

1: // Reduce-scatter G on DP for correct gradients
2: g = reduce_scatter(G, dp_group)
3: // Apply momentum to g using local partitioned momentum m
4:  $g' = update_with_momentum(g, m, \mu)$ 
5: // DP Gather: gathering  $g'$  across DP into a full matrix G
6:  $G = gather(g', dp_{group})$ 
7: // Calculate Muon update
8: U = Newton-Schulz(G)
9: // Discard the rest of U and only keep the local partition u, then apply the update rule
10:  $p' = apply\_update(p, u)$ 
11: // All-gather updated  $p'$  into P
12:  $P = all\_gather(p', dp_{group})$ 
13: // Return the update RMS for logging
14: return  $\sqrt{u^{2}.mean()}$
</div>

We propose a new distributed solution based on ZeRO-1 for Muon, referred to as Distributed Muon. Distributed Muon follows ZeRO-1 to partition the optimizer states on DP, and introduces two additional operations compared to a vanilla Zero-1 AdamW optimizer:

1. DP Gather. For a local DP partitioned master weight (1/DP the size of the model weight), this operation is to gather the corresponding partitioned gradients into a full gradient matrix.

2. Calculate Full Update. After the above gathering, perform Newton-Schulz iteration steps on the full gradient matrix as described in Sec 2.1. Note that we will then discard part of the full update matrix, as we only need the partition corresponding to the local parameters to perform update.

The implementation of Distributed Muon is described in Algorithm 1. The additional operations introduced by Distributed Muon are colored in blue.

Analysis We compared Distributed Muon to a classic ZeRO-1 based distributed AdamW (referred as Distributed AdamW for simplicity) in several aspects:

\- Memory Usage. Muon uses only one momentum buffer, while AdamW uses two momentum buffers. Therefore, the additional memory used by the Muon optimizer is half of Distributed AdamW.

\- Communication Overhead. For each device, the additional DP gathering is only required by the local DP partitioned parameters p. Therefore, the communication cost is less than the reduce-scatter of G or the all-gather of P. Besides, Muon only requires the Newton-Schulz iteration steps in bf16, thus further reducing the communication overhead to 50% comparing to fp32. Overall, the communication workload of Distributed Muon is $(1, 1.25]$ of that of Distributed AdamW. The upper-bound is calculated as that the communication of Distributed Muon is 4 (fp32 G reduce-scatter) + 2 (bf16 Muon gather) + 4 (fp32 P all-gather), while Distributed AdamW is $4 + 4$ . In practice, as we usually train with multiple DP, the empirical additional cost usually is closer to the lower-bound $1^{5}$ .

\- Latency. Distributed Muon has larger end-to-end latencies than Distributed AdamW because it introduces additional communication and requires running Newton-Schulz iteration steps. However, this is not a significant issue because (a) only about 5 Newton-Schultz iteration steps are needed for a good result (discussed in Sec 2.2), and (b) the end-to-end latency caused by the optimizer is negligible compared to the model's forward-backward pass time (e.g. usually $1\%$ to $3\%$ ). Moreover, several engineering techniques, such as overlapping gather and computation, and overlapping optimizer reduce-scatter with parameter gather, can further reduce latency.

When training large-scale models in our distributed cluster, Distributed Muon has no noticeable latency overhead compared to its AdamW counterparts. We will soon release a pull request that implements Distributed Muon for the open-source Megatron-LM (Shoeybi et al. 2020) project.

Table 1: Controlling Muon's Update RMS Across Different Model Params

<table><tr><td>Methods</td><td>Training loss</td><td>Validation loss</td><td>query weight RMS</td><td>MLP weight RMS</td></tr><tr><td>Baseline</td><td>2.734</td><td>2.812</td><td>3.586e-2</td><td>2.52e-2</td></tr><tr><td>Update Norm</td><td>2.72</td><td>2.789</td><td>4.918e-2</td><td>5.01e-2</td></tr><tr><td>Adjusted LR</td><td>2.721</td><td>2.789</td><td>3.496e-2</td><td>4.89e-2</td></tr></table>

## 3 Experiments

## 3.1 Consistent Update RMS

As discussed in Sec 2.2, we aim to match the update RMS across all matrix parameters and also match it with that of AdamW. We experimented with two methods to control the Muon update RMS among parameters and compared them to a baseline that only maintains a consistent RMS with AdamW:

1. Baseline. We multiplied the update matrix by $0.2 \cdot \sqrt{H}$ (H is the model hidden size) to maintain a consistent update RMS with AdamW. Note that $\max(A, B)$ equals to H for most matrices.

$$
\mathbf {W} _ {t} = \mathbf {W} _ {t - 1} - \eta_ {t} (0. 2 \cdot \mathbf {O} _ {t} \cdot \sqrt {H} + \lambda \mathbf {W} _ {t - 1})\tag{5}
$$

2. Update Norm. We can directly normalize the updates calculated via Newton-Schulz iterations so its RMS strictly becomes 0.2;

$$
\mathbf {W} _ {t} = \mathbf {W} _ {t - 1} - \eta_ {t} (0. 2 \cdot \mathbf {O} _ {t} / \operatorname{RMS} (\mathbf {O} _ {t}) + \lambda \mathbf {W} _ {t - 1})\tag{6}
$$

3. Adjusted LR. For each update matrix, we can scale its learning rate by a factor of $0.2 \cdot \sqrt{\max(A, B)}$ based on its shape.

$$
\mathbf {W} _ {t} = \mathbf {W} _ {t - 1} - \eta_ {t} (0. 2 \cdot \mathbf {O} _ {t} \cdot \sqrt {\max (A , B)} + \lambda \mathbf {W} _ {t - 1})\tag{7}
$$

Analysis We designed experiments to illustrate the impact of Muon update RMS at an early training stage, because we observed that unexpected behaviors happened very quickly when training models at larger scale. We experimented with small scale 800M models as described in 3.2. The problem of inconsistent update RMS is more pronounced when the disparity between matrix dimensions increases. To highlight the problem for further study, we slightly modify the model architecture by replacing the Swiglu MLP with a standard 2-layer MLP, changing the shape of its matrix parameters from $[H, 2.6H]$ to $[H, 4H]$ . We evaluated the model's loss and monitored a few of its parameters' RMS, specifically, attention query (shape $[H, H]$ ) and MLP (shape $[H, 4H]$ ). We evaluated the model after training for 4B tokens out of a 20B-token schedule. From Table 1, we observed several interesting findings:

1. Both Update Norm and Adjusted LR achieved better performances than Baseline;

2. For the MLP weight matrix of shape $[H, 4H]$ , both Update Norm and Adjusted LR obtain a weight RMS that is roughly doubled comparing to Baseline. This is reasonable as $\sqrt{\max(H, 4H)} / \sqrt{H} = 2$ , so the update RMS of Update Norm and Adjusted LR is roughly two times of Baseline;

3. For the attention query weight matrix of shape $[H, H]$ , Update Norm still norms the update, while Adjusted LR does not because $\sqrt{\max(H, H)}/\sqrt{H} = 1$ . As a result, Adjusted LR results in a similar weight RMS as Baseline, but Update Norm has a larger weight rms similar to its MLP.

Based on these findings, we choose the Adjusted LR method for future experiments because it has lower cost.

## 3.2 Scaling Law of Muon

For a fair comparison with AdamW, we performed scaling law experiments on a series of dense models in Llama (Grattafiori et al. 2024) architecture. Building a strong baseline is of crucial importance in optimizer research. Hence, we perform a grid search for hyper-parameters of AdamW, following the compute-optimal training setup (Kaplan et al. 2020) (the grid search experiments can be found in Appendix B). Details of the model architecture and hyper-parameters can be found in Table 2. For Muon, as discussed in Sec 2.2, since we matched Muon's update RMS to AdamW, we directly reused the hyper-parameters that are optimal for the AdamW baseline.

The fitted scaling law curve can be found in figure 3, and the fitted equations are detailed in table 3. As shown in Figure 1a, Muon only requires about 52% training FLOPs to match the performance of AdamW under compute-optimal setting.

Table 2: Scaling Law Models and Hyper-Parameters

<table><tr><td># Params. w/o Embedding</td><td>Head</td><td>Layer</td><td>Hidden</td><td>Tokens</td><td>LR</td><td>Batch Size*</td></tr><tr><td>399M</td><td>12</td><td>12</td><td>1536</td><td>8.92B</td><td>9.503e-4</td><td>96</td></tr><tr><td>545M</td><td>14</td><td>14</td><td>1792</td><td>14.04B</td><td>9.143e-4</td><td>128</td></tr><tr><td>822M</td><td>16</td><td>16</td><td>2048</td><td>20.76B</td><td>8.825e-4</td><td>160</td></tr><tr><td>1.1B</td><td>18</td><td>18</td><td>2304</td><td>28.54B</td><td>8.561e-4</td><td>192</td></tr><tr><td>1.5B</td><td>20</td><td>20</td><td>2560</td><td>38.91B</td><td>8.305e-4</td><td>256</td></tr></table>

\*In terms of number of examples in 8K context length.

![](images/229c31380edfa44de41602b8719918e948326392e94bb0f9ee6e9ee5b33eab6b.jpg)  
Figure 3: Fitted scaling law curves for Muon and AdamW optimizers.

## 3.3 Pretraining with Muon

Model Architecture To evaluate Muon against contemporary model architectures, we pretrained from scratch using the deepseek-v3-small architecture (DeepSeek-AI et al. 2024) as it demonstrates strong performance and the original results serve as a reference for comparison. Our pretrained model has 2.24B activated and 15.29B total parameters (3B activated and 16B total when including embedding). Minor modifications to the architecture are detailed in Appendix C.

Pretraining Data Our pretraining data details can be found in K. Team 2025. The maximum context length during pretraining is 8K.

Pretraining The model is trained in several stages. We use a 1e-3 auxfree bias update rate in stage 1 and 2, and 0.0 auxfree bias update rate in stage 3. The weight decay is set to 0.1 for all stages. More details and discussions of model training can be found in the Appendix D.

1. 0 to 33B tokens: In this stage, the learning rate linearly increases to 4.2e-4 in 2k steps. The batch size is kept at 2048 examples;

Table 3: Fitted parameters of the scaling law curves

<table><tr><td></td><td>Muon</td><td>AdamW</td></tr><tr><td>LM loss (seqlen=8K)</td><td> $2.506 \times C^{-0.052}$ </td><td> $2.608 \times C^{-0.054}$ </td></tr></table>

2. 33B to 5.2T tokens: In this stage, the learning rate decays from 4.2e-4 to 4.2e-5 in a cosine style. We keep the batch size at 2048 until 200B tokens, and then doubled to 4096 for the remaining;

3. 5.2T to 5.7T tokens: In this stage (also referred as the cooldown stage), the learning rate increases to 1e-4 in in 100 steps, and then linearly decays to 0 in 500B tokens, and we keep a constant 4096 batch size. In this stage, we use the highest quality data, focusing on math, code, and reasoning.

Evaluation Benchmarks Our evaluation encompasses four primary categories of benchmarks, each designed to assess distinct capabilities of the model:

\- English Language Understanding and Reasoning: MMLU(5-shot)(Hendrycks, Burns, Basart, et al. 2021), MMLU-pro(5-shot) (Wang et al. 2024), BBH(3-shot) (Suzgun et al. 2022), TriviaQA(5-shot) (Joshi et al. 2017)

• Code Generation: HumanEval(pass@1) (M. Chen et al. 2021), MBPP(pass@1)(Austin et al. 2021)

\- Mathematical Reasoning: GSM8K(4-shot) (Cobbe et al. 2021) MATH (Hendrycks, Burns, Kadavath, et al. 2021), CMATH (Wei et al. 2023)

\- Chinese Language Understanding and Reasoning: C-Eval(5-shot) (Y. Huang et al. 2023), CMMLU(5-shot)(H. Li et al. 2024)

Performance We named our model trained with Muon “Moonlight”. We compared Moonlight with different public models on a similar scale. We first evaluated Moonlight at 1.2T tokens and compared it with the following models that have the same architecture and trained with comparable number of tokens:

\- Deepseek-v3-Small (DeepSeek-AI et al. 2024) is a 2.4B/16B-parameter MoE model trained with 1.33T tokens;

\- Moonlight-A follows the same training settings as Moonlight, except that it uses the AdamW optimizer.

For Moonlight and Moonlight-A, we used the intermediate 1.2T token checkpoint of the total 5.7T pretraining, where the learning rate is not decayed to minimal and the model has not gone through the cooldown stage yet.

Table 4: Comparison of different models at around 1.2T tokens.

<table><tr><td></td><td>Benchmark (Metric)</td><td>DSV3-Small</td><td>Moonlight-A@1.2T</td><td>Moonlight@1.2T</td></tr><tr><td rowspan="4"></td><td> $Activated\ Params^†$ </td><td>2.24B</td><td>2.24B</td><td>2.24B</td></tr><tr><td> $Total\ Params^†$ </td><td>15.29B</td><td>15.29B</td><td>15.29B</td></tr><tr><td>Training Tokens</td><td>1.33T</td><td>1.2T</td><td>1.2T</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>Muon</td></tr><tr><td rowspan="4">English</td><td>MMLU</td><td>53.3</td><td>60.2</td><td>60.4</td></tr><tr><td>MMLU-pro</td><td>-</td><td>26.8</td><td>28.1</td></tr><tr><td>BBH</td><td>41.4</td><td>45.3</td><td>43.2</td></tr><tr><td>TriviaQA</td><td>-</td><td>57.4</td><td>58.1</td></tr><tr><td rowspan="2">Code</td><td>HumanEval</td><td>26.8</td><td>29.3</td><td>37.2</td></tr><tr><td>MBPP</td><td>36.8</td><td>49.2</td><td>52.9</td></tr><tr><td rowspan="3">Math</td><td>GSM8K</td><td>31.4</td><td>43.8</td><td>45.0</td></tr><tr><td>MATH</td><td>10.7</td><td>16.1</td><td>19.8</td></tr><tr><td>CMath</td><td>-</td><td>57.8</td><td>60.2</td></tr><tr><td rowspan="2">Chinese</td><td>C-Eval</td><td>-</td><td>57.2</td><td>59.9</td></tr><tr><td>CMMLU</td><td>-</td><td>58.2</td><td>58.8</td></tr></table>

$^{\dagger}$ The reported parameter counts exclude the embedding parameters.

As shown in Table 4, Moonlight-A, our AdamW-trained baseline model, demonstrates strong performance compared to similar public models. Moonlight performs significantly better than Moonlight-A, proving the scaling effectiveness of Muon. We observed that Muon especially excels on Math and Code related tasks, and we encourage the research community to further investigate this phenomena. After Moonlight is fully trained to 5.7T tokens, we compared it with public models at similar scale and showed the results in Table 5:

\- LLAMA3-3B from Grattafiori et al. 2024 is a 3B-parameter dense model trained with 9T tokens.

\- Qwen2.5-3B from Yang et al. 2024 is a 3B-parameter dense model trained with 18T tokens.

Table 5: Comparison of different models on various benchmarks.

<table><tr><td></td><td>Benchmark (Metric)</td><td>Llama3.2-3B</td><td>Qwen2.5-3B</td><td>DSV2-Lite</td><td>Moonlight</td></tr><tr><td rowspan="4"></td><td> $Activated Param^†$ </td><td>2.81B</td><td>2.77B</td><td>2.24B</td><td>2.24B</td></tr><tr><td> $Total Params^†$ </td><td>2.81B</td><td>2.77B</td><td>15.29B</td><td>15.29B</td></tr><tr><td>Training Tokens</td><td>9T</td><td>18T</td><td>5.7T</td><td>5.7T</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>Unknown</td><td>AdamW</td><td>Muon</td></tr><tr><td rowspan="4">English</td><td>MMLU</td><td>54.7</td><td>65.6</td><td>58.3</td><td>70.0</td></tr><tr><td>MMLU-pro</td><td>25.0</td><td>34.6</td><td>25.5</td><td>42.4</td></tr><tr><td>BBH</td><td>46.8</td><td>56.3</td><td>44.1</td><td>65.2</td></tr><tr><td> $TriviaQA^‡$ </td><td>59.6</td><td>51.1</td><td>65.1</td><td>66.3</td></tr><tr><td rowspan="2">Code</td><td>HumanEval</td><td>28.0</td><td>42.1</td><td>29.9</td><td>48.1</td></tr><tr><td>MBPP</td><td>48.7</td><td>57.1</td><td>43.2</td><td>63.8</td></tr><tr><td rowspan="3">Math</td><td>GSM8K</td><td>34.0</td><td>79.1</td><td>41.1</td><td>77.4</td></tr><tr><td>MATH</td><td>8.5</td><td>42.6</td><td>17.1</td><td>45.3</td></tr><tr><td>CMath</td><td>-</td><td>80.0</td><td>58.4</td><td>81.1</td></tr><tr><td rowspan="2">Chinese</td><td>C-Eval</td><td>-</td><td>75.0</td><td>60.3</td><td>77.2</td></tr><tr><td>CMMLU</td><td>-</td><td>75.0</td><td>64.3</td><td>78.2</td></tr></table>

$^{\dagger}$ The reported parameter counts exclude the embedding parameters. $^{\ddagger}$ We tested all listed models with the full set of TriviaQA.

\- Deepseek-v2-Lite from DeepSeek-AI 2024 is a 2.4B/16B-parameter MOE model trained with 5.7T tokens.

As shown in Table 5, Moonlight outperforms models with similar architectures trained with an equivalent number of tokens. Even when compared to dense models trained on substantially larger datasets, Moonlight maintains competitive performance. Detailed comparisons can be found in Appendix E. The performance of Moonlight is further compared with other well-known language models on MMLU and GSM8k, as illustrated in Figure 1b and Appendix E Figure 8. $^{6}$ . Notably, Moonlight lies on the Pareto frontier of model performance versus training budget, outperforming many other models across various sizes.

## 3.4 Dynamics of Singular Spectrum

In order to validate the intuition that Muon can optimize the weight matrices in more diverse directions, we conducted a spectral analysis of the weight matrices trained with Muon and AdamW. For a weight matrix with singular values $\sigma = (\sigma_{1}, \sigma_{2}, \cdots, \sigma_{n})$ , we calculate the SVD entropy (Alter et al. 2000; Roy et al. 2007) of this matrix as follows:

$$
H (\sigma) = - \frac {1}{\log n} \sum_ {i = 1} ^ {n} \frac {\sigma_ {i} ^ {2}}{\sum_ {j = 1} ^ {n} \sigma_ {j} ^ {2}} \log \frac {\sigma_ {i} ^ {2}}{\sum_ {j = 1} ^ {n} \sigma_ {j} ^ {2}}
$$

As shown in Figure 4, we visualized the average SVD entropy of the weight matrices across different training checkpoints during pretraining with 1.2T tokens. We can see that across all training checkpoints and all groups of weight matrices, the SVD entropy of Muon is higher than that of AdamW, which verifies the intuition that Muon can provide a more diverse spectrum of updates for the weight matrices. This discrepancy is more significant in the router weights for expert selection, which indicates that mixture-of-expert models can benefit more from Muon.

Moreover, we visualized the singular value distributions of each weight matrix at the checkpoint trained with 1.2T tokens as demonstrated in Appendix F. We find that, for over $90\%$ of the weight matrices, the SVD entropy when optimized by Muon is higher than that of AdamW, providing strong empirical evidence for Muon's superior capability in exploring diverse optimization directions.

## 3.5 Supervised Finetuning (SFT) with Muon

In this section, we present ablation studies on the Muon optimizer within the standard SFT stage of LLM training. Our findings demonstrate that the benefits introduced by Muon persist during the SFT stage. Specifically, a model that is both Muon-pretrained and Muon-finetuned outperforms others in the ablation studies. However, we also observe that when the SFT optimizer differs from the pretraining optimizer, SFT with Muon does not show a significant advantage over AdamW. This suggests that there is still considerable room for further exploration, which we leave for future work.

![](images/7641d6515f25171b418748e998aa5c0f8d347a98e280dfb8bf625ca50d4ef48d.jpg)

![](images/d7c1d68bcc69eed11ba861c9c0635b872f3b5dff5d2e879644e9697934328032.jpg)

![](images/8db668e23c096df4d908fb9371130ffaa2ffe65ae353f815135c3c05681fb798.jpg)

![](images/12e5c761d1865c265dbe9bd7b98c470f6896ae90fec87e2eb95a0ba8b20f04be.jpg)  
Training Iterations (K)

![](images/68bf0ad645586e93d3b212cd534b14e3370f359c51ffde19302f3fd9adfb3d34.jpg)

![](images/581595ba46de1f11230a42d17231af891197e86a3ad24104d2f89a6a9f03a0dc.jpg)  
Figure 4: SVD entropy of weight matrices across different training iterations. We categorize the weight matrices into 6 different groups: 1) AttnQO denotes the weight matrices related to the query and output projection in the attention layer; 2) AttnKV denotes the weight matrices related to the key and value projection in the attention layer; 3) Experts denotes the weight matrices in expert models; 4) SharedExperts denotes the weight matrices in shared expert models; 5) Router denotes the weight matrices in the router; 6) Dense denotes the weight matrices in the first dense layer. The SVD entropy is calculated as the macro-average of the weight matrices in each group across all layers. For weights in expert models, we only calculate 3 out of 64 experts in different layers for efficiency.

## 3.5.1 Ablation Studies on the Interchangeability of Pretrain and SFT Optimizers

To further investigate Muon's potential, we finetuned Moonlight@1.2T and Moonlight-A@1.2T using both the Muon and AdamW optimizers. These models were finetuned for two epochs on the open-source tulu-3-sft-mixture dataset (Lambert et al. 2024), which contains 4k sequence length data. The learning rate followed a linear decay schedule, starting at $5 \times 10^{-5}$ and gradually reducing to 0. The results, shown in Table 6, highlight the superior performance of Moonlight@1.2T compared to Moonlight-A@1.2T.

Table 6: Examining the impact of optimizer interchangeability between pretraining and SFT phases.

<table><tr><td>Benchmark (Metric)</td><td># Shots</td><td colspan="4">Moonlight-1.2T</td></tr><tr><td>Pretraining Optimizer</td><td>-</td><td>Muon</td><td>AdamW</td><td>Muon</td><td>AdamW</td></tr><tr><td>SFT Optimzier</td><td>-</td><td>Muon</td><td>Muon</td><td>AdamW</td><td>AdamW</td></tr><tr><td>MMLU (EM)</td><td>0-shot (CoT)</td><td>55.7</td><td>55.3</td><td>50.2</td><td>52.0</td></tr><tr><td>HumanEval (Pass@1)</td><td>0-shot</td><td>57.3</td><td>53.7</td><td>52.4</td><td>53.1</td></tr><tr><td>MBPP (Pass@1)</td><td>0-shot</td><td>55.6</td><td>55.5</td><td>55.2</td><td>55.2</td></tr><tr><td>GSM8K (EM)</td><td>5-shot</td><td>68.0</td><td>62.1</td><td>64.9</td><td>64.6</td></tr></table>

## 3.5.2 SFT with Muon on public pretrained models

We further applied Muon to the supervised fine-tuning (SFT) of a public pretrained model, specifically the Qwen2.5-7B base model (Yang et al. 2024), using the open-source tulu-3-sft-mixture dataset (Lambert et al. 2024). The dataset was packed with an 8k sequence length, and we employed a cosine decay learning rate schedule, starting at $2 \times 10^{-5}$ and gradually decreasing to $2 \times 10^{-6}$ . The results are presented in Table 7. For comparison, we show that the Muon-finetuned model achieves performance on par with the Adam-finetuned model. These results indicate that for optimal performance, it is more effective to apply Muon during the pretraining phase rather than during supervised fine-tuning.

Table 7: Comparison of Adam and Muon optimizers applied to the SFT of the Qwen2.5-7B pretrained model.

<table><tr><td>Benchmark (Metric)</td><td># Shots</td><td>Adam-SFT</td><td>Muon-SFT</td></tr><tr><td>Pretrained Model</td><td>-</td><td colspan="2">Qwen2.5-7B</td></tr><tr><td>MMLU (EM)</td><td>0-shot (CoT)</td><td>71.4</td><td>70.8</td></tr><tr><td>HumanEval (Pass@1)</td><td>0-shot</td><td>79.3</td><td>77.4</td></tr><tr><td>MBPP (Pass@1)</td><td>0-shot</td><td>71.9</td><td>71.6</td></tr><tr><td>GSM8K (EM)</td><td>5-shot</td><td>89.8</td><td>85.8</td></tr></table>

## 4 Discussions

There are several possible directions for future research that could further explore and expand upon the current findings.

Incorporating All Parameters into the Muon Framework Currently, the Muon optimizer is utilized in conjunction with the Adam optimizer, where certain parameters remain under the purview of Adam optimization. This hybrid approach, while functional, presents an opportunity for improvement. The integration of the optimization of all parameters exclusively within the Muon framework is a topic of significant research interest.

Extending Muon to Schatten Norms The Muon optimizer can be interpreted as the steepest descent method under the spectral norm. Given the broad applicability and versatility of Schatten norms, extending Muon to encompass the general Schatten norm is a promising direction. This extension may unlock additional optimization capabilities and potentially yield superior results compared to the current spectral norm-based implementation.

Understanding and Solving the Pretraining-Finetuning Mismatch A notable phenomenon observed in practice is the suboptimal performance of models pretrained with AdamW when fine-tuned with Muon, and vice versa. This optimizer mismatch presents a significant barrier to effectively leveraging the extensive repository of AdamW-pretrained checkpoints, thereby necessitating a rigorous theoretical investigation. A precise understanding of the underlying mechanisms is essential for devising robust and effective solutions.

## 5 Conclusions

In this technical report, we presented a comprehensive study on the scalability of Muon in LLM training. Through systematic analysis and improvements, we successfully applied Muon to a 3B/16B-parameter MoE model trained on 5.7 trillion tokens. Our results demonstrate that Muon can effectively replace AdamW as the standard optimizer for large-scale LLM training, offering significant advantages in both training efficiency and model performance. By open-sourcing our implementation, the Moonlight model, and intermediate training checkpoints, we aim to facilitate further research in scalable optimization techniques and accelerate the development of training methods for LLMs.

## References

Alter, Orly, Patrick O. Brown, and David Botstein. “Singular value decomposition for genome-wide expression data processing and modeling”. In: Proceedings of the National Academy of Sciences 97.18 (2000), pp. 10101–10106. DOI: 10.1073/pnas.97.18.10101.eprint: https://www.pnas.org/doi/pdf/10.1073/pnas.97.18.10101. URL: https://www.pnas.org/doi/abs/10.1073/pnas.97.18.10101.

Austin, Jacob et al. Program Synthesis with Large Language Models. 2021. arXiv: 2108.07732 [cs.PL]. URL: https://arxiv.org/abs/2108.07732.

Bernstein, Jeremy and Laker Newhouse. Old Optimizer, New Norm: An Anthology. 2024. arXiv: 2409.20325 [cs.LG]. URL: https://arxiv.org/abs/2409.20325.

Bi, Xiao et al. “Deepseek llm: Scaling open-source language models with longtermism”. In: arXiv preprint arXiv:2401.02954 (2024).

Cesista, Franz Louis. Deep Learning Optimizers as Steepest Descent in Normed Spaces. 2024. URL: http://leloykun.github.io/ponder/steepest-descent-opt/.

Chen, Mark et al. “Evaluating Large Language Models Trained on Code”. In: (2021). arXiv: 2107.03374 [cs.LG].

Cobbe, Karl et al. Training Verifiers to Solve Math Word Problems. 2021. arXiv: 2110.14168 [cs.LG]. URL: https://arxiv.org/abs/2110.14168.

DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. 2024. arXiv:2405.04434 [cs.CL].

DeepSeek-AI et al. DeepSeek-V3 Technical Report. 2024. arXiv: 2412.19437 [cs.CL]. URL: https://arxiv.org/abs/2412.19437.

Franz, Louis Cesista. The Case for Muon. Oct. 2024. URL: https://x.com/leloykun/status/1846842887839125941 (visited on 02/18/2025).

Grattafiori, Aaron et al. The Llama 3 Herd of Models. 2024. arXiv: 2407.21783 [cs.AI]. URL: https://arxiv.org/abs/2407.21783.

Hendrycks, Dan, Collin Burns, Steven Basart, et al. Measuring Massive Multitask Language Understanding. 2021. arXiv: 2009.03300 [cs.CY]. URL: https://arxiv.org/abs/2009.03300.

Hendrycks, Dan, Collin Burns, Saurav Kadavath, et al. Measuring Mathematical Problem Solving With the MATH Dataset. 2021. arXiv: 2103.03874 [cs.LG]. URL: https://arxiv.org/abs/2103.03874.

Hoffmann, Jordan et al. Training Compute-Optimal Large Language Models. 2022. arXiv: 2203.15556 [cs.CL]. URL: https://arxiv.org/abs/2203.15556.

Huang, Yuzhen et al. C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models. 2023. arXiv: 2305.08322 [cs.CL]. URL: https://arxiv.org/abs/2305.08322.

Jordan, Keller et al. Muon: An optimizer for hidden layers in neural networks. 2024. URL: https://kellerjordan.github.io/posts/muon/.

Joshi, Mandar et al. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. 2017. arXiv: 1705.03551 [cs.CL]. URL: https://arxiv.org/abs/1705.03551.

Kaplan, Jared et al. Scaling Laws for Neural Language Models. 2020. arXiv: 2001.08361 [cs.LG]. URL: https://arxiv.org/abs/2001.08361.

Kingma, Diederik P. and Jimmy Ba. “Adam: A Method for Stochastic Optimization”. In: 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings. Ed. by Yoshua Bengio and Yann LeCun. 2015. URL: http://arxiv.org/abs/1412.6980.

Lambert, Nathan et al. “Tülu 3: Pushing Frontiers in Open Language Model Post-Training”. In: (2024).

Li, Haonan et al. CMMLU: Measuring massive multitask language understanding in Chinese. 2024. arXiv: 2306.09212 [cs.CL]. URL: https://arxiv.org/abs/2306.09212.

Li, Xi-Lin. “Preconditioned Stochastic Gradient Descent”. In: IEEE Transactions on Neural Networks and Learning Systems 29.5 (May 2018), pp. 1454–1466. ISSN: 2162-2388. DOI: 10.1109/tnnls.2017.2672978. URL: http://dx.doi.org/10.1109/TNNLS.2017.2672978.

- Preconditioner on Matrix Lie Group for SGD. 2018. arXiv: 1809.10232 [stat.ML]. URL: https://arxiv.org/abs/1809.10232.

- Stochastic Hessian Fittings with Lie Groups. 2024. arXiv: 2402.11858 [stat.ML]. URL: https://arxiv.org/abs/2402.11858.

Li, Xilin. Black Box Lie Group Preconditioners for SGD. 2022. arXiv: 2211.04422 [stat.ML]. URL: https://arxiv.org/abs/2211.04422.

Liu, Hong et al. “Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training”. In: The Twelfth International Conference on Learning Representations. 2024. URL: https://openreview.net/forum?id=3xHDeA8Noi.

Loshchilov, Ilya and Frank Hutter. “Decoupled Weight Decay Regularization”. In: International Conference on Learning Representations. 2019. URL: https://openreview.net/forum?id=Bkg6RiCqY7.

OLMo, Team et al. “2 OLMo 2 Furious”. In: arXiv preprint arXiv:2501.00656 (2024).

OpenAI et al. GPT-4 Technical Report. 2024. arXiv: 2303.08774 [cs.CL]. URL: https://arxiv.org/abs/2303.08774.

Pethick, Thomas et al. Training Deep Learning Models with Norm-Constrained LMOs. 2025. arXiv: 2502.07529 [cs.LG]. URL: https://arxiv.org/abs/2502.07529.

Pooladzandi, Omead and Xi-Lin Li. Curvature-Informed SGD via General Purpose Lie-Group Preconditioners. 2024. arXiv: 2402.04553 [cs.LG]. URL: https://arxiv.org/abs/2402.04553.

Rajbhandari, Samyam et al. “ZeRO: Memory optimizations Toward Training Trillion Parameter Models”. In: (Nov. 2020), pp. 1–16. DOI: 10.1109/sc41405.2020.00024. URL: http://dx.doi.org/10.1109/SC41405.2020.00024.

Roy, Olivier and Martin Vetterli. “The effective rank: A measure of effective dimensionality”. In: 2007 15th European Signal Processing Conference. 2007, pp. 606–610.

Shazeer, Noam. Fast Transformer Decoding: One Write-Head is All You Need. 2019. arXiv: 1911.02150 [cs.NE]. URL: https://arxiv.org/abs/1911.02150.

Shoeybi, Mohammad et al. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. 2020. arXiv: 1909.08053 [cs.CL]. URL: https://arxiv.org/abs/1909.08053.

Suzgun, Mirac et al. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. 2022. arXiv:2210.09261 [cs.CL]. URL: https://arxiv.org/abs/2210.09261.

Team, Gemini et al. Gemini: A Family of Highly Capable Multimodal Models. 2024. arXiv: 2312.11805 [cs.CL]. URL: https://arxiv.org/abs/2312.11805.

Team, Gemma et al. “Gemma 2: Improving open language models at a practical size”. In: arXiv preprint arXiv:2408.00118 (2024).

Team, Kimi. "Kimi k1.5: Scaling Reinforcement Learning with LLMs". In: (2025).

Vyas, Nikhil et al. “SOAP: Improving and Stabilizing Shampoo using Adam”. In: The Thirteenth International Conference on Learning Representations. 2025. URL: https://openreview.net/forum?id=IDxZhXrpNf.

Wang, Yubo et al. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. 2024. arXiv: 2406.01574 [cs.CL]. URL: https://arxiv.org/abs/2406.01574.

Wei, Tianwen et al. CMATH: Can Your Language Model Pass Chinese Elementary School Math Test? 2023. arXiv:2306.16636 [cs.CL]. URL: https://arxiv.org/abs/2306.16636.

Yang, An et al. “Qwen2.5 Technical Report”. In: arXiv preprint arXiv:2412.15115 (2024).

You, Jiacheng. Jiacheng You's discussion on Muon's Update RMS. 2025. URL: https://x.com/YouJiacheng/status/1890094769386451309.

Yuan, Huizhuo et al. MARS: Unleashing the Power of Variance Reduction for Training Large Models. 2024. arXiv:2411.10438 [cs.LG].

## A Update RMS

## Proof of Lemma 1

Proof. Without loss of generality, consider the orthogonal matrices $U \in \mathbb{R}^{n \times n}$ and $V \in \mathbb{R}^{m \times m}$ where $n \geq m \geq r$ . We will show that for $X = U_{[:,:r]}V_{[:r,:]}$ (the update of the Muon has the same format), the RMS value is $\sqrt{r / mn}$ . From the definition of matrix multiplication:

$$
X _ {i, j} = \sum_ {k = 1} ^ {r} U _ {i, k} V _ {k, j}
$$

The RMS can be expressed as:

$$
\begin{array}{l} \operatorname{RMS} (X) ^ {2} = \frac {1}{m n} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \sum_ {k = 1} ^ {r} U _ {i, k} ^ {2} V _ {k, j} ^ {2} \\ \qquad = \frac {1}{m n} \sum_ {k = 1} ^ {r} \left(\sum_ {i = 1} ^ {n} U _ {i, k} ^ {2}\right) \left(\sum_ {j = 1} ^ {m} V _ {k, j} ^ {2}\right) \\ \qquad = \frac {1}{m n} \sum_ {k = 1} ^ {r} 1 \\ \qquad = \frac {r}{m n} \end{array}
$$

Therefore, $\mathrm{RMS}(X) = \sqrt{r / mn}$ . For the common case where the matrices are full-rank, $r = m$ , yielding $\mathrm{RMS}(X) = \sqrt{1 / n}$ .

Consistent Update RMS Across Muon and AdamW As discussed in 2.2, we'd like to match the update RMS between Muon and AdamW optimizers. This is validated by experiments on small-scale models. We set Muon's Update RMS in the range of [0.05, 0.1, 0.2, 0.4, 0.8] and AdamW as baseline. We reported the loss and representative weight matrix RMS at 2k steps (about 2B tokens) in the Table 8. From the results, we find that 0.2 RMS and 0.4 RMS performed similarly and much better than other settings. These findings are consistent with our empirical observation that AdamW's update RMS is in the range of $0.2 \sim 0.4$ . We opted to control the update RMS of Muon to 0.2.

Table 8: Muon Update RMS Experiments

<table><tr><td>Optimizer</td><td>AdamW</td><td>0.05 RMS*</td><td>0.1 RMS</td><td>0.2 RMS</td><td>0.4 RMS</td><td>0.8 RMS</td></tr><tr><td>LM training loss</td><td>3.512</td><td>3.355</td><td>3.239</td><td>3.198</td><td>3.199</td><td>3.386</td></tr><tr><td>LM validation loss</td><td>3.679</td><td>3.503</td><td>3.374</td><td>3.325</td><td>3.314</td><td>3.543</td></tr><tr><td>AttnQ weight RMS</td><td>1.01e-2</td><td>5.74e-3</td><td>8.44e-3</td><td>1.57e-2</td><td>2.95e-2</td><td>7.23e-2</td></tr><tr><td>Mlp weight RMS</td><td>1.25e-2</td><td>8.01e-3</td><td>1.27e-2</td><td>2.35e-2</td><td>4.51e-2</td><td>8.73e-2</td></tr></table>

\*Except the first column, all other candidates are using Muon with controlled RMS.

## B AdamW Baseline Scaling Law

To ensure the fairness and accuracy of our experiments, we conducted a series of experiments on our proprietary dataset to derive scaling law parameters that are optimal for AdamW. This includes determining the optimal model size(N), number of training tokens(D), learning rate( $\eta$ ), batch size(B) under a constrained computational budget (FLOPs, C). (Kaplan et al. 2020; Hoffmann et al. 2022; Bi et al. 2024) Table 9 presents the results of our systematic parameter search process.

Table 9: Empirical Relationships Between Scaling Law Parameters and Computational Budget (FLOPs)

<table><tr><td>N(C)</td><td>D(C)</td><td>η(C)</td><td>B(C)</td></tr><tr><td>0.0483359 ·  $C^{0.5112684}$ </td><td>3.4480927 ·  $C^{0.4887316}$ </td><td>0.0127339 ·  $C^{-0.0574752}$ </td><td>0.0065202 ·  $C^{0.4137915}$ </td></tr></table>

![](images/d479e6f85b1f48ae6bf036df172dfb196d10e5afe0a8bd334b625981a6f8f380.jpg)  
Figure 5: Optimization Landscapes for Scaling Law Hyper-parameters Across FLOPs Budgets

Hyper-Parameters Search To systematically identify optimal scaling law hyper-parameters in the AdamW baseline, we adopted a multistage search protocol. First, we selected multiple computational budgets (FLOPs levels) and initialized model sizes, learning rates, and batch sizes based on empirical guidelines from prior studies. For each fixed FLOPs constraint, we varied the model size N while adjusting the training token count D inversely to maintain C = 6ND, thereby exploring the trade-off between model capacity and data efficiency. Each configuration was trained to convergence, and the validation loss was recorded to determine the Pareto-optimal combinations of N and D. Subsequently, with the optimal N - D pairs fixed, we refined the learning rate and batch size through grid searches, ensuring stability and convergence across configurations. To mitigate local minima and enhance robustness, this iterative procedure was repeated 2–3 times, progressively narrowing the hyper-parameter space.

The optimization process is further illustrated in Figure 5, which depicts the loss landscapes as functions of training tokens, learning rate, and batch size across varying FLOPs budgets. Each bowl-shaped curve represents the loss surface for a specific FLOPs level, with a distinct global minimum corresponding to the optimal hyper-parameter configuration.

## C Model Architecture

Muon is agnostic to model architectures, and we used a model similar to Deepseek-V3-Small as described in DeepSeek-AI et al. 2024, because it is a strong model with open weights as a baseline. We made several small modifications in the Moonlight model and listed them here:

Multi-token Prediction (MTP) MTP has not shown significant benefits to pretraining in our experiments. For simplicity, we do not introduce MTP layers into the Moonlight model.

Auxfree Bias Update In DeepSeek-AI et al. 2024, auxfree bias is updated by: $b_{i} = b_{i} + u \times \text{sign}(e_{i})$ , where $u$ is the update ratio, $b_{i}$ is the bias for the ith expert, and $e_{i}$ is the expert's violating ratio. We slightly modified the update rule as: $b_{i} = b_{i} + u \times (\text{sign}(e_{i}) - \text{sign}(e).\text{mean}()$ , where $\text{sign}(e).\text{mean}()$ is the average of the signs of all expert's violating ratio, in order to control the magnitude of the bias, while does not change the topk selection logic.

Gate Scaling Factor Deepseek-V2-Lite did not use the gate scaling factor, and Deepseek-V3 used a scaling factor of 2.5. We used a scaling factor of 2.446 to control a similar output rms like dense models. The code for calculating our gate scaling factor can be found in Figure 6.

## D Training Stability

No Loss or Grad Norm Spike The Moonlight training process was very smooth and we did not meet any loss spike or gradient norm spike. The loss and grad norm curve can be seen in Figure 7 (Moonlight is colored in blue and Moonlight-A trained by AdamW is colored in red)

Max Attention Logit During training, we observed that while both the training loss and gradient norm remained stable throughout the process, the maximum attention logit (computed as the single largest logit value across the global batch) exhibited a distinct upward trajectory in specific layers during the initial training phase, exceeding a threshold of 100. Notably, AdamW demonstrated healthier behavior in controlling this metric compared to alternative optimizers.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_gate_scaling_factor(num_experts: int, topk: int, iter_times: int):
    """Calculate the gate scaling factor for MoE.

    Args:
    num_experts(int): The number of experts.
    topk(int): The number of experts to select.
    iter_timers(int): The number of iterations.

    Returns:
    float: The gate scaling factor.
    """
    factors = []
    for _ in range(iter_times):
    # mock gaussian logits
    logits = np.random.randn(num_experts)
    # select topk logits
    p = np.sort(sigmoid(logits))[::-1]
    p = p[:topk]
    # renormalize
    p = p / p.sum()
    # calculate the scaling factor
    factors.append(1/ (p**2).sum()**0.5)
    return np.mean(factors)
```  
Figure 6: Python implementation for calculating the gate scaling factor.

To further investigate the impacts of this phenomenon, we introduced the large attention logits ratio metric, defined as the proportion of attention logits exceeding 100 within a batch. As shown in Fig.7, this ratio remained consistently low (about $10^{-4}$ ), indicating that extreme large logit values were sparse. Furthermore, the maximum logit values gradually decrease as training progressed, suggesting that the optimization dynamics become healthier.

RMSNorm Gamma Weight Decay It is noteworthy that applying weight decay to the RMSNorm gamma parameter is crucial for ensuring training stability, as it effectively prevents excessively high output RMS values in each layer.

## E Comparison with More Expensive Models

Table 10 presents a comparative analysis between our Moonlight model (optimized with Muon) and publicly available models trained with greater computational resources, including LLama3.1-8B (Grattafiori et al. 2024), Gemma-9B (Gemma Team et al. 2024) and Qwen2.5-7B (Yang et al. 2024). Figure 8 illustrates the GSM8k performance benchmarks of Moonlight against comparable models in the field.

## F Singular Value Distributions of Weight Matrices

We visualize the singular value distributions of weight matrices by plotting a line graph of its singular values in descending order for each matrix, normalized by the largest one. As shown in Figures 9 and 10, we find that, for most of the weight matrices, the singular value distributions of them optimized by Muon are more flattened than that of AdamW, which further confirms the hypothesis that Muon can provide a more diverse spectrum of updates.

![](images/8c7c514933072f3508dc4da8ecf580157f532e111a3a8278c88d6d061f9da080.jpg)  
(a) Training Loss

![](images/43e101592baf264bbb859bad0bbcc60d0f8799c3aa7997ee284d489f2a9fa510.jpg)  
(b) Gradient Norm

![](images/dba9dad9a44fe0f4aa0e3d438c1c7814ad9bd9d6c70a2eac638db70bf4420ac2.jpg)  
(c) Max Attention Logit (Layer 1)

![](images/3e018b963f523e826d23356e9c2bc8476fbc28fe842845b5cddd9b0e17a9b0e7.jpg)  
(d) Large Attention Logits Ratio (Layer 1)  
Figure 7: Training dynamics comparison between Moonlight and Moonlight-A

Table 10: Comparison of different models on various benchmarks.

<table><tr><td rowspan="2"></td><td rowspan="2">Benchmark (Metric)</td><td rowspan="2">Moonlight</td><td>LLAMA3.1-8B</td><td>Gemma2-9B</td><td>Qwen2.5-7B</td></tr><tr><td colspan="3">Larger Training Compute Model</td></tr><tr><td rowspan="4"></td><td> $Activated Param^†$ </td><td>2.24B</td><td>7.38B</td><td>8.32B</td><td>6.83B</td></tr><tr><td> $Total Params^†$ </td><td>15.29B</td><td>7.38B</td><td>8.32B</td><td>6.83B</td></tr><tr><td>Training Tokens</td><td>5.7T</td><td>15T</td><td>8T</td><td>18T</td></tr><tr><td>Optimizer</td><td>Muon</td><td>AdamW</td><td>Unknown</td><td>Unknown</td></tr><tr><td rowspan="4">English</td><td>MMLU</td><td>70.0</td><td>66.7</td><td>71.3</td><td>74.2</td></tr><tr><td>MMLU-pro</td><td>42.4</td><td>37.1</td><td>44.7</td><td>45.0</td></tr><tr><td>BBH</td><td>65.2</td><td>57.7</td><td>68.2</td><td>70.4</td></tr><tr><td> $TriviaQA^‡$ </td><td>66.3</td><td>70.3</td><td>-</td><td>60.0</td></tr><tr><td rowspan="2">Code</td><td>HumanEval</td><td>48.1</td><td>37.2</td><td>37.8</td><td>57.9</td></tr><tr><td>MBPP</td><td>63.8</td><td>47.6</td><td>62.2</td><td>74.9</td></tr><tr><td rowspan="2">Math</td><td>GSM8K</td><td>77.4</td><td>57.2</td><td>70.7</td><td>85.4</td></tr><tr><td>MATH</td><td>45.3</td><td>20.3</td><td>37.7</td><td>49.8</td></tr></table>

$^{\dagger}$ The reported parameter counts exclude the embedding parameters. $^{\ddagger}$ We test all listed models with the full set of TriviaQA.

![](images/fddfd4f4de31b7e5b3fcf729c5b040ba178783d5d5539f225e76b8ed79738b39.jpg)  
Figure 8: The GSM8k performance of our Moonlight model optimized with Muon and other comparable models.

![](images/b361f8c8e83ed171273fdd7ad819deccdfaf7f376718d121e2465e4d7e73ddb0.jpg)  
Figure 9: Distribution of singular values for each weight matrix in the attention layers. We use WC to denote the weight matrices at each layer that compress the hidden states to the shared latent spaces for keys and values, WV to denote the weight matrices up-projecting the values from the latent space, WO to denote the output projection matrices, and WKR, WKC, WQR and WQC to denote the projection matrices for the part of keys and queries with and without RoPE respectively. We set the spines of each line graph red if the corresponding weight matrix optimized by Muon has a lower singular entropy than AdamW.

![](images/076141d4fbc3bbb0afe5a5d94d17ccae48582e34b41179ce1d1d966a378da2af.jpg)  
Figure 10: Distribution of singular values for each weight matrix in the feed-forward network (FFN) layers. We use WI, WV and WO to denote the weight matrices involved in the FFN layer with SwiGLU activation function, where WI represents the input projection to the Swish $_{1}$ function, WV represents the extra input projection interacting with Swish $_{1}$ activations, and WO represents the output projection. We use E0, E2, E3 to denote three arbitrarily selected expert models and SE to denote the weights in the shared expert model. We use RW to denote the weights in the router. We set the spines of each line graph red if the corresponding weight matrix optimized by Muon has a lower singular entropy than AdamW.