# Nemotron-Labs-Difusion: A Tri-Mode Language Model Unifying Autoregressive, Difusion, and Self-Speculation Decoding

Yonggan Fu, Lexington Whalen<sup>1†</sup>, Abhinav Garg, Chengyue Wu<sup>2†</sup>, Maksim Khadkevich, Nicolai Oswald, Enze Xie, Daniel Egert, Sharath Turuvekere Sreenivas, Shizhe Diao, Chenhan Yu, Ye Yu, Weijia Chen, Sajad Norouzi, Shiyi Lan, Ligeng Zhu, Jin Wang<sup>2†</sup>, Jindong Jiang, Morteza Mardani, Mehran Maghoumi, Song Han<sup>3</sup>, Ante Jukić, Nima Tajbakhsh, Jan Kautz, Pavlo Molchanov

We introduce Nemotron-Labs-Difusion, a tri-mode language model (LM) that unifies AR, difusion, and self-speculation decoding within a single architecture. Trained with a joint AR-difusion objective, Nemotron-Labs-Difusion can switch modes to sustain high throughput across deployment settings and concurrency levels. Our study shows that (1) AR and difusion objectives are complementary: difusion improves lookahead planning, while AR provides left-to-right linguistic priors. (2) In self-speculation mode, difusion drafts while AR verifies, outperforming multi-token prediction (MTP) methods in both acceptance rate and real-device eficiency. (3) A speed-of-light analysis further demonstrates difusion’s long-term potential, with up to 76.5% more tokens per forward pass than self-speculation under an optimal sampler. Scaling to 3B, 8B, and 14B parameters, our Nemotron-Labs-Difusion family, including base, instruct, and vision-language models, consistently outperforms state-of-the-art open-source AR and difusion LMs in both accuracy and speed. For example, Nemotron-Labs-Difusion-8B decodes 6× more tokens per forward than Qwen3-8B with comparable accuracy, translating to 4× higher throughput on SPEED-Bench with SGLang on a GB200 GPU.

Models on Hugging Face: Nemotron-Labs-Difusion Model Family

![](images/a476c8fc002e60b14f7ee9a9d59dc5788652450a1098020e012af30519bb6504.jpg)

![](images/7dccbf488cb566e8038511dc88f5811a3b56b0682ea1652f9b92f19b4170f3c4.jpg)

![](images/0f913f481e0cc8102fba89dc9d0ac1b5e960b6d70ff0cb9311145462aced1fef.jpg)  
Figure 1 | (a) An illustration of three modes in one model. (b) Accuracy–throughput trade-of measured on general benchmarks at batch size 1 on an NVIDIA H100 using PyTorch. NLD denotes our model and Linear/Quad SS denotes linear/quadratic self-speculation. <sup>⋆</sup> indicates the difusion mode with diferent denoising block sizes (8, 16, 32), where smaller symbols correspond to smaller block sizes. (c) Trade-of between system vs. per-user throughput of the AR and Linear SS modes of our 8B model and Qwen3-8B Eagle3, measured on SPEED-Bench [1] at diferent concurrency ?? on an NVIDIA GB200 GPU using SGLang.

## 1. Introduction

The strictly sequential, token-by-token decoding process of autoregressive (AR) language models (LMs) fundamentally limits their inference parallelism, resulting in resource under-utilization and low throughput, especially in low-batch-size deployment scenarios. Difusion LMs [2, 3, 4] have recently emerged as a promising alternative, enabling parallel generation by decoding multiple tokens per forward pass. Nevertheless, difusion LMs often lag behind AR models in accuracy and learning eficiency, requiring substantially more data to reach comparable performance [5]. A key reason is that difusion training treats all token permutations uniformly [6], rather than leveraging the strong left-to-right prior inherent in natural language. Moreover, existing difusion LMs still lack clear advantages over multi-token prediction (MTP) methods and often fall behind them in practical eficiency–accuracy trade-ofs.

![](images/1a9dc05c40bd9c4bd57bc61c9bba0f8553dbf9282c3d1dd6c70b7e8ad4cfea69.jpg)  
(a)

![](images/3614b9340bb861a6e0ee696bf42d7318e87bbee5903a33562b1995fb1b8b0c1b.jpg)  
(b)

![](images/8b24f936e72fab1692e95677b5730f33107d46aae1b65efd1fa56fc5e1ae25ce.jpg)  
(c)  
Figure 2 | Benchmarking our Nemotron-Labs-Difusion-8B (Instruct) against SOTA AR and difusion instruct LMs across diferent benchmarks. (a) shows the average accuracy across all 10 tasks (HumanEval, MBPP, LiveCodeBench-CPP, GSM8K, Math500, AIME24, AIME25, GPQA, IFEval, MMLU) and the tokens per forward (TPF) of diferent models, while (b) and (c) show the average accuracy across coding (HumanEval, MBPP, LiveCodeBench-CPP) and math (GSM8K, Math500, AIME24, AIME25) domains, respectively.

These limitations raise three critical questions for understanding the role of difusion LMs:

Q1: Should difusion LMs compete with AR LMs, or can the two paradigms be harmonized?

Q2: Can difusion LMs provide a stronger acceleration mechanism than MTP methods?

Q3: Does difusion decoding have enough long-term potential to justify deeper exploration?

Answering these questions is critical for judging the true promise of difusion LMs and guiding their correct and wide adoption. This work studies these questions by unifying AR and difusion modeling within a single model that preserves the strengths of AR LMs while exploring the benefits and potential of parallel decoding. The motivation is that AR and difusion LMs might not be competing paradigms in which one should replace the other; instead, they can be mutually beneficial and unified within a single model by switching between causal and bidirectional attention. Specifically, AR models inherently learn to plan ahead for future tokens [7], and difusion training can further enhance this capability. Conversely, preserving AR objectives during training injects strong left-to-right linguistic priors into difusion modeling.

Based on this insight, we introduce Nemotron-Labs Difusion, a tri-mode LM that jointly optimizes difusion and AR losses under a unified training framework. Our training scheme employs a global loss-averaging strategy that treats all tokens across batches equally to stabilize optimization. We further adopt a twostage training procedure: we first strengthen AR capabilities to establish strong left-to-right linguistic priors, and then enable joint difusion and AR training to fully integrate both objectives. The resulting models support tri-mode decoding as shown in Fig. 1 (a): (1) AR decoding, (2) parallel difusion-based decoding, which can be paired with a sampler optimized on sampling trajectories for improved parallelism, and (3) self-speculation, where difusion drafts candidate tokens and AR predictions verify them.

We leverage this training scheme to deliver the Nemotron-Labs-Difusion model family, including base, instruct, and vision-language variants at 3B/8B/14B scales. As shown in Fig. 2, our models outperform state-of-the-art (SOTA) open-source AR / difusion LMs in both accuracy and inference speed across a wide range of benchmarks. For example, our Nemotron-Labs-Difusion-8B delivers 6× tokens per forward over Qwen3-8B while maintaining comparable or better accuracy on general benchmarks, translating to 4× throughput on SPEED-Bench [1] measured with SGLang on an NVIDIA GB200 GPU.

The results and analysis of our tri-mode LMs ofer rich insights to answer the above questions. First, AR and difusion LMs can be harmonized rather than treated as competing alternatives and unifying AR and difusion objectives is a promising pathway: AR contributes strong next-token modeling and linguistic priors, while difusion unlocks parallel generation without sacrificing benchmark performance. Beyond accuracy, the joint objective naturally enables selfspeculation, allowing tri-mode LMs to adapt to different deployment regimes with diferent levels of concurrency: self-speculation is especially efective in low-concurrency settings, as shown in Fig. 1 (b), while AR remains well suited for compute-bound highconcurrency scenarios. As such, tri-mode models can serve as drop-in replacements for conventional AR LMs, requiring no architectural or pipeline changes while ofering consistently high throughput across deployment scenarios.

Our speed-of-light (SOL) analysis, which estimates the upper bound of difusion decoding when equipped with an optimal sampler, shows that difusion decoding has strong potential and substantial headroom beyond current parallel decoding methods. Specifically, difusion-based decoding with an optimal sampler can correctly predict over 76.5% more tokens per forward pass than the self-speculation mode, indicating that current samplers still leave a large fraction of the available parallelism unused. These results highlight the long-term promise of difusion decoding.

Notably, we find that sampling tokens from the difusion mode to approach the SOL upper bound remains an open challenge, and that the most efective approach is to verify the decoded tokens using the same model in AR mode. This observation motivates the aforementioned self-speculation mode, where diffusion generates high-quality multi-token drafts while AR verifies them within a single model, eliminating the need for the auxiliary prediction heads used by prior MTP methods [8]. As shown in Fig. 1 (c), this yields higher acceptance rates and more favorable trade-ofs between system throughput and per-user throughput, making tri-mode LMs a stronger and more flexible alternative to existing MTP approaches.

We hope the above insights can shed light on the proper adoption of difusion objectives in language modeling and on future directions that fully unlock the potential of difusion decoding.

Paper structure. The rest of the paper is organized as follows. Sec. 2 and Sec. 3 detail our joint AR-difusion training framework and tri-mode inference algorithms; Sec. 4 presents the speed-of-light analysis; Sec. 5 and Sec. 6 introduce the Nemotron-Labs-Difusion model family and present experiments, including comparisons between self-speculation and MTP; Sec. 7 reviews related work and Sec. 8 concludes with insights and future directions.

## 2. Tri-Mode LM Training

## 2.1. Training Objectives

Motivation. We hypothesize that AR and difusion objectives are complementary rather than competing. AR pretraining induces an implicit ability to plan ahead [7], which the difusion objective further unlocks by forcing the model to reason about future tokens; in turn, the AR objective anchors difusion training to the left-to-right structure of language and prevents wasted capacity on arbitrary token permutations. Therefore, we train Nemotron-Labs-Difusion on a weighted combination of an AR next-token loss and a block-wise difusion denoising loss.

AR objective. For a token sequence $x ,$ the AR objective maximizes the likelihood under the left-toright factorization:

$$
\mathcal {L} _ {\mathrm{AR}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}} \left[ - \sum_ {i = 1} ^ {| x |} \log p _ {\theta} (x _ {i} \mid x _ {<   i}) \right].\tag{1}
$$

![](images/a659f9dd7bb0c198fa0386971e4cd49e809961d40e55f70f2a471033a9339830.jpg)  
Figure 3 | Visualizing the attention pattern of Nemotron-Labs-Difusion, where denotes attention among noisy tokens, denotes attention from noisy tokens to clean-context tokens, and denotes attention within the clean context.

Difusion objective. As shown in Fig. 3, we adopt the block-wise difusion formulation [9, 10], which partitions the sequence into ?? contiguous blocks $\{ x ^ { b } \} _ { b = 1 } ^ { B }$ and trains the model to denoise one block at a time conditioned on its clean prefix. At noise level $t \sim \mathcal { U } [ 0 , 1 ]$ , only the tokens in the current block are corrupted via a forward noising process ??, i.e., $\tilde { x } _ { t } ^ { b } \sim q ( \cdot | x ^ { b } )$ , while the prefix $x ^ { < b }$ remains clean:

$$
\mathcal{L}_{\mathrm{diff}}(\theta) = \mathbb{E}_{\substack{t\sim \mathcal{U}[0,1]\\ \tilde{x}_{t}^{b}\sim q(\cdot |x^{b})}}\left[-\frac{1}{t}\sum_{b = 1}^{B}\log p_{\theta}\big(x^{b}\mid \tilde{x}_{t}^{b}, x^{<  b}\big)\right].\tag{2}
$$

This block-wise design is bidirectional within each block to enable parallel intra-block prediction, and causal across blocks so that previously generated blocks can reuse their KV cache during inference.

Joint objective. We optimize a weighted combination of the two losses:

$$
\mathcal {L} (\theta) = \mathcal {L} _ {\mathrm{AR}} (\theta) + \alpha \mathcal {L} _ {\mathrm{diff}} (\theta),\tag{3}
$$

where the AR loss has coeficient 1 and ?? controls the strength of difusion supervision. This design choice is motivated by the observation that the difusion loss is often larger than the AR loss, and selecting an ?? that aligns the magnitudes of the two losses yields the best results, i.e., enabling difusion-style parallel decoding while maintaining AR accuracy. We set ?? = 0.3 across all training stages.

Two-stage training. To strengthen left-to-right priors and improve learning eficiency, we adopt a two-stage training strategy that first trains with the AR objective, which anchors the representation to the language’s inherent left-to-right inductive bias, and then switches to the joint objective. In terms of Eq. 3, Stage 1 sets $\alpha = 0 ;$ , reducing the optimization to the pure AR objective in Eq. 1. In Stage 2, we turn on difusion supervision by setting ?? to align the magnitudes of the two losses, as mentioned above, so that difusion gradients complement rather than overwrite the AR priors.

Table 1 | Ablation study of each training technique during continuous pretraining on 25B tokens.

<table><tr><td>Technique</td><td>HumanEval</td><td>HumanEval+</td><td>MBPP</td><td>MBPP+</td><td>GSM8K</td><td>Minerva Math</td><td>Avg</td></tr><tr><td>Block-wise attention</td><td>39.02</td><td>37.80</td><td>53.40</td><td>67.72</td><td>82.87</td><td>44.58</td><td>54.23</td></tr><tr><td>+ Global Loss Avg</td><td>42.07</td><td>39.02</td><td>56.20</td><td>71.69</td><td>83.78</td><td>45.36</td><td>56.35</td></tr><tr><td>+ DP-rank Varying Masking Ratios</td><td>45.12</td><td>43.29</td><td>55.80</td><td>70.90</td><td>81.58</td><td>45.66</td><td>57.06</td></tr><tr><td>+ Two-stage training</td><td>58.54</td><td>52.44</td><td>53.00</td><td>73.81</td><td>83.17</td><td>55.84</td><td>62.80</td></tr><tr><td>+ AR loss</td><td>64.02</td><td>57.93</td><td>65.60</td><td>80.95</td><td>86.73</td><td>66.44</td><td>70.28</td></tr></table>

Global loss averaging. Since the difusion objective involves randomly sampling masked tokens, diferent training examples may have diferent numbers of tokens contributing to the difusion loss. As a result, the strategy for averaging token-wise losses matters, analogous to how the choice of aggregation in on-policy RL objectives (e.g., GRPO [11] vs. DAPO [12]) can afect training stability. We consider two loss averaging strategies. Let a batch contain ?? sequences, each of length $L ,$ and let $\ell _ { n , i }$ denote the token-level loss for token ?? in sequence ?? based on Eq. 3. One choice is to first average token losses within each sequence and then average them over sequences:

$$
\mathcal {L} _ {\text { seq - avg }} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left(\frac {1}{L} \sum_ {i = 1} ^ {L} \ell_ {n, i}\right).\tag{4}
$$

Another choice is to treat all tokens across the batch equally and globally average over the ???? token losses:

$$
\mathcal {L} _ {\mathrm{global}} = \frac {1}{N L} \sum_ {n = 1} ^ {N} \sum_ {i = 1} ^ {L} \ell_ {n, i}.\tag{5}
$$

While $\operatorname { E q . }$ . 4 and Eq. 5 coincide when every sequence has the same number of loss-contributing tokens, they difer once masking yields variable numbers of noisy tokens across samples, which is common in the diffusion objective. In particular, in Eq. 2, the loss includes $\begin{array} { r } { \dot { \mathrm { ~ a ~ } } \frac { 1 } { t } } \end{array}$ reweighting, and the number of noisy tokens is approximately proportional to ??. When ?? is small, each noisy token tends to carry a larger weight (due to <sup>1</sup> ), but there are fewer such tokens in the sample. Sequence-wise averaging can therefore amplify the influence of these small-?? samples: their per-token losses are larger, yet the per-sequence normalization assigns them the same weight as other samples, increasing batch-to-batch fluctuations and gradient variance. In contrast, global averaging efectively weights each training example in proportion to its number of contributing tokens, preventing samples with only a few highly weighted noisy tokens from disproportionately influencing the batch loss.

## 2.2. Attention Pattern

Following [9], at training time we use a dual-stream input by concatenating a corrupted/noised view and a clean view of the same sequence, and apply a structured attention pattern, as shown in Fig. 3.

The Noisy→Noisy and Noisy→Clean parts follow the standard block difusion design [9]: we partition the sequence into ?? contiguous blocks $\{ x ^ { b } \} _ { b = 1 } ^ { B } ;$ in the noisy stream, tokens attend bidirectionally within each block and causally across blocks; and for denoising block ??, noisy tokens additionally attend to the clean-prefix blocks $x ^ { < b }$ in the clean stream to achieve clean-context conditioning.

The key diference lies in the Clean→Clean mask. Prior designs [9, 10, 13] allow the clean stream to attend to future tokens using block-wise attention. In contrast, we enforce a strictly causal mask within the clean stream [14, 7]. This enables us to compute the AR objective on ?? in this clean-context part together with the difusion objective on $\tilde { x } _ { t }$ in the same forwardbackward pass, without label leakage.

Relationship with prior works. Our attention pattern follows the pioneering work of [14], which also performs joint difusion and AR training. The key diferences in our work lie in (1) proposing tri-mode inference, particularly self-speculation decoding, along with post-training enhancements for improved parallelism, including the samplers and LoRA-enhanced drafters introduced in Sec. 3; (2) the overall training pipeline used to develop the full model family described in Sec. $5 ;$ and (3) the systematic studies and SOL analysis conducted to address critical questions regarding the true potential of difusion LMs.

## 2.3. Ablation Study on Training Techniques

We ablate the contribution of each training technique by progressively adding them during continuous pretraining on 25B tokens, starting from the oficial Ministral3-8B base model. Detailed training/evaluation settings will be elaborated in Sec. 5.1 and Sec. 6.3. All models are evaluated in difusion mode on coding and math benchmarks.

Observations. As shown in Tab. 1, we progressively add each technique and observe that (1) block-wise attention serves as the baseline at 54.23% average accuracy, following the setting of [10, 4] and providing a functional difusion LM; (2) global loss averaging improves the average by 2.12%, confirming tha treating all tokens equally across the batch reduces gradient variance from variable masking ratios, as analyzed in Sec. 2.1; (3) DP-rank varying masking ratios, which applies diferent noise levels across dataparallel ranks, further improves the average by 0.71%; (4) two-stage training, which provides a better AR initialization with 1T-token AR objective training, yields a substantial 5.74% gain, implying that stronger AR initialization can enable better future planning and ease AR-to-difusion conversion; (5) the addition of AR loss contributes the largest single improvement of 7.48%, significantly boosting difusion decoding abilities, echoing our analysis in Sec. 2.

![](images/dd72c6647a5111d822e81262c79300fdd68c7aeabe865d7247edf80d17afdf93.jpg)

![](images/df2b422cceb669bdc22a2f4a37ace69c325e32804a5a44e2047fdeafb17df53d.jpg)  
Figure 4 | Visualizing the evolution of the AR and difusion losses across training steps under diferent difusion loss coeficients ??. The AR loss coeficient is set to 1 by default across all settings, except in the ‘no AR’ setting, where the AR loss is removed and only the difusion loss is used.

Table 2 | Impact of difusion loss weight ?? during 25B-token continuous pretraining.

<table><tr><td>α</td><td>Mode</td><td>Human Eval</td><td>Human Eval+</td><td>MBPP</td><td>MBPP+</td><td>GSM8K</td><td>Minerva Math</td><td>Avg</td></tr><tr><td rowspan="2">0.1</td><td>Diff.</td><td>56.71</td><td>51.83</td><td>64.80</td><td>81.22</td><td>87.64</td><td>67.02</td><td>68.20</td></tr><tr><td>AR</td><td>58.54</td><td>53.05</td><td>64.60</td><td>80.42</td><td>87.64</td><td>68.02</td><td>68.71</td></tr><tr><td rowspan="2">0.2</td><td>Diff.</td><td>60.37</td><td>54.27</td><td>63.40</td><td>80.69</td><td>86.43</td><td>66.58</td><td>68.62</td></tr><tr><td>AR</td><td>60.98</td><td>57.93</td><td>66.40</td><td>83.60</td><td>87.04</td><td>67.04</td><td>70.50</td></tr><tr><td rowspan="2">0.3</td><td>Diff.</td><td>61.59</td><td>58.54</td><td>64.60</td><td>80.42</td><td>87.64</td><td>65.86</td><td>69.77</td></tr><tr><td>AR</td><td>62.80</td><td>57.93</td><td>65.80</td><td>82.54</td><td>87.79</td><td>66.84</td><td>70.62</td></tr><tr><td rowspan="2">0.5</td><td>Diff.</td><td>59.76</td><td>54.27</td><td>64.40</td><td>80.16</td><td>87.11</td><td>66.98</td><td>68.78</td></tr><tr><td>AR</td><td>58.54</td><td>53.05</td><td>65.40</td><td>82.80</td><td>86.81</td><td>67.14</td><td>68.96</td></tr><tr><td rowspan="2">1.0</td><td>Diff.</td><td>56.10</td><td>48.78</td><td>65.00</td><td>80.69</td><td>84.91</td><td>66.12</td><td>66.93</td></tr><tr><td>AR</td><td>54.27</td><td>50.61</td><td>64.80</td><td>80.69</td><td>86.58</td><td>66.36</td><td>67.22</td></tr></table>

Cumulatively, the full pipeline improves the baseline by 16.05% in average accuracy, with the AR loss and two-stage training contributing the most. This validates our core insight that preserving the AR objective during difusion training anchors the model to linguistically coherent trajectories and is a critical factor for achieving strong difusion LM accuracy.

## 2.4. Mutual Impact of AR/Difusion Losses

In this subsection, we examine whether the AR and difusion objectives compete for model capacity or reinforce each other: the impact of adding the AR loss on the difusion mode, and the impact of adding the difusion loss on AR-mode accuracy.

AR loss boosts difusion accuracy. As studied in Sec. 2.3 and Tab. 1, AR loss can significantly boost difusion accuracy. As a complement to this study, we also vary ?? in Eq. 3 during 25B-token continuous pretraining on top of the two-stage training setting in Tab. 2. We observe that both modes peak at ??=0.3. This implies that the two modes do not necessarily compete with each other or achieve the best performance at the two extremes; instead, there exists a sweet spot where both are well harmonized. Similarly, no value of ?? in [0.1, 0.5] improves one mode at the expense of the other, and the two objectives rise and fall together, indicating that they are complementary rather than competing for model capacity.

We also visualize the training loss curves in Fig. 4. We observe that the aforementioned setting ??=0.3, which achieves the best accuracy, provides a good balance between the two losses. Setting ?? too small or too large leads to increased difusion or AR loss, respectively. In addition, without the AR or difusion loss, the corresponding AR or difusion capabilities are degraded or lost.

Difusion loss preserves AR accuracy. To study whether difusion loss can hurt or preserve AR accuracy, we compare models trained w/o and w/ the difusion loss (??=0.3) under two settings: continuous pretraining on 25B tokens on top of the two-stage training setting in Tab. 1, and further SFT on 45B tokens, following training/evaluation settings in Sec. 5.2 and Sec. 6.1. We ensure that all settings are trained on the same number of tokens.

As shown in Tab. 3, we observe that: (1) In both settings, the average AR accuracy is preserved or slightly boosted, with 0.14% and 0.43% improvements for the base and instruct models, respectively, indicating that difusion training, when properly integrated, can enhance the future prediction abilities of the AR mode, similar to observations in DeepSeek-V3 [15]; (2) At the per-benchmark level, the instruct model shows gains on coding and math benchmarks, e.g., 4.24% higher on LCB-CPP and 1.59% higher on MBPP, but drops on IFEval (3.01% lower) and HumanEval (2.44% lower), suggesting that strict instruction-following compliance is slightly afected by the difusion objective.

Table 3 | AR-mode accuracy with and without the difusion loss (??=0.3). Base: 25B-token continuous pretraining from Ministral3-8B. Instruct: further SFT on 45B tokens.

<table><tr><td colspan="12">Base Model</td></tr><tr><td>Training</td><td>HumanEval</td><td>HumanEval+</td><td>MBPP</td><td>MBPP+</td><td>GSM8K</td><td>Minerva Math</td><td>MMLU</td><td>Hellaswag</td><td>PIQA</td><td>Winogrande</td><td>Avg</td></tr><tr><td>AR only</td><td>60.37</td><td>56.10</td><td>67.00</td><td>81.48</td><td>87.64</td><td>67.90</td><td>76.34</td><td>76.54</td><td>79.71</td><td>71.98</td><td>72.50</td></tr><tr><td>+ Diff. loss</td><td>62.80</td><td>57.93</td><td>65.80</td><td>82.54</td><td>87.79</td><td>66.84</td><td>75.99</td><td>76.59</td><td>79.82</td><td>70.32</td><td>72.64</td></tr><tr><td colspan="12">Instruct Model</td></tr><tr><td>Training</td><td>GPQA</td><td>IFEval</td><td>HumanEval</td><td>MBPP</td><td>Math500</td><td>GSM8K</td><td>AIME24</td><td>AIME25</td><td>MMLU</td><td>LCB-CPP</td><td>Avg</td></tr><tr><td>AR only</td><td>44.44</td><td>71.66</td><td>82.93</td><td>83.60</td><td>87.80</td><td>93.63</td><td>36.67</td><td>26.67</td><td>79.77</td><td>24.61</td><td>63.18</td></tr><tr><td>+ Diff. loss</td><td>44.44</td><td>68.65</td><td>80.49</td><td>85.19</td><td>88.00</td><td>94.01</td><td>33.33</td><td>33.33</td><td>79.85</td><td>28.85</td><td>63.61</td></tr></table>

These results, together with the ?? sensitivity analysis above, support the conclusion that joint AR–difusion training is not a zero-sum trade-of: the difusion loss enables parallel decoding modes (Sec. 3) at negligible cost to AR-mode accuracy, and the two objectives share a common optimal operating point.

## 3. Tri-Mode LM Inference

The joint AR and difusion training enables decoding in three modes: AR, difusion, and self-speculation decoding, as shown in Fig. 5.

## 3.1. Mode 1: AR Decoding

Tri-mode LMs fully preserve standard left-to-right generation: at step ??, they sample $x _ { i } \sim p _ { \theta } ( \cdot \mid x _ { < i } )$ with causal attention. This mode is preferred when serving with high concurrency.

## 3.2. Mode 2: Block-wise Difusion Denoising

Confidence-based sampling. Following [13, 10], the difusion decoding mode proceeds block by block. For the current block, we initialize its positions as mask tokens and iteratively denoise multiple tokens in parallel per step based on a confidence threshold [16]. When a block is completed, its KV cache will be refreshed, and decoding proceeds to the next block.

Sampling with a trained sampler. A fixed confidence threshold is an implicit signal that is not explicitly optimized during training. We therefore train a lightweight sampler that, for every masked position in the current block, predicts whether the top-1 prediction at the current denoising step is correct. Here, correct means that the decoded token matches the token that will eventually be committed at this position when decoding only the highest-confidence token at each step. At inference, we commit positions whose predicted probability from the sampler exceeds a predefined threshold, which trades of TPF against per-token error rate. The sampler can be viewed as a learned classifier to approach the greedy-acceptance criterion that we later analyze in Sec. 4. The sampler architecture, feature engineering, and training trajectory data collection are detailed in Appendix A.

## 3.3. Mode 3: Self-Speculation Decoding

Linear self-speculation. The simplest selfspeculative mode separates difusion-based drafting and AR-based verification into two forward passes. Let $[ x _ { 1 } , \ldots , x _ { n } ]$ denote the currently verified prefix and let ?? be the speculative width.

Drafting with difusion. We append ?? mask tokens to the verified prefix, forming the input $\left[ x _ { 1 } , \ldots , x _ { n } , m _ { 1 } , \ldots , m _ { k } \right]$ . The model denoises all ?? mask positions in parallel using the difusion pathway, producing draft tokens $\{ \hat { x } _ { n + 1 } , . . . , \hat { x } _ { n + k } \}$

Verification with AR. We then run a second forward pass over the draft tokens $[ \hat { x } _ { n + 1 } , . . . , \hat { x } _ { n + k } ]$ with causal attention, again reusing the prefix KV cache. The AR logits at each position yield next-token predictions $\{ \bar { x } _ { n + j } ^ { \mathrm { A R } } \} _ { j = 1 } ^ { k }$ We accept the longest prefix of draft tokens that passes the verification criterion $( \mathrm { e . g . } , x _ { n + j } ^ { \mathrm { A R } } = \hat { x } _ { n + j } )$ and commit the accepted tokens to the verified prefix. As in standard speculative decoding [17], the AR prediction at the first rejected position provides one additional verified token, so each step produces between 1 and ??+1 tokens. Note that both the drafting and verification passes can reuse the cached prefix KVs from prior verified steps.

Enhance linear self-speculation w/ LoRA. We further enhance linear self-speculation by tuning a LoRA adapter [18] on top of the difusion draft pathway to better align its drafts with the AR verifier, thereby extending the accepted prefix length per step. We apply LoRA only to the $o _ { \mathrm { p r o j } }$ layer of the attention module (rank 128, ??=512, ∼36M trainable parameters/∼0.4% of the backbone), leaving the AR pathway unchanged. The training loss combines an LK-hybrid distribution-matching term [19] with a token-level cross-entropy term, both applied to the accepted prefix plus the first rejected position of each draft block, as shown in Fig. 11 in Appendix B.

Drafter–verifier setup. For each position $j \in$ $\overline { { \{ 1 , \ldots , k \} } }$ in the draft block, the LoRA-augmented drafter produces logits $z _ { j } ^ { d } \in \mathbb { R } ^ { | \nu | }$ over the vocabulary ??, while the frozen AR verifier produces target logits $z _ { j } ^ { t }$ on the same context. We define the temperaturescaled distributions

![](images/4bdc132826d0b83cafbcfd82a3356fc5213ddbcfc353e5070008323355b1122a.jpg)  
Figure 5 | Visualizing tri-mode inference: (a) left-to-right AR decoding, (b) parallel difusion decoding, and (c) linear self-speculation decoding (quadratic self-speculation is visualized in Fig. 12).

$$
q _ {j} = \mathrm{softmax} (z _ {j} ^ {d} / \tau), \qquad p _ {j} = \mathrm{softmax} (z _ {j} ^ {t} / \tau),
$$

with $\tau { = } 3 . 0$ . The target $p _ { j }$ is treated as fixed (stopgradient), so only the drafter $q _ { j }$ carries gradient through the LoRA parameters.

Active position mask: “accepted + 1”. As shown in Fig. 11, both loss terms are computed only on the accepted prefix plus the first rejected position. Letting $j ^ { * }$ denote the position of the first mismatch (the smallest $j$ with $\hat { x } _ { n + j } \neq x _ { n + j } ^ { \mathrm { A R } } )$ , the active set is ${ \mathcal { A } } =$ $\{ 1 , \ldots , j ^ { * } \}$ when there is a rejection in the block, and $A = \{ 1 , \ldots , k \}$ otherwise; all other positions are masked out and contribute neither to the loss numerator nor to its denominator. This mask is essential because, at inference, the verifier’s KV cache is rebuilt at the rejection point: logits at positions $j > j ^ { * }$ are conditioned on a continuation the deployed loop never observes, so training on them would bias the drafter toward a counterfactual distribution.

LK-hybrid distribution-matching loss. The distribution-matching term adapts the LK-hybrid loss of [19] to a truncated top-?? support. We retain the union of the top-?? token indices, $\bar { \mathcal { U } } _ { j } = \mathcal { S } _ { j } ^ { t } \cup \mathcal { S } _ { j } ^ { d }$ , where $S _ { j } ^ { t } \ ( \mathrm { r e s p . } \ S _ { j } ^ { d } )$ is the set of ?? indices with the largest probability under $p _ { j } ~ ( \mathrm { r e s p . } ~ q _ { j } )$ . Setting ??=200 gives $| \mathcal { U } _ { j } | \le 2 K = 4 0 0$ . We zero both distributions outside $\mathcal { U } _ { j }$ and renormalize to obtain ${ \tilde { p } } _ { j }$ and $\tilde { q } _ { j }$ . Truncation to the union avoids the K $\mathsf { L } ( \tilde { p } _ { j } \parallel \tilde { q } _ { j } ) = \infty$ catastrophe of full-vocabulary KL. The per-position hybrid loss is

$$
\mathcal {L} _ {j} ^ {\mathrm{LK}} = \lambda_ {j} \cdot \mathrm{KL} (\tilde {p} _ {j} \| \tilde {q} _ {j}) + (1 - \lambda_ {j}) \cdot \frac {1}{2} \sum_ {v \in \mathcal {U} _ {j}} | \tilde {p} _ {j} (v) - \tilde {q} _ {j} (v) |,\tag{6}
$$

where the right-hand total-variation (TV) term equals $1 \ - \ \alpha _ { j } ,$ with $\begin{array} { r c l } { \alpha _ { j } } & { = } & { \sum _ { v \in { \mathcal U } _ { i } } } \end{array}$ min $\left( \tilde { p } _ { j } ( v ) , \tilde { q } _ { j } ( v ) \right)$ the standard speculative-decoding acceptance probability [17]—the probability that a token sampled from $\tilde { q } _ { j }$ is accepted by the speculative-decoding rejection rule against ${ \tilde { p } } _ { j }$ . The adaptive coeficient $\lambda _ { j } =$ $\mathrm { e x p } ( - \eta \cdot \mathrm { s g } [ \alpha _ { j } ] )$ with $\eta { = } 0 . 5$ makes the loss behave like the (forward) KL early in training (when $\alpha _ { j } \approx 0$ and $\lambda _ { j } \approx 1$ , providing a stronger distribution-matching gradient) and like TV as the drafter approaches the verifier $( \alpha _ { j }  1 $ and $\lambda _ { j } \to e ^ { - \eta }$ ≈ 0.6, directly minimizing the acceptance-rate gap).

Cross-entropy term. The LK-hybrid term matches the full top-?? output distribution. We additionally apply a token-level cross-entropy at each active position against the verifier’s argmax target $y _ { j } = \arg \operatorname* { m a x } _ { v } z _ { j } ^ { t } ( v )$ on the same union support:

$$
\ell_ {j}   =   \left\{ \begin{array}{l l} - \log q _ {j} ^ {(\mathcal {U} _ {j})} (y _ {j}) & \text { if } y _ {j} \in \mathcal {U} _ {j}, \\ 0 & \text { otherwise }, \end{array} \right.\tag{7}
$$

where $q _ { j } ^ { ( \mathcal { U } _ { j } ) }$ is the drafter softmax restricted to $\mathcal { U } _ { j }$ with $\tau { = } 1 . 0$ . The cross-entropy provides a strong teacher-forcing signal toward the verifier’s modal token, complementing the soft distribution-matching of LK. When the truncation excludes $y _ { j } \ ( \mathrm { r a r e } , < 2 \%$ at $K { = } 2 0 0 ) , ~ \ell _ { j }$ is set to zero; that position is also dropped from the CE denominator below $\left( \mathrm { E q . ~ } 8 \right)$ , so occasional truncation misses cannot blow up the loss.

Total loss and training-time drafter sampling. The two terms are aggregated as masked means over the active positions of each draft block:

$$
\mathcal {L} _ {\mathrm{LK}} = \frac {1}{| \mathcal {A} |} \sum_ {j \in \mathcal {A}} \mathcal {L} _ {j} ^ {\mathrm{LK}}, \qquad \mathcal {L} _ {\mathrm{CE}} = \frac {\sum_ {j \in \mathcal {A}} \ell_ {j}}{\sum_ {j \in \mathcal {A}} \mathbb {1} \{y _ {j} \in \mathcal {U} _ {j} \}}.\tag{8}
$$

These per-block means are further averaged across the inner training batch. The total loss is

$$
\mathcal {L} = \lambda_ {\mathrm{KL}} \cdot \mathcal {L} _ {\mathrm{LK}} + \lambda_ {\mathrm{CE}} \cdot \mathcal {L} _ {\mathrm{CE}}, \quad \lambda_ {\mathrm{KL}} = \lambda_ {\mathrm{CE}} = 1.\tag{9}
$$

At training time, 90% of the inner-batch slots draw their drafter tokens by sampling from softmax $( z _ { j } ^ { d } / T _ { \mathrm { d r a f t } } )$ with $T _ { \mathrm { { d r a f t } } } { = } 1 . 0 ;$ the rest stay greedy. Sampling exposes the LK gradient to a broader empirical distribution of drafter outputs and yields adapters that remain robust when the verifier is itself sampled at inference.

![](images/1bbf3b1dfa6aec8a276612fdbeda730de14c1ec790dbe5da33ec75c8a89e73fe.jpg)  
Figure 6 | An example of using the recursive dynamic compaction method to identify the SOL path.

## 3.4. Variant: Quadratic Self-Speculation

A variant of linear self-speculation is quadratic selfspeculation, which leverages quadratic decoding [7] with single-forward drafting and verification, following the same process as [20]. This decoding scheme pre pares for the worst case by predicting the next block at all possible acceptance positions with a quadratic cost. Specifically, quadratic self-speculation performs speculative drafting and verification simultaneously within a single forward pass by using a structured attention mask, where causal predictions verify previous drafts while parallel difusion predictions generate new draft tokens for the next iteration. The interleaved quadratic layout ensures that each iteration consistently produces ?? speculative tokens even when verification terminates early due to mismatches. In addition to standard AR-based verification, our tri-mode model further supports an AR-difusion ensemble verifier that combines causal and difusion predictions through weighted interpolation. More details are provided in Appendix C and Fig. 12.

## 4. Speed-of-Light Analysis

We conduct a speed-of-light (SOL) analysis to quantify the maximum acceptance rate / token-perforward achievable by the difusion mode. We apply this analysis to the difusion mode of Nemotron-Labs-Difusion-8B delivered in Sec. 3.2. The SOL ceiling tells us how much intrinsic parallelism the current confidence-based sampling is leaving on the table. SOL is computed entirely within the difusion model, i.e., no AR verifier is involved, so it isolates the diffusion model’s own parallel-decoding capability and provides a reference ceiling for any scheme that targets its converged output. Compared with linear self-speculation in Sec. 3, which only commits a contiguous prefix of the draft and is therefore truncated at the first rejection, difusion-mode decoding can commit any subset of masked positions per pass.

## 4.1. Difusion SOL Construction

Oracle target via serial denoising. We first define the difusion model’s converged output for each block of length ??. Let $f _ { \theta }$ denote the difusion model, which on a partially masked input outputs a categorical distribution over the vocabulary at every masked position; let [M] denote the mask token. Starting from an all-mask input $\mathbf { x } ^ { ( 0 ) } = \left[ \mathsf { M } \right] ^ { B }$ , at each step we identify the masked position whose output distribution has the highest peak probability (across positions and vocabulary), commit its argmax to that position, and re-evaluate $f _ { \theta }$ on the resulting partially-unmasked sequence; we repeat until all ?? positions are filled. This serial denoising procedure uses exactly ?? forward passes—one position committed per pass—and yields a target sequence $\mathbf { t } ~ = ~ ( t _ { 1 } , \ldots , t _ { B } )$ that the model would converge to in the absence of any parallel commits. The SOL acceptance ratio is then the average TPF needed to reproduce t from the same all-mask input under a parallel scheme.

Greedy parallel acceptance. The simplest parallel scheme is greedy acceptance. At iteration $k ,$ the model produces argmax predictions $\hat { \mathbf { t } } ^ { ( k ) }$ for every position from the current input $\mathbf { x } ^ { ( k ) }$ , and we commit every masked position whose prediction matches the serial target, $\mathcal { A } ^ { ( k ) } = \{ j : x _ { j } ^ { ( k ) } = [ \mathtt { M } ] \ \wedge \ \hat { t } _ { j } ^ { ( k ) } = t _ { j } \}$ ; if no position matches, we commit the single highestconfidence position as a fallback. After ?? iterations the block is fully unmasked and the realized TPF is $B / K$ . Greedy is fast but is not always exact: each committed token becomes part of the context for the next forward pass, and committing several contextdependent tokens at once can shift the conditional distribution at the remaining positions away from t. This motivates a second scheme that recovers t exactly on every block.

Recursive dynamic compaction. To recover t exactly on every block, we replace greedy acceptance with a strategic search for the largest safe subset of matching positions, as shown in Fig. 6. At each iteration, we rank the ?? matched positions by model confidence as $\left( p _ { 1 } , \ldots , p _ { N } \right)$ and search for the largest prefix $\{ p _ { 1 } , \ldots , p _ { k } \}$ whose commit is safe—where “safe” means that continuing decoding on the remaining positions still arrives at t. Each safety check itself runs the decoder one level shallower (greedy acceptance) under a simulation budget of 5000 forward passes per block, which is what makes the scheme recursive; if the budget is exceeded, the candidate is treated as unsafe and the binary search shrinks the prefix. Because the top-1 match is always safe (committing one position is no diferent from a serial step), the scheme commits at least one position per iteration and its TPF dominates greedy whenever greedy is exact, while recovering exact-match cases that greedy misses. We use this scheme to report SOL throughout this section; greedy serves as a fast lower-bound proxy.

![](images/6b4e93318e90baf0d25f25873c2036b3bd64353b7e6477dd1ecc5854a8e20ba5.jpg)

![](images/5f3d4346499efe154fba5df53f64aeea6169a97ec819d02c0fb08ff800ca9309.jpg)  
Figure 7 | Visualizing the acceptance rate and TPF across diferent SPEED-Bench categories for difusion SOL and linear self-speculation. The average metrics across all categories are highlighted in red.

## 4.2. SOL Evaluation on SPEED-Bench

We apply recursive dynamic compaction to 713 SPEED-Bench [1] samples spanning 11 categories on the difusion mode of Nemotron-Labs-Difusion-8B (instruct), sweeping the block length ?? ∈ {4, 8, 16, 32}. We also report the average accuracy achieved by SOL under diferent block lengths on the 10 instruct LM benchmarks in Sec. 6.1 to understand their impact.

Observations on difusion SOL. From Tab. 4, we observe that (1) The difusion mode exhibits substantial intrinsic parallelism: the SOL acceptance rate reaches 7.60× on average and exceeds 10× on multilin gual and coding content at ?? = 32, and grows nearly linearly with ??, from 2.89× at ?? = 4 to 7.60× at ?? = 32. (2) Confidence-based sampling, by contrast, achieves only ∼3× TPF at comparable accuracy in the block-32 difusion-mode results of Sec. 6.1, leaving a notable gap to the 7.60× SOL ceiling. This indicates that confidence-based sampling is far from optimal and that there is substantial headroom for better-designed samplers to capture. (3) Per-category

Table 4 | Per-category SOL acceptance ratio on SPEED-Bench under recursive dynamic compaction. Benchmark accuracy is averaged over 10 instruct LM benchmarks in Sec. 6.1.

<table><tr><td>Category</td><td>BL=32</td><td>BL=16</td><td>BL=8</td><td>BL=4</td></tr><tr><td>coding</td><td>10.24</td><td>7.50</td><td>5.32</td><td>3.32</td></tr><tr><td>humanities</td><td>6.93</td><td>5.00</td><td>3.76</td><td>2.76</td></tr><tr><td>math</td><td>9.30</td><td>7.02</td><td>4.90</td><td>3.20</td></tr><tr><td>multilingual</td><td>11.26</td><td>8.08</td><td>5.46</td><td>3.37</td></tr><tr><td>qa</td><td>5.63</td><td>4.43</td><td>3.46</td><td>2.61</td></tr><tr><td>rag</td><td>7.32</td><td>5.67</td><td>4.14</td><td>2.91</td></tr><tr><td>reasoning</td><td>7.22</td><td>5.06</td><td>3.87</td><td>2.79</td></tr><tr><td>roleplay</td><td>3.49</td><td>2.76</td><td>2.41</td><td>2.00</td></tr><tr><td>stem</td><td>8.01</td><td>5.68</td><td>4.11</td><td>2.98</td></tr><tr><td>summarization</td><td>6.02</td><td>4.48</td><td>3.25</td><td>2.71</td></tr><tr><td>writing</td><td>6.13</td><td>4.71</td><td>3.58</td><td>2.62</td></tr><tr><td>Acceptance rate</td><td>7.60</td><td>5.68</td><td>4.17</td><td>2.89</td></tr><tr><td>Benchmark Acc</td><td>61.81</td><td>63.18</td><td>65.43</td><td>64.04</td></tr></table>

SOL spans a ∼3.2× range, from 3.49× on roleplay to 11.26× on multilingual content. We hypothesize this reflects token-level entropy: templated content has more positions that are confidently determined by partial context, while open-ended generation does not. If this holds, samplers that adapt to the local content type could exploit this variance. (4) A moderately small block length achieves the best accuracy, while larger block lengths degrade it. The trade-of between benchmark accuracy and eficiency, i.e., acceptance rate, indicates that improving difusion performance under larger block lengths is a critical direction.

Difusion SOL vs. linear self-speculation. We additionally compare the SOL ceiling with linear selfspeculation (Sec. 3) on SPEED-Bench at ??=32, visualized in Fig. 7. Two distinct metrics matter here: The acceptance rate counts how many tokens are committed per acceptance step. SOL commits multiple positions in a single difusion forward pass, while linear self-speculation accepts up to ?? draft tokens per draft+verify cycle. The real TPF, in contrast, is the average number of tokens committed per single model forward pass: SOL incurs only a per-block KV-cache recompute on top of one forward per acceptance step, so its real TPF is close to its acceptance rate; linear self-speculation, however, uses two forwards per cycle (one difusion draft, one AR verification), so its real TPF is the acceptance rate divided by two. The two settings also target diferent correctness signals: SOL agrees with the difusion mode’s own serial-denoising output, while linear self-speculation agrees with the AR mode verification. We view this as a fair headto-head measurement of parallel-decoding potential, considering that the difusion and AR modes achieve comparable accuracy in Sec. 6.1 and thus both targets are equally credible references for a correct token.

From Fig. 7, we observe that: (1) At the acceptancerate level, linear self-speculation approaches SOL, achieving 6.82× vs. 7.60× overall (approximately 10.3% below the upper bound), with similarly small gaps across categories. Thus, on top of difusion drafting, applying AR verification is an efective way to approach the SOL ceiling. (2) The real TPF gap, however, is much larger: 6.02× for SOL vs. 3.41× for linear self-speculation, i.e., a 76.5% improvement. Beyond the doubled forward-pass cost, linear selfspeculation only commits a contiguous prefix of the draft, discarding confident tokens beyond the first rejection. These two efects together motivate stronger difusion-mode samplers that can safely commit tokens within a single forward pass, including at nonprefix positions.

Implications for the tri-mode framework. The SOL analysis highlights two key takeaways for the trimode framework. (1) Difusion-mode decoding can be a promising approach for highly parallel decoding, provided that an optimized sampler can close the gap between pure confidence-based sampling and the SOL ceiling. (2) AR verification is an efective way to sample from difusion drafts and can approach SOL. However, its additional verification cost and prefixonly acceptance pattern fundamentally cap its real TPF below SOL, even when the difusion drafter and the AR verifier are well aligned.

## 5. Nemotron-Labs-Difusion Family

We deliver the Nemotron-Labs-Difusion model family in 3B, 8B, and 14B sizes, including base and instruct models as well as VLMs.

## 5.1. Base Models

To speed up pretraining, we start from the pretrained Ministral3 [21] base models and apply the two-stage training strategy introduced in Sec. 2.1. Specifically, we adopt the pretraining dataset in [22] and perform continuous pretraining for 1T tokens in Stage 1 (pure AR) and 300B tokens in Stage 2 (joint AR and difusion training with an ?? of 0.3). The initial learning rate is set to 1e-5 and decayed to 3e-6 using a WSD schedule [23] with the AdamW optimizer and a weight decay of 0.1. We adopt a global batch size of 512 and a sequence length of 4096. The training is performed on 256 NVIDIA H100 GPUs. We release the training and inference pipeline through Megatron Bridge.

## 5.2. Instruct Models

We perform supervised fine-tuning (SFT) on top of our base models to deliver instruct models. Specifically, we adopt joint AR and difusion training with an ?? of 0.3 throughout the SFT process. The initial learning rate is set to 2.5e-6 and decayed to 2.5e-7 using the WSD schedule [23] with the AdamW optimizer and a weight decay of 0.1. We train the mode on 45B tokens from the SFT dataset of [24], with a global batch size of 256 and a sequence length of 16k. Following [2], the training pipeline is the same as pretraining except that we do not mask any tokens from the prompt, and the loss is computed only on the answer parts. The training is performed on 256 NVIDIA H100 GPUs.

## 5.3. Vision-Language Models

We extend Nemotron-Labs-Difusion to the visionlanguage setting by adding a vision encoder and a multimodal projector to the difusion LM backbone. The resulting VLM inherits the joint AR-difusion training objective, the dual-stream attention pattern, and the tri-mode inference capability. Below, we describe how the VLM is initialized from pretrained components and how image features are integrated into the dual-stream training layout.

Architecture. The VLM augments Nemotron-Labs-Difusion with a vision encoder and a two-layer MLP projector with 2 × 2 patch merging. The difusion training and inference pipeline is shared with the text-only model, with only the vision frontend added.

Weight initialization. We initialize the LM backbone and LM head from the Nemotron-Labs-Difusion-8B instruct model, which carries the difusion-aware representations learned during joint AR-difusion training, and initialize the vision encoder and projector from the corresponding AR VLM (Ministral3-8B-Instruct-2512) from the same model family, which provides fully trained visual perception and cross-modal alignment weights.

Because the LM architectures are identical between the two sources, the merge is exact with no parameter mismatch or interpolation. No new parameters are introduced; the vocabulary and embedding dimensions remain unchanged.

Table 5 | Benchmark our Nemotron-Labs-Difusion-8B instruct model against SOTA AR and difusion instruct LMs across scientific QA, instruction following, coding, and math reasoning benchmarks.

<table><tr><td>Model</td><td>Qwen2.5 7B</td><td>Qwen3 8B</td><td>Ministral3-8B Instruct-2512</td><td>LLaDA-8B Instruct</td><td>Dream-7B Instruct</td><td colspan="2">SDAR-8B Chat</td><td colspan="4">Nemotron-Labs-Diffusion-8B (Tri-Mode in One Model)</td></tr><tr><td>Gen. Mode</td><td>AR</td><td>AR</td><td>AR</td><td>Diff.</td><td>Diff.</td><td>Diff.</td><td>Diff.</td><td>AR</td><td>Diff.</td><td>Linear SS</td><td>Quad. SS</td></tr><tr><td colspan="12">Scientific QA &amp; Instruction Following</td></tr><tr><td>GPQA</td><td>37.12</td><td>49.24</td><td>42.87</td><td>33.30</td><td>33.00</td><td>40.20</td><td>30.80</td><td>44.44</td><td>43.94</td><td>40.40</td><td>44.30</td></tr><tr><td>IFEval</td><td>74.58</td><td>87.38</td><td>64.31</td><td>59.90</td><td>62.50</td><td>61.40</td><td>60.07</td><td>68.65</td><td>68.32</td><td>69.13</td><td>71.00</td></tr><tr><td>MMLU</td><td>74.86</td><td>76.66</td><td>73.90</td><td>65.50</td><td>67.00</td><td>78.60</td><td>78.83</td><td>79.85</td><td>78.71</td><td>79.01</td><td>79.95</td></tr><tr><td colspan="12">Coding</td></tr><tr><td>HumanEval</td><td>77.44</td><td>81.71</td><td>71.04</td><td>49.40</td><td>55.50</td><td>78.70</td><td>79.27</td><td>80.49</td><td>78.66</td><td>81.71</td><td>79.27</td></tr><tr><td>MBPP</td><td>81.55</td><td>81.88</td><td>78.97</td><td>41.00</td><td>58.80</td><td>72.00</td><td>67.32</td><td>85.19</td><td>83.86</td><td>84.92</td><td>85.19</td></tr><tr><td>LCB-CPP</td><td>12.33</td><td>21.09</td><td>20.76</td><td>4.19</td><td>1.25</td><td>13.44</td><td>11.89</td><td>28.85</td><td>26.16</td><td>24.89</td><td>27.70</td></tr><tr><td colspan="12">Math</td></tr><tr><td>Math500</td><td>75.10</td><td>84.80</td><td>83.60</td><td>39.20</td><td>43.00</td><td>78.60</td><td>72.40</td><td>88.00</td><td>85.80</td><td>87.60</td><td>88.80</td></tr><tr><td>GSM8K</td><td>91.89</td><td>92.42</td><td>92.42</td><td>79.91</td><td>81.00</td><td>91.30</td><td>88.48</td><td>94.01</td><td>93.03</td><td>93.78</td><td>94.16</td></tr><tr><td>AIME24</td><td>13.75</td><td>30.21</td><td>27.71</td><td>0.00</td><td>0.00</td><td>16.67</td><td>13.33</td><td>33.33</td><td>46.67</td><td>36.67</td><td>33.33</td></tr><tr><td>AIME25</td><td>6.88</td><td>22.08</td><td>24.58</td><td>0.00</td><td>3.33</td><td>10.00</td><td>3.33</td><td>33.33</td><td>26.67</td><td>30.00</td><td>36.67</td></tr><tr><td colspan="12">Average over All Tasks</td></tr><tr><td>Accuracy</td><td>54.55</td><td>62.75</td><td>58.02</td><td>37.24</td><td>40.54</td><td>54.09</td><td>50.57</td><td>63.61</td><td>63.18</td><td>62.81</td><td>64.04</td></tr><tr><td>TPF</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.75</td><td>1.00</td><td>2.57</td><td>5.99</td><td>6.38</td></tr></table>

Continued SFT. Starting from the merged initialization, we finetune the full model (LM backbone, vision encoder, and projector) with the same joint AR-difusion objective used for the text-only instruct model, on multimodal instruction-following data [25].

Asymmetric dual stream. A straightforward extension doubles all tokens in the noisy half, including vision tokens. However, vision tokens are never masked; only text response tokens are subject to forward corruption. Carrying vision tokens in the noisy half, therefore, adds FLOPs without contributing to the difusion loss. For high-resolution images, this overhead is substantial. We address this with an asymmetric dual-stream layout that strips all vision token positions from the noisy half:

$$
\big [ \tilde {x} _ {t} ^ {\mathrm{(text,} L _ {\mathrm{text}})} \mid x ^ {(L)} \big ], \qquad L _ {\mathrm{text}} = L - N _ {\mathrm{vis}},\tag{10}
$$

where $N _ { \mathrm { v i s } }$ is the number of vision tokens. The clean half retains the full sequence, including vision tokens, preserving complete visual context for the AR objective and for cross-stream conditioning. The total sequence length becomes $L _ { \mathrm { t e x t } } + L$ instead of 2??, and the reduction of FLOPs of attention scales with the vision tokens $N _ { \mathrm { v i s } } / L$

## 6. Evaluation and Analysis

## 6.1. Benchmark Instruct Models

Baselines and benchmarks. We compare our Nemotron-Labs-Difusion-8B instruct model against

SOTA AR instruct models (Qwen3-8B, Qwen2.5-7B, and Ministral3-8B Instruct) and SOTA difusion instruct models (LLaDA-8B Instruct [2], Dream-7B Instruct [3], and SDAR-8B Chat [4]). We evaluate all modes of our model: AR, difusion, and selfspeculation, including both linear and quadratic selfspeculation modes (denoted as linear SS and quadratic SS). We use LoRA-enhanced linear self-speculation by default. For the difusion model evaluation of Nemotron-Labs-Difusion-8B and SDAR-8B Chat [4], we also report tokens per forward (TPF) by selecting diferent denoising thresholds [16]. All models are evaluated in the non-thinking mode.

We evaluate across scientific QA and instruction following (GPQA, IFEval, MMLU), coding (HumanEval, MBPP, LiveCodeBench-CPP), and math reasoning (Math500, GSM8K, AIME24, AIME25). We use NeMo-Skills [26] as the evaluation framework for all AR baselines and our Nemotron-Labs-Difusion-8B, and use the oficial evaluation pipelines provided in the original papers for the difusion baselines [2, 3, 4].

Observations. As shown in Tab. 5, we observe that, compared to SOTA AR and difusion instruct LMs, our Nemotron-Labs-Difusion-8B achieves both higher accuracy and eficiency across all modes. More specifically, (1) In terms of AR performance, Nemotron-Labs-Difusion-8B in AR mode delivers +0.86% higher average accuracy than Qwen3-8B and outperforms all other AR baselines, demonstrating that the joint AR–difusion training objective efectively preserves strong AR accuracy. In fact, the ablation study under a controlled setting in Sec. 2.4 indicates that adding the difusion objective can maintain or slightly improve AR accuracy, potentially due to an improved ability to predict the future. (2) The difusion mode decodes 2.57× TPF while achieving +0.43% higher average accuracy than Qwen3-8B. Compared to existing difusion LMs, Nemotron-Labs-Difusion-8B outperforms SDAR-8B Chat by +9.09% in average accuracy and better maintains accuracy under larger decoding parallelism, as shown in Fig. 1 (b). (3) LoRA-tuned linear self-speculation maintains comparable accuracy to the difusion mode while further boosting TPF to 5.99×, indicating the efectiveness of aligning the difusion drafter with the AR target via lightweight LoRA tuning. (4) Quadratic self-speculation can achieve the highest TPF of 6.38×, as it prepares the next block for all possible acceptance positions at a quadratic cost. However, due to the use of FlexAttention with less optimized kernels for the dedicated attention mask [20], the real-device eficiency of quadratic self-speculation falls behind the linear one according to Fig. 1 (b). As such, we use linear self-speculation by default.

Table 6 | Per-task TPF achieved by linear self-speculation w/ and w/o LoRA tuning across 3B/8B/14B scales.

<table><tr><td>Model Scale</td><td>Setting</td><td>GPQA</td><td>IFEval</td><td>HumanEval</td><td>MBPP</td><td>Math500</td><td>GSM8K</td><td>AIME24</td><td>AIME25</td><td>MMLU</td><td>LCB-CPP</td><td>Avg TPF</td><td>Avg Acc</td></tr><tr><td rowspan="2">3B</td><td>w/o LoRA</td><td>3.07</td><td>2.97</td><td>4.77</td><td>3.74</td><td>4.94</td><td>4.01</td><td>4.33</td><td>4.53</td><td>2.42</td><td>3.34</td><td>3.81</td><td>55.00</td></tr><tr><td>w/ LoRA</td><td>3.57</td><td>3.30</td><td>5.56</td><td>4.23</td><td>5.63</td><td>4.55</td><td>4.99</td><td>5.11</td><td>2.74</td><td>3.95</td><td>4.36</td><td>55.00</td></tr><tr><td rowspan="2">8B</td><td>w/o LoRA</td><td>5.10</td><td>4.32</td><td>4.62</td><td>3.44</td><td>5.43</td><td>4.47</td><td>5.38</td><td>5.05</td><td>3.25</td><td>4.14</td><td>4.52</td><td>62.88</td></tr><tr><td>w/ LoRA</td><td>6.64</td><td>5.52</td><td>5.82</td><td>4.44</td><td>7.36</td><td>5.89</td><td>7.44</td><td>6.92</td><td>4.08</td><td>5.70</td><td>5.99</td><td>62.81</td></tr><tr><td rowspan="2">14B</td><td>w/o LoRA</td><td>5.31</td><td>3.86</td><td>6.75</td><td>4.42</td><td>5.01</td><td>4.54</td><td>4.47</td><td>3.92</td><td>4.95</td><td>3.42</td><td>4.67</td><td>66.35</td></tr><tr><td>w/ LoRA</td><td>6.07</td><td>4.74</td><td>8.11</td><td>5.22</td><td>6.72</td><td>5.79</td><td>5.88</td><td>5.41</td><td>7.22</td><td>4.46</td><td>5.96</td><td>66.36</td></tr></table>

Remark. The tri-mode design enables Nemotron-Labs-Difusion to serve diferent deployment needs within a single model: (1) The AR mode matches or surpasses SOTA AR LMs in accuracy, meaning that Nemotron-Labs-Difusion can serve as a drop-in replacement for any application that currently uses an AR model, with no pipeline changes required. (2) The difusion mode enables one-for-all flexibility: by adjusting the denoising threshold, a single model can achieve a range of accuracy-throughput trade-ofs, as illustrated in Fig. 1 (b). (3) Self-speculation is promising for achieving significant inference speedup through the synergy between AR and difusion. While it sacrifices the flexibility of the difusion mode by only accepting prefix tokens, it provides a reliable mechanism to verify difusion drafts, which can lead to substantial inference acceleration, as demonstrated in Sec. 6.5.

Improve difusion decoding with a better sampler. We evaluate the efectiveness of the proposed sampler in Sec. 3.2. We apply it on top of our instruct model with a block size of 32 and report the average accuracy across all ten tasks under diferent denoising thresholds to obtain the accuracy–TPF trade-of. As shown in Fig. 8, the trained sampler shifts the entire Pareto frontier upward, delivering higher TPF at the same accuracy (e.g., 1.3× TPF) or higher accuracy at the same TPF (e.g., +10.6% accuracy). The improvements from this simple design suggest that part of the gap between the realized difusion-mode TPF and the SOL ceiling in Sec. 4 can be closed by learning the acceptance policy itself.

![](images/96c8e9332ca5c95f140a29bb91384da9896290a26323006173f12c7102bdb92d.jpg)  
Figure 8 | Comparing the accuracy-TPF trade-ofs achieved w/ and $\mathrm { w / o }$ a sampler.

The impact of LoRA tuning for linear selfspeculation. We perform an ablation study on the impact of LoRA tuning for aligning difusion drafters and AR verifiers. As shown in Tab. 6, we observe that (1) even without LoRA adapters, linear self-speculation already achieves nontrivial TPF, e.g., 4.67× for our 14B model, with larger model scales generally leading to higher TPF; and (2) adding LoRA tuning consistently improves TPF, yielding 14.4%/32.5%/27.6% relative gains at the 3B/8B/14B scales. We also note that the small accuracy gap between diferent self-speculation settings and the AR mode is due to kernel mismatches between 1-token decoding and multi-token prefilling.

## 6.2. Extend to More Model Scales

Baselines and benchmarks. We extend the evaluation to two additional scales, Nemotron-Labs-Difusion-3B/14B, under the same evaluation protocol as Sec. 6.1, and benchmark against SOTA open-source AR instruct models at the corresponding scales.

Observations. As shown in Tab. 7, we observe that: (1) Nemotron-Labs-Difusion maintains consistent improvements in accuracy and eficiency across scales and generation modes. For example, using LoRAtuned linear self-speculation, our Nemotron-Labs-Difusion-3B/14B outperforms the strongest baselines, Qwen3-4B/14B, by +1.77%/+1.19% in accuracy while achieving 4.36×/5.96× TPF, respectively.

Table 7 | Benchmark our Nemotron-Labs-Difusion-3B/14B instruct models against SOTA AR instruct models.

<table><tr><td>Model</td><td>Gen. Mode</td><td>GPQA</td><td>IFEval</td><td>HumanEval</td><td>MBPP</td><td>Math500</td><td>GSM8K</td><td>AIME24</td><td>AIME25</td><td>MMLU</td><td>LCB-CPP</td><td>Avg Acc</td><td>Avg TPF</td></tr><tr><td colspan="14">3B Scale</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>AR</td><td>27.78</td><td>69.69</td><td>65.85</td><td>66.93</td><td>32.60</td><td>62.09</td><td>0.00</td><td>0.00</td><td>62.45</td><td>9.25</td><td>39.90</td><td>1.00</td></tr><tr><td>Phi-4-mini-Instruct</td><td>AR</td><td>40.91</td><td>71.16</td><td>77.44</td><td>75.13</td><td>70.80</td><td>76.57</td><td>10.00</td><td>6.67</td><td>53.37</td><td>9.91</td><td>49.20</td><td>1.00</td></tr><tr><td>Ministral3-3B-Instruct-2512</td><td>AR</td><td>28.79</td><td>57.55</td><td>64.79</td><td>68.39</td><td>71.90</td><td>87.41</td><td>12.50</td><td>12.71</td><td>63.21</td><td>12.50</td><td>47.97</td><td>1.00</td></tr><tr><td>Qwen3-4B</td><td>AR</td><td>37.18</td><td>72.20</td><td>75.91</td><td>72.49</td><td>68.40</td><td>92.19</td><td>10.63</td><td>10.63</td><td>77.05</td><td>15.58</td><td>53.23</td><td>1.00</td></tr><tr><td>Nemotron-Labs</td><td>AR</td><td>39.39</td><td>69.39</td><td>76.22</td><td>71.16</td><td>77.60</td><td>87.87</td><td>23.33</td><td>16.67</td><td>71.70</td><td>21.37</td><td>55.50</td><td>1.00</td></tr><tr><td rowspan="3">-Diffusion-3B (Tri-Mode)</td><td>Diff.</td><td>33.84</td><td>68.93</td><td>74.39</td><td>73.54</td><td>74.80</td><td>88.40</td><td>16.67</td><td>10.00</td><td>72.06</td><td>16.30</td><td>52.90</td><td>1.91</td></tr><tr><td>Linear SS</td><td>35.86</td><td>68.96</td><td>75.00</td><td>70.11</td><td>77.40</td><td>87.79</td><td>26.67</td><td>16.67</td><td>71.89</td><td>20.04</td><td>55.00</td><td>4.36</td></tr><tr><td>Quad. SS</td><td>42.93</td><td>71.36</td><td>79.27</td><td>76.46</td><td>78.20</td><td>88.25</td><td>13.33</td><td>16.67</td><td>72.12</td><td>19.60</td><td>55.80</td><td>5.42</td></tr><tr><td colspan="14">14B Scale</td></tr><tr><td>Gemma-3-12B-IT</td><td>AR</td><td>38.32</td><td>85.73</td><td>58.23</td><td>85.45</td><td>84.55</td><td>90.45</td><td>23.75</td><td>16.88</td><td>76.41</td><td>20.54</td><td>58.03</td><td>1.00</td></tr><tr><td>Phi-3-Medium-14B</td><td>AR</td><td>37.56</td><td>85.75</td><td>70.43</td><td>76.65</td><td>43.35</td><td>89.69</td><td>1.88</td><td>0.62</td><td>76.67</td><td>10.79</td><td>49.34</td><td>1.00</td></tr><tr><td>Phi-4-14B</td><td>AR</td><td>56.94</td><td>68.96</td><td>84.60</td><td>83.93</td><td>79.95</td><td>92.27</td><td>19.17</td><td>15.83</td><td>84.76</td><td>21.75</td><td>60.82</td><td>1.00</td></tr><tr><td>Ministral3-14B-Instruct-2512</td><td>AR</td><td>52.02</td><td>71.51</td><td>72.56</td><td>82.47</td><td>86.30</td><td>92.80</td><td>36.25</td><td>29.38</td><td>79.88</td><td>26.54</td><td>62.97</td><td>1.00</td></tr><tr><td>Qwen3-14B</td><td>AR</td><td>50.51</td><td>88.36</td><td>83.54</td><td>87.30</td><td>85.40</td><td>94.31</td><td>33.33</td><td>20.00</td><td>81.51</td><td>27.42</td><td>65.17</td><td>1.00</td></tr><tr><td>Nemotron-Labs</td><td>AR</td><td>54.55</td><td>68.50</td><td>86.59</td><td>85.19</td><td>88.40</td><td>91.36</td><td>46.67</td><td>43.33</td><td>82.51</td><td>27.48</td><td>67.46</td><td>1.00</td></tr><tr><td rowspan="3">-Diffusion-14B (Tri-Mode)</td><td>Diff.</td><td>48.99</td><td>69.03</td><td>83.54</td><td>82.80</td><td>85.80</td><td>93.71</td><td>43.33</td><td>50.00</td><td>82.17</td><td>25.77</td><td>66.51</td><td>2.74</td></tr><tr><td>Linear SS</td><td>47.47</td><td>70.06</td><td>85.37</td><td>84.66</td><td>86.60</td><td>92.04</td><td>50.00</td><td>40.00</td><td>81.11</td><td>26.32</td><td>66.36</td><td>5.96</td></tr><tr><td>Quad. SS</td><td>52.02</td><td>72.15</td><td>87.20</td><td>85.45</td><td>88.00</td><td>92.12</td><td>53.33</td><td>40.00</td><td>82.45</td><td>28.74</td><td>68.15</td><td>6.92</td></tr></table>

Table 8 | Benchmark our Nemotron-Labs-Difusion-8B base model against SOTA AR and difusion base LMs across coding, math, knowledge, and commonsense reasoning benchmarks.

<table><tr><td>Model</td><td>Gen. Mode</td><td>Human Eval</td><td>Human Eval+</td><td>MBPP</td><td>MBPP+</td><td>GSM8K</td><td>Minerva Math</td><td>MMLU</td><td>ARC-E</td><td>ARC-C</td><td>Hella swag</td><td>PIQA</td><td>Wino grande</td><td>Avg Acc</td><td>Avg TPF</td></tr><tr><td>Llama-3.1-8B</td><td>AR</td><td>35.37</td><td>28.66</td><td>48.80</td><td>61.90</td><td>54.06</td><td>18.22</td><td>65.15</td><td>81.31</td><td>53.41</td><td>78.93</td><td>81.18</td><td>77.43</td><td>57.04</td><td>1.00</td></tr><tr><td>Ministral3-8B</td><td>AR</td><td>42.68</td><td>38.41</td><td>61.60</td><td>76.98</td><td>80.21</td><td>44.58</td><td>76.39</td><td>86.15</td><td>60.75</td><td>79.01</td><td>80.74</td><td>73.48</td><td>66.75</td><td>1.00</td></tr><tr><td>Qwen3-8B</td><td>AR</td><td>64.63</td><td>56.71</td><td>69.40</td><td>83.07</td><td>86.73</td><td>52.94</td><td>76.93</td><td>81.90</td><td>53.16</td><td>78.59</td><td>79.22</td><td>75.69</td><td>71.58</td><td>1.00</td></tr><tr><td>LLaDA-8B</td><td>Diff.</td><td>32.32</td><td>27.44</td><td>40.80</td><td>51.85</td><td>70.96</td><td>27.30</td><td>65.86</td><td>73.78</td><td>49.15</td><td>71.05</td><td>73.88</td><td>74.66</td><td>54.92</td><td>1.00</td></tr><tr><td>Dream-7B</td><td>Diff.</td><td>54.88</td><td>49.39</td><td>56.80</td><td>74.60</td><td>77.18</td><td>39.60</td><td>67.00</td><td>82.20</td><td>59.13</td><td>73.73</td><td>75.52</td><td>73.56</td><td>65.30</td><td>1.00</td></tr><tr><td rowspan="4">Nemotron-Labs -Diffusion-8B (Tri-Mode)</td><td>AR</td><td>60.37</td><td>53.05</td><td>68.20</td><td>82.54</td><td>88.25</td><td>66.00</td><td>74.68</td><td>83.38</td><td>58.11</td><td>76.08</td><td>80.09</td><td>71.98</td><td>71.89</td><td>1.00</td></tr><tr><td>Diff.</td><td>62.80</td><td>57.32</td><td>67.00</td><td>81.75</td><td>87.26</td><td>65.16</td><td>74.68</td><td>83.38</td><td>58.11</td><td>76.08</td><td>80.09</td><td>71.98</td><td>72.13</td><td>2.06</td></tr><tr><td>Linear SS</td><td>63.41</td><td>56.10</td><td>67.20</td><td>81.75</td><td>88.17</td><td>67.38</td><td>74.68</td><td>83.38</td><td>58.11</td><td>76.08</td><td>80.09</td><td>71.98</td><td>72.36</td><td>4.67</td></tr><tr><td>Quad. SS</td><td>62.20</td><td>54.88</td><td>67.60</td><td>81.48</td><td>88.48</td><td>66.24</td><td>74.68</td><td>83.38</td><td>58.11</td><td>76.08</td><td>80.09</td><td>71.98</td><td>72.10</td><td>7.04</td></tr></table>

(2) Based on the performance of Nemotron-Labs-Difusion-3B/8B/14B, larger LMs generally more readily unlock parallel difusion abilities, as the TPF of difusion/self-speculation modes grows broadly with scale. For example, the TPF of linear self-speculation increases from 4.36× to 5.96× when scaling from 3B to 14B. We attribute this to the stronger futureprediction abilities of larger models, which yield more reliable draft predictions.

## 6.3. Benchmark Base Models

Baselines and benchmarks. We compare Nemotron-Labs-Difusion-8B against SOTA AR base LMs and two representative difusion LMs (LLaDA-8B [2] and Dream-7B [3]). We evaluate on coding benchmarks (HumanEval, HumanEval+, MBPP, MBPP+), math reasoning (GSM8K, Minerva Math), knowledge (MMLU), and commonsense reasoning (ARC-E, ARC-C, Hellaswag, PIQA, Winogrande).

Observations. As shown in Tab. 8, we observe findings consistent with the instruct model results. Our Nemotron-Labs-Difusion-8B base model achieves both higher accuracy and eficiency across all modes: (1) the AR mode delivers +5.14%/+0.31% higher average accuracy than Ministral3-8B/Qwen3-8B; (2) the difusion mode delivers +17.21%/+6.83% higher accuracy than LLaDA-8B/Dream-7B; and (3) selfspeculation achieves 4.67× TPF (linear) and 7.04× TPF (quadratic) with over 0.5% higher average accuracy compared to the strongest baseline, Qwen3-8B.

## 6.4. Benchmark VLMs

Benchmarks and evaluation settings. We evaluate on a diverse set of VLM benchmarks spanning two categories. Short-answer benchmarks require brief, factual responses: AI2D [27], ChartQA [28], DocVQA [29], MMMU [30], MathVista [31], and RealWorldQA [32]. Long-answer benchmarks require extended chain-of-thought reasoning: MMMU-Pro-V [33]. We benchmark against existing difusion VLMs [34, 35, 36, 37]. All benchmarks are evaluated using VLMEvalKit [38] under the same prompts and post-processing as the AR baseline (Ministral3 VLM). Throughput (tokens per second, TPS) is measured on a single NVIDIA H100 GPU with identical prompt batching for fair comparison.

Observations. As shown in Tab. 9, we compare our Nemotron-Labs-Difusion-VLM against existing difusion VLMs in three modes: difusion, AR, and linear self-speculation decoding. We observe that (1) In terms of AR performance, our model delivers 1.3% higher average accuracy than the strongest baseline, LLaDA-V-8B. (2) The difusion mode provides 2.46×–3.15× TPF while maintaining competitive accuracy. (3) Linear self-speculation preserves near-AR accuracy, with only a 0.1% average accuracy drop, while further increasing decoding parallelism to 3.63×–7.45× TPF, where the higher end is achieved for responses exceeding 200 tokens. This implies that the advantage of our model is most pronounced on tasks requiring longer reasoning. These results demonstrate that the joint AR–difusion training framework extends efectively to the vision-language setting, preserving the broad capabilities of the LM backbone while enabling eficient multi-token decoding.

Table 9 | Benchmarking discrete difusion VLMs and Nemotron-Labs-Difusion-VLM across tasks. The difusion mode of Nemotron-Labs-Difusion-VLM uses denoising threshold $\tau { = } 0 . 9$

<table><tr><td>Model</td><td>Gen. Mode</td><td>AI2D</td><td>ChartQA</td><td>DocVQA</td><td>MMMU</td><td>MMMU Pro-10c</td><td>MMMU Pro-V-CoT</td><td>Math Vista</td><td>RealWorld QA</td><td>TPF</td><td>Acc</td></tr><tr><td>MMaDA</td><td>Diff.</td><td>67.4</td><td>9.6</td><td>9.5</td><td>30.2</td><td>16.5</td><td>8.5</td><td>33.4</td><td>49.2</td><td>1</td><td>28.0</td></tr><tr><td>LaViDa</td><td>Diff.</td><td>70.0</td><td>59.0</td><td>64.6</td><td>43.3</td><td>28.7</td><td>10.5</td><td>44.8</td><td>54.5</td><td>1</td><td>46.9</td></tr><tr><td>Dimple</td><td>Diff.</td><td>74.4</td><td>63.4</td><td>37.7</td><td>45.2</td><td>23.8</td><td>12.4</td><td>42.3</td><td>55.4</td><td>1</td><td>44.3</td></tr><tr><td>LLaDA-V-8B</td><td>Diff.</td><td>77.8</td><td>78.3</td><td>83.9</td><td>48.6</td><td>35.2</td><td>18.6</td><td>59.7</td><td>63.2</td><td>1</td><td>58.2</td></tr><tr><td></td><td>AR</td><td>75.0</td><td>81.3</td><td>89.2</td><td>50.3</td><td>32.6</td><td>24.3</td><td>60.4</td><td>62.6</td><td>1</td><td>59.5</td></tr><tr><td rowspan="2">Nemotron-Labs-Diffusion-VLM-8B</td><td>Diff.</td><td>74.7</td><td>76.6</td><td>88.3</td><td>50.4</td><td>31.7</td><td>22.2</td><td>58.5</td><td>60.3</td><td rowspan="2">2.46 all samples 2.80 tok&gt;100 3.15 tok&gt;200 3.63 all samples 6.03 tok&gt;100 7.45 tok&gt;200</td><td>57.9</td></tr><tr><td>Linear SS</td><td>74.9</td><td>81.2</td><td>89.3</td><td>50.0</td><td>32.8</td><td>24.1</td><td>60.7</td><td>62.4</td><td>59.4</td></tr></table>

## 6.5. Inference Eficiency

We analyze and compare the deployment eficiency of Nemotron-Labs-Difusion against MTP/Eagle3-style speculative decoding.

Self-speculation vs. MTP. MTP methods such as Eagle3 [8] have become the default choice for efi cient LLM deployment at low concurrency, where a small model is used to draft multiple future tokens, and then a single forward pass of a larger AR model verifies and accepts some of them. This schedule is more eficient at low concurrency because the memory transfer cost is similar between token-by-token generation and verification passes, while the latter can accept multiple tokens in a single pass. The two main bottlenecks of Eagle3 are: (1) the draft model has limited capacity and is less reliable beyond a short horizon; and (2) proposals are generated recursively, so even if the draft model is tiny, it still incurs the cost of the embedding layer and LM head. In contrast, Nemotron-Labs-Difusion provides unique advantages through (1) significantly higher acceptance length, and (2) token-parallel drafting that enables better GPU utilization.

Setup. We deploy Nemotron-Labs-Difusion-8B with the SGLang server and profile it on NVIDIA GB200, RTX Pro 6000, and DGX Spark under diferent concurrency levels, comparing against Qwen3-8B-Eagle3, with results shown in Fig. 1 (c) and Fig. 9. Evaluations are conducted on SPEED-Bench [1] across four categories (math, coding, reasoning, and multilingual), and limiting generation length to 1024 tokens to avoid repetition/hallucinations. We perform a grid search over hyperparameters for Eagle3, whereas for Nemotron-Labs-Difusion we only vary the block length. We additionally report a SOL throughput estimate mentioned in Sec. 4, providing a reference ceiling for current self-speculation infrastructure.

Observations. As shown in Fig. 1 (c) and Fig. 9, we observe that: (1) Linear self-speculation consistently improves user throughput over the AR mode across all three GPUs, achieving up to 3.3× speedup over AR on GB200 (3.97× speedup and 1015 tok/sec with an optimized kernel), as shown in Fig. 9 (c), and pushing the absolute throughput at batch size 1 to 277/525 tok/sec on RTX Pro 6000 $( 3 . 4 6 \times / 2 . 3 5 \times$ over AR) and 77.5/112.5 tok/sec on DGX Spark (3.14×/2.69× over AR) under FP8/INT4 quantization, demonstrating its efectiveness as a drop-in low-concurrency acceleration scheme. (2) Compared with Eagle3, linear self-speculation delivers a 2.4×/2.3×/1.8× speedup at batch size 1 on GB200/RTX Pro 6000/DGX Spark and achieves better trade-ofs between system throughput and per-user throughput, as shown in Fig. 1 (c) and Fig. 9 (a). This indicates that difusion drafting paired with AR verification is a more efective acceleration mechanism than auxiliary-head MTP due to its higher acceptance length. (3) The SOL ceiling reveals substantial remaining headroom: on RTX Pro 6000, the projected SOL throughput reaches 7.09×/12.36× over AR under FP8/INT4 quantiza tion, roughly 2× above linear self-speculation.

![](images/3400906b4f8acd018c4df8e903937937cf8958b2a92173e8f017557a3900f667.jpg)

![](images/43e288fc5b91d7af3217394e04ca25c30ca7799e35fe0b2567ea619f9d06b96b.jpg)

![](images/7426928a16b523c3ba7e111dd9e51276340e7a383eee51fabb1884bebbb1b68f.jpg)

![](images/04b05fc3828bda511faf234f5657389befd7d65b35077e9247b6811952860035.jpg)  
Figure 9 | (a) System vs. per-user throughput trade-of on an NVIDIA RTX Pro 6000 GPU; (b)/(c)/(d): The throughput under a concurrency of 1 on NVIDIA RTX Pro 6000, GB200, and DGX Spark, respectively.

Table 10 | Per-category acceptance length on SPEED-Bench [1]. Comparing Native / LoRA for Nemotron-Labs-Difusion-8B and Qwen3-8B-Eagle3 / Qwen3- 9B-MTP, all with draft length 31.

<table><tr><td>Category</td><td>Native</td><td>LoRA</td><td>Eagle3</td><td>MTP</td></tr><tr><td>coding</td><td>6.61</td><td>8.57</td><td>3.14</td><td>5.97</td></tr><tr><td>math</td><td>6.24</td><td>8.14</td><td>2.79</td><td>4.80</td></tr><tr><td>reasoning</td><td>6.18</td><td>7.99</td><td>3.40</td><td>3.68</td></tr><tr><td>multilingual</td><td>7.96</td><td>10.06</td><td>1.91</td><td>4.47</td></tr><tr><td>humanities</td><td>5.01</td><td>6.31</td><td>3.12</td><td>3.76</td></tr><tr><td>qa</td><td>4.01</td><td>4.65</td><td>2.63</td><td>3.50</td></tr><tr><td>rag</td><td>5.07</td><td>6.15</td><td>3.06</td><td>4.75</td></tr><tr><td>roleplay</td><td>4.66</td><td>5.54</td><td>2.10</td><td>2.32</td></tr><tr><td>stem</td><td>5.55</td><td>7.02</td><td>2.92</td><td>4.45</td></tr><tr><td>summarization</td><td>4.47</td><td>5.48</td><td>2.66</td><td>3.69</td></tr><tr><td>writing</td><td>4.28</td><td>5.07</td><td>2.81</td><td>3.21</td></tr><tr><td>Average</td><td>5.46</td><td>6.82</td><td>2.75</td><td>4.24</td></tr><tr><td>4 category avg</td><td>6.75</td><td>8.69</td><td>2.81</td><td>4.73</td></tr></table>

Acceptance length per category. As shown in Tab. 10, Nemotron-Labs-Difusion achieves significantly higher acceptance length than both Eagle3 and MTP across all categories, with average acceptance lengths of 5.46/6.82 for Native/LoRA-tuned Nemotron-Labs-Difusion versus 2.75/4.24 for Ea gle3/MTP. The gap further widens to 6.75/8.69 vs. 2.81/4.73 on the four difusion-friendly categories (coding, math, reasoning, multilingual), implying that difusion drafting yields more reliable multi-token proposals, especially on structured tasks with strong syntactic or semantic constraints.

## 7. Related Work

Difusion language models. To overcome the token-by-token decoding nature of AR LMs, diffusion LMs, both continuous [39, 40, 41] and discrete [42, 43, 44, 45, 46], have been proposed to perform non-AR decoding and thus enable parallel token generation. Among them, masked difusion LMs [43, 44, 2, 3] have been successfully scaled up (e.g., LLaDA [2] and Dream [3]). Follow-up work has further explored alternative difusion LM paradigms [47, 48, 6], and scaled them to larger scales [49, 50, 51] or domain-specific specialists such as coding agents [52, 53, 54, 55], explored dedicated reinforcement learning schemes [56, 57], and extended them to more modalities [36, 34]. Compared to AR LMs, difusion LMs have been demonstrated to be better learners under data-constrained settings [58] and show improved performance in planning [3] and text embedding [59].

Difusion language model acceleration. Despite the acceleration potential of large difusion LMs [2, 3], the gap between bidirectional attention and KV caching, along with the one-token-per-step denoising process, limits their achievable speed-up. To address these challenges, dedicated caching strategies [60, 61, 16] have been developed to reuse computations and approximate bidirectional attention. In addition, to realize the potential of parallel token generation, confidence-based sampling [16], guidance from AR models [62], and adaptive decoding with certainty and positional priors [63] have been proposed. Beyond these training-free methods, [64, 3] propose initializing difusion LMs from AR models with token shifts to accelerate difusion LM training. Block Diffusion [9] combines AR and difusion by performing block-wise AR and in-block difusion to support native KV caching. Follow-up works [13, 10, 4, 65, 66] also convert pretrained AR models or difusion LMs into block-wise ones. [14, 7] further explore combining AR and difusion through either joint training or LoRA modules dedicated to difusion.

## 8. Insights and Future Directions

We deliver Nemotron-Labs-Difusion, a tri-mode language model family trained via joint AR-difusion optimization that unifies AR, difusion, and selfspeculation within a single model. The resulting base, instruct, and vision-language models outperform SOTA open-source AR/difusion LMs in both accuracy and eficiency. The training and analysis of tri-mode LMs reveal the following insights:

1. Tri-mode generation arises naturally from joint AR-difusion training. By enabling both AR and non-AR parallel token prediction within a single model, the joint training objective simultaneously produces three inference modes without any mode-specific architectural modifications.

2. AR and difusion losses are complementary, not competing. The two objectives mutually benefit each other and peak at the same loss coeficient (??=0.3). Adding the AR loss induces left-to-right linguistic priors for difusion, and the difusion loss preserves or slightly improves AR accuracy through better future planning.

3. Self-speculation outperforms MTP methods. Instead of relying on auxiliary prediction heads, self-speculation leverages difusion to generate highquality multi-token drafts and uses AR verification to ensure correctness, achieving higher acceptance rates and better eficiency.

4. Variance reduction is critical for difusion training. The difusion loss introduces intrinsically high variance due to random masking with variable noise levels. More suficiently trained AR starting points (e.g., via two-stage training) or variance-reduction training techniques (e.g., global loss averaging) can improve training efectiveness.

5. Linear self-speculation is currently the most eficient mode. Linear self-speculation achieves the best eficiency in the current infrastructure. Quadratic self-speculation achieves higher TPF per step, making it more promising at batch size 1 with improved infrastructure support.

6. Difusion-mode decoding has substantial headroom. Our SOL analysis shows the potential to correctly predict 76.5% more tokens per forward pass than the current best strategy (linear self-speculation), indicating a more promising upper bound for parallel decoding than speculative decoding based only on prefix decoding.

Looking forward, these insights shed light on several promising directions for further improvements:

1. Closing the gap between practical difusion decoding and its SOL upper bound. Our SOL analysis suggests that difusion-mode decoding could ofer a more attractive path toward par allel decoding than linear decoding, due to its non-prefix acceptance pattern and therefore higher upper bound. However, current confidence-based samplers remain far from this upper bound. Developing optimized samplers that more reliably identify correct tokens, or more advanced training schemes that enable more aggressive parallel sampling of conditionally independent tokens, is a promising direction for closing this gap.

2. Improving draft-verification alignment for self-speculation. Given the strong practical speedup achieved by self-speculation, an important future direction is to better align the difusion draft mode with the AR verification mode during training, thereby improving the acceptance rate. In addition, the cost of drafting can be further reduced by using nested subnets, where weightshared smaller subnets generate drafts through specialized training techniques [67, 68].

3. Beyond prefix-only AR verification. Current AR verification accepts drafted tokens only in a prefix-wise manner, which does not fully exploit the non-AR nature of difusion drafts. A promising direction is to explore difusion-mode verification, potentially using another difusion verifier, to validate multiple non-contiguous drafted tokens and further improve the efective acceptance rate.

4. Enabling higher-level parallelism in difusion generation. Although difusion decoding enables parallel token prediction, its generation order still exhibits a strong left-to-right tendency and mainly provides token-level parallelism. Future training algorithms that encourage segment-level or paragraph-level parallelism could better unlock the global planning ability and eficiency potential of difusion-mode generation.

## References

[1] Talor Abramovich, Maor Ashkenazi, Benjamin Chislett, Tiyasa Mitra, Bita Darvish Rouhani, Ran Zilberstein, Yonatan Geifman, et al. Speed-bench: A unified and diverse benchmark for speculative decoding. arXiv preprint arXiv:2604.09557, 2026.

[2] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language difusion models. arXiv preprint arXiv:2502.09992, 2025.

[3] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Difusion large language models. arXiv preprint arXiv:2508.15487, 2025.

[4] Shuang Cheng, Yihan Bian, Dawei Liu, Linfeng Zhang, Qian Yao, Zhongbo Tian, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, et al. Sdar: A synergistic difusion-autoregression paradigm for scalable sequence generation. arXiv preprint arXiv:2510.06303, 2025.

[5] Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked difusion models on text. arXiv preprint arXiv:2410.18514, 2024.

[6] Shuchen Xue, Tianyu Xie, Tianyang Hu, Zijin Feng, Jiacheng Sun, Kenji Kawaguchi, Zhenguo Li, and Zhi-Ming Ma. Any-order gpt as masked difusion model: Decoupling formulation and architecture. arXiv preprint arXiv:2506.19935, 2025.

[7] Mohammad Samragh, Arnav Kundu, David Harrison, Kumari Nishu, Devang Naik, Minsik Cho, and Mehrdad Farajtabar. Your llm knows the future: Uncovering its multi-token prediction potential. arXiv preprint arXiv:2507.11851, 2025.

[8] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test. arXiv preprint arXiv:2503.01840, 2025.

[9] Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block difusion: Interpolating between autoregressive and difusion language models. arXiv preprint arXiv:2503.09573, 2025.

[10] Yonggan Fu, Lexington Whalen, Zhifan Ye, Xin Dong, Shizhe Diao, Jingyu Liu, Chengyue Wu, Hao Zhang, Enze Xie, Song Han, et al. Eficient-dlm: From autoregressive to difusion language models, and beyond in speed. arXiv preprint arXiv:2512.14067, 2025.

[11] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Push ing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[12] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An opensource llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

[13] Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Eficient block-difusion llm. arXiv preprint arXiv:2509.26328, 2025.

[14] Itai Gat, Heli Ben-Hamu, Marton Havasi, Daniel Haz iza, Jeremy Reizenstein, Gabriel Synnaeve, David Lopez-Paz, Brian Karrer, and Yaron Lipman. Set block decoding is a language model inference acceler ator. arXiv preprint arXiv:2509.04185, 2025.

[15] DeepSeek-AI. Deepseek-v3 technical report, 2024.

[16] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of difusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025.

[17] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative de coding. In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.

[18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.

[19] Aleksei Samarin et al. Lk losses: Direct acceptance rate optimization for speculative decoding. arXiv preprint arXiv:2602.23881, 2026.

[20] Jingyu Liu, Xin Dong, Zhifan Ye, Rishabh Mehta, Yonggan Fu, Vartika Singh, Jan Kautz, Ce Zhang, and Pavlo Molchanov. Tidar: Think in difusion, talk in autoregression. arXiv preprint arXiv:2511.08923, 2025.

[21] Alexander H Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jefares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, et al. Ministral 3. arXiv preprint arXiv:2601.08584, 2026.

[22] Aarti Basant, Abhijit Khairnar, Abhijit Paithankar, Abhinav Khattar, Adithya Renduchintala, Aditya Malte, Akhiad Bercovich, Akshay Hazare, Alejandra Rico, Aleksander Ficek, et al. Nvidia nemotron nano 2: An accurate and eficient hybrid mamba-transformer reasoning model. arXiv preprint arXiv:2508.14444, 2025.

[23] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

[24] Aakshita Chandiramani, Aaron Blakeman, Abdullahi Olaoye, Abhibha Gupta, Abhilash Somasamudramath, Abhinav Khattar, Adeola Adesoba, Adi Renduchintala, Adil Asif, Aditya Agrawal, et al. Nemotron 3 super: Open, eficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning. arXiv preprint arXiv:2604.12374, 2026.

[25] Luis Wiedmann, Orr Zohar, Amir Mahla, Xiaohan Wang, Rui Li, Thibaud Frere, Leandro von Werra, Aritra Roy Gosthipaty, and Andrés Marafioti. Finevision: Open data is all you need, 2025.

[26] NVIDIA Corporation. Nemo-skills: A toolkit for improving skills of large language models. https: //github.com/NVIDIA-NeMo/Skills, 2024. GitHub repository.

[27] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European Conference on Computer Vision (ECCV), pages 235–251, 2016.

[28] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263– 2279, 2022.

[29] Minesh Mathew, Dimosthenis Karatzas, and C.V. Jawahar. Docvqa: A dataset for vqa on document images. In IEEE Winter Conference on Applications of Computer Vision (WACV), pages 2200–2209, 2021.

[30] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A mas sive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9556–9567, 2024.

[31] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chun yuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

[32] xAI. Realworldqa, 2024.

[33] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025.

[34] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large difusion language models. arXiv preprint arXiv:2505.15809, 2025.

[35] Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, and Aditya Grover. Lavida: A large difusion language model for multimodal understanding. arXiv preprint arXiv:2505.16839, 2025.

[36] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language difusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.

[37] Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dim ple: Discrete difusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025.

[38] Haodong Duan, Xinyu Fang, Junming Yang, Xiangyu Zhao, Yuxuan Qiao, Mo Li, Amit Agarwal, Zhe Chen, Lin Chen, Yuan Liu, Yubo Ma, Hailong Sun, Yifan Zhang, Shiyin Lu, Tack Hwa Wong, Weiyun Wang, Peiheng Zhou, Xiaozhe Li, Chaoyou Fu, Junbo Cui, Jixuan Chen, Enxin Song, Song Mao, Shengyuan Ding, Tianhao Liang, Zicheng Zhang, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. arXiv preprint arXiv:2407.11691, 2024.

[39] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Difusion-lm improves controllable text generation. Advances in neural information processing systems, 35:4328–4343, 2022.

[40] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and LingPeng Kong. Difuseq: Sequence to sequence text generation with difusion models. arXiv preprint arXiv:2210.08933, 2022.

[41] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplex-based difusion language model for text generation and modular control. arXiv preprint arXiv:2210.17432, 2022.

[42] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising difusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

[43] Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. Difusionbert: Improving generative masked language models with difusion models. arXiv preprint arXiv:2211.15029, 2022.

[44] Subham Sahoo, Marianne Arriola, Yair Schif, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and efective masked difusion language models. Advances in Neural Information Processing Systems, 37:130136– 130184, 2024.

[45] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete difusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

[46] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete difusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.

[47] Subham Sekhar Sahoo, Zhihan Yang, Yash Akhauri, Johnna Liu, Deepansha Singh, Zhoujun Cheng, Zhengzhong Liu, Eric Xing, John Thickstun, and Arash Vahdat. Esoteric language models. arXiv preprint arXiv:2506.01928, 2025.

[48] Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and Volodymyr Kuleshov. The difusion duality. arXiv preprint arXiv:2506.10892, 2025.

[49] Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. Llada2. 0: Scaling up difusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025.

[50] Tiwei Bie, Maosong Cao, Xiang Cao, Bingsen Chen, Fuyuan Chen, Kun Chen, Lun Du, Daozhuo Feng,

Haibo Feng, Mingliang Gong, et al. Llada2. 1: Speed ing up text difusion via token editing. arXiv preprint arXiv:2602.08676, 2026.

[51] Google DeepMind. Gemini difusion, 2025. Model page: state-of-the-art, experimental text difusion model.

[52] Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, et al. Mercury: Ultra-fast language models based on difusion. arXiv preprint arXiv:2506.17298, 2025.

[53] Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, et al. Seed difusion: A large-scale difusion language model with high-speed inference. arXiv preprint arXiv:2508.02193, 2025.

[54] Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, and Yizhe Zhang. Difucoder: Understanding and improving masked difusion models for code generation. arXiv preprint arXiv:2506.20639, 2025.

[55] Zhihui Xie, Jiacheng Ye, Lin Zheng, Jiahui Gao, Jingwei Dong, Zirui Wu, Xueliang Zhao, Shansan Gong, Xin Jiang, Zhenguo Li, et al. Dream-coder 7b: An open difusion language model for code. arXiv preprint arXiv:2509.01142, 2025.

[56] Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in difusion large language models via reinforcement learning. arXiv preprint arXiv:2504.12216, 2025.

[57] Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, et al. Llada 1.5: Variance reduced preference optimization for large language difusion models. arXiv preprint arXiv:2505.19223, 2025.

[58] Mihir Prabhudesai, Mengning Wu, Amir Zadeh, Katerina Fragkiadaki, and Deepak Pathak. Difusion beats autoregressive in data-constrained settings. arXiv preprint arXiv:2507.15857, 2025.

[59] Siyue Zhang, Yilun Zhao, Liyuan Geng, Arman Co han, Anh Tuan Luu, and Chen Zhao. Difusion vs. autoregressive language models: A text embedding perspective. arXiv preprint arXiv:2505.15045, 2025.

[60] Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating difusion large language models with adaptive caching. arXiv preprint arXiv:2506.06295, 2025.

[61] Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for difusion language models. arXiv preprint arXiv:2505.15781, 2025.

[62] Daniel Israel, Guy Van den Broeck, and Aditya Grover. Accelerating difusion llms via adaptive parallel decoding. arXiv preprint arXiv:2506.00413, 2025.

[63] Qingyan Wei, Yaojie Zhang, Zhiyuan Liu, Dongrui Liu, and Linfeng Zhang. Accelerating difusion large language models with slowfast: The three golden principles. arXiv preprint arXiv:2506.10848, 2025.

[64] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, and Lingpeng Kong. Scaling difusion language models via adaptation from autoregressive models. In The Thirteenth International Conference on Learning Representations, 2025.

[65] Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. Difusion llms can do fasterthan-ar inference via discrete difusion forcing. arXiv preprint arXiv:2508.09192, 2025.

[66] Yu-Yang Qian, Junda Su, Lanxiang Hu, Peiyuan Zhang, Zhijie Deng, Peng Zhao, and Hao Zhang. d3llm: Ultra-fast difusion llm using pseudo-trajectory distillation. arXiv preprint arXiv:2601.07568, 2026.

[67] Ruisi Cai, Saurav Muralidharan, Greg Heinrich, Hongxu Yin, Zhangyang Wang, Jan Kautz, and Pavlo Molchanov. Flextron: Many-in-one flexible large language model. arXiv preprint arXiv:2406.10260, 2024.

[68] Yonggan Fu, Zhongzhi Yu, Junwei Li, Jiayi Qian, Yongan Zhang, Xiangchi Yuan, Dachuan Shi, Roman Yakunin, and Yingyan Celine Lin. Amoeballm: Constructing any-shape large language models for eficient and instant deployment. Advances in Neu ral Information Processing Systems, 37:78299–78319, 2024.

[69] Dhruv Nathawani, Shuoyang Ding, Vitaly Lavrukhin, Igor Gitman, Somshubra Majumdar, Evelina Bakhturina, Boris Ginsburg, and Jane Polak Scowcroft. Nemotron-Post-Training-Dataset-v2, August 2025.

## A. Difusion Sampler Details

This appendix details the sampler introduced in Sec. 3.2, including its architecture, input features, and training trajectory collection.

Architecture and feature engineering. As shown in Fig. 10, the sampler operates on top of the frozen backbone and adds negligible parameter overhead (∼0.06%, with 4.8M compared to the 8B backbone). It is a 4-layer lightweight Transformer with a hidden dimension of ??=384. It attends bidirectionally over the current block, followed by a per-position linear head with a sigmoid output. Each input position is represented by a 144-dimensional feature: PCA-compressed semantic embeddings of the top-3 predictions, as well as statistics summarizing the output distribution (e.g., top-1 probability, margin, top-3 mass, and entropy). The semantic embedding of the model’s own top-1 prediction is by far the most informative feature. We also find cross-position attention to be essential as an MLP-only ablation drops accuracy-TPF AUC by 10 percentage points, indicat ing that the sampler must jointly reason about which positions are mutually safe to commit.

Data collection for sampler training. To train the sampler, we collect ∼20M denoising trajectories from Nemotron-Labs-Difusion-8B on [69] (math, code, STEM, and chat subsets) at block lengths $B \in 8 , 3 2$ We use two complementary trajectory policies: (i) standard confidence decoding (??=1), where the block is decoded one token at a time in confidence order; and (ii) a hybrid policy that first commits groundtruth tokens whenever the model’s top-1 prediction already agrees with them, then falls back to confidence for the remaining positions. At each intermediate step of every trajectory, we store the 144- dimensional per-position features and the binary label 1[current top-1 = final ID], where the final ID is the token ultimately committed at that position once the block is fully decoded under the same policy. Training uses per-position binary cross-entropy on masked positions, with AUC on a held-out trajectory split as the early stopping criterion. The resulting accuracy–TPF gains over confidence thresholding are empirically reported in Sec. 6.1 and Fig. 8.

![](images/70218a405095b8a4b005e4035783b8f114bf514549d3688d7429fbd6b543d465.jpg)  
Figure 10 | An illustration of the sampler design on top of the difusion mode.

## B. LoRA-Enhanced Linear SS

We provide a visualization of the enhanced linear self-speculation $\mathrm { w } / $ LoRA in Fig. 11.

## C. Quadratic SS Details

We provide more details about quadratic selfspeculation, which is visualized in Fig. 12. Specifically, let $[ x _ { 1 } , \ldots , x _ { n } ]$ denote the currently verified prefix, and let ?? be the speculative width. At generation step ??+1, we reuse the ?? speculative tokens from the previous step, denoted $\{ \bar { x } _ { n + j } ^ { t } \} _ { j = 2 } ^ { k + 1 }$ , and interleave ?? fresh mask tokens after each speculative token, yielding the quadratic input:

$$
\begin{array}{r l} X _ {m} ^ {t + 1} = & [ x _ {1}, \ldots , x _ {n}, x _ {n + 1} ] + [ x _ {n + 2} ^ {t}, m _ {1}, \ldots , m _ {k} ] \\ & + \dots + [ x _ {n + k + 1} ^ {t}, m _ {1}, \ldots , m _ {k} ], \end{array}\tag{11}
$$

where $x _ { n + 1 }$ is the next token generated autoregressively at step ??+1 (and is thus immediately verified), and the total number of inserted masks is $k ^ { 2 }$

Parallel draft and verification. Feeding $X _ { m } ^ { t + 1 }$ via a structured attention mask into our model produces two types of outputs in a single forward pass. First, the model generates next-token predictions for the speculative tokens in a causal manner, yielding $\{ x _ { n + j } ^ { t + 1 } \} _ { j = 2 } ^ { k + 1 }$ . These tokens are used to verify the previous speculative draft through sequential comparison [7]: we accept the longest prefix that satisfies a verification criterion $( \mathrm { e . g . } , \bar { x } _ { n + j } ^ { t + \bar { 1 } } = x _ { n + j } ^ { t }$ , as detailed later), commit the accepted tokens to the verified prefix, and stop verification at the first mismatch. Second, in the same forward pass, the model predicts the tokens corresponding to the newly inserted masks $\{ m _ { r } \} _ { r = 1 } ^ { k }$ in parallel; we treat these predictions as the draft tokens that will serve as $\{ x _ { n + j } ^ { t + 1 } \} _ { j = 2 } ^ { k + 1 }$ in the next iteration. The interleaved quadratic layout ensures that, even if verification fails early at some position, newly inserted mask positions remain that still yield fresh speculative tokens for the next step, so each iteration consistently produces ?? tokens to verify [7].

Verification with the AR-difusion ensemble. For verification, the simplest choice is to use the AR predictions on the speculative tokens. In addition, our tri-mode model provides a complementary verification signal from the difusion pathway: for each speculative token $x _ { n + j } ^ { t }$ in Eq. 11, we can use the difusion prediction at the first newly inserted mask position $m _ { 1 }$ immediately following it as an alternative verifier for the same token. Concretely, the AR verifier uses the causal logits $p _ { \theta } ^ { \mathrm { A R } } ( \cdot \mid x _ { < n + j } )$ at position ??+??, while the difusion verifier uses the denoising logits $p _ { \theta } ^ { \mathrm { d i f f } } ( \cdot \mid X _ { m } ^ { t + 1 } , t )$ produced for the corresponding ?? position. Generally, we can form an AR-difusion ensemble verifier by combining the two distributions:

![](images/2f764156af47b01086d85ca49043cbddb0cb676771443b4aa5797aa2db09907a.jpg)  
Figure 11 | An illustration of LoRA training on the difusion drafter of the linear self-speculation mode.

![](images/e8d569eb8f6dcb2acd0c1cdd448b5943afc82edd899d855c9283f2903f2395d5.jpg)  
Figure 12 | An illustration of quadratic selfspeculation with simultaneous drafting and verification. denotes draft tokens that match AR verification, and denotes the block that originates from the last matched token and is to be verified in the next iteration.

$$
p _ {\theta} ^ {\mathrm{ens}} (\cdot) = \lambda p _ {\theta} ^ {\mathrm{AR}} (\cdot) + (1 - \lambda) p _ {\theta} ^ {\mathrm{diff}} (\cdot),
$$

where $\lambda \in [ 0 , 1 ]$ controls the interpolation.