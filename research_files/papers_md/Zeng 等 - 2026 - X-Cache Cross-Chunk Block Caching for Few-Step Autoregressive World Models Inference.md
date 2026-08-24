# X-Cache: Cross-Chunk Block Caching for Few-Step Autoregressive World Models Inference

AI Infra Team, XPeng Inc. https://x-cache-1.github.io/

## Abstract

Real-time world simulation is becoming a key infrastructure for scalable evaluation and online reinforcement learning of autonomous driving systems. Recent driving world models built on autoregressive video diffusion achieve high-fidelity, controllable multi-camera generation, but their inference cost remains a bottleneck for interactive deployment. However, existing diffusion caching methods are designed for offline video generation with multiple denoising steps, and do not transfer to this scenario. Few-step distilled models have no inter-step redundancy left for these methods to reuse, and sequence-level parallelization techniques require future conditioning that closed-loop interactive generation does not provide. We present X-Cache, a training-free acceleration method that caches along a different axis: across consecutive generation chunks rather than across denoising steps. X-Cache maintains per-block residual caches that persist across chunks, and applies a dualmetric gating mechanism over a structure- and action-aware block-input fingerprint to independently decide whether each block should recompute or reuse its cached residual. To prevent approximation errors from permanently contaminating the autoregressive KV cache, X-Cache identifies KV update chunks (the forward passes that write clean keys and values into the persistent cache) and unconditionally forces full computation on these chunks, cutting off error propagation. We implement X-Cache on X-world, a production multi-camera action-conditioned driving world model built on multi-block causal DiT with few-step denoising and rolling KV cache. X-Cache achieves 71% block skip rate with 2.6× wall-clock speedup while maintaining minimum degradation.

![](images/0844be93e93fb2254c76733c191e31f9737eba0554695c68973aaebd93078542.jpg)  
Figure 1: Visual comparison between baseline and X-cache on different simulation scenario

## 1 Introduction

Generative world models are emerging as a key infrastructure for autonomous driving. By conditioning on ego actions and sensor history, these models generate photorealistic future observations, enabling two capabilities that are difficult to achieve through real-world testing alone: scalable closed-loop evaluation of end-to-end driving policies, and online reinforcement learning where a policy explores counterfactual maneuvers in a controlled, repeatable environment [21, 15, 10]. In both settings, the world model must operate as an interactive simulator: generating observations, receiving actions, and responding in real time over long-horizon rollouts.

Autoregressive video diffusion [19, 9, 1, 3] has become a compelling backbone for such simulators. Unlike bidirectional video diffusion that generates an entire clip jointly, autoregressive formulations produce video chunk by chunk with causal attention, conditioning each new chunk solely on previously generated content. This causal, streaming structure naturally supports interactive simulation, where the model must respond to each newly issued action without waiting for a full clip to complete. Combined with a persistent key-value (KV) cache [8, 9, 21] that bounds memory regardless of sequence length, these models enable stable, unbounded-length generation. To meet throughput demands, they are further distilled to few-step denoising schedules [18, 19, 9]. This work focuses on accelerating inference for few-step autoregressive video diffusion deployed in interactive, closed-loop world simulation.

A growing line of training-free caching methods [12, 4, 11, 5] accelerates diffusion inference by reusing block outputs in Diffusion Transformers (DiTs) [14] across adjacent denoising steps, where structural changes between steps are often small enough [5, 11] to permit substantial reuse. Recent extensions to autoregressive video follow the same cross-step design: FlowCache [13] applies chunk wise caching policies, while SCOPE [6] combines tri-modal scheduling with predictive extrapolation along the denoising trajectory. However, this strategy is poorly suited to the few-step regime required for real-time interactive world simulation. With only four denoising steps, the cross-step similarity that these methods rely on is greatly reduced. Each step contributes substantial and non-redundant structural updates, leaving little room for safe reuse. FlowCache and SCOPE are subject to the same limitation: both are built on the assumption of cross-step redundancy and have only been evaluated under many-step schedules.

Beyond the few-step barrier, accelerating interactive world simulation exposes two structural compli cations. First, a major subset of training-free caching methods [11, 7, 6, 13] relies on extrapolating cached feature trajectories under a local smoothness assumption. In interactive simulation, the per-chunk action stream emitted by an external policy is by design non-smooth at chunk boundaries (discrete braking, steering, or lane-change commands enter every DiT block via adaLN-Zero), so the smoothness assumption fails precisely at the chunk boundaries where the world model is supposed to respond to a new control input. Second, specific to interactive simulation, the generation loop imposes a strict causal dependency: the world model must wait for an external policy to observe the fully generated current chunk and output a corresponding action before the next generation step can begin. This sequential dependency rules out sequence-level parallelization techniques (e.g., Block Cascading [2]) that rely on knowing future conditionals upfront to concurrently decode multiple chunks, so all acceleration must stay within the current generation boundary.

Despite these limitations, we observe that few-step autoregressive world models exhibit strong redundancy along a different axis: across consecutive generation steps rather than across denoising steps. In autonomous driving the scene evolves slowly relative to the generation rate, so the physical world changes continuously but smoothly between adjacent chunks. DiT block inputs at matching (denoising step, block) positions are then largely similar between consecutive generations. This cross-chunk redundancy comes from physical scene continuity rather than from proximity along the denoising trajectory, so it survives the few-step distillation that erases cross-step redundancy. Based on this observation, we propose X-Cache, a training-free block-level caching method that exploits cross-chunk redundancy for few-step autoregressive world model inference. Our contributions are:

1. Cross-chunk block caching: We cache DiT block residuals across consecutive generation steps at matching denoising step indices, exploiting temporal redundancy that persists under few-step distillation. To our knowledge, this is the first caching method that operates along the cross-chunk axis.

2. Structure-aware, action-aware fingerprint: To make cross-chunk similarity tractable at every (denoising step, block) position, we introduce a compact fingerprint that subsamples block inputs on the latent’s 3D (F, H, W) grid rather than along the flattened 1D token axis, giving geometrically balanced coverage across frames and spatial locations. We further attach a global mean channel to catch bulk drift that the sparse spatial sample may miss, and an action-condition channel that lifts the per-chunk action vector (consumed inside each block via adaLN-Zero) back into fingerprint space. This restores direct sensitivity to per-step control maneuvers that flat input-similarity metrics would otherwise miss, while dynamic-object and lane conditioning, which have unstable padding across chunks, are handled by the step-0 protection and anchor-block mechanisms introduced next.

3. KV update frame protection: We identify the KV cache writing pass as a critical error amplification point unique to autoregressive inference and unconditionally force full computation on these frames, cutting off permanent error propagation.

4. Adaptive per-position thresholding: To avoid hand-tuning a single global similarity threshold across all blocks and denoising steps, we introduce a per-(denoising step, block) adaptive cosine threshold that auto-regulates from the block’s own cross-chunk similarity history via an exponential moving average, clamped from below by a global quality floor. Blocks whose cross-chunk similarity is consistently high accumulate aggressive thresholds and contribute most of the skips, while blocks with volatile similarity remain conservative, yielding a stable per-chunk compute profile without any explicit budget controller.

5. Validation on a production driving world model: We validate X-Cache on a multicamera, action-conditioned driving world model with few-step causal denoising and rolling KV cache. X-Cache achieves 71% block skip rate with 2.6× DiT wall-clock speedup at negligible quality degradation in closed-loop simulation.

We note that direct comparison with existing caching methods is not applicable: all prior methods operate along the cross-step axis under many-step, one-shot generation settings, and cannot be directly applied to the interactive, few-step autoregressive world model that X-Cache targets. X-Cache exploits a different redundancy axis entirely, making it complementary to rather than competitive with stepwise approaches. Our experiments therefore focus on ablation studies and absolute performance rather than head-to-head comparisons.

## 2 Method

We first establish notation for the AR video diffusion inference loop, then describe each component of the caching mechanism.

## 2.1 Preliminaries: AR Video Diffusion Inference

Let n index the generation step (i.e., the n-th chunk of video), $t \in \{ 0 , \ldots , S { - } 1 \}$ the denoising step within a chunk, and $b \in \{ 0 , \ldots , B { - } 1 \}$ } the DiT block index. At generation step n, chunk latents are initialized from Gaussian noise and iteratively refined through S denoising steps, where each step passes through all B DiT blocks:

$$
\mathbf {x} _ {t, b} ^ {(n)} = \mathbf {x} _ {t, b - 1} ^ {(n)} + f _ {b} \left(\mathbf {x} _ {t, b - 1} ^ {(n)}; \mathbf {c} _ {t} ^ {(n)}\right),\tag{1}
$$

where $\mathbf { x } _ { t , 0 } ^ { ( n ) }$ is the latent at the start of block 0 at denoising step $t , f _ { b } ( \cdot ; \mathbf { c } )$ denotes the b-th DiT block parameterized by conditioning c (timestep embedding, action embedding, text embedding, etc.), and the operation is a standard residual connection. We define the block residual as:

$$
\mathbf {r} _ {t, b} ^ {(n)} = f _ {b} \Big (\mathbf {x} _ {t, b - 1} ^ {(n)}; \mathbf {c} _ {t} ^ {(n)} \Big) = \mathbf {x} _ {t, b} ^ {(n)} - \mathbf {x} _ {t, b - 1} ^ {(n)}.\tag{2}
$$

After all S denoising steps complete for chunk $n ,$ the model performs a KV update pass: it runs one additional forward pass through all blocks using the fully denoised (clean) latent, computing key-value projections that are appended to the persistent KV cache for conditioning future chunks via cross-attention. In rolling KV cache implementations, oldest entries are evicted via FIFO when the cache reaches capacity.

![](images/7204b769a4019d7eddddf7aecbe7963dbbf1bbf2793c87407354ea23de7a207e.jpg)  
Figure 2: Overall architecture of X-Cache

## 2.2 Cross-Chunk Residual Caching

## 2.2.1 Key observation

In AR video generation of physically continuous environments (especially in autonomous driving), consecutive chunks depict scenes that change smoothly relative to the generation rate. The block input $\mathbf { x } _ { t , b - 1 } ^ { ( n ) }$ at position (t, b) is therefore highly similar to $\mathbf { x } _ { t , b - 1 } ^ { ( n - 1 ) }$ at the same position in the previous generation step. This cross-chunk redundancy is independent of the number of denoising steps S and persists even under aggressive few-step distillation (e.g., S = 4), unlike cross-step redundancy, which diminishes as S decreases.

## 2.2.2 Caching mechanism

When block b is fully computed at generation step n and denoising step $t ,$ we cache its residual:

$$
\hat {\mathbf {r}} _ {t, b} \leftarrow \mathbf {r} _ {t, b} ^ {(n)} = \mathbf {x} _ {t, b} ^ {(n)} - \mathbf {x} _ {t, b - 1} ^ {(n)}.\tag{3}
$$

At the next generation step n+1, if the gating mechanism (Section 2.3) determines that block b at denoising step t can be skipped, we approximate its output by additive reuse:

$$
\tilde {\mathbf {x}} _ {t, b} ^ {(n + 1)} = \mathbf {x} _ {t, b - 1} ^ {(n + 1)} + \hat {\mathbf {r}} _ {t, b}.\tag{4}
$$

The cached residual is indexed by the pair (t, b), ensuring reuse occurs between matching positions in the denoising trajectory.

## 2.2.3 Initialization

No cached residuals exist for the first generation step (n = 0). During this warmup phase (configurable to $W \geq 1$ steps), all blocks compute fully, populating the cache for every (t, b) pair.

## 2.3 Dual-Metric Gating

Beyond the warmup phase, X-Cache evaluates whether each block can be safely skipped using a dual-metric similarity test between the current input fingerprint and the cached fingerprint from the same (t, b) position at the previous generation.

## 2.3.1 Fingerprint

Each block input $\mathbf { x } _ { t , b - 1 } ^ { ( n ) }$ has shape $B { \times } V { \times } L { \times } C$ , where V is the number of cameras in a view group and the token axis L is a flattened $( F _ { g } \times H _ { g } \times W _ { g } )$ spatio-temporal grid. Comparing the full tensor at every (t, b) is too expensive, and uniform 1D subsampling along $\bar { L }$ would cover frames and spatial locations unevenly. We instead subsample on the 3D grid: given a target budget of K tokens, we allocate the three axes in proportion to the grid’s aspect ratio $( k _ { F } : k _ { H } : k _ { W } \overset { \cdot } { \approx } F _ { g } : H _ { g } : W _ { g }$ with $k _ { F } k _ { H } k _ { W } \approx K )$ , take uniformly spaced indices along each axis, and index x with their Cartesian product, yielding a compact fingerprint $\phi ( \mathbf { x } )$ of shape $B { \times } V { \times } ( k _ { F } k _ { H } k _ { W } ) { \times } C$ (we use $K { = } 3 2 )$ . For blocks with $L { \le } K$ we keep the full x; if the grid shape is unavailable at runtime we fall back to 1D linspace along L. The fingerprint is computed independently per view group: in the multi-camera setting, the seven cameras form three view groups (front, side, and rear) by shared grid shape, with cameras within a group stacked along V and sharing one per-group fingerprint.

## 2.3.2 Auxiliary channels

Because $\phi ( \mathbf { x } )$ is sparse and depends only on the block input, we attach two cheap auxiliary signals to close blind spots:

• Global channel: The per-view-group sequence-mean of the block input, obtained by averaging along the token axis. It captures bulk latent drift that the sparse spatial sample may miss.

• Condition channel: The per-chunk action vector consumed by adaLN-Zero is flattened and appended as an additional fingerprint entry. Its shape is stable (fixed by the action dimensionality and the frames-per-chunk) and it changes once per generation, so the overhead is negligible. Including it lets the input-similarity metric react directly to per-step control maneuvers that would otherwise only propagate through the block-0 cascade. Other conditioning signals (dynamic-object and lane embeddings injected via cross-attention, text via cross-attention) have variable padding across chunks and slower change timescales; we leave them to the cascade and anchor-block mechanisms described in Section 2.5.

All fingerprint entries are flattened before metric computation.

## 2.3.3 Metric 1: Cosine similarity (global direction)

$$
s _ {\cos} = \frac {\phi (\mathbf {x} _ {t , b - 1} ^ {(n)}) \cdot \phi (\mathbf {x} _ {t , b - 1} ^ {(n - 1)})}{\| \phi (\mathbf {x} _ {t , b - 1} ^ {(n)}) \| \cdot \| \phi (\mathbf {x} _ {t , b - 1} ^ {(n - 1)}) \|},\tag{5}
$$

computed per fingerprint entry (each spatial view group, plus the global and action-condition channels) and aggregated by taking the minimum across all entries: $s _ { \mathrm { c o s } } = \operatorname* { m i n } _ { k } s _ { \mathrm { c o s } } ^ { ( k ) }$

## 2.3.4 Metric 2: Maximum token deviation (local outlier)

$$
d _ {\max} = \frac {\max \left| \phi (\mathbf {x} _ {t , b - 1} ^ {(n)}) - \phi (\mathbf {x} _ {t , b - 1} ^ {(n - 1)}) \right|}{\text {mean} \left| \phi (\mathbf {x} _ {t , b - 1} ^ {(n - 1)}) \right| + \epsilon},\tag{6}
$$

computed only over the per-group spatial fingerprints (the global and action-condition channels use cosine alone) and aggregated by taking the maximum across view groups: $d _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { g } d _ { \operatorname* { m a x } } ^ { ( g ) }$ .

## 2.3.5 Skip decision

Block b at denoising step t is skipped if and only if both metrics pass:

$$
\operatorname{skip} (t, b) = \left(s _ {\cos} \geq \tau_ {\cos} (t, b)\right) \wedge \left(d _ {\max} <   \tau_ {\mathrm{dev}}\right),
$$

(7)

where $\tau _ { \mathrm { c o s } } ( t , b )$ is an adaptive threshold (Section 2.4) and $\tau _ { \mathrm { d e v } }$ is a fixed deviation threshold. The conservative aggregation (min-cosine over all fingerprint entries, max-deviation over spatial view groups) ensures that anomalous change in any single view group, in the global summary, or in the action-condition channel triggers recomputation.

## 2.4 Adaptive Threshold

Rather than a fixed cosine threshold, X-Cache learns a per-position threshold from the block’s own history. For each (t, b) position, we maintain an exponential moving average of the observed cosine similarity:

$$
\bar {s} _ {t, b} \gets \alpha \cdot s _ {\cos} ^ {(n)} + (1 - \alpha) \cdot \bar {s} _ {t, b},\tag{8}
$$

updated each time the test at (t, b) is evaluated $( \alpha = 0 . 3 )$ . The adaptive threshold is:

$$
\tau_ {\mathrm{cos}} (t, b) = \max (\tau_ {\mathrm{floor}}, \bar {s} _ {t, b} - m),\tag{9}
$$

where m is a configurable margin and $\tau _ { \mathrm { f l o o r } }$ is a hard quality floor. In our implementation, $\tau _ { \mathrm { f l o o r } } = 0 . 9 5$ and $m = 0 . 0 2$

Blocks with consistently high cross-chunk similarity accumulate high EMAs and their thresholds settle just below their typical similarity, maximizing skips. Blocks with volatile similarity remain conservative. The quality floor provides an absolute safety guarantee independent of history.

## 2.5 Safety Mechanisms

Condition visibility gap External conditioning signals are injected inside each DiT block, not into the block input x. The spatial fingerprint ϕ(x) therefore does not directly reflect condition changes. The action-condition channel from Section 2.3 lifts the action vector into fingerprint space, closing the gap for per-step controls. Dynamic-object and lane embeddings have variable padding across chunks and change on slower timescales, so including them directly in the fingerprint is neither shape-stable nor cost-effective; together with the text condition they are handled instead by the following mechanisms:

1. Denoising step 0 protection: $\mathbf { A } \mathbf { t } t = 0 .$ , the input latent is dominated by high-level noise, and conditioning signals have maximal relative influence on the block output. Noise is also resampled at each KV update cycle and new context from the previous chunk’s output is embedded in the noisy input, so consecutive generations naturally exhibit low cosine similarity at $t = 0$ . By default, X-Cache forces full computation at $t = 0$ . An optional relaxation mode applies a strict threshold $\tau _ { 0 } ^ { \mathrm { s t r i c t } } = 0 . 9 9 9$ , which is near-unity: even minor latent changes drop the similarity below 0.999, keeping step 0 effectively protected. Once step 0 computes with updated conditions, its output cascades through subsequent steps: each block’s input is derived from the prior step’s output, so fingerprints diverge from the cached versions and trigger recomputation where needed.

2. Anchor blocks $( F _ { n } ) \colon$ The first $F _ { n }$ blocks are unconditionally computed at all denoising steps (default $F _ { n } = 1 )$ . With $F _ { n } = 1$ , block 0 always processes the current conditioning via adaLN-Zero, and its changed output cascades through subsequent blocks’ fingerprints. This provides a per-step guarantee independent of step 0 protection. The last $B _ { n }$ blocks can similarly be designated as tail anchors (default $B _ { n } = 0 )$

3. KV update frame protection: The generation step whose clean latent will be used for the KV update pass is critical: its KV projections are attended to by all future chunks. During this generation, X-Cache enters force-compute mode: all blocks compute fully, but the cache remains attached so that fingerprints and residuals are refreshed for the next generation. This ensures both (a) precise KV entries and (b) that the subsequent generation starts with fresh cache data, avoiding a double-heavy-frame penalty.

4. Maximum staleness: A counter tracks consecutive skips per (t, b) position. If it exceeds threshold M, the block is forced to recompute.

## 3 Experiments

## 3.1 Settings

## 3.1.1 Hardware

All of our experiments run on the Zhenwu (<sup>真武</sup>) 810E, an AI accelerator developed by Alibaba T-Head. In the remainder of this paper we will refer to the device as a Parallel Processing Unit (PPU). Each PPU integrates 96 GB of HBM2e on-chip memory and natively supports FP16, BF16, and INT8 with hardware acceleration. In the following experiments, all DiT forward passes are executed in BF16 on a single PPU.

## 3.1.2 Experiment Model

We evaluate on X-World [21], a controllable ego-centric multi-camera world model for autonomous driving built on WAN 2.2 [17]. X-World follows the latent video diffusion paradigm, combining a causal VAE with a DiT denoiser to generate synchronized 360° video from seven cameras at 12 FPS. During streaming autoregressive rollout, it generates video chunk by chunk: each chunk starts from Gaussian noise, run 4-step denoising, and updates a fixed-size rolling KV cache with FIFO eviction to support long-horizon simulation under bounded memory.

## 3.1.3 Dataset

We evaluate on an internal held-out test split drawn from the X-World training distribution, exercised under the interactive closed-loop inference protocol X-World is designed for. At each generation chunk the causal DiT attends only to one frame of initial 7-camera history, the conditioning that applies to that chunk alone (ego action state, dynamic-agent poses, static road-element annotations, a scene-level text caption), and the rolling KV cache of past chunks; the causal architecture forbids any look-ahead to future conditioning. The per-chunk actions are not known to the model in advance: they are replayed one chunk at a time from a recorded trajectory so each run is reproducible, which is indistinguishable from a live policy stream from the DiT’s perspective. Each clip produces 264 frames (around 22 seconds at 12 FPS) of future multi-camera video, with no per-frame visual ground truth supplied during rollout. The split covers three scenario groups: 7 urban-street clips with dense surrounding traffic, pedestrians, and storefronts; 3 highway clips covering elevated urban express ring-road and ordinary motorway; and 3 urban u-turn clips where the ego vehicle executes sharp heading changes and consecutive chunks change the most across views. We use the u-turn group to stress-test the gating assumption under the largest cross-chunk motion in the split.

## 3.1.4 Metrics

Since X-Cache only modifies the DiT denoiser, we evaluate only metrics that directly reflect its effect on denoising. We do not include VAE encoding or decoding, data loading, post-processing, or inter-device transfer in any reported numbers. On compute side, we report the block skip rate, defined as the fraction of DiT block evaluations that reuse a cached residual during a rollout, excluding warmup chunks. We also show how this skip rate varies across denoising steps and block indices. For efficiency, we report the average per-chunk DiT wall-clock time on a single PPU, as well as the resulting DiT speedup relative to a full-compute run with the same seed, conditioning, and KV state. For fidelity, we compare each X-Cache rollout with its corresponding full-compute reference rollout. Because both branches share the same decoder, any difference comes only from the DiT side. We report three frame-level metrics on the decoded images, averaged over all seven cameras: PSNR for pixel-level fidelity, SSIM for structural fidelity, and LPIPS [20] for perceptual fidelity on deep features.

## 3.1.5 Default Parameters

Unless otherwise specified, X-Cache is run with the default hyperparameters given in Table 1.

<table><tr><td>Symbol</td><td>Meaning</td><td>Default</td></tr><tr><td> $F_n$ </td><td>Number of front anchor blocks (always computed)</td><td>1</td></tr><tr><td> $B_n$ </td><td>Number of back anchor blocks (always computed)</td><td>0</td></tr><tr><td>W</td><td>Warmup chunks (all blocks fully computed)</td><td>1</td></tr><tr><td> $τ_{floor}$ </td><td>Adaptive-threshold quality floor</td><td>0.97</td></tr><tr><td>m</td><td>Adaptive-threshold margin below EMA</td><td>0.02</td></tr><tr><td>α</td><td>EMA update coefficient</td><td>0.30</td></tr><tr><td> $τ_{dev}$ </td><td>Max-token deviation threshold</td><td>2.00</td></tr><tr><td>M</td><td>Max consecutive skips per (t,b)</td><td>Off</td></tr><tr><td>KV-update-chunk protection</td><td>Force full compute on KV-update chunk</td><td>On</td></tr><tr><td>Denoising step-0 protection</td><td>Force full compute at t=0</td><td>Off</td></tr></table>

Table 1: Default X-Cache hyperparameters used throughout the experiments.

![](images/d1801aaf46ccdde893eb987fe580c2320c93ccde8319a6b6d966bb7d69d290e2.jpg)  
Figure 3: Qualitative frame-level comparison in three scenarios. Rows, top to bottom: highway, urban street, and u-turn. Each row: left baseline full-compute, middle X-Cache under identical seed and conditioning, right absolute pixel difference amplified 20×. Per-view labels match Table 2.

## 3.2 Results

X-Cache gives 2.65 to 2.70× DiT speedup across the three splits while keeping decoded quality in the imperceptible range (SSIM above 0.9990, LPIPS below 4e-4, 7-camera PSNR between 51 and 55 dB in Table 2). Scene complexity produces a small but real spread in quality, with the shaded 7-camera PSNR increasing from 51.37 dB for urban street to 52.04 dB for u-turn and 54.67 dB for highway. Skip rate and DiT time do not follow this spread: they stay within 0.3 percentage points and 30 ms, respectively, across all three splits. The quality ordering reflects how each scenario’s decoded pixels absorb a common cache-induced latent perturbation, not a scenario-specific slowdown or tightening of the gate itself. The qualitative residuals in Figure 3 also agree with this: the three difference panels show the same sparsity pattern, with the residual sitting on lane edges and far-field foliage and disappearing at display scale even under 20 × amplification.

<table><tr><td>Scenario</td><td>Camera</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Skip (%)</td><td>DiT (s)</td><td>Speedup</td></tr><tr><td rowspan="8">Urban Street (n=7)</td><td>F-C</td><td>53.83</td><td>0.9988</td><td>3.6e-4</td><td></td><td></td><td></td></tr><tr><td>F-W</td><td>50.27</td><td>0.9987</td><td>4.3e-4</td><td></td><td></td><td></td></tr><tr><td>S-FL</td><td>49.49</td><td>0.9985</td><td>5.1e-4</td><td></td><td></td><td></td></tr><tr><td>S-FR</td><td>48.69</td><td>0.9984</td><td>5.2e-4</td><td>71.4</td><td>1.392</td><td>2.65×</td></tr><tr><td>S-RL</td><td>48.59</td><td>0.9985</td><td>4.8e-4</td><td></td><td></td><td></td></tr><tr><td>S-RR</td><td>48.07</td><td>0.9985</td><td>5.2e-4</td><td></td><td></td><td></td></tr><tr><td>Rear</td><td>51.77</td><td>0.9986</td><td>4.7e-4</td><td></td><td></td><td></td></tr><tr><td>7-cam</td><td>51.37</td><td>0.9990</td><td>3.3e-4</td><td></td><td></td><td></td></tr><tr><td rowspan="8">Highway (n=3)</td><td>F-C</td><td>54.87</td><td>0.9989</td><td>2.6e-4</td><td></td><td></td><td></td></tr><tr><td>F-W</td><td>54.38</td><td>0.9988</td><td>2.3e-4</td><td></td><td></td><td></td></tr><tr><td>S-FL</td><td>53.08</td><td>0.9987</td><td>2.8e-4</td><td></td><td></td><td></td></tr><tr><td>S-FR</td><td>52.20</td><td>0.9987</td><td>2.9e-4</td><td>71.6</td><td>1.365</td><td>2.66×</td></tr><tr><td>S-RL</td><td>52.48</td><td>0.9987</td><td>2.5e-4</td><td></td><td></td><td></td></tr><tr><td>S-RR</td><td>51.90</td><td>0.9986</td><td>3.0e-4</td><td></td><td></td><td></td></tr><tr><td>Rear</td><td>53.42</td><td>0.9987</td><td>3.2e-4</td><td></td><td></td><td></td></tr><tr><td>7-cam</td><td>54.67</td><td>0.9991</td><td>1.9e-4</td><td></td><td></td><td></td></tr><tr><td rowspan="8">U-turn (n=3)</td><td>F-C</td><td>54.60</td><td>0.9987</td><td>4.3e-4</td><td></td><td></td><td></td></tr><tr><td>F-W</td><td>51.79</td><td>0.9987</td><td>3.6e-4</td><td></td><td></td><td></td></tr><tr><td>S-FL</td><td>49.29</td><td>0.9985</td><td>4.6e-4</td><td></td><td></td><td></td></tr><tr><td>S-FR</td><td>49.18</td><td>0.9985</td><td>4.7e-4</td><td>71.3</td><td>1.364</td><td>2.70×</td></tr><tr><td>S-RL</td><td>48.87</td><td>0.9985</td><td>4.0e-4</td><td></td><td></td><td></td></tr><tr><td>S-RR</td><td>48.82</td><td>0.9984</td><td>4.9e-4</td><td></td><td></td><td></td></tr><tr><td>Rear</td><td>52.51</td><td>0.9986</td><td>4.2e-4</td><td></td><td></td><td></td></tr><tr><td>7-cam</td><td>52.04</td><td>0.9990</td><td>3.1e-4</td><td></td><td></td><td></td></tr></table>

Table 2: X-Cache vs. baseline full-compute reference. Per-camera rows are F-C (front centre), F-W (front wide), S-FL/FR/RL/RR (four sides), and Rear. The shaded "7-cam" row is computed on the stitched seven-view image used in the qualitative figures (not the arithmetic mean of the per-camera rows). Skip rate excludes warmup; DiT wall-clock is per-chunk on a single PPU, excluding the first chunk. Baseline DiT is 3.682 s / 3.633 s / 3.688 s per chunk on urban / Highway / u-turn.

![](images/13328fce5f0d316fbc52c82722ec0d242fda3b41578f2f7aa12441cd9cf17bf4.jpg)

![](images/60b1b350060bce0682204f9dcad9728c377d618a43177486f187c435992ae70f.jpg)

![](images/b51f90153ee6cfc651098978b7041f62fe6c433936080038e8b29beeb4143f4b.jpg)  
Figure 4: Per-frame combined-7-view PSNR of X-Cache rollouts against their baseline full-compute references, on one representative session per scenario. Dashed line marks the per-session mean.

It is counter-intuitive that the u-turn split slightly outperforms the urban split, given that u-turn exhibits the largest cross-chunk motion in the dataset; this can be explained by two clip-level properties. First, the turn occupies only a minority of each u-turn clip, with most of the 22 seconds spent in straight driving or lane change, so the clip-averaged PSNR averages urban-like frames with a smaller stretch of genuine turn. Second, during the yaw the cameras register real-world motion blur across the lateral field. The blur smooths fine texture in the reference frames and reduces the high-frequency content that cache-induced latent perturbations would otherwise surface as pixel error, so the in-turn frames themselves are easier to match than a sharp urban frame of comparable complexity. The combined effect places the u-turn between urban and highway rather than below urban. Figure 4 shows the structure behind this average: the u-turn curve sits near the urban level outside the turn window and has a visible drop between frames 50 and 140 where the per-chunk fingerprint change exceeds the adaptive threshold more often, then recovers to the pre-turn level once the fingerprint settles.

![](images/a05f7c1901be1a08742877879bec9e14ad206aa6af3d76c62b64a97f0625a90c.jpg)

![](images/64790d81327d133921c226a7b2a52d691dcc675ec9f85e95d14a7db21762dd70.jpg)

![](images/f0fab90d8f344f5ece1ef83e96a7190ccfc610fb82b5719958cc1c5eaea54b9b.jpg)

Figure 5: Per-(step, block) cross-chunk cosine similarity, one panel per scenario, averaged over post-warmup chunks of a representative clip. Block 0 is omitted because the front anchor $( F _ { n } { = } 1 )$ never participates in the gate. Color scale is shared across the three panels.  
![](images/b15d3d11ff3a8f9ffa4dcc3deebba96b1a01a6ba4665ad77ace2b944ffd81e69.jpg)  
Figure 6: Per-block skip rate, averaged across the four denoising steps and across post-warmup chunks of one representative clip per scenario. Block 0 (front anchor, $F _ { n } { = } 1 )$ is omitted. Dashed line marks the theoretical 0.75 plateau implied by W=1 warmup over the 4-chunk reuse window.

The gate behaves nearly identically across scenarios: all three skip the same set of blocks. Figure 5 shows nearly identical heatmap shapes across panels. Blocks 1 to 19 remain above 0.95, whereas blocks 20 to 26 drop to around 0.90, and variation along the denoising-step axis stays within about 0.02 for any fixed block. Figure 6 shows the same partition at the decision level: blocks 1 to 19 stay on the 0.75 plateau, whereas blocks 20 to 26 fall to about 0.69, with all three scenarios following essentially the same curve. The 0.3 percentage-point spread in skip rate in Table 2 therefore comes from small movements within this fixed set of blocks, rather than from scenario-dependent changes in which blocks are cacheable. The per-chunk reuse budget also binds before the cosine test does: even u-turn, which has the largest inter-chunk motion, does not fall to a lower skip rate.

Two further observations reinforce this interpretation. First, the per-frame PSNR traces show no drift at chunk boundaries in any of the three clips, consistent with KV-update protection remaining effective over the 22-second rollouts we test. Second, the per-camera PSNR ranking within each scenario (front-center highest, rear intermediate, and side cameras 2 to 6 dB lower) should not be interpreted as a cache-side signal, because the gate makes a single decision per (t, b) cell using a 7-camera-aware fingerprint (Section 2.3), rather than separate decisions for each camera. When a single PSNR per scenario is needed, the shaded 7-camera column is therefore the appropriate value to report.

## 4 Ablation

We keep all settings from Section 3.1.5 fixed and vary one protection mechanism or hyperparameter at a time on a single held-out rollout (another highway clip with 264 frames). Table 3 reports the same fidelity metrics as Table 2, together with per-chunk DiT wall-clock time and DiT speedup

relative to the no-cache baseline on the same clip. Decoder and I/O are excluded, so the speedup column reflects only the portion affected by X-Cache.

<table><tr><td>Configuration</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Skip (%)</td><td>DiT (s)</td><td>Speedup</td></tr><tr><td>Baseline (no cache)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>3.637</td><td>1.00×</td></tr><tr><td>Default (step-0 off,  $\tau_{floor}=0.97$ )</td><td>53.384</td><td>0.9990</td><td>2.0e-4</td><td>71.3</td><td>1.406</td><td>2.59×</td></tr><tr><td>+ Step-0 protection</td><td>53.389</td><td>0.9990</td><td>2.0e-4</td><td>53.5</td><td>1.975</td><td>1.84×</td></tr><tr><td>- KV-update protection</td><td>21.461</td><td>0.8067</td><td>1.77e-1</td><td>62.8</td><td>1.670</td><td>2.18×</td></tr><tr><td>- Front anchor ( $F_n=0$ ) $^{\dagger}$ </td><td>53.622</td><td>0.9991</td><td>2.0e-4</td><td>55.4</td><td>1.902</td><td>1.91×</td></tr><tr><td> $\tau_{floor} \in \{0.90,\dots,0.96\}^{\dagger}$ </td><td>53.37–53.40</td><td>0.9990</td><td>2.0e-4</td><td>53.5</td><td>1.96–1.98</td><td>1.84×</td></tr></table>

Table 3: X-Cache ablations on the single 264-frame clip. <sup>†</sup> The $F _ { n }$ row and the $\tau _ { \mathrm { { f l o o r } } }$ sweep keep step-0 protection on, so the effect of each knob is isolated rather than entangled with the step-0 decision; the Default and +Step-0 rows already measure that decision on its own.

## 4.1 Denoising step-0 protection

By default, step-0 skipping is enabled. Enabling step-0 protection removes all step-0 skips, reducing the overall skip rate from 71.3% to 53.5% and increasing per-chunk DiT wall-clock time from 1.41 s to 1.98 s (speedup from 2.59× to 1.84×). On this clip, however, quality remains unchanged to the fourth decimal place of SSIM and LPIPS. In practice, whenever reuse is valid, the step-0 residual fingerprints already lie above the adaptive threshold, so the cosine-similarity gate alone is sufficient to suppress the cases of step-0 reuse that would otherwise matter. We nevertheless retain this option because the step-0 residual initializes all downstream blocks. Under larger distribution shift (for example, night, heavy rain, or sudden trajectory changes) or in longer rollouts where small errors may accumulate, forcing step-0 recomputation provides an additional safety margin without requiring any other change.

## 4.2 KV-update-chunk protection

This is the only setting whose removal causes a substantial loss in quality. Allowing the gate to skip tokens on the chunk where the rolling KV cache is updated reduces PSNR from 53.4 dB to 21.5 dB and increases LPIPS by three orders of magnitude, while improving skip rate by only 9 percentage points and changing DiT speedup from 2.59× to 2.18×. This trade-off is clearly unfavorable, so KV-update-chunk protection remains enabled.

## 4.3 Front anchor $F _ { n } = 1$

Setting $F _ { n } = 0$ increases skip rate by 1.9 percentage points and saves about 70 ms of DiT time per chunk at essentially the same PSNR. However, block 0 defines the residual basis on which all later blocks build. If block 0 is skipped during a KV-update cycle, downstream error propagation is no longer naturally bounded. Our current clips do not strongly expose this failure mode, but we set the default to $F _ { n } = 1$ so that deployment on rarer scenarios does not rely on these clips being benign.

## 4.4 Adaptive-threshold floor

Sweeping $\tau _ { \mathrm { { f l o o r } } }$ from 0.90 to 0.96 leaves all reported metrics unchanged at the precision shown here. Under the current workload, the floor is effectively inactive for two reasons. First, the measured per-chunk cosine-similarity distribution is heavily concentrated at 1.0 (both the median and $\mathsf { p } 7 5$ are 1.0 across all 7 urban-street clips), so very few tokens fall in the 0.94–0.97 range where lowering the floor would alter the skip decision. Second, the effective threshold is max $( \tau _ { \mathrm { f l o o r } } , \mathrm { E M A } _ { \mathrm { c o s } } - m )$ and on most (t, b) cells the EMA-minus-margin term lies around 0.94–0.95. As a result, any τ<sub>floor</sub> below about 0.94 is dominated by the EMA term rather than by the floor itself. The sweep therefore shows that the gate is insensitive to this parameter on the current test distribution, but not that it will remain insensitive on data farther from training. The floor still defines the minimum threshold the gate may fall to when the cosine-similarity distribution shifts, so we keep it configurable and choose a conservative default.

## 5 Limitations

All rollouts we report are 22-second clips from an internal held-out split of the X-World training distribution, across urban street, highway, and u-turn. We have not measured X-Cache on longer horizons or on clips outside that distribution such as night-time, adverse weather, aggressive driving, or sustained highway cruising; in those scenario cross-chunk similarity patterns may shift and the adaptive EMA threshold would need time to recalibrate before the reported skip rate and quality numbers would apply. The defaults in Table 1 are also tuned on a single held-out clip and are chosen to be safe rather than aggressive: the 7-camera PSNR settles around 53 dB with SSIM above 0.999 and LPIPS below $4 \times 1 0 ^ { - 4 }$ , well inside the imperceptible range, so a direct parameter sweep (a lower $\tau _ { \mathrm { { f l o o r } } } .$ , a relaxed $\tau _ { \mathrm { d e v } } .$ , or dropping the front anchor on blocks whose cross-chunk cos stays high) should trade a small amount of the current quality margin for more skip rate and more wall-clock. We have not mapped out this Pareto frontier on our test set.

## 6 Conclusion

X-Cache targets a redundancy axis that cross-step caching methods do not reach: block-level residual reuse across consecutive generation chunks of a few-step autoregressive video diffusion model. Validated on X-World, a production driving world model, the default configuration reuses 71% of DiT block evaluations and delivers 2.6–2.7× DiT wall-clock speedup against the no-cache baseline with no visible quality degradation. The behaviour holds across urban streets, highway driving, and u-turn rollouts, and the underlying cross-chunk cosine pattern is driven by block position in the DiT rather than by scene content. Our ablations identify KV-update-chunk protection as the one hard safety requirement and treat step-0 protection, the front anchor, and the adaptive-threshold floor as soft knobs that the current workload does not exercise but that the design retains as margins against distribution shift, long-horizon drift, and extreme operating conditions not covered by the current test split. Other few-step interactive AR world models that share this template, such as WorldPlay [16], are natural next extension targets, although the achievable speedup and quality floor under their larger and more abrupt action transitions still need to be verified.

## 7 Contributors

We extend our sincere gratitude to the entire team for their dedication and hard work. This project is a testament to our collective effort in pushing the boundaries of world model research and engineering.

Advisors and Project Leads: Yu Zhang, Boyang Wang, Linkun Xu, Siyuan Lu, Bo Tian, Xianming Liu

Contributors: Yixiao Zeng<sup>∗</sup>, Jianlei Zheng, Chaoda Zheng, Shijia Chen, Mingdian Liu, Tongping Liu, Tengwei Luo

<sup>∗</sup>Core contributor

## References

[1] Sand. ai, Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W. Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, Dai Shi, Guoli Su, Hanwen Sun, Hong Pan, Jie Wang, Jiexin Sheng, Min Cui, Min Hu, Ming Yan, Shucheng Yin, Siran Zhang, Tingting Liu, Xianping Yin, Xiaoyu Yang, Xin Song, Xuan Hu, Yankai Zhang, and Yuqiao Li. Magi-1: Autoregressive video generation at scale, 2025.

[2] Hmrishav Bandyopadhyay, Nikhil Pinnaparaju, Rahim Entezari, Jim Scott, Yi-Zhe Song, and Varun Jampani. Block cascading: Training free acceleration of block-causal video models, 2025.

[3] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. Skyreels-v2: Infinite-length film generative model, 2025.

[4] Pengtao Chen, Mingzhu Shen, Peng Ye, Jianjian Cao, Chongjun Tu, Christos-Savvas Bouganis, Yiren Zhao, and Tao Chen. δ-dit: A training-free acceleration method tailored for diffusion transformers, 2024.

[5] Hanshuai Cui, Zhiqing Tang, Zhifei Xu, Zhi Yao, Wenyi Zeng, and Weijia Jia. Bwcache: Accelerating video diffusion transformers through block-wise caching, 2026.

[6] Hanshuai Cui, Zhiqing Tang, Zhi Yao, Fanshuai Meng, Weijia Jia, and Wei Zhao. Not all frames deserve full computation: Accelerating autoregressive video generation via selective computation and predictive extrapolation, 2026.

[7] Weilun Feng, Guoxin Fan, Haotong Qin, Chuanguang Yang, Mingqiang Wu, Yuqi Li, Xiangqi Li, Zhulin An, Libo Huang, Dingrui Wang, Longlong Liao, Michele Magno, and Yongjun Xu. Worldcache: Accelerating world models for free via heterogeneous token caching, 2026.

[8] Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen. Ca2- vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing, 2025.

[9] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

[10] Chiyu (Max) Jiang, Xander Masotto, and Bo Sun. The waymo world model: A new frontier for autonomous driving simulation, Feb 2026.

[11] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model, 2024.

[12] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[13] Yuexiao Ma, Xuzhe Zheng, Jing Xu, Xiwei Xu, Feng Ling, Xiawu Zheng, Huafeng Kuang, Huixia Li, Xing Wang, Xuefeng Xiao, Fei Chao, and Rongrong Ji. Flow caching for autoregressive video generation, 2026.

[14] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023.

[15] Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving, 2025.

[16] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling, 2025.

[17] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[18] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024.

[19] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025.

[20] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

[21] Chaoda Zheng, Sean Li, Jinhao Deng, Zhennan Wang, Shijia Chen, Liqiang Xiao, Ziheng Chi, Hongbin Lin, Kangjie Chen, Boyang Wang, Yu Zhang, and Xianming Liu. X-world: Controllable ego-centric multi-camera world models for scalable end-to-end driving, 2026.