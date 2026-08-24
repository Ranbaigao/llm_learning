# X-Foresight: A Joint Vision-Action Causal Forecasting Network via Predictive World Modeling

PWM Team, XPeng Inc. https://x-foresight-1.github.io

## Abstract

Physical world knowledge resides mainly in videos. Equipping Vision-Language-Action (VLA) models with such knowledge is fundamental for safe and generalizable planning. To extract this knowledge from video data, predictive world modeling enables VLA to internalize physical dynamics and long-term causality by predicting future video from past observations. However, naive next-frame prediction faces two challenges: 1) unlike semantically distinct text tokens, video tokens are inherently low-entropy and redundant, causing prediction to easily degenerate into trivial extrapolation. 2) world modeling poses a temporal dilemma: instantaneous dynamics need dense frame prediction, while long-term causality unfolds over long and variable-size horizons that dense prediction cannot efficiently cover. To learn world knowledge effectively, we introduce X-Foresight, a predictive world model integrated directly into the VLA architecture to jointly learn world modeling and real-time action control. At its core lies long-horizon chunk-wise auto-regressive strategy, that addresses both challenges: By predicting across semantically distant chunks rather than adjacent frames, it escapes trivial extrapolation, while preserving dense intra-chunk frames to capture instantaneous dynamics and sparse inter-chunk transitions to capture long-term causality, resolving both the entropy and temporal-dilemma challenges at tractable training cost. A curriculum learning schedule is further utilized to progressively extend prediction horizons and stabilize long-horizon training. To capture long-term causality effectively, we present temporal importance sampling, which concentrates supervision on safety-critical chunks identified by ego-motion and behavioral signals. We further delegate photorealistic synthesis to a diffusion-based multi-view renderer, enriching photorealistic details and improving pixel-level appearance. Compre hensive experiments demonstrate that X-Foresight significantly outperforms VLA baselines in planning performance while maintaining strong generative fidelity, establishing a robust paradigm for world-knowledge-driven autonomous systems.

## 1 Introduction

The pursuit of artificial intelligence in physical world has driven significant interest in world models. With world models, internal representations of the environment can be learned, which enforces agents prediction, planning, and control. Recent world models employ diverse strategies to represent internal dynamics of the world: video-based generative models such as Sora [2], Genie [3], and X-World [22] learn to synthesize photorealistic future frames with large diffusion models; 3D-centric frameworks like Marble [12] explicitly model spatial geometry for physically consistent world reconstruction, while latent-based methods such as JEPA [1] bypass pixel-space generation entirely, learning semantic state representations through self-supervised predictive frameworks.

Vision-Language-Action (VLA) models have emerged as powerful paradigms for embodied decisionmaking, unifying perception, reasoning, and control within a single auto-regressive framework. Foundational works such as RT-2 [23], PaLM-E [6], and OpenVLA [11] have demonstrated remark able generalization by transferring web-scale vision-language knowledge to robotic manipulation. In the autonomous driving domain, industry systems such as XPeng’s VLA 2.0 [19] exemplify the adoption of VLA architectures for end-to-end perception-to-control pipelines. However, prevailing VLA architectures remain fundamentally reactive, lacking deep world understanding and proactive danger-avoidance mechanisms. This limitation stems from their inability to simulate future states before acting, forcing policies to rely solely on historical observations. Without forward-looking foresight, models struggle to anticipate collisions, navigate through complex scenarios, or exploit long-horizon environmental causality to improve agent control.

![](images/870a26754cbc70f6e5f7a83ebb4fc33d8215039eb8c1b0346b43d898fdad76a5.jpg)  
Figure 1: (A) Inference pipeline of X-Foresight. The main contributions resided in the Large Drive Model (LDM) and the Vision Renderer. (B) Closed-loop inference visualization of predicted future frames at t=2 s, t=4 s, and t=6 s; only the front camera was shown. (C) X-Foresight outperformed the baselines across multiple benchmarks.

Among the representations commonly used in robotics and autonomous driving, video provides one of the most holistic descriptions of the surrounding environment. It captures both low-level visual details, such as color, texture, and object shape, and high-level semantic information, such as the motion patterns of vehicles and vulnerable road users. Moreover, the ego vehicle’s trajectory can often be inferred directly from dash-camera observations. Therefore, we argue that video serves as a primary carrier of physical-world knowledge, encoding rich spatial, temporal, and semantic cues that are essential for future prediction and decision-making. We thus introduce X-Foresight, a predictive world model built atop the VLA architecture that seamlessly integrates multi-view future prediction with closed-loop action control. By actively forecasting future camera images, the model develops an internalized understanding of long-horizon environmental causality, enabling proactive, safety-critical decision-making while preserving the real-time responsiveness required for autonomous systems.

Inspired by the success of large language models, future-token prediction has become a central paradigm for generative sequence modeling. However, directly transferring this paradigm to videobased world modeling introduces an important modality gap. Language tokens are semantically discrete, relatively sparse, and high-entropy, whereas video tokens are often redundant, with high frame-to-frame similarity. This difference can easily lead to degenerate future rollouts, where the model learns only trivial frame-level extrapolation rather than meaningful physical dynamics. A further temporal challenge in world modeling arises from the mismatch between instantaneous dynamics and world transitions: effective world representation learning must capture short-term dynamics and long-term causality simultaneously. Instantaneous dynamics rely primarily on shortterm dense visual evidence, whereas world transitions demand long-horizon causal reasoning over extended temporal contexts. To reconcile this discrepancy, X-Foresight employs a chunk-wise autoregressive rollout that jointly forecasts future vision and action tokens. The proposed long-horizon chunk-wise auto-regressive strategy not only avoids trivial extrapolation by preserving dense intrachunk frames that capture instantaneous dynamics, but also enables efficient training and low-latency inference.

We further improve world-knowledge learning through targeted training innovations. Since predictive capability develops progressively, we adopt a curriculum learning schedule in which the model first learns to forecast short horizons within each chunk and is then gradually exposed to longer prediction horizons. This progressive training strategy improves the stability of long-horizon forecasting and enhances the robustness of the learned policy. Additionally, because temporal segments do not contribute equally to world modeling, we introduce a hybrid sampling strategy that combines random frame selection with importance-weighted sampling. This approach encourages broad temporal coverage while emphasizing safety-critical transitions, thereby improving the model’s ability to learn informative world knowledge relevant to long-term causality and decision-making. We argue that world representations should remain highly abstract; excessive visual detail at the latent level dilutes the model’s capacity for structural world understanding. To reconcile abstraction with perceptual fidelity, we delegate photorealistic synthesis to a dedicated diffusion module. This enables highfidelity multi-view rendering, where a diffusion-based renderer reconstructs detailed surround-view cameras within the auto-regressive loop, providing dense latent-space supervision.

In summary, the key contributions of this work are:

• We introduce a long-horizon chunk-wise auto-regressive strategy that leverages extended future horizons for world modeling. This design both mitigates the collapse of worldknowledge learning under naive next-frame prediction and resolves the temporal dilemma: dense intra-chunk frames allow the model to capture instantaneous dynamics, while longhorizon chunk-level prediction promotes the learning of broader world causality.

• To improve long-horizon forecasting stability and enhance policy robustness, we exploit a curriculum-based learning strategy, starting with short-horizon prediction across chunks and gradually extending to longer strides.

• We adopt temporal importance sampling, a hybrid sampling mechanism that accounts for uneven contributions of future frames to world knowledge learning, combining random selection with importance-weighted focus on critical temporal transitions.

• The proposed method can acquire high-fidelity multi-view future images. Integration of a diffusion-based renderer into the auto-regressive pipeline is developed to reconstruct photorealistic surround-view details.

Comprehensive experiments demonstrate that X-Foresight significantly outperforms the reactive VLA baseline in planning safety, generative fidelity as shown in Fig 1, establishing a robust paradigm for world-knowledge-driven autonomous systems.

## 2 Data

## 2.1 Data scale/resolution

![](images/074013250235183a77e719bda1f5e28f2d5d317b057f6247041cd65256f9ee54.jpg)  
Figure 2: Overview of the large-scale multi-camera driving dataset. The dataset contained approximately 280,000 hours of driving data, segmented into 34M clips. Multi-view images from seven cameras (front fisheye, front narrow, left/right front, left/right rear, and rear) were processed and tokenized into 13.8T tokens, which were used for training the world model.

We built an industry-level dataset with approximately 280,000 hours of in-house driving data, spanning various traffic conditions and environmental scenarios, as illustrated in Fig 2. In total, the data was segmented into 34M clips of up to 30 seconds each and tokenized into 13.8T tokens from multi-view observations. All camera streams were stored at a native frequency of 12 Hz and downsampled to 4 Hz for training. This setup provided sufficient temporal context for long-horizon world modeling while maintaining a tractable sequence length for efficient training.

Training Scenario Distribution Each bar shows the total share of one scenario group; colors split the share into urban and highway clips  
![](images/e84f8ef677be0ef16290bef17bd7f90374be31319372db5356c8efe5fcd49f75.jpg)  
Figure 3: Training scenario distribution of X-Foresight. Fine-grained auto-tags were grouped into eight scenario categories. Bar length denoted the total share of the category, colors split the share by road class (urban / highway), and numbers in parentheses denoted the number of fine-grained labels assigned to each category.

The dataset contained two native resolution configurations. To balance computational efficiency and spatial fidelity, images from different cameras were resized to viewpoint-specific resolutions to better preserve spatial characteristics. This temporal configuration provided a favorable trade-off between motion fidelity and computational cost, allowing the model to capture fine-grained dynamics while keeping sequence lengths tractable for training.

Data was collected using a 7-camera surround-view system that provides full 360-degree coverage. The setup consisted of one front-facing fisheye camera for wide field-of-view perception, one frontfacing narrow camera for long-range observation, four side cameras for cross-traffic awareness, and one rear camera. The camera layout and mounting configuration were consistent with X-World [22]. All cameras were geometrically calibrated with respect to the ego vehicle frame. This multi-camera configuration enabled consistent multi-view observations and reduced occlusions, which was critical for learning a spatially coherent and temporally consistent world model.

## 2.2 Distribution

Following the auto-tagging and distribution-analysis protocol of X-World [22], we analyzed the scenario distribution of the X-Foresight training set from auto-tagged driving clips. The original tags covered nearly 200 fine-grained scenarios, including road type, ego maneuver, road layout, surrounding traffic, and long-tail safety cases. Here, we aggregated these fine-grained tags into eight scenario categories according to the main driving context of each tag. This grouping preserved the main driving regimes in the dataset while retaining explicit categories for interaction-heavy and long-tail scenarios.

Fig 3 showed the aggregated distribution. Routine driving still took the largest share: open-road lane keeping accounted for 21.0% of the data and lane changes accounted for 20.1%. Constrainedlane driving, such as line straddling and unmarked roads, made up 16.0%, while intersections and turns contributed 13.1%. The rest of the data covered interaction and long-tail scenarios, including obstacles and cut-ins (9.9%), following and congestion (9.6%), VRU interactions (6.2%), and rare road layouts such as ramps, toll stations, roundabouts, auxiliary roads, and high-curvature segments (4.1%). Together, common driving and maneuvering cases accounted for about 70% of the training data, while the remaining portion provided coverage for more challenging agent interactions and road structures.

![](images/9676e8245d496c5956e448a4e12e4dfddff7e27d4c0b591b3badf1df8b442294.jpg)  
Figure 4: The training pipelines of the proposed X-Foresight. X-Foresight consisted of two main components, a Large Drive Model (LDM) that performed joint world prediction and action planning in a unified token space, and a Vision Renderer that decoded the LDM’s predicted camera tokens into photorealistic multi-view frames. In stage I and stage II, the LDM and the Vision Renderer were trained separately, with the renderer conditioned on ground-truth future trajectories; in stage III, the LDM was frozen and the renderer’s conditioning was swapped to the LDM’s predicted future camera tokens.

The road-class split further showed that the dataset was dominated by urban driving (86.8%), with highway clips accounting for 13.2%. This mix reflected the operating regimes most frequently encountered in large-scale public-road driving, while still preserving explicit coverage of safetycritical and long-tail scenarios such as VRU interactions, cut-ins, ramps, and toll stations. The resulting distribution provided broad supervision for world-model learning and supported scenarioaware evaluation of X-Foresight across both common driving patterns and challenging edge cases.

## 3 Methodology

This section presented the methods employed in X-Foresight. X-Foresight consisted of two main components: a Large Drive Model (LDM) that performed joint world prediction and action planning in a unified token space, and a Vision Renderer that decoded the LDM’s predicted camera tokens into photorealistic multi-view frames as shown in Fig 4. The LDM was an auto-regressive transformer that consumed a temporal stream of multi-view camera observations, language instructions, egovehicle state, and predicted three kinds of future targets at every step: ego actions used for real-time control, a bird’s-eye-view (BEV) plot that captured the surrounding spatial structure, and per-camera latent tokens that summarized the predicted future appearance from each viewpoint. The Vision Renderer was a diffusion-based decoder whose rendered frames were fed back into the LDM as fresh observations, closing the auto-regressive prediction loop. We further described the training strategy and the inference pipeline of the proposed X-Foresight.

## 3.1 Large Drive Model

## 3.1.1 Design Rationale

To enable world understanding in X-Foresight, Large Drive Model (LDM) was explicitly trained to predict future images, bird’s-eye-view (BEV) representations, and actions. We adopted a long-horizon chunk-wise auto-regressive strategy to perform future rollouts over extended horizons. Training on long-horizon data allowed the model to capture world transition causality while improving data utilization. Further details were provided in Section 3.1.3.

The chunk-wise design addressed an inherent temporal dilemma: instantaneous dynamics required densely sampled short-range frames, whereas long-term causality unfolded over variable-length horizons. Chunk-wise future rollout served as the key mechanism to reconcile these two requirements. Inspired by the success of autoregressive modeling in LLMs and VLMs, we adopted an autoregressive formulation that further enabled extensible future rollout generation.

The long-horizon chunk-wise auto-regressive strategy within LDM also facilitated rapid deployment. Since world knowledge was encoded during training, LDM at inference time was capable of both producing high-frequency instantaneous control signals and generating long-horizon rollouts with high-fidelity multi-view image synthesis, with the support of Vision Renderer.

## 3.1.2 Multi-modal Prompt Design

The core component of the X-Foresight framework was the LDM trained to predict future ego actions and future observations simultaneously. As illustrated in Fig 4, we formulated this problem as a multi-modal prompting task:

$$
[ \text { SYSTEM   PROMPT } ] \mid [ l _ {0}, O _ {0}, A _ {0}, Q _ {0} ] \mid [ l _ {1}, O _ {1}, A _ {1}, Q _ {1} ] \mid \dots \mid [ l _ {i}, O _ {i}, A _ {i}, Q _ {i} ].
$$

The prompt consisted of a global system prompt followed by a sequence of temporal chunks. The system prompt provided general task instructions and high-level context about the ego vehicle, including its long-horizon navigation objective. Each temporal chunk contained four types of tokens. The text tokens $\mathbf { \overline { { \boldsymbol { l } } } } _ { i }$ specified the prediction horizon or temporal window. The observation tokens $O _ { i }$ represented multi-camera video tokens extracted from multiple frames within the chunk using a ViT [5] encoder. The action/state tokens $A _ { i }$ encoded the ego-vehicle states, which can be derived from historical trajectories. The query tokens $Q _ { i }$ were used to trigger the prediction of future variables. To reduce the overall token budget, video tokens were predicted in place rather than through additional visual query tokens.

## 3.1.3 Extension to Longer Horizon

While our prompt design could naturally extend to longer horizons, directly increasing the prediction duration was computationally prohibitive: both memory footprint and computational cost grew quadratically with sequence length. In this section, we introduced several key designs which not only improved model performances, but also optimized LDM training on longer horizon.

![](images/398e330569b7e87229e8254480643e62dfe69e73b180e0db8580b46a42b4c631.jpg)  
Figure 5: Prompt formulations for future frame prediction. (a) Frame-wise foresight one frame at each step. (b) Frame-wise longer foresight increased the temporal stride s. (c) Chunk-wise foresight predicted a chunk of K consecutive frames in parallel. (d) Chunk-wise longer foresight combined chunk length K with stride s for longer-horizon prediction. (e) Chunk-wise temporal importance sampling.

Chunk-Wise Prediction In Large Drive Model (LDM), we adopted a chunk-wise formulation in which the prompts were segmented into short, fixed-length units (typically of 1 second duration)

that were predicted by the model in an auto-regressive manner. This design was motivated by both modeling considerations and computational efficiency. As illustrated in $\mathrm { F i g } \bar { 5 } ( \mathrm { a } )$ , frame-wise foresight provided only a weak learning signal at each step, since adjacent camera frames typically differed only marginally over short time intervals. A simple alternative (Fig 5 (b)) was to downsample the frame sequence and perform frame-wise longer foresight at a coarser temporal resolution. However, this could degrade performance because trajectory prediction relied on temporally coherent motion cues, which might be lost or aliased under aggressive temporal downsampling. In contrast, chunkwise foresight, shown in $\mathrm { F i g } 5 \left( \mathrm { c } \right)$ , preserved short-term temporal structure within each chunk while requiring the model to predict a longer segment of future evolution at each auto-regressive step.

Curriculum Learning with Extended Foresight To further improve long-horizon prediction, we adopted a short-to-long curriculum learning strategy. Training started with short-horizon sequences in which adjacent chunks were temporally contiguous and separated by 1 second. We then extended the effective prediction horizon by increasing the temporal stride between neighboring chunks to 3 seconds, which was called Chunk-wise longer foresight as shown in $\mathrm { F i g } 5 \left( \mathrm { d } \right)$ . This larger temporal gap between chunks extended the prediction horizon while it did not increase the computational budgets.

Under this temporally strided setting, the observation targets were predicted across the extended time gaps between chunks. In contrast, action prediction remained at the immediate next control step, since closed-loop control required temporally dense, unstrided trajectory outputs. This asymmetric design allowed the model to learn long-horizon visual evolution while preserving the temporal resolution required for ego-action prediction.

Temporal Importance Sampling Chunk-wise longer foresight let LDM train on long horizons at tractable cost. A uniform sampler, however, wasted most of the budget on near-cruise segments and under-supervised the rare events that drove safety-critical decisions. We replaced it with an importance-weighted sampler as in Fig 5 (e).

For each candidate step $k ,$ we computed an importance score $w _ { k }$ based on the longitudinal and lateral accelerations of the ego trajectory, denoted by $a _ { x }$ and $a _ { y } ,$ , respectively. To capture safety-relevant dynamics at different temporal scales, we evaluated the trajectory over three temporal windows around the candidate step, denoted as $W _ { 1 } ^ { k } , \dot { W } _ { 2 } ^ { k }$ , and $\boldsymbol { W } _ { 3 } ^ { k }$ . Within each window, we took the maximum value of a weighted acceleration magnitude, and then summed the resulting scores across windows:

$$
w _ {k} = \sum_ {W \in \{W _ {1} ^ {k}, W _ {2} ^ {k}, W _ {3} ^ {k} \}} \max _ {t \in W} (\lambda_ {x} | a _ {x} (t) | + \lambda_ {y} | a _ {y} (t) |).\tag{1}
$$

where $\lambda _ { x } \geq 0$ and $\lambda _ { y } \geq 0$ controlled the relative importance of longitudinal and lateral accelerations. The three temporal windows were designed to capture complementary maneuver phases. The nearfuture window $W _ { 1 } ^ { k }$ emphasized imminent events such as braking, acceleration, or sudden swerve onsets. The mid-horizon window $\boldsymbol { W } _ { 2 } ^ { k }$ captured upcoming maneuver commitments, including turn-in and brake-in behavior. The recent-history window $\boldsymbol { W } _ { 3 } ^ { k }$ captured the aftermath of maneuvers that had just been executed. Together, these windows assigned higher sampling probability to steps that were temporally close to safety-relevant trajectory changes.

Given the per-step importance scores $\{ w _ { k } \}$ , we sampled candidate steps according to a temperaturescaled distribution:

$$
p _ {k} = \frac {w _ {k} ^ {1 / \tau}}{\sum_ {j} w _ {j} ^ {1 / \tau}},\tag{2}
$$

where $\tau > 0$ controlled the sharpness of the distribution. In practice, we additionally constrained the maximum temporal gap between consecutive selected steps. This prevented the sampled sequence from becoming overly sparse and ensured that future prediction remained well-conditioned on recent temporal context.

Semi-Causal Block Sparse Attention To further speed up efficient long-horizon training, we adopted a semi-causal block-sparse attention mask implemented using block sparse attention [8], inspired by the sparse attention patterns in [13]. The token sequence was first padded and partitioned into fixed-size blocks, and sparsity was imposed at the block level.

Semi-causal block-sparse attention mask  
![](images/f84ea1499aa5e7d82be87e995f5787659b97050a856dbb27fc9c01c2d8cc71f9.jpg)  
Figure 6: Semi-causal block-sparse attention mask for long-sequence training. Each colored pixel denoted one token block, where attention was allowed. The mask preserved bidirectional attention within each temporal chunk, allowed access to the global system prompt and previous prompt tokens, and prohibited attention between query tokens across different chunks. The two panels showed complementary sparse patterns assigned to different attention-head groups.

As illustrated in Fig 6, the attention mask was designed to preserve the temporal structure of the prediction task while reducing unnecessary long-range interactions. The system prompt was treated as a set of global sink tokens and was visible to all subsequent temporal chunks. Within each chunk, token blocks were connected with fully bidirectional self-attention, allowing the model to jointly reason over the text, observation, action/state, and query tokens associated with the same temporal window.

A temporally causal design was employed for interactions across different chunks. Because every chunk shared an identical internal layout, each prompt-side token, including $l _ { i } , O _ { i }$ and $A _ { i } .$ , had a positional counterpart in every earlier chunk; cross-chunk attention was defined with respect to this counterpart, covering the same position and a surrounding spatial neighborhood whose width shrank with temporal distance; this design allowed both temporal tracking and spatial reasoning while keeping total compute manageable. Query tokens $Q _ { i }$ retained full access to all preceding prompt-side tokens, enabling future predictions to condition on the available observation and action history. However, attention from $Q _ { i }$ to previous query tokens $Q _ { 1 : i - 1 }$ was masked to prevent predictions from directly depending on earlier predicted targets.

To further reduce computation, we partitioned attention heads into two disjoint groups based on the parity of the timestep difference between query and key blocks. Each group attended to a complementary subset of temporal offsets, which reduced the number of active block connections while maintaining coverage over the temporal context. As a result, the number of attended blocks grew approximately linearly with sequence length, rather than quadratically.

## 3.1.4 Training Objectives

During training, we optimized the model to predict future observations and ego actions under a teacher-forcing formulation. Specifically, ground-truth observations and actions were encoded and inserted into the multi-modal prompt for previous temporal chunks, while the model was trained to predict the targets at future prediction steps. The training loss was applied to both the predicted observation tokens $\hat { \mathbf { o } } _ { i } ^ { v }$ and the decoded ego actions $\hat { \mathbf { a } } _ { i }$

$$
L _ {\mathrm{cam}} = \frac {1}{H V} \sum_ {i = 1} ^ {H} \sum_ {v = 1} ^ {V} \| \hat {\mathbf {o}} _ {i} ^ {v} - g (\mathbf {I} _ {i} ^ {v}) \| _ {2},\tag{3}
$$

$$
L _ {\mathrm{act}} = \frac {1}{H} \sum_ {i = 1} ^ {H} \left\| \hat {\mathbf {a}} _ {i} - \mathbf {a} _ {i} \right\| _ {1}.\tag{4}
$$

where H was the total horizon used in training, $V$ was the number of camera views. $g ( \cdot )$ was the frozen ViT from the base vision language model.

We additionally incorporated an auxiliary bird’s-eye-view (BEV) prediction loss during training to encourage the model to learn geometrically meaningful scene representations. The predicted BEV map $\hat { \mathbf { b } } _ { i }$ was decoded from a subset of the query tokens in $Q _ { i }$ <sub>i</sub> and supervised by the corresponding ground-truth BEV target $\mathbf { b } _ { i } \mathbf { : }$

$$
L _ {\mathrm{bev}} = \frac {1}{H} \sum_ {i = 1} ^ {H} \left\| \hat {\mathbf {b}} _ {i} - \mathbf {b} _ {i} \right\| _ {2}.\tag{5}
$$

The final training objective was a weighted sum of the action, camera, and BEV losses:

$$
L _ {\mathrm{total}} = L _ {\mathrm{act}} + \alpha L _ {\mathrm{cam}} + \beta L _ {\mathrm{bev}},\tag{6}
$$

where $\alpha$ and $\beta$ controlled the relative weights of the visual prediction and BEV prediction losses, respectively.

## 3.2 Vision Renderer

The Vision Renderer was the terminal module of X-Foresight’s auto-regressive pipeline: it converted the LDM’s predicted camera latent tokens into high-fidelity multi-view future frames that could be fed back into the vision encoder to close the inference loop. Camera latent tokens were optimized for auto-regressive reasoning and therefore compressed away much of the high-frequency appearance detail required for photorealistic closed-loop feedback; decoding them directly in the vision decoder would accumulate reconstruction error over long rollouts and degrade downstream perception. Unlike prior generative world models that coupled action-conditioned dynamics with pixel synthesis in a single network, we intentionally separated the two roles — imagination and control lived in the LDM, while the renderer acted as a conditional refiner that, conditioned on a short window of history images and the LDM-predicted camera tokens, restored the high-frequency visual detail lost during latent compression.

## 3.2.1 Design Rationale

A central design decision was that the renderer was not conditioned on the LDM’s action tokens. The camera tokens already encoded the ego-vehicle transformation implicitly, together with surrounding scene geometry and dynamic agents; in our setting we found that the camera token was the only LDM-produced conditioning signal required, with multi-view history continuing to enter through the standard I2V pathway and no additional action conditioning needed. We adopted this restriction to avoid the risk that, if both signals were exposed, the renderer might exploit action tokens as a low-entropy shortcut and ignore the camera tokens, causing the LDM and renderer to predict futures independently and defeating the purpose of the closed loop. As a consistency check, we compared the renderer’s frames against the pixel-space decode of the same camera tokens produced by the Camera Latent Decoder (Section 4.1.3); their agreement was consistent with the camera token acting as the effective information bottleneck through which the LDM controlled the rendered future.

## 3.2.2 Model Architecture

Backbone The Vision Renderer was built upon X-World, a Diffusion Transformer (DiT) [15] video generator that coupled a 3D causal VAE with a latent denoiser trained under a rectified-flow objective [14]. The encoder and decoder were based on the 3D causal VAE from the WAN2.2 release of WAN [18], which compressed input videos along both spatial and temporal dimensions into a compact latent space for efficient denoising. Rectified flow learned a time-dependent velocity field $v _ { \theta } ( \mathbf { y } _ { t } , t , \mathbf { c } )$ that mapped a Gaussian prior to the data distribution along a straight interpolation path

$$
\mathbf {y} _ {t} = (1 - t) \mathbf {y} _ {0} + t   \mathbf {y} _ {1}, \quad \mathbf {y} _ {0} \sim p _ {\text { data }} (\mathbf {y} \mid \mathbf {c}), \mathbf {y} _ {1} \sim \mathcal {N} (\mathbf {0}, \mathbf {I}), t \sim \mathcal {U} (0, 1),\tag{7}
$$

by minimizing

$$
\mathcal {L} _ {\mathrm{velocity}} (\theta) = \mathbb {E} _ {\mathbf {y} _ {0}, \mathbf {y} _ {1}, t, \mathbf {c}} \left[ \| v _ {\theta} (\mathbf {y} _ {t}, t, \mathbf {c}) - (\mathbf {y} _ {1} - \mathbf {y} _ {0}) \| _ {2} ^ {2} \right],\tag{8}
$$

where c denoted the conditioning inputs passed to the renderer. Relative to standard denoising diffusion, rectified flow yielded straighter probability paths, enabling high-quality samples with fewer function evaluations.

Cross-view Attention To enforce geometric consistency across synchronized surround-view cameras, we adopted the view-temporal self-attention design of X-World: self-attention was applied alternately along the temporal and cross-view axes over latent tokens from multiple cameras, encour aging coherent geometry, object identity, and motion across viewpoints while preserving temporal smoothness.

Conditioning The renderer departed from X-World in what it conditioned on. X-World was a full generative simulator and therefore accepted vectorized conditions describing dynamic agents (bounding boxes, categories) and static road elements (lane lines, boundaries) alongside text prompts, actions, and camera parameters. In X-Foresight, these scene descriptors were already encoded by the LDM into the camera token stream: the tokens carried both the ego-vehicle pose and the predicted placement and behavior of surrounding agents. Consequently, in its final configuration the renderer retained only a single cross-attention branch that injected the LDM’s camera tokens at each DiT block, with X-World’s action, dynamic-agent, static-element, and text-conditioning branches all removed. The action-conditioning branch was retained temporarily during fine-tuning so that the renderer could be adapted under ground-truth ego actions before being spliced onto the LDM’s camera-token stream (Section 4.2.1). Multi-view history frames were provided through the standard image-to-video (I2V) input pathway of WAN.

Temporal Alignment Most pre-trained video generation models operated at 12 Hz, whereas X-Foresight’s LDM emitted one camera token per rendered frame at 4 Hz, matching the output rate of the action policy head. We adapted the renderer to 4 Hz during fine-tuning (Section 4.2.1) so that each rendered frame was tightly anchored one-to-one to a corresponding LDM camera-token prediction.

Rollout Drift Mitigation Under closed-loop deployment, each rendered frame became part of the conditioning history for the next step, exposing the renderer to its own samples rather than ground-truth frames. This distributional gap accumulated over the rollout and caused predicted frames to drift off the data manifold. Recent works such as CausVid [20] and Self-Forcing [10] addressed this via distribution matching distillation (DMD). For X-Foresight, we instead adopted two lighter-weight modifications: a latent sink that anchored a stable reference context across rollout steps, and latent augmentation on the current-step latent during training, following Helios [21], which exposed the renderer to perturbed conditioning resembling its inference-time distribution.

## 3.3 Pipeline

## 3.3.1 Training

X-Foresight coupled a Large Drive Model (LDM), which jointly modeled actions and high-level visual semantics, with a diffusion-based vision renderer that produced photorealistic multi-view frames. Training proceeded in three stages. In stage I, the LDM was trained alone under teacher forcing to emit future actions together with compact camera tokens that summarized the mean of plausible future appearances. In stage II, the vision renderer was independently adapted to synthesize photorealistic multi-view imagery conditioned on ground-truth future ego actions. In stage III, the LDM wasfrozen and only the renderer was further fine-tuned, with its action-conditioning branch swapped for the LDM’s predicted future camera tokens, so that the renderer learned to produce frames consistent with the token distribution it would actually consume under the closed-loop auto-regressive rollout used at inference. This decoupled-then-aligned schedule kept LDM and renderer training tractable in stages I and II while letting stage III close the gap between training-time teacher forcing and inference-time rollout, without disturbing the LDM’s learned action and semantic predictions.

Stage I: LDM Pretraining We trained the Large Drive Model (LDM) under teacher forcing to jointly predict future ego actions, multi-view camera tokens, and an auxiliary BEV latent, using the chunk-wise multi-modal prompt and the combined loss $L _ { \mathrm { t o t a l } } = L _ { \mathrm { a c t } } + \alpha L _ { \mathrm { c a m } } + \beta L _ { \mathrm { b e v } } $ Three components made this stage tractable on long horizons: a semi-causal block-sparse attention mask, with heads partitioned by parity so block density grew linearly with sequence length; a short-to-long curriculum that progressively increased the inter-chunk stride from 1 s to 3 s, shifting supervision from near-frame extrapolation toward longer-horizon dynamics; and an importanceweighted temporal sampler that scored candidate AR steps by peak weighted longitudinal/lateral acceleration, concentrating supervision on safety-critical futures. Because each predicted camera token was regressed against many plausible futures under an L2 objective, it converged to a low entropy summary of the conditional future — enough for action grounding and scene structure, but visually blurry when decoded directly.

Stage II: Renderer Pretraining In parallel, the renderer was adapted to synthesize photorealistic multi-view frames conditioned on ground-truth ego actions, rather than on camera tokens predicted by the LDM. We initialized from X-World, a DiT video generator that already provided crossview temporal attention, and adapted it in two steps. (i) Temporal alignment. We fine-tuned the renderer from WAN’s native 12 Hz to the LDM’s 4 Hz emission rate, so each generated frame was anchored one-to-one to a single LDM-predicted token in stage III. (ii) Rollout-drift mitigation. We attached a latent sink that anchored a stable reference context across rollout steps, and applied latent augmentation to the current-step latent during training so the renderer saw perturbed conditioning resembling its inference-time distribution.

Stage III: Renderer Alignment In the third stage, the LDM was frozen and the vision renderer was further finetuned so that rendered frames stayed consistent with the camera tokens the LDM actually produced under the closed-loop rollout used at inference. The renderer now conditioned on camera tokens predicted by the LDM rather than ground-truth actions, and was supervised solely by the rectified-flow velocity-matching objective $\bar { \mathcal { L } } _ { \mathrm { v e l o c i t y } }$ in Eq. (8). The stage-I action and BEV losses on the LDM were dropped: the LDM no longer received gradients, and spatial structure was anchored through the renderer’s pixel-space supervision. Finetuning at the 4 Hz cadence established in stage II closed the train/inference gap left by teacher forcing and yielded the tightly interleaved prediction-rendering loop deployed at inference time.

## 3.3.2 Inference

At inference time, X-Foresight performed an auto-regressive rollout that strictly interleaved prediction and rendering at the 4 Hz cadence established in stage III. Each step began with a text prompt and a short window of multi-view history frames. The LDM consumed this context using the same chunk-wise prompt format as in training and emitted, in a single forward pass, the next ego action and one second of multi-view latent tokens — four frames per camera at 4 Hz, all seven camera views per frame, each camera-frame represented by several tokens — summarizing the predicted future appearance from each viewpoint. Decoding the camera token directly through the Camera Latent Decoder yielded a blurry image, because the token represented the mean of plausible futures rather than any specific realization. The vision renderer then conditioned on this token together with the multi-view history and ran a small number of rectified-flow sampling steps to draw one concrete future from the implied distribution, producing high-fidelity, view-consistent surround imagery. Because the renderer was conditioned only on the camera token — not on the predicted action — the rendered future was fully determined by the LDM’s latent imagination, which also let us audit alignment by comparing the renderer’s frames against the Camera Latent Decoder’s decode of the same token. The rendered frames and the new action were appended to the rolling context and fed back into the LDM for the next step. Iterating this loop produced arbitrarily long closed-loop rollouts of paired action trajectories and photorealistic surround-view video.

## 4 Experiments

This section presented the implementation details of both LDM and Vision Renderer. Comprehensive experiments and ablation studies were conducted on our in-house dataset.

## 4.1 Large Drive Model

## 4.1.1 Trajectory Results

This section evaluated X-Foresight along three axes. We first examined the effect of training with longer chunk-wise horizons. We then ablated three refinements of long-horizon training: curriculum learning (CL), curriculum learning with extended foresight (CLEF) and temporal importance sampling (TIS). Finally, we reported the combined system at production scale, both quantitatively and through two qualitative scenarios.

We reported two complementary sets of metrics. ADE and FDE (in meters) measured the displacement between predicted and ground-truth ego trajectories. The CCES suite measured driving quality across four axes: Compliance (traffic-rule adherence, e.g., lane centering, line crossings, red-light running), Comfort (ride quality, e.g., steering instability, motion sickness), Efficiency (progress toward destination, e.g., under-acceleration, failure to overtake slow vehicles), and Safety (avoidance of dangerous events, e.g., collisions, time-to-collision violations, hitting road boundaries). Each underlying metric was a per-frame fail-rate proportion. To put metrics of very different magnitudes on a common scale, we took the first row of Table 1 as the shared reference and reported every other entry across all three tables as its ratio to that reference; each CCES category cell was then the unweighted mean of these ratios across the curated metrics in that category, and Total was the sum of the four category means. By construction the reference row therefore had 1 in every category cell and 4 in the Total column, and lower was better everywhere. Collision rate was additionally reported as a standalone safety headline in its raw percentage fail-rate form. To ensure a fair comparison, all models were evaluated using only a single forward step at inference time, isolating the effect of training-time supervision on single-step inference quality.

Effect of Long-Horizon Training. Table 1 evaluated the effect of scaling the training-time horizon $H \in \{ 1 , 6 , 2 1 \}$ , where each step predicted one chunk of one second. This covered no long-horizon supervision $( H { = } 1$ , single-chunk training), a moderate horizon $\left( H { = } 6 \right)$ , and our full long-horizon configuration $\left( H { = } 2 1 \right)$ . All three rows shared architecture, data, hardware (128 GPUs), and number of training steps; only H varied. ADE, FDE, and the standalone collision rate all improved monotonically with H, dropping by roughly 2–7% from H=1 to H=21. The Safety $( 1 . 0 0 0 0  0 . 9 4 8 1 )$ ) and Compliance $( 1 . 0 0 0 0  0 . 9 5 3 3 )$ categories, the axes most directly aligned with long-horizon worldcausality supervision, also improved monotonically. Comfort and Efficiency saw small regressions at H=21, revealing the optimization difficulty of naively scaling H; these partially offset the Safety/Compliance gains in the summed Total, which dipped from 4.0000 at $H { = } \bar { 1 }$ to 3.9396 at $H { = } 6$ but regressed marginally to 3.9524 at H=21. The overall direction confirmed our central claim that long-horizon chunk-wise supervision benefited the safety- and compliance-related axes, and the small Total regression at H=21 motivated the curriculum-learning and importance-sampling designs ablated next.

Table 1: Effect of training-time horizon H. All models shared architecture, data, hardware (128 GPUs), and number of training steps; only H varied. ADE/FDE in meters; Coll. is the percentage collision fail rate (%); Compl., Comfort, Eff., and Safety are each the unweighted mean of permetric ratios to $H { = } 1$ (lower = better); Total sums the four category means and equals 4 at $H { = } 1$ by construction. Lower is better throughout.

<table><tr><td rowspan="2">Method</td><td colspan="2">ADE ↓</td><td colspan="2">FDE ↓</td><td rowspan="2">Coll. ↓</td><td rowspan="2">Compl. ↓</td><td rowspan="2">Comfort ↓</td><td rowspan="2">Eff. ↓</td><td rowspan="2">Safety ↓</td><td rowspan="2">Total ↓</td></tr><tr><td>Lat.</td><td>Long.</td><td>Lat.</td><td>Long.</td></tr><tr><td>H=1</td><td>0.1923</td><td>1.2409</td><td>0.4881</td><td>3.1935</td><td>0.263</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>4.0000</td></tr><tr><td>H=6</td><td>0.1864</td><td>1.2196</td><td>0.4691</td><td>3.1178</td><td>0.262</td><td>0.9756</td><td>0.9880</td><td>0.9833</td><td>0.9927</td><td>3.9396</td></tr><tr><td>H=21</td><td>0.1810</td><td>1.2110</td><td>0.4571</td><td>3.0988</td><td>0.245</td><td>0.9533</td><td>1.0416</td><td>1.0094</td><td>0.9481</td><td>3.9524</td></tr></table>

Ablating CL, CLEF and TIS. Table 2 ablated how the $H { = } 2 1$ horizon should be supervised. All four rows were initialized from the $H { = } 6$ checkpoint and additionally trained on 128 GPUs; only the second-stage strategy differed. Row 1 (Cont. $\scriptstyle { \bar { H } } = 6 )$ continued training at $H { = } 6$ . Row $2 \ : ( + H ) = 2 1$ CL) applied basic curriculum learning (short-to-long H schedule at fixed 1 s inter-chunk stride). Row 3 (+ H=21, CLEF) applied our curriculum learning with extended foresight, progressively widening the inter-chunk stride from 1 s to 3 s. Row 4 $\scriptstyle ( + H = 2 1$ , TIS) added our temporal importance sampling on top of CLEF, concentrating supervision on safety-critical chunks identified by peak weighted lateral/longitudinal acceleration. Continuing H=6 (Row 1) reduced ADE/FDE relative to the $H { = } 6$ checkpoint while collision rate plateaued (0.262 → 0.270), but did not match Rows 3 and 4, ruling out additional training time as the sole explanation. Row 2 (CL) further reduced collision rate $( 0 . 2 7 0  0 . 2 3 8 )$ and Total CCES (3.9523 → 3.8745). CLEF (Row 3) delivered the strongest ADE/FDE gains (lowest lateral ADE/FDE), indicating that progressively extending the foresight horizon stabilized trajectory prediction, though Safety regressed slightly versus CL $( 0 . 9 3 1 0  0 . 9 3 8 7 )$ , motivating the safety-targeted sampler in Row 4. TIS (Row 4) further reduced the collision rate from 0.230 to 0.216, a 6.1% relative reduction, and achieved the lowest Total CCES (3.8447); the Safety ratio improved to 0.9264, the lowest among the four rows, consistent with the design intent of biasing supervision toward safety-critical chunks.

Table 2: Ablating CL, CLEF, and TIS. All four rows were initialized from the H=6 checkpoint and additionally trained on 128 GPUs. ADE/FDE in meters; Coll. is the percentage collision fail rate (%); Compl., Comfort, Eff., and Safety are unitless ratios as in Table 1. Lower is better. Cont.: continue training at H=6; CL: basic curriculum learning; CLEF: curriculum learning with extended foresight; TIS: temporal importance sampling.

<table><tr><td rowspan="2">Method</td><td colspan="2">ADE ↓</td><td colspan="2">FDE ↓</td><td rowspan="2">Coll. ↓</td><td rowspan="2">Compl. ↓</td><td rowspan="2">Comfort ↓</td><td rowspan="2">Eff. ↓</td><td rowspan="2">Safety ↓</td><td rowspan="2">Total ↓</td></tr><tr><td>Lat.</td><td>Long.</td><td>Lat.</td><td>Long.</td></tr><tr><td>Cont. H=6</td><td>0.1741</td><td>1.1807</td><td>0.4344</td><td>3.0087</td><td>0.270</td><td>0.9302</td><td>1.0515</td><td>0.9980</td><td>0.9726</td><td>3.9523</td></tr><tr><td>+H=21, CL</td><td>0.1718</td><td>1.1671</td><td>0.4277</td><td>2.9856</td><td>0.238</td><td>0.9326</td><td>1.0106</td><td>1.0003</td><td>0.9310</td><td>3.8745</td></tr><tr><td>+H=21, CLEF</td><td>0.1692</td><td>1.1571</td><td>0.4181</td><td>2.9421</td><td>0.230</td><td>0.9320</td><td>1.0076</td><td>0.9951</td><td>0.9387</td><td>3.8734</td></tr><tr><td>+H=21, TIS</td><td>0.1696</td><td>1.1578</td><td>0.4195</td><td>2.9413</td><td>0.216</td><td>0.9187</td><td>1.0043</td><td>0.9953</td><td>0.9264</td><td>3.8447</td></tr></table>

Production-Scale Comparison. Table 3 reported the headline comparison at production scale, combining long-horizon training (H=21) with CLEF and TIS, trained on 1024 GPUs against the baseline trained at the same scale. The full X-Foresight model reduced lateral/longitudinal ADE by 6.4%/3.6% (0.1675 → 0.1567 lat; 1.1387 → 1.0982 long) and lateral/longitudinal FDE by 8.8%/4.1% (0.4153 → 0.3789 lat; 2.9117 → 2.7924 long). Collision rate dropped from 0.228% to 0.191%, a 16.2% relative reduction. All four CCES categories improved: Safety by 9.1%, Compliance by 8.2%, Comfort by 1.0%, and Efficiency by 0.4%. The aggregate Total CCES improved from 3.8296 to 3.6535, a 4.6% relative reduction, with the gains concentrated in Safety and Compliance, consistent with the central claim that long-horizon world-causality supervision improved safety-critical decision quality.

Table 3: Production-scale comparison against the baseline, both trained on 1024 GPUs. X-Foresight (Ours) used H=21 with CLEF and TIS. ADE/FDE in meters; Coll. is the percentage collision fail rate (%); Compl., Comfort, Eff., and Safety are unitless ratios as in Table 1. Lower is better.

<table><tr><td rowspan="2">Method</td><td colspan="2">ADE ↓</td><td colspan="2">FDE ↓</td><td rowspan="2">Coll. ↓</td><td rowspan="2">Compl. ↓</td><td rowspan="2">Comfort ↓</td><td rowspan="2">Eff. ↓</td><td rowspan="2">Safety ↓</td><td rowspan="2">Total ↓</td></tr><tr><td>Lat.</td><td>Long.</td><td>Lat.</td><td>Long.</td></tr><tr><td>Baseline</td><td>0.1675</td><td>1.1387</td><td>0.4153</td><td>2.9117</td><td>0.228</td><td>0.9483</td><td>0.9505</td><td>0.9867</td><td>0.9441</td><td>3.8296</td></tr><tr><td>X-Foresight</td><td>0.1567</td><td>1.0982</td><td>0.3789</td><td>2.7924</td><td>0.191</td><td>0.8708</td><td>0.9413</td><td>0.9831</td><td>0.8583</td><td>3.6535</td></tr></table>

The two scenarios in Fig 7 illustrated how this aggregate gain manifested qualitatively, both turning on correct action toward events that lay ahead in space or in time, the central capability the baseline lacked by construction. In the top panel, the ego entered a multi-exit roundabout under a navigation instruction that directed it to take a later exit rather than the first one. The baseline committed to the closest visible exit and emitted a trajectory that veered off the lane, diverging substantially from the ground truth. X-Foresight held the roundabout lane and tracked the ground truth all the way to the intended exit, anchoring its action on the long-horizon navigation target rather than on the most salient local cue. In the bottom panel, the ego approached a nighttime intersection while the traffic light was still red at the current frame; the underlying log confirmed that the light would turn green before the ego reached the stop line, so the ground-truth trajectory continued through the intersection without braking. The baseline treated the present red signal as an immediate stop and truncated its prediction well before the stop line. X-Foresight anticipated the signal transition through its predictive rollout and produced a continuous trajectory matching the ground truth.

![](images/ca1ba66698b4b94e6b353cfe4e235a71f2a9a3b7d054c4882d7413ebce298908.jpg)  
Figure 7: Qualitative comparisons. The baseline failed on events lying ahead in space (top, a far-exit instruction in a multi-exit roundabout) or in time (bottom, a red light that turned green before arrival); X-Foresight tracked the ground truth in both.

Table 4: Training throughput comparison over long-horizon sequences.

<table><tr><td>Attention implementation</td><td>Per-step time (s) ↓</td><td>Speedup ↑</td></tr><tr><td>FlashAttention-2 [4]</td><td>24.50</td><td>1.00×</td></tr><tr><td>BSA w/ mask (ours)</td><td>15.40</td><td>1.59×</td></tr></table>

## 4.1.2 Acceleration

We optimized the Block Sparse Attention (BSA) kernel for our target hardware and integrated it as a drop-in replacement for FlashAttention-2 (FA2). Table 4 reported the wall-clock time per training step before and after this change, with the FA2 baseline using a standard causal mask. Per-step time was measured under identical model and training configurations, with the attention implementation and associated mask being the only changes. Pairing BSA with our tailored sparsity pattern reduced per-step time from 24.50 s to 15.40 s, a 1.59× speedup, confirming that attention was the major bottleneck and that the savings from our sparsity design translated directly into end-to-end training throughput.

## 4.1.3 Camera Latent Decoder

In long-horizon rollouts, future observations were generated recursively from the model’s own decoded latent, making the structure and consistency of the latent space critical to stable future prediction and downstream planning performance. However, our vision renderer was diffusion-based, where rendered images were influenced by strong generative priors and sampling stochasticity rather than being deterministic projections of the latent representation. Consequently, rendered outputs could not reliably reflect the underlying latent space.

To address this issue, we introduced a lightweight Camera Latent Decoder to visualize the predicted latent space, following prior works [7] [16]. Unlike diffusion-based rendering, our latent decoder provided a more faithful and stable projection of the latent representation, enabling direct inspection of spatial structure, temporal consistency, and multi-view alignment. This capability was essential for diagnosing latent drift, understanding error accumulation in long-horizon rollouts, and ensuring that the learned latent space satisfied the structural assumptions required by the downstream vision renderer.

To enable direct inspection of the predicted latent space, we trained a lightweight latent decoder that reconstructed image sequences from the model’s spatio-temporal latent representations. Given latent representations extracted from the world model, the decoder progressively upsampled along both temporal and spatial dimensions to reconstruct high-resolution image frames from the compressed latent representation. The decoder adopted a lightweight convolutional architecture with progressive spatio-temporal upsampling based on causal 3D ResNet blocks. A lightweight spatial self-attention module was introduced at the lowest-resolution stage to enable global feature interaction, allowing efficient reconstruction of high-resolution image sequences from the predicted latent representations.

## 4.2 Vision Renderer

## 4.2.1 Training Process

We initialized the Vision Renderer from X-World weights, retaining all transferable parameters of the DiT backbone and the 3D causal VAE. The newly introduced camera-token cross-attention branch was randomly initialized, while the action, dynamic-agent, static-element, and text-conditioning branches inherited from X-World were discarded in stage III. Training data consisted of synchronized multi-view driving clips paired with LDM-produced camera tokens for the same intervals, so that the renderer learned to denoise future frames under the exact token distribution it would consume at inference.

To bridge the gap between X-World’s pre-training regime and the closed-loop deployment setting of X-Foresight, we followed a decoupled-then-aligned schedule spanning the two renderer training stages: in stage II the renderer was adapted independently of the LDM under ground-truth action conditioning, while in stage III the LDM was frozen and only the renderer was fine-tuned to consume the LDM’s predicted camera tokens.

Stage II: Renderer Pretraining. We first fine-tuned the diffusion renderer on the multi-view driving corpus to adapt it from X-World’s 12 Hz pre-training cadence to the 4 Hz cadence emitted by X-Foresight’s LDM. We optimized the rectified-flow objective on subsampled video streams with the VAE temporal stride adjusted accordingly. The LDM was frozen throughout this stage. The renderer retained X-World’s original action-conditioning branch (consuming ground-truth ego actions rather than LDM-predicted camera tokens); the camera-token cross-attention branch that would replace it in stage III was not yet active. We used the Muon optimizer with a constant learning rate of $\bar { 8 } \times 1 0 ^ { - 5 }$ batch size 1 per device, and 128 GPUs.

Stage III: Renderer Alignment. In the third stage, we spliced the renderer onto the LDM, swapped the conditioning source, and froze the LDM so that only the renderer received gradients. The actionconditioning branch used during stage II, together with X-World’s dynamic-agent, static-element, and text-conditioning branches, were all removed; the renderer was exposed exclusively to (i) multi-view history latents and (ii) LDM-produced camera tokens through the newly activated cross-attention branch, committing the DiT to the LDM’s latent imagination as its sole control signal. The renderer was then finetuned under the rectified-flow velocity-matching objective $\mathcal { L } _ { \mathrm { v e l o c i t y } }$ in Eq. (8), now conditioned on the LDM’s predicted camera tokens. To fit the model into device memory, we applied activation checkpointing on every DiT block and sharded the renderer’s parameters, gradients, and optimizer states with FSDP; the LDM was kept in inference mode and contributed only forward activations. We optimized with Muon under a one-cycle cosine learning-rate schedule, tuned for stable convergence of the diffusion velocity field. From stage III onward we monitored the controllability of the renderer by periodically comparing its samples against the Camera Latent Decoder’s pixel decode of the same camera tokens, as described in Section 4.1.3; the renderer’s samples consistently aligned with the Camera Latent Decoder’s pixel decode of the same camera tokens, confirming that the rendered frames faithfully tracked the LDM’s camera-token condition.

## 4.2.2 Results

Fig 8 visualized a 6-second rollout from the Vision Renderer across all seven surround-view cameras (left to right: left rear, left front, front narrow, front fisheye, rear, right front, right rear). The vertical axis was time, with the ground-truth observation at $t = 0 \ : s$ at the top and 6 s of predicted future extending downward. Each AR step emitted one second of camera tokens at 4 Hz, which the renderer decoded into four frames per camera; six consecutive steps yielded the full horizon shown.

![](images/f780bdf70997b551977d9033531075aa5492c61a8918e2f7103cfb7d26208e87.jpg)  
Figure 8: Visualization of the Vision Renderer conditioned on camera tokens. The horizontal axis spans the seven surround-view cameras (left rear, left front, front narrow, front fisheye, rear, right front, right rear), and the vertical axis spans time from t = 0 s (ground truth) to t = 6 s of predicted future. The blue overlay on the front-narrow and front-fisheye views shows the trajectory predicted by the LDM from action tokens, which the renderer never sees.

To probe whether camera tokens encoded the ego-vehicle’s intended motion, we overlaid on the front-narrow and front-fisheye views the trajectory predicted by the LDM from its action tokens (blue line). The renderer was not conditioned on these action tokens (Section 3), so the trajectory served as an independent reference: it revealed the LDM’s intent but could not influence the rendered scene. Despite this separation, the trajectory remained tightly aligned with the synthesized frames across the full 6-second horizon, consistently tracing a path geometrically consistent with the lane configuration, surrounding agents, and ego pose. This indicated that the camera tokens already encoded the ego trajectory and scene dynamics, and that the renderer was genuinely controlled by the LDM’s latent imagination rather than hallucinating from history alone.

Table 5 reported quantitative image-quality metrics for both the Camera Latent Decoder and the Vision Renderer, evaluated at a 1-second horizon (single AR step) and at the 6-second horizon. FID [9] was computed over all seven surround-view cameras at 4 Hz (four frames per camera per second), with ground-truth references sampled from 200k random clips in the test dataset. FVD [17] was computed per camera over its temporal sequence and then averaged across the seven cameras. The decoder reflected reconstruction quality of camera tokens in isolation, while the renderer measured end-to-end generation quality from the autoregressive rollout.

Table 5: Image-quality metrics for the Camera Latent Decoder and Vision Renderer at 1 s and 6 s horizons.

<table><tr><td rowspan="2">Method</td><td colspan="2">FID ↓</td><td colspan="2">FVD ↓</td></tr><tr><td>1 s</td><td>6 s</td><td>1 s</td><td>6 s</td></tr><tr><td>Camera Latent Decoder</td><td>10.97</td><td>11.82</td><td>135.56</td><td>158.39</td></tr><tr><td>Vision Renderer</td><td>1.51</td><td>2.84</td><td>11.28</td><td>29.52</td></tr></table>

Taken together, Fig 8 and Table 5 demonstrated that the Vision Renderer produced high-fidelity multi-view imagery suitable for closed-loop inference within X-Foresight. Quantitatively, the renderer reached FID/FVD of 1.51/11.28 at a 1 s horizon and 2.84/29.52 at 6 s, reducing FID/FVD relative to the Camera Latent Decoder by 10.97/135.56 at 1 s and 11.82/158.39 at 6 s. The modest degradation from 1 s to 6 s—an absolute increase of 1.33 FID points and 18.24 FVD points—indicated that the autoregressive rollout accumulated only limited drift across the full prediction window, an essential property for closed-loop deployment where each step’s output conditions the next. Qualitatively, the LDM-predicted trajectory in Fig 8 remained geometrically consistent with the rendered scene throughout the 6-second horizon despite the renderer never receiving action tokens, confirming that the LDM’s camera tokens already encoded the ego trajectory and surrounding scene dynamics with sufficient fidelity for the DiT to faithfully visualize the LDM’s latent imagination. Together, these results established that the Vision Renderer met the fidelity, controllability, and temporal-stability requirements needed to serve as the pixel-space front-end of X-Foresight’s closed-loop inference loop.

## 5 Conclusion

We presented X-Foresight, a predictive world model that equipped VLA models with an internalized understanding of the physical world by forecasting high-fidelity multi-view camera observations while maintaining real-time action control. The proposed chunk-wise auto-regressive strategy addressed two key challenges in video-based world modeling: degeneration in future rollouts and the temporal mismatch between instantaneous dynamics and long-horizon world causality. Furthermore, we introduced a curriculum learning schedule that progressively extended prediction horizons, stabilizing training and improving control robustness. We also employed temporal importance sampling to concentrate supervision on safety-critical chunks, thereby enhancing the model’s ability to learn decision-relevant dynamics and better understand world causality. Finally, photorealistic synthesis was delegated to a diffusion-based multi-view renderer, which provided pixel-level supervision and improved rollout stability.

Our work extended world model research to industrial-scale data and established a foundation for several future research directions. First, incorporating additional supervision during closed-loop rollout might further improve long-tail performance in autonomous systems. Second, integrating richer modalities, such as 3D geometric supervision, might enhance the model’s physical world knowledge. Third, more advanced training strategies, such as coarse-to-fine world reconstruction, might help unlock the model’s full potential for comprehensive world understanding.

## Acknowledgments

We sincerely thank our predictive world model team for their passionate exploration, valuable discussions, and hard work throughout this project. We also thank our sibling GWM team for their excellent work on X-World [22], which we build upon for pretraining our diffusion-based renderer. In particular, we are grateful to our advisors, Yu Zhang and Xianming Liu, for their insightful guidance on world model design, which was instrumental to the development of this work. We also thank XPeng for providing the platform, tools, and support that made this work possible.

We further acknowledge the broader research community, whose progress in world modeling, visionlanguage-action models, and generative modeling has inspired and informed our contribution to predictive world modeling.

## Contributors

Advisors: Yu Zhang, Xianming Liu

Project Lead: Zhuangzhuang Ding, Pengkun Zheng

Contributors: Baolu Li<sup>\*</sup>, Jingyu Qian<sup>\*</sup>, Rui Guo<sup>\*</sup>, Yilun Chen<sup>\*</sup>, Hanpeng Liu, Yuan Lin, Junhong Zhou, Ruixin Liu, Liu Yang, Yutong Zheng, Zhenli Zhang, Sean Li, Chaoda Zheng, Boyang Wang

Technical Program Manager: Tenglong (Victor) Gu

Core contribution. The first four authors are listed in alphabetical order.

## References

[1] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15619–15629, 2023.

[2] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Leo Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.

[3] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In International Conference on Machine Learning (ICML), 2024.

[4] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, volume 2024, pages 35549–35562, 2024.

[5] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929, 2020.

[6] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv:2303.03378, 2023.

[7] Yuan Gao, Chen Chen, Tianrong Chen, and Jiatao Gu. One layer is enough: Adapting pretrained visual encoders for image generation. ArXiv, 2512.07829, 2025.

[8] Junxian Guo, Haotian Tang, Shang Yang, Zhekai Zhang, Zhijian Liu, and Song Han. Block Sparse Attention. https://github.com/mit-han-lab/Block-Sparse-Attention, 2024.

[9] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems (NeurIPS), 30, 2017.

[10] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. Advances in Neural Information Processing Systems (NeurIPS), 38:167283–167308, 2026.

[11] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv:2406.09246, 2024.

[12] World Labs. Marble: World Labs Spatial Intelligence. https://www.worldlabs.ai/, 2026.

[13] Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: O(n log n) sparse attention with energy decay for long video generation. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[14] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

[15] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, 2023.

[16] Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Jiwen Lu. SVG: Latent diffusion model without variational autoencoder. arXiv:2510.15301, 2025.

[17] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv:1812.01717, 2018.

[18] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv:2503.20314, 2025.

[19] XPeng Inc. XPeng VLA 2.0. https://www.xpeng.com, 2026.

[20] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22963–22974, 2025.

[21] Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan. Helios: Real real-time long video generation model. arXiv:2603.04379, 2026.

[22] Chaoda Zheng, Sean Li, Jinhao Deng, Zhennan Wang, Shijia Chen, Liqiang Xiao, Ziheng Chi, Hongbin Lin, Kangjie Chen, Boyang Wang, et al. X-world: Controllable ego-centric multi-camera world models for scalable end-to-end driving. arXiv:2603.19979, 2026.

[23] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), pages 2165–2183. PMLR, 2023.