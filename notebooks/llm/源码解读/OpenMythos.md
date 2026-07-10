<div class="om-demo-wrapper ds-container" id="om-app">
    <style>
        /* ==========================================================
           1. 自定义属性 (CSS Variables)
           ========================================================== */
        .om-demo-wrapper {
            --ds-bg-color: #f8fafc;
            --ds-panel-bg: #ffffff;
            --ds-text-primary: #334155;
            --ds-text-secondary: #64748b;
            --ds-border-color: #cbd5e1;
            --ds-highlight: #3b82f6;
            --ds-success: #16a34a;
            --ds-panel-shadow: 0 4px 6px rgba(0,0,0,0.05);
            
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, system-ui, sans-serif;
            width: 100%;
            max-width: calc(100vw - 35rem); /* 适配 MkDocs 左右侧栏 */
            margin: 0 auto;
            padding: 24px 20px 120px; 
            border-radius: 12px;
            line-height: 1.6;
            background: var(--ds-bg-color);
            position: relative;
            box-sizing: border-box;
            color: var(--ds-text-primary);
        }
        
        @media (max-width: 60rem) {
            .om-demo-wrapper { max-width: 100vw; }
        }

        /* ==========================================================
           2. Partition 色系定义
           ========================================================== */
        /* P0: 天蓝 (数据张量) */
        .om-demo-wrapper .ds-c0 { background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }
        /* P1: 翠绿 (Transformer核心计算) */
        .om-demo-wrapper .ds-c1 { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
        /* P2: 暖黄 (位置/深度与微调参数) */
        .om-demo-wrapper .ds-c2 { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
        /* P3: 浅紫 (控制流/概率输出) */
        .om-demo-wrapper .ds-c3 { background: #f3e8ff; color: #6b21a8; border: 1px solid #d8b4fe; }

        /* ==========================================================
           3. 核心布局与组件
           ========================================================== */
        .om-demo-wrapper .ds-header {
            text-align: center;
            margin-bottom: 24px;
        }
        .om-demo-wrapper .ds-header-title {
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 8px;
        }
        .om-demo-wrapper .ds-header-desc {
            font-size: 14px;
            color: var(--ds-text-secondary);
        }

        /* 信息面板 */
        .om-demo-wrapper .ds-info-panel {
            background: var(--ds-panel-bg);
            padding: 16px 24px;
            border-radius: 12px;
            border: 1px solid var(--ds-border-color);
            border-left: 5px solid var(--ds-highlight);
            box-shadow: var(--ds-panel-shadow);
            min-height: 100px;
            margin-bottom: 24px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .om-demo-wrapper .ds-info-title {
            font-size: 16px;
            font-weight: 700;
            color: #1e40af;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .om-demo-wrapper .ds-info-desc {
            font-size: 13px;
            color: #475569;
        }

        /* 可视化舞台 (Stage) */
        .om-demo-wrapper .om-stage {
            background: #ffffff;
            border: 1px solid var(--ds-border-color);
            border-radius: 12px;
            padding: 30px 20px;
            min-height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
            overflow: hidden;
            position: relative;
        }

        .om-demo-wrapper .om-stage-content {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            width: 100%;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            animation: ds-fadeIn 0.5s ease-out;
        }

        /* 张量与操作块样式 */
        .om-demo-wrapper .om-tensor,
        .om-demo-wrapper .om-op {
            padding: 12px 16px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            min-width: 110px;
            position: relative;
        }
        .om-demo-wrapper .om-tensor {
            border-radius: 6px;
        }
        .om-demo-wrapper .om-op {
            border-radius: 20px; /* 操作采用大圆角区别于张量 */
            border-width: 2px;
            font-weight: 700;
        }
        .om-demo-wrapper .om-block-title {
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .om-demo-wrapper .om-block-dim {
            font-size: 11px;
            font-weight: 600;
            opacity: 0.85;
            font-family: monospace;
        }
        .om-demo-wrapper .om-arrow {
            color: #94a3b8;
            font-size: 20px;
            font-weight: 800;
            display: flex;
            align-items: center;
        }
        .om-demo-wrapper .om-col {
            display: flex;
            flex-direction: column;
            gap: 12px;
            align-items: center;
        }
        .om-demo-wrapper .om-row {
            display: flex;
            flex-direction: row;
            gap: 12px;
            align-items: center;
        }

        /* 动画与状态 */
        @keyframes ds-fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes ds-soft-pulse {
            0%   { box-shadow: 0 0 0 0 rgba(59,130,246,0.4); }
            70%  { box-shadow: 0 0 0 10px rgba(59,130,246,0); }
            100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); }
        }
        
        .om-demo-wrapper .ds-receiving {
            transform: scale(1.05);
            box-shadow: 0 0 12px rgba(59,130,246,0.6);
            border-color: var(--ds-highlight);
            animation: ds-soft-pulse 2s infinite;
        }
        .om-demo-wrapper .ds-reduced {
            transform: scale(1.05);
            box-shadow: 0 0 8px rgba(22,163,74,0.5);
            border-color: var(--ds-success);
        }

        /* ==========================================================
           4. Dock 底部导航
           ========================================================== */
        .om-demo-wrapper .ds-dock-container {
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
        }
        .om-demo-wrapper .ds-dock {
            display: flex;
            align-items: center;
            gap: 16px;
            background: rgba(255, 255, 255, 0.94);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 999px;
            border: 1px solid rgba(203, 213, 225, 0.95);
            box-shadow: 0 12px 28px -10px rgba(15,23,42,0.22),
                        0 4px 10px -6px rgba(15,23,42,0.16);
            padding: 8px 16px;
        }
        .om-demo-wrapper .ds-dot {
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            color: #64748b;
            background: transparent;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: none;
        }
        .om-demo-wrapper .ds-dot:hover {
            background: #f1f5f9;
            color: #334155;
        }
        .om-demo-wrapper .ds-dot.ds-active {
            background: var(--ds-highlight);
            color: white;
            transform: scale(1.1);
            box-shadow: 0 4px 8px rgba(59,130,246,0.3);
        }
        .om-demo-wrapper .ds-nav-btn {
            background: transparent;
            border: none;
            color: #475569;
            font-size: 16px;
            cursor: pointer;
            padding: 4px 8px;
            transition: color 0.2s;
            display: flex;
            align-items: center;
        }
        .om-demo-wrapper .ds-nav-btn:hover { color: var(--ds-highlight); }
        .om-demo-wrapper .ds-nav-btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
            color: #94a3b8;
        }
    </style>

    <!-- Header -->
    <div class="ds-header">
        <div class="ds-header-title">OpenMythos 架构前向传播图解</div>
        <div class="ds-header-desc">从输入 Token 到 Logits 的 Recurrent-Depth Transformer (RDT) 矩阵变换全景</div>
    </div>

    <!-- Info Panel -->
    <div class="ds-info-panel" id="om-info-panel">
        <div class="ds-info-title" id="om-info-title">加载中...</div>
        <div class="ds-info-desc" id="om-info-desc">请稍候...</div>
    </div>

    <!-- Visual Stage -->
    <div class="om-stage" id="om-stage">
        <!-- JS 动态注入节点 -->
    </div>

    <!-- Dock Navigation -->
    <div class="ds-dock-container">
        <div class="ds-dock">
            <button class="ds-nav-btn" id="om-btn-prev">◀</button>
            <button class="ds-dot" data-step="0">0</button>
            <button class="ds-dot" data-step="1">1</button>
            <button class="ds-dot" data-step="2">2</button>
            <button class="ds-dot" data-step="3">3</button>
            <button class="ds-dot" data-step="4">4</button>
            <button class="ds-dot" data-step="5">5</button>
            <button class="ds-nav-btn" id="om-btn-next">▶</button>
        </div>
    </div>

    <script>
        (function() {
            const stepsData = [
                {
                    title: "Step 0: 词表嵌入 (Embedding)",
                    desc: "将离散的 Token ID 映射为连续的稠密向量。输出矩阵 `x` 的维度为 <code>[Batch, Seq_Len, Dim]</code>。此层的权重后续将与最后的分类头 (Head) 共享 (Weight Tying)。",
                    html: `
                        <div class="om-stage-content">
                            <div class="om-tensor ds-c0">
                                <div class="om-block-title">input_ids</div>
                                <div class="om-block-dim">[B, T]</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-op ds-c1">
                                <div class="om-block-title">Embedding</div>
                                <div class="om-block-dim">Vocab_Size → 2048</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-tensor ds-c0 ds-receiving">
                                <div class="om-block-title">x (隐状态)</div>
                                <div class="om-block-dim">[B, T, 2048]</div>
                            </div>
                        </div>
                    `
                },
                {
                    title: "Step 1: 前奏层 (Prelude Blocks)",
                    desc: "初始状态 `x` 与预计算的复数旋转位置编码 `freqs_cis` 进入标准的 Transformer 层（不含 MoE），提取基础语义。提取后的特征被冻结为 `e` (Encoded Input)，用于后续深层循环的残差注入。",
                    html: `
                        <div class="om-stage-content">
                            <div class="om-col">
                                <div class="om-tensor ds-c0">
                                    <div class="om-block-title">x</div>
                                    <div class="om-block-dim">[B, T, 2048]</div>
                                </div>
                                <div class="om-tensor ds-c2" style="padding:6px; min-width:80px;">
                                    <div class="om-block-title">RoPE 频率</div>
                                    <div class="om-block-dim">freqs_cis</div>
                                </div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-op ds-c1">
                                <div class="om-block-title">Prelude Layers</div>
                                <div class="om-block-dim">Dense FFN</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-tensor ds-c0 ds-receiving">
                                <div class="om-block-title">e (冻结特征)</div>
                                <div class="om-block-dim">[B, T, 2048]</div>
                            </div>
                        </div>
                    `
                },
                {
                    title: "Step 2: 循环体注入 (Loop Injection)",
                    desc: "进入 <code>RecurrentBlock</code>。为了区分当前是第几次循环，模型对隐状态 `h_t` 注入与深度 <code>loop_t</code> 相关的正弦位置编码。为了防止深层退化，同时拼接上一步的冻结特征 `e`。",
                    html: `
                        <div class="om-stage-content">
                            <div class="om-col">
                                <div class="om-tensor ds-c0">
                                    <div class="om-block-title">h_t</div>
                                    <div class="om-block-dim">[B, T, 2048]</div>
                                </div>
                                <div class="om-tensor ds-c2" style="padding:6px; min-width:80px;">
                                    <div class="om-block-title">Loop Index</div>
                                    <div class="om-block-dim">t = 0...N</div>
                                </div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-op ds-c1" style="border-radius:6px;">
                                <div class="om-block-title">RoPE (Depth) + Add e</div>
                                <div class="om-block-dim">Norm(h_loop + e)</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-tensor ds-c0 ds-receiving">
                                <div class="om-block-title">Combined</div>
                                <div class="om-block-dim">[B, T, 2048]</div>
                            </div>
                        </div>
                    `
                },
                {
                    title: "Step 3: 核心路由计算 (Attention + MoE + LoRA)",
                    desc: "特征经过 MLA(极简KV缓存) 或 GQA，再进入细粒度的 MoEFFN (混合专家) 进行动态路由计算。由于权重是跨层共享的，最后加上与深度 $t$ 相关的 <code>LoRAAdapter</code> 来微调该层的行为偏置。",
                    html: `
                        <div class="om-stage-content">
                            <div class="om-tensor ds-c0">
                                <div class="om-block-title">Combined</div>
                                <div class="om-block-dim">[B, T, 2048]</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-col">
                                <div class="om-op ds-c1">
                                    <div class="om-block-title">MLA / GQA</div>
                                    <div class="om-block-dim">Attention</div>
                                </div>
                                <div class="om-op ds-c1">
                                    <div class="om-block-title">MoEFFN</div>
                                    <div class="om-block-dim">Top-K Routed</div>
                                </div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-col">
                                <div class="om-tensor ds-c0 ds-receiving">
                                    <div class="om-block-title">trans_out</div>
                                    <div class="om-block-dim">[B, T, 2048]</div>
                                </div>
                                <div class="om-tensor ds-c2" style="padding:6px; min-width:80px;">
                                    <div class="om-block-title">+ LoRA(t)</div>
                                    <div class="om-block-dim">Depth Shift</div>
                                </div>
                            </div>
                        </div>
                    `
                },
                {
                    title: "Step 4: LTI 稳定更新与 ACT 停机",
                    desc: "使用 <code>LTIInjection</code> 进行隐状态更新 <code>$h_{t+1} = A·h_t + B·e + out$</code>，严格保证谱半径 $ρ(A)<1$ 防止爆炸。同时计算 <code>ACTHalting</code> 概率 $p$，一旦累加概率超过阈值，该 Token 终止思考。",
                    html: `
                        <div class="om-stage-content" style="flex-wrap: wrap;">
                            <div class="om-col">
                                <div class="om-tensor ds-c0" style="padding:6px; min-width:60px;">h_t</div>
                                <div class="om-tensor ds-c0" style="padding:6px; min-width:60px;">e</div>
                                <div class="om-tensor ds-c0" style="padding:6px; min-width:60px;">trans_out</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-op ds-c1">
                                <div class="om-block-title">LTI Stable Update</div>
                                <div class="om-block-dim">A·h + B·e + out</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-tensor ds-c0 ds-reduced">
                                <div class="om-block-title">h_{t+1}</div>
                                <div class="om-block-dim">[B, T, 2048]</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-col">
                                <div class="om-op ds-c3">
                                    <div class="om-block-title">ACT Halting</div>
                                    <div class="om-block-dim">p = Sigmoid(W·h)</div>
                                </div>
                                <div class="om-tensor ds-c3" style="padding:6px; min-width:80px;">
                                    <div class="om-block-title">∑p ≥ 0.99 ?</div>
                                    <div class="om-block-dim">Weight Remainder</div>
                                </div>
                            </div>
                        </div>
                    `
                },
                {
                    title: "Step 5: 尾声与输出分类 (Coda & Head)",
                    desc: "累加所有循环产生的隐状态得到最终的 $h_{final}$，送入 <code>Coda Layers</code>（常规Transformer）进行润色，最后使用与 Embedding 共享权重的线性分类头映射回词表维度产生概率分布。",
                    html: `
                        <div class="om-stage-content">
                            <div class="om-tensor ds-c0">
                                <div class="om-block-title">h_final</div>
                                <div class="om-block-dim">[B, T, 2048]</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-op ds-c1">
                                <div class="om-block-title">Coda Layers</div>
                                <div class="om-block-dim">Dense FFN</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-op ds-c2" style="border-radius: 6px;">
                                <div class="om-block-title">Linear Head</div>
                                <div class="om-block-dim">Tied Weight</div>
                            </div>
                            <div class="om-arrow">➔</div>
                            <div class="om-tensor ds-c3 ds-reduced">
                                <div class="om-block-title">Logits</div>
                                <div class="om-block-dim">[B, T, 32000]</div>
                            </div>
                        </div>
                    `
                }
            ];

            let currentStep = 0;
            const titleEl = document.getElementById('om-info-title');
            const descEl = document.getElementById('om-info-desc');
            const stageEl = document.getElementById('om-stage');
            const dots = document.querySelectorAll('.om-demo-wrapper .ds-dot');
            const btnPrev = document.getElementById('om-btn-prev');
            const btnNext = document.getElementById('om-btn-next');

            function renderStep(index) {
                // Update text
                titleEl.innerHTML = `<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--ds-highlight); margin-right:8px;"></span> ${stepsData[index].title}`;
                descEl.innerHTML = stepsData[index].desc;
                
                // Animate stage
                const oldContent = stageEl.querySelector('.om-stage-content');
                if (oldContent) {
                    oldContent.style.opacity = '0';
                    oldContent.style.transform = 'translateY(-10px)';
                    setTimeout(() => {
                        stageEl.innerHTML = stepsData[index].html;
                    }, 200);
                } else {
                    stageEl.innerHTML = stepsData[index].html;
                }

                // Update Dock
                dots.forEach(dot => {
                    if (parseInt(dot.dataset.step) === index) {
                        dot.classList.add('ds-active');
                    } else {
                        dot.classList.remove('ds-active');
                    }
                });

                // Update Buttons
                btnPrev.disabled = index === 0;
                btnNext.disabled = index === stepsData.length - 1;
            }

            // Event Listeners
            btnPrev.addEventListener('click', () => {
                if (currentStep > 0) {
                    currentStep--;
                    renderStep(currentStep);
                }
            });

            btnNext.addEventListener('click', () => {
                if (currentStep < stepsData.length - 1) {
                    currentStep++;
                    renderStep(currentStep);
                }
            });

            dots.forEach(dot => {
                dot.addEventListener('click', (e) => {
                    currentStep = parseInt(e.target.dataset.step);
                    renderStep(currentStep);
                });
            });

            // Init
            renderStep(0);
        })();
    </script>
</div>