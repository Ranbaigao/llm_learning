<div class="rb-demo-wrapper" id="rb-demo-root">
    <style>
        /* =========================================
           1. 绝对沙盒化 CSS，全部以 rb- 开头
           2. 视觉色彩规范：Deepspeed Zero2 现代清新浅色系
           ========================================= */
        .rb-demo-wrapper {
            --rb-bg-color: #f8fafc;
            --rb-panel-bg: #ffffff;
            --rb-text-primary: #334155;
            --rb-text-strong: #0f172a;
            --rb-text-secondary: #64748b;
            --rb-border-color: #cbd5e1;
            --rb-highlight: #3b82f6;       /* blue-500 */
            --rb-highlight-dark: #1d4ed8;  /* blue-700 */
            --rb-success: #16a34a;         /* green-600 */
            --rb-danger: #ef4444;          /* red-500 */
            --rb-panel-shadow: 0 4px 6px rgba(0,0,0,0.05);

            /* 4色切片配色 */
            --rb-c0: #e0f2fe; --rb-c0-text: #0369a1; --rb-c0-border: #7dd3fc;
            --rb-c1: #dcfce7; --rb-c1-text: #15803d; --rb-c1-border: #86efac;
            --rb-c2: #fef9c3; --rb-c2-text: #a16207; --rb-c2-border: #fde047;
            --rb-c3: #f3e8ff; --rb-c3-text: #6b21a8; --rb-c3-border: #d8b4fe;

            font-family: "Segoe UI", Tahoma, Geneva, Verdana, system-ui, sans-serif;
            background-color: var(--rb-bg-color);
            color: var(--rb-text-primary);
            width: 100%;
            margin: 0 auto;
            padding: 18px 18px 96px;
            box-sizing: border-box;
            line-height: 1.6;
            border-radius: 12px;
            position: relative;
            min-height: 700px;
        }

        .rb-demo-wrapper * { box-sizing: border-box; }

        /* --- 标题区域 --- */
        .rb-demo-wrapper .rb-header-title { text-align: center; font-size: 22px; font-weight: 800; color: var(--rb-text-strong); margin-bottom: 5px; }
        .rb-demo-wrapper .rb-header-desc { text-align: center; color: var(--rb-text-secondary); font-size: 13px; margin-bottom: 20px; }

        /* --- 分页内容容器 --- */
        .rb-demo-wrapper .rb-step-content {
            display: none;
            animation: rb-fadeIn 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }
        .rb-demo-wrapper .rb-step-content.rb-active { display: block; }

        /* --- 核心组件：信息面板 --- */
        .rb-demo-wrapper .rb-info-panel {
            background: var(--rb-panel-bg);
            padding: 15px 20px;
            border-radius: 12px;
            border: 1px solid var(--rb-border-color);
            border-left: 5px solid var(--rb-highlight);
            box-shadow: var(--rb-panel-shadow);
            margin-bottom: 25px;
            position: relative;
            min-height: 95px;
            display: flex;
            flex-direction: column;
        }
        .rb-demo-wrapper .rb-info-title { font-size: 16px; font-weight: 700; color: var(--rb-highlight-dark); margin-bottom: 6px; padding-right: 90px; }
        .rb-demo-wrapper .rb-info-desc { font-size: 13px; color: var(--rb-text-secondary); margin-bottom: 10px; }
        
        .rb-demo-wrapper .rb-text-box {
            font-size: 13px; color: var(--rb-text-secondary); margin-bottom: 15px;
            background: #f1f5f9; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;
        }

        /* --- 实时动画解说条 (Action Log) --- */
        .rb-demo-wrapper .rb-action-log {
            background: #eff6ff; color: #1d4ed8; padding: 8px 12px; border-radius: 6px;
            font-size: 13px; font-weight: 700; border: 1px dashed #93c5fd;
            opacity: 0; transition: opacity 0.3s ease; display: flex; align-items: center; gap: 8px;
        }
        .rb-demo-wrapper .rb-action-log.rb-active { opacity: 1; }

        /* --- 重播按钮 --- */
        .rb-demo-wrapper .rb-replay-btn {
            position: absolute; top: 15px; right: 15px; display: flex; align-items: center; gap: 4px;
            padding: 5px 10px; font-size: 12px; font-weight: 600; color: var(--rb-highlight);
            background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; cursor: pointer; transition: all 0.2s;
        }
        .rb-demo-wrapper .rb-replay-btn:hover { background: #e2e8f0; transform: translateY(-1px); }
        .rb-demo-wrapper .rb-replay-btn svg { width: 13px; height: 13px; }

        /* --- 流水线动画区域 (Pipeline) --- */
        .rb-demo-wrapper .rb-pipeline-container {
            display: flex; flex-direction: column; gap: 10px; align-items: center; margin: 15px 0;
            background: var(--rb-bg-color); padding: 20px; border-radius: 10px; border: 1px solid var(--rb-border-color);
        }
        .rb-demo-wrapper .rb-pipe-row {
            display: flex; align-items: center; justify-content: center; gap: 6px; width: 100%; flex-wrap: wrap;
        }
        .rb-demo-wrapper .rb-module {
            background: #ffffff; border: 2px solid var(--rb-border-color); padding: 10px 12px;
            border-radius: 8px; font-size: 12px; font-weight: 700; color: var(--rb-text-primary);
            text-align: center; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 4px rgba(0,0,0,0.02); min-width: 100px; position: relative;
        }
        .rb-demo-wrapper .rb-arrow { 
            font-size: 18px; color: #cbd5e1; font-weight: 900; transition: all 0.3s; 
        }
        .rb-demo-wrapper .rb-arrow.rb-flowing {
            color: var(--rb-highlight); text-shadow: 0 0 8px rgba(59,130,246,0.6); transform: scale(1.2);
        }
        
        /* 模块激活态光晕 (Deepspeed 风格) */
        .rb-demo-wrapper .rb-module.rb-receiving {
            transform: scale(1.15); z-index: 5;
            box-shadow: 0 0 12px rgba(59, 130, 246, 0.6);
            border: 2px solid var(--rb-highlight);
            color: var(--rb-highlight-dark);
        }
        .rb-demo-wrapper .rb-dimmed { opacity: 0.4; filter: grayscale(50%); }

        /* --- 静态步骤卡片 (不随动画改变内容) --- */
        .rb-demo-wrapper .rb-static-desc-grid {
            display: flex; flex-direction: column; gap: 8px; margin-top: 15px;
        }
        .rb-demo-wrapper .rb-step-card {
            background: #f8fafc; border: 1px solid var(--rb-border-color); padding: 12px 16px;
            border-radius: 8px; font-size: 13px; color: var(--rb-text-secondary);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex; flex-direction: column;
        }
        .rb-demo-wrapper .rb-step-card strong {
            color: var(--rb-text-strong); margin-bottom: 4px; font-size: 14px;
        }
        /* 静态卡片的联动高亮特效 */
        .rb-demo-wrapper .rb-step-card.rb-highlight {
            background: #ffffff;
            border-color: var(--rb-highlight);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
            color: var(--rb-text-primary);
            transform: scale(1.02);
            border-left: 4px solid var(--rb-highlight);
        }
        .rb-demo-wrapper .rb-step-card.rb-highlight strong { color: var(--rb-highlight-dark); }


        /* --- ACT 模拟器样式 --- */
        .rb-demo-wrapper .rb-act-grid { display: grid; gap: 10px; margin-top: 15px; }
        .rb-demo-wrapper .rb-act-row {
            display: flex; align-items: center; gap: 10px; background: #f8fafc;
            padding: 10px; border-radius: 8px; border: 1px solid var(--rb-border-color);
        }
        .rb-demo-wrapper .rb-token-label {
            font-size: 12px; font-weight: 700; width: 140px; display: flex; align-items: center; gap: 6px;
        }
        .rb-demo-wrapper .rb-chunk {
            padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 800; border: 1px solid transparent;
        }
        .rb-demo-wrapper .rb-progress-track {
            flex: 1; height: 16px; background: #e2e8f0; border-radius: 999px; position: relative; overflow: hidden;
        }
        .rb-demo-wrapper .rb-progress-fill {
            height: 100%; width: 0%; background: var(--rb-highlight); transition: all 0.4s ease;
        }
        .rb-demo-wrapper .rb-progress-fill.rb-halted {
            background-color: var(--rb-success);
            background-image: linear-gradient(135deg, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0) 100%);
        }
        .rb-demo-wrapper .rb-limit-line { position: absolute; left: 99%; top: 0; bottom: 0; width: 2px; background: var(--rb-danger); z-index: 10; }
        .rb-demo-wrapper .rb-prob-val { font-size: 11px; font-weight: 700; width: 40px; text-align: right; font-family: monospace; }

        /* --- Dock 风格底部导航 --- */
        .rb-demo-wrapper .rb-dock-container {
            position: fixed; left: 50%; bottom: calc(18px + env(safe-area-inset-bottom));
            z-index: 10000; width: min(420px, calc(100vw - 32px)); display: flex; justify-content: center;
            transform: translateX(-50%); pointer-events: none;
        }
        .rb-demo-wrapper .rb-dock {
            display: flex; align-items: center; gap: 12px; padding: 8px 16px;
            background: rgba(255, 255, 255, 0.94); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border-radius: 999px; border: 1px solid rgba(203, 213, 225, 0.95);
            box-shadow: 0 12px 28px -10px rgba(15,23,42,0.22), 0 4px 10px -6px rgba(15,23,42,0.16);
            pointer-events: auto;
        }
        .rb-demo-wrapper .rb-dock-btn {
            background: transparent; border: none; color: #475569; cursor: pointer;
            display: flex; align-items: center; justify-content: center; padding: 6px; border-radius: 50%; transition: all 0.2s;
        }
        .rb-demo-wrapper .rb-dock-btn:hover:not(:disabled) { background: #f1f5f9; color: #0f172a; }
        .rb-demo-wrapper .rb-dock-btn:disabled { opacity: 0.3; cursor: not-allowed; }
        .rb-demo-wrapper .rb-dock-btn svg { width: 18px; height: 18px; }
        
        .rb-demo-wrapper .rb-dock-steps { display: flex; gap: 8px; }
        .rb-demo-wrapper .rb-dot {
            width: 26px; height: 26px; display: flex; align-items: center; justify-content: center;
            font-size: 12px; font-weight: 600; color: #64748b; background: transparent; border-radius: 50%;
            cursor: pointer; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); border: none;
        }
        .rb-demo-wrapper .rb-dot:hover { background: #e2e8f0; }
        .rb-demo-wrapper .rb-dot.rb-active { background: var(--rb-highlight); color: white; transform: scale(1.1); box-shadow: 0 4px 8px rgba(59,130,246,0.3); }

        @keyframes rb-fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
    </style>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <div class="rb-header-title">OpenMythos：RecurrentBlock 解析</div>
    <div class="rb-header-desc">单物理层网络，如何通过时间循环与动态路由实现隐式深度推理？</div>

    <!-- 步骤 0: 宏观架构 -->
    <div id="rb-step-0" class="rb-step-content rb-active">
        <div class="rb-info-panel">
            <div class="rb-info-title">阶段 0: 宏观结构 (时间换空间)</div>
            <div class="rb-info-desc">在传统的 Transformer 中，深度是由物理层数堆叠而成的（例如 12 层）。</div>
            
            <div class="rb-text-box">
                <b>Recurrent-Depth Transformer (RDT) 颠覆了这一点：</b><br>
                它的核心大脑仅仅是一个 <code>TransformerBlock</code>，但在前向传播时，代码会通过 <code>for t in range(n_loops)</code> 让数据在这个块里反复迭代。<br><br>
                <b>输入要素：</b><br>
                • <code>h</code>：隐状态（潜变量），在每次循环中不断更新。<br>
                • <code>e</code>：原始输入的编码特征，在所有循环中被<b>冻结并强行注入</b>，防止跑偏。
            </div>
            <div style="text-align: center; margin-top: 10px;">
                <button onclick="rbChangeStep(1)" style="padding: 8px 16px; background: var(--rb-highlight); color: white; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; box-shadow: 0 2px 4px rgba(59,130,246,0.3);">
                    查看内部微观流水线 ➔
                </button>
            </div>
        </div>
    </div>

    <!-- 步骤 1: 5步微观流水线动画 -->
    <div id="rb-step-1" class="rb-step-content">
        <div class="rb-info-panel">
            <button class="rb-replay-btn" onclick="rbPlayPipeline()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"></path><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path></svg>
                运行循环动画
            </button>
            <div class="rb-info-title">阶段 1: 循环迭代的 5 步流水线</div>
            <div class="rb-info-desc">下方静态卡片展示了 RecurrentBlock 的 5 大核心模块原理。点击上方按钮可观看数据流转动画。</div>
            
            <div class="rb-action-log" id="rb-action-log-1">
                <svg style="width:16px; height:16px;" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                <span id="rb-action-log-1-text">等待执行循环...</span>
            </div>

            <!-- 动态流水线容器 -->
            <div class="rb-pipeline-container">
                <div style="font-size: 11px; font-weight: 700; color: var(--rb-text-secondary); margin-bottom: 5px;">
                    当前循环深度: <span id="rb-t-display" style="color: var(--rb-highlight-dark); font-size: 14px;">t = 0</span>
                </div>
                <div class="rb-pipe-row">
                    <div class="rb-module" id="rb-mod-1">1. Loop-Index</div>
                    <div class="rb-arrow" id="rb-arr-1">→</div>
                    <div class="rb-module" id="rb-mod-2">2. Trans+MoE</div>
                    <div class="rb-arrow" id="rb-arr-2">→</div>
                    <div class="rb-module" id="rb-mod-3">3. LoRA Delta</div>
                    <div class="rb-arrow" id="rb-arr-3">→</div>
                    <div class="rb-module" id="rb-mod-4">4. LTI Inject</div>
                    <div class="rb-arrow" id="rb-arr-4">→</div>
                    <div class="rb-module" id="rb-mod-5">5. ACT Halt</div>
                </div>
            </div>
            
            <!-- 静态讲解卡片区（内容永远可见，不随动画改变） -->
            <div class="rb-static-desc-grid">
                <div class="rb-step-card" id="rb-desc-1">
                    <strong>1. Loop-Index Embedding (赋予时间感知)</strong>
                    <span>把当前的循环次数 t 编码成正弦波注入特征。这让同一套物理参数能在循环前期做特征提取，后期做逻辑总结。同时强制加回原始输入 e，防止思考过度导致“语义漂移”。</span>
                </div>
                <div class="rb-step-card" id="rb-desc-2">
                    <strong>2. Transformer + MoE 路由 (动态路由计算)</strong>
                    <span>核心的推理大脑。依靠上方赋予的时间感知，MoE 路由器会在循环的早期自动激活“基础知识专家”，而在循环晚期激活“逻辑推导专家”，在单一网络层内动态拓展了知识域。</span>
                </div>
                <div class="rb-step-card" id="rb-desc-3">
                    <strong>3. Depth-wise LoRA (打破权重共享僵化)</strong>
                    <span>完全一样的权重会让模型思维僵化。这里引入了一个随 t 变化的微小专属缩放向量 <code>scale[t]</code>，通过降维/升维矩阵，以极低的参数成本微调每一步的推理方向。</span>
                </div>
                <div class="rb-step-card" id="rb-desc-4">
                    <strong>4. LTI Injection (抗爆炸的物理稳定器)</strong>
                    <span>如果简单累加残差，几十次循环必定引发数值爆炸 (NaN)。此模块运用零阶保持器 (ZOH) 离散化映射，强制将隐状态衰减系数 A 压死在 (0, 1) 区间内，化爆炸为平稳衰减。</span>
                </div>
                <div class="rb-step-card" id="rb-desc-5">
                    <strong>5. ACT Halting (自适应算力早退判断)</strong>
                    <span>计算当前 Token 是否已经收敛（累积概率 p ≥ 0.99）。如果是“the”这种简单词，可能循环 1 次就退出；如果是数学公式，则持续循环到满。极大节省了全局算力。</span>
                </div>
            </div>
        </div>
    </div>

    <!-- 步骤 2: 深度外推与 LoRA -->
    <div id="rb-step-2" class="rb-step-content">
        <div class="rb-info-panel">
            <div class="rb-info-title">阶段 2: 深度外推 (Depth Extrapolation) 的秘密</div>
            <div class="rb-info-desc">为什么训练时只循环 8 次，推理时循环 16 次也能解开更难的题？</div>
            
            <div class="rb-text-box">
                秘密在于上文提到的 <b>Depth-wise LoRA</b>。它用极少的参数 <code>scale[t]</code> 在不同的循环轮次微调模型。当推理循环次数超过训练时，代码会触发 <code>Clamp</code> 机制（复用最后一个缩放向量），保持模型稳定外推。
            </div>
            
            <div style="position: relative; width: 100%; height: 260px; margin-top: 10px;">
                <canvas id="rb-lora-chart"></canvas>
            </div>
        </div>
    </div>

    <!-- 步骤 3: ACT 算力模拟器 -->
    <div id="rb-step-3" class="rb-step-content">
        <div class="rb-info-panel">
            <button class="rb-replay-btn" onclick="rbResetACT()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"></path><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path></svg>
                重置 Tokens
            </button>
            <div class="rb-info-title">阶段 3: ACT (自适应计算时间) 模拟</div>
            <div class="rb-info-desc">不同难度的 Token 在同一 Batch 内的“早退”机制。</div>
            
            <div class="rb-text-box" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span>当累积概率 ≥ 0.99 时，该 Token 停止更新 (Halted)。</span>
                <button id="rb-act-btn" onclick="rbNextACTLoop()" style="padding: 5px 12px; background: var(--rb-highlight); color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">执行 Loop +1 ➔</button>
            </div>

            <div class="rb-act-grid">
                <!-- Token 1 -->
                <div class="rb-act-row">
                    <div class="rb-token-label">
                        <span class="rb-chunk rb-c0">T0</span> <span style="color:var(--rb-c0-text);">"The"</span> <br><span style="font-weight:normal; font-size:10px; color:#64748b; margin-left:4px;">(简单词)</span>
                    </div>
                    <div class="rb-progress-track">
                        <div class="rb-progress-fill" id="rb-bar-0"></div>
                        <div class="rb-limit-line"></div>
                    </div>
                    <div class="rb-prob-val" id="rb-prob-0">0.00</div>
                </div>
                <!-- Token 2 -->
                <div class="rb-act-row">
                    <div class="rb-token-label">
                        <span class="rb-chunk rb-c1">T1</span> <span style="color:var(--rb-c1-text);">"Therefore"</span> <br><span style="font-weight:normal; font-size:10px; color:#64748b; margin-left:4px;">(逻辑词)</span>
                    </div>
                    <div class="rb-progress-track">
                        <div class="rb-progress-fill" id="rb-bar-1"></div>
                        <div class="rb-limit-line"></div>
                    </div>
                    <div class="rb-prob-val" id="rb-prob-1">0.00</div>
                </div>
                <!-- Token 3 -->
                <div class="rb-act-row">
                    <div class="rb-token-label">
                        <span class="rb-chunk rb-c2">T2</span> <span style="color:var(--rb-c2-text);">"Quantum"</span> <br><span style="font-weight:normal; font-size:10px; color:#64748b; margin-left:4px;">(深层词)</span>
                    </div>
                    <div class="rb-progress-track">
                        <div class="rb-progress-fill" id="rb-bar-2"></div>
                        <div class="rb-limit-line"></div>
                    </div>
                    <div class="rb-prob-val" id="rb-prob-2">0.00</div>
                </div>
            </div>
            
            <div class="rb-action-log" id="rb-act-log" style="margin-top: 15px;">
                <svg style="width:16px; height:16px;" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                <span id="rb-act-log-text">等待执行循环...</span>
            </div>
        </div>
    </div>

    <!-- 底栏 Dock 导航 -->
    <div class="rb-dock-container">
        <div class="rb-dock">
            <button class="rb-dock-btn" id="rb-btn-prev" onclick="rbChangeStep(-1)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
            </button>
            <div class="rb-dock-steps">
                <button class="rb-dot rb-active" id="rb-dot-0" onclick="rbGoToStep(0)">0</button>
                <button class="rb-dot" id="rb-dot-1" onclick="rbGoToStep(1)">1</button>
                <button class="rb-dot" id="rb-dot-2" onclick="rbGoToStep(2)">2</button>
                <button class="rb-dot" id="rb-dot-3" onclick="rbGoToStep(3)">3</button>
            </div>
            <button class="rb-dock-btn" id="rb-btn-next" onclick="rbChangeStep(1)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </button>
        </div>
    </div>

    <script>
        (function() {
            // --- 核心工具 ---
            const rbSleep = ms => new Promise(r => setTimeout(r, ms));
            let rbCurrentStep = 0;
            const rbTotalSteps = 4;
            let rbChartInitialized = false;

            // --- 步骤导航逻辑 ---
            window.rbGoToStep = function(step) {
                if(step < 0 || step >= rbTotalSteps) return;
                rbCurrentStep = step;
                
                document.getElementById('rb-btn-prev').disabled = (step === 0);
                document.getElementById('rb-btn-next').disabled = (step === rbTotalSteps - 1);

                document.querySelectorAll('.rb-dot').forEach(d => d.classList.remove('rb-active'));
                document.getElementById(`rb-dot-${step}`).classList.add('rb-active');

                document.querySelectorAll('.rb-step-content').forEach(c => c.classList.remove('rb-active'));
                document.getElementById(`rb-step-${step}`).classList.add('rb-active');

                // 按需初始化图表
                if (step === 2 && !rbChartInitialized) {
                    rbInitChart();
                    rbChartInitialized = true;
                }
                
                // 每次切换到步骤 1 时如果不在播放状态，可提示用户点击，或自动播放
                if (step === 1) rbPlayPipeline();
            }
            window.rbChangeStep = function(delta) { rbGoToStep(rbCurrentStep + delta); }

            // --- 日志更新 ---
            function rbSetLog(id, msg) {
                const logBox = document.getElementById(id);
                const logText = document.getElementById(id + '-text');
                if (msg) {
                    logText.innerHTML = msg;
                    logBox.classList.add('rb-active');
                } else {
                    logBox.classList.remove('rb-active');
                }
            }

            // --- 阶段 1：静态区与联动流水线动画 ---
            let rbAnimVersion = 0;

            window.rbPlayPipeline = async function() {
                rbAnimVersion++;
                let myVer = rbAnimVersion;
                
                // 演示跑 3 个内部循环，展示数据流转感
                const totalLoopsToAnimate = 3;
                
                for (let t = 0; t < totalLoopsToAnimate; t++) {
                    if(myVer !== rbAnimVersion || rbCurrentStep !== 1) return;
                    
                    document.getElementById('rb-t-display').innerText = `t = ${t}`;
                    rbSetLog('rb-action-log-1', `🔄 正在执行第 <b>${t}</b> 轮内部循环的数据流转...`);
                    await rbSleep(500);

                    for (let m = 1; m <= 5; m++) {
                        if(myVer !== rbAnimVersion || rbCurrentStep !== 1) return;
                        
                        // 重置所有模块、卡片、箭头的状态
                        for(let i=1; i<=5; i++) {
                            document.getElementById(`rb-mod-${i}`).classList.remove('rb-receiving');
                            document.getElementById(`rb-mod-${i}`).classList.add('rb-dimmed');
                            document.getElementById(`rb-desc-${i}`).classList.remove('rb-highlight');
                            if(i < 5) document.getElementById(`rb-arr-${i}`).classList.remove('rb-flowing');
                        }
                        
                        // 点亮当前箭头 (流入模块 m)
                        if (m > 1) {
                            document.getElementById(`rb-arr-${m-1}`).classList.add('rb-flowing');
                        }

                        // 激活当前模块与对应静态卡片
                        document.getElementById(`rb-mod-${m}`).classList.remove('rb-dimmed');
                        document.getElementById(`rb-mod-${m}`).classList.add('rb-receiving');
                        document.getElementById(`rb-desc-${m}`).classList.add('rb-highlight');
                        
                        // 流转节奏 (0.7秒)
                        await rbSleep(700); 
                    }
                }
                
                if(myVer === rbAnimVersion) {
                    // 恢复全亮待机状态
                    for(let i=1; i<=5; i++) {
                        document.getElementById(`rb-mod-${i}`).classList.remove('rb-receiving', 'rb-dimmed');
                        document.getElementById(`rb-desc-${i}`).classList.remove('rb-highlight');
                        if(i < 5) document.getElementById(`rb-arr-${i}`).classList.remove('rb-flowing');
                    }
                    rbSetLog('rb-action-log-1', `✅ 循环演示结束。隐状态 h 已更新完毕，准备进入下一阶段。`);
                }
            };

            // --- 阶段 2：LoRA Chart ---
            window.rbInitChart = function() {
                const initChartInterval = setInterval(() => {
                    if (typeof Chart !== 'undefined') {
                        clearInterval(initChartInterval);
                        const ctx = document.getElementById('rb-lora-chart').getContext('2d');
                        const labels = Array.from({length: 12}, (_, i) => `t=${i}`);
                        
                        const generateData = (peak, speed) => {
                            let data = [];
                            for(let i=0; i<12; i++) {
                                let t = i > 8 ? 8 : i; // 模拟 Clamp 保护
                                data.push(Math.sin(t * speed) * peak + peak);
                            }
                            return data;
                        };

                        new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [
                                    {
                                        label: 'LoRA Scale [某浅层通道]',
                                        data: generateData(1.5, 0.8),
                                        borderColor: '#3b82f6', backgroundColor: 'transparent',
                                        tension: 0.4, borderWidth: 3
                                    },
                                    {
                                        label: 'LoRA Scale [某深层通道]',
                                        data: generateData(2.0, 0.3),
                                        borderColor: '#10b981', backgroundColor: 'transparent',
                                        tension: 0.4, borderWidth: 3
                                    }
                                ]
                            },
                            options: {
                                responsive: true, maintainAspectRatio: false,
                                plugins: {
                                    legend: { position: 'top', labels: { font: { family: 'system-ui' } } },
                                    annotation: {
                                        annotations: {
                                            line1: {
                                                type: 'line', xMin: 8, xMax: 8,
                                                borderColor: '#ef4444', borderWidth: 2, borderDash: [5, 5],
                                                label: { content: 'Training Limit (Clamp)', enabled: true, position: 'top' }
                                            }
                                        }
                                    }
                                },
                                scales: {
                                    x: { title: { display: true, text: 'Loop Iteration (t)' }, grid: { display: false } },
                                    y: { title: { display: true, text: 'Scale Value (s)' } }
                                }
                            }
                        });
                    }
                }, 100);
            };

            // --- 阶段 3：ACT 模拟器 ---
            let actT = 0;
            const threshold = 0.99;
            let tokens = [
                { id: 0, sum: 0, halted: false, rate: [0.6, 0.4, 0.0, 0.0, 0.0, 0.0] }, // T0: The (快)
                { id: 1, sum: 0, halted: false, rate: [0.3, 0.3, 0.2, 0.15, 0.05, 0.0] }, // T1: Therefore (中)
                { id: 2, sum: 0, halted: false, rate: [0.1, 0.2, 0.2, 0.2, 0.2, 0.1] }  // T2: Quantum (慢)
            ];

            window.rbNextACTLoop = function() {
                if(actT >= 6) return;
                let allHalted = true;

                tokens.forEach(tok => {
                    if (!tok.halted) {
                        let p = tok.rate[actT] || 0;
                        let remainder = Math.max(0, 1.0 - tok.sum);
                        
                        if (tok.sum + p >= threshold) {
                            tok.sum += remainder; // ACT Remainder 补齐
                            tok.halted = true;
                            document.getElementById(`rb-bar-${tok.id}`).classList.add('rb-halted');
                        } else {
                            tok.sum += p;
                            allHalted = false;
                        }

                        document.getElementById(`rb-prob-${tok.id}`).innerText = tok.sum.toFixed(2);
                        document.getElementById(`rb-bar-${tok.id}`).style.width = `${Math.min(100, tok.sum * 100)}%`;
                    }
                });

                actT++;
                rbSetLog('rb-act-log', `经过 <b>${actT}</b> 轮循环计算，更新各 Token 累积收敛概率。`);
                
                if (allHalted) {
                    document.getElementById('rb-act-btn').disabled = true;
                    document.getElementById('rb-act-btn').innerText = "已全退出";
                    document.getElementById('rb-act-btn').style.background = "#94a3b8";
                    rbSetLog('rb-act-log', `✅ 所有 Token 均已越过 0.99 阈值，网络整体提前短路退出 (Short-circuit Exit)！`);
                }
            };

            window.rbResetACT = function() {
                actT = 0;
                document.getElementById('rb-act-btn').disabled = false;
                document.getElementById('rb-act-btn').innerText = "执行 Loop +1 ➔";
                document.getElementById('rb-act-btn').style.background = "var(--rb-highlight)";
                rbSetLog('rb-act-log', "");
                
                tokens.forEach(tok => {
                    tok.sum = 0; tok.halted = false;
                    document.getElementById(`rb-prob-${tok.id}`).innerText = "0.00";
                    const bar = document.getElementById(`rb-bar-${tok.id}`);
                    bar.style.width = "0%";
                    bar.classList.remove('rb-halted');
                });
            };

            // 启动初始化
            rbGoToStep(0);

        })();
    </script>
</div>