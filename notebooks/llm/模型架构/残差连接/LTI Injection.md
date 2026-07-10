<div class="lti-demo-wrapper" id="lti-demo-root">
    <style>
        /* =========================================
           1. 绝对沙盒化 CSS
           2. 视觉色彩规范
           ========================================= */
        .lti-demo-wrapper {
            --lti-bg-color: #f8fafc;
            --lti-panel-bg: #ffffff;
            --lti-text-primary: #334155;
            --lti-text-strong: #0f172a;
            --lti-text-secondary: #475569;
            --lti-border-color: #cbd5e1;
            --lti-highlight: #3b82f6;
            --lti-highlight-dark: #1e40af;
            --lti-success: #16a34a;
            --lti-danger: #ef4444;
            --lti-panel-shadow: 0 4px 6px rgba(0,0,0,0.05);

            --lti-c0-bg: #e0f2fe; --lti-c0-text: #0369a1; --lti-c0-border: #7dd3fc; 
            --lti-c1-bg: #f3e8ff; --lti-c1-text: #6b21a8; --lti-c1-border: #d8b4fe; 
            --lti-c2-bg: #dcfce7; --lti-c2-text: #15803d; --lti-c2-border: #86efac; 
            --lti-c3-bg: #fef9c3; --lti-c3-text: #a16207; --lti-c3-border: #fde047; 

            font-family: "Segoe UI", Tahoma, Geneva, Verdana, system-ui, sans-serif;
            background-color: var(--lti-bg-color);
            color: var(--lti-text-primary);
            padding: 32px 24px 100px 24px;
            border-radius: 12px;
            border: 1px solid var(--lti-border-color);
            box-sizing: border-box;
            line-height: 1.6;
            font-size: 14px;
            position: relative;
            min-height: 700px;
        }

        .lti-demo-wrapper * { box-sizing: border-box; }

        /* --- 标题 --- */
        .lti-demo-wrapper .lti-header-title {
            font-size: 24px;
            font-weight: 800;
            color: var(--lti-text-strong);
            text-align: center;
            margin-bottom: 8px;
        }
        
        .lti-demo-wrapper .lti-header-desc {
            font-size: 14px;
            font-weight: 600;
            color: var(--lti-highlight-dark);
            text-align: center;
            margin-bottom: 24px;
        }

        /* --- 分页系统控制 --- */
        .lti-demo-wrapper .lti-step-content {
            display: none;
            animation: lti-fadeIn 0.5s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }
        .lti-demo-wrapper .lti-step-content.lti-active {
            display: block;
        }

        /* --- 文本与面板 --- */
        .lti-demo-wrapper .lti-text-block {
            font-size: 14px;
            color: var(--lti-text-secondary);
            margin-bottom: 20px;
            text-align: justify;
            background: rgba(255, 255, 255, 0.5);
            padding: 16px;
            border-radius: 8px;
            border-left: 3px solid #e2e8f0;
        }
        .lti-demo-wrapper .lti-highlight-word {
            color: var(--lti-highlight-dark);
            font-weight: 700;
            background: #eff6ff;
            padding: 0 4px;
            border-radius: 4px;
        }
        .lti-demo-wrapper .lti-info-title {
            font-size: 18px;
            font-weight: 800;
            color: var(--lti-text-strong);
            margin: 0 0 16px 0;
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 1px solid var(--lti-border-color);
            padding-bottom: 12px;
        }
        .lti-demo-wrapper .lti-info-panel {
            background: var(--lti-panel-bg);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--lti-border-color);
            border-left: 5px solid var(--lti-highlight);
            box-shadow: var(--lti-panel-shadow);
            margin-bottom: 24px;
            position: relative;
        }

        /* --- 剖析面板 (Deep Dive) --- */
        .lti-demo-wrapper .lti-deep-dive {
            background: #f8fafc;
            padding: 16px 20px;
            border-radius: 10px;
            border: 1px solid var(--lti-border-color);
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
            display: none;
            margin-top: 20px;
            font-size: 14px;
            color: var(--lti-text-secondary);
            line-height: 1.7;
        }
        .lti-demo-wrapper .lti-deep-dive.lti-active {
            display: block;
            animation: lti-fadeIn 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }

        /* --- 公式块 --- */
        .lti-demo-wrapper .lti-formula-row {
            display: flex; align-items: center; justify-content: center;
            gap: 14px; flex-wrap: wrap; font-size: 16px; font-weight: 700;
            background: #f1f5f9; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;
        }
        .lti-demo-wrapper .lti-chunk {
            padding: 8px 16px; border-radius: 6px; border: 1px solid transparent;
            cursor: pointer; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            white-space: nowrap; font-family: 'Cambria Math', 'Times New Roman', serif; font-style: italic;
        }
        .lti-demo-wrapper .lti-c0 { background: var(--lti-c0-bg); color: var(--lti-c0-text); border-color: var(--lti-c0-border); }
        .lti-demo-wrapper .lti-c1 { background: var(--lti-c1-bg); color: var(--lti-c1-text); border-color: var(--lti-c1-border); }
        .lti-demo-wrapper .lti-c2 { background: var(--lti-c2-bg); color: var(--lti-c2-text); border-color: var(--lti-c2-border); }
        .lti-demo-wrapper .lti-c3 { background: var(--lti-c3-bg); color: var(--lti-c3-text); border-color: var(--lti-c3-border); }
        .lti-demo-wrapper .lti-chunk:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .lti-demo-wrapper .lti-chunk.lti-receiving { transform: scale(1.1); z-index: 10; }
        .lti-demo-wrapper .lti-dimmed { opacity: 0.3; filter: grayscale(80%); transform: scale(0.95); }

        /* --- ZOH 滑块控制 --- */
        .lti-demo-wrapper .lti-slider-group {
            background: #f8fafc; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0;
        }
        .lti-demo-wrapper .lti-slider-header {
            display: flex; justify-content: space-between; margin-bottom: 8px;
            font-weight: 700; font-size: 13px; color: var(--lti-text-strong);
        }
        .lti-demo-wrapper .lti-val-tag {
            background: #ffffff; border: 1px solid var(--lti-border-color);
            padding: 2px 10px; border-radius: 4px; font-family: monospace; font-weight: 800; color: var(--lti-highlight-dark);
        }
        .lti-demo-wrapper input[type=range] {
            -webkit-appearance: none; width: 100%; background: var(--lti-border-color); height: 6px; border-radius: 3px; outline: none;
        }
        .lti-demo-wrapper input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%;
            background: var(--lti-highlight); cursor: pointer; border: 2px solid #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: transform 0.2s;
        }
        .lti-demo-wrapper input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.2); }

        /* --- 进度条 --- */
        .lti-demo-wrapper .lti-progress-track {
            width: 100%; height: 24px; background: #e2e8f0; border-radius: 999px;
            overflow: hidden; position: relative; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); margin-top: 16px;
        }
        .lti-demo-wrapper .lti-progress-fill {
            height: 100%; background: var(--lti-highlight); transition: width 0.1s linear, background-color 0.4s ease;
        }
        .lti-demo-wrapper .lti-updated-param { background-color: var(--lti-success); background-image: linear-gradient(135deg, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0) 100%); }
        .lti-demo-wrapper .lti-danger-param { background-color: var(--lti-danger); animation: lti-soft-pulse 1.5s infinite; }
        .lti-demo-wrapper .lti-limit-line { position: absolute; right: 0; top: 0; bottom: 0; width: 3px; background: #7f1d1d; z-index: 10; }

        /* --- Dock 导航 --- */
        .lti-demo-wrapper .lti-dock-container {
            position: absolute; left: 50%; bottom: 24px; transform: translateX(-50%); z-index: 10000;
            width: min(420px, calc(100vw - 32px)); display: flex; justify-content: center;
        }
        .lti-demo-wrapper .lti-dock {
            display: flex; align-items: center; gap: 12px; padding: 8px 16px;
            background: rgba(255, 255, 255, 0.94); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border-radius: 999px; border: 1px solid rgba(203, 213, 225, 0.95);
            box-shadow: 0 12px 28px -10px rgba(15,23,42,0.22), 0 4px 10px -6px rgba(15,23,42,0.16);
        }
        .lti-demo-wrapper .lti-dock-btn {
            background: transparent; border: none; color: #475569; cursor: pointer;
            display: flex; align-items: center; justify-content: center; padding: 6px; border-radius: 50%; transition: all 0.2s;
        }
        .lti-demo-wrapper .lti-dock-btn:hover:not(:disabled) { background: #f1f5f9; color: #0f172a; }
        .lti-demo-wrapper .lti-dock-btn:disabled { opacity: 0.3; cursor: not-allowed; }
        .lti-demo-wrapper .lti-dock-btn svg { width: 18px; height: 18px; }
        .lti-demo-wrapper .lti-dock-steps { display: flex; gap: 8px; }
        .lti-demo-wrapper .lti-dot {
            width: 28px; height: 28px; font-size: 13px; font-weight: 700; color: var(--lti-text-secondary);
            background: transparent; border-radius: 50%; border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .lti-demo-wrapper .lti-dot:hover { background: #f1f5f9; }
        .lti-demo-wrapper .lti-dot.lti-active { background: var(--lti-highlight); color: white; transform: scale(1.15); box-shadow: 0 4px 8px rgba(59,130,246,0.3); }

        @keyframes lti-fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes lti-soft-pulse { 0% { box-shadow: inset 0 0 0 1px rgba(0,0,0,0.05); } 50% { box-shadow: inset 0 0 12px rgba(239,68,68,0.5); } 100% { box-shadow: inset 0 0 0 1px rgba(0,0,0,0.05); } }
    </style>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <div class="lti-header-title">Parcae 架构：LTI 注入机制解析</div>
    <div class="lti-header-desc">非线性时变动力系统与双重指数 ZOH 离散化映射</div>

    <!-- 步骤 0：背景介绍 -->
    <div id="lti-step-0" class="lti-step-content lti-active">
        <div class="lti-info-panel">
            <h3 class="lti-info-title">
                <span style="color:var(--lti-highlight); margin-right:4px;">00</span> 背景：循环大模型的致命瓶颈
            </h3>
            <div class="lti-text-block" style="border-left-color: var(--lti-highlight);">
                <strong>大背景：</strong>传统的 Transformer（如 GPT）通过“堆叠数百层不同的网络”来提升能力，导致巨大的内存开销。
                <span class="lti-highlight-word">循环大模型 (Looped Transformers)</span> 提出了一种省内存的绝妙思路：<strong>只建极少层网络，但让数据在这些层里反复循环计算几十次</strong>。
                <br><br>
                <strong>致命痛点：</strong>如果隐状态在同一个权重矩阵里反复滚雪球，极易导致<strong>梯度/隐状态爆炸</strong>（数值飙升至 NaN 崩溃）。接下来的分页将向您展示 <strong>LTI Injection（线性时不变注入）</strong> 机制是如何通过严谨的数学控制论，彻底解决这个爆炸难题的。
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <button onclick="ltiChangeStep(1)" style="padding: 10px 20px; background: var(--lti-highlight); color: #fff; border: none; border-radius: 6px; font-weight: 700; cursor: pointer; box-shadow: 0 4px 6px rgba(59,130,246,0.3);">
                    开始探索 LTI 机制核心 🚀
                </button>
            </div>
        </div>
    </div>

    <!-- 步骤 1：公式拆解 -->
    <div id="lti-step-1" class="lti-step-content">
        <div class="lti-info-panel">
            <h3 class="lti-info-title">
                <span style="color:var(--lti-highlight); margin-right:4px;">01</span> 循环更新规则：打破传统残差的魔咒
            </h3>
            <div class="lti-text-block">
                在传统网络中，层与层之间通过简单加法连接：<code style="background:#e2e8f0; padding:2px 6px; border-radius:4px;">h = h + out</code>。
                但在循环架构中沿用此法，旧信息会被无限制放大。LTI 机制重构了更新公式，引入了<strong>受控遗忘</strong>和<strong>持续的原始输入注入</strong>。
                <br><em>👇 点击下方公式中的彩色模块，查看其具体物理含义。</em>
            </div>

            <div class="lti-formula-row">
                <div class="lti-chunk lti-c0" data-target="lti-desc-hnext">h<sub>t+1</sub></div>
                <div style="color:var(--lti-text-secondary)">=</div>
                <div class="lti-chunk lti-c1" data-target="lti-desc-decay">A · h<sub>t</sub></div>
                <div style="color:var(--lti-text-secondary)">+</div>
                <div class="lti-chunk lti-c2" data-target="lti-desc-inject">B · e</div>
                <div style="color:var(--lti-text-secondary)">+</div>
                <div class="lti-chunk lti-c3" data-target="lti-desc-trans" style="font-family: inherit; font-style: normal;">Trans_out</div>
            </div>

            <div class="lti-deep-dive lti-active" id="lti-desc-default" style="text-align: center; background: #f8fafc; border: 1px dashed var(--lti-border-color);">
                请点击上方的高亮色块，这里将显示其详细的机制剖析。
            </div>

            <div class="lti-deep-dive" id="lti-desc-hnext" style="border-left: 4px solid var(--lti-c0-border);">
                <strong style="color: var(--lti-c0-text); font-size: 15px;">输出隐状态 (h<sub>t+1</sub>)</strong><br>
                模型在第 <span style="font-family: monospace;">t</span> 次循环思考后打包好的“短期记忆”，作为下一次循环（<span style="font-family: monospace;">t+1</span> 步）的输入。
            </div>
            <div class="lti-deep-dive" id="lti-desc-decay" style="border-left: 4px solid var(--lti-c1-border);">
                <strong style="color: var(--lti-c1-text); font-size: 15px;">状态遗忘/衰减项 (A · h<sub>t</sub>) —— 稳定性的核心</strong><br>
                控制模型保留多少“上一轮旧记忆”。为了防止循环时记忆无限放大，<strong>衰减因子 A 必须严格限制在 (0, 1) 区间内</strong>。
            </div>
            <div class="lti-deep-dive" id="lti-desc-inject" style="border-left: 4px solid var(--lti-c2-border);">
                <strong style="color: var(--lti-c2-text); font-size: 15px;">原始输入注入项 (B · e) —— 防止“语义漂移”</strong><br>
                模型在一个问题上循环几十次后容易忘了最初的上下文。此项在每次循环时按固定比例强制混入<strong>最原始的提问特征 (e)</strong>进行锚定。
            </div>
            <div class="lti-deep-dive" id="lti-desc-trans" style="border-left: 4px solid var(--lti-c3-border);">
                <strong style="color: var(--lti-c3-text); font-size: 15px;">计算增量 (Trans_out) —— 新的思考结果</strong><br>
                当前轮次通过 Transformer 块（自注意力 + FFN）推理得出的一阶新信息增量。
            </div>
        </div>
    </div>

    <!-- 步骤 2：图表 -->
    <div id="lti-step-2" class="lti-step-content">
        <div class="lti-info-panel">
            <h3 class="lti-info-title">
                <span style="color:var(--lti-highlight); margin-right:4px;">02</span> 爆炸模拟：为什么我们如此害怕 A ≥ 1？
            </h3>
            <div class="lti-text-block">
                如果让神经网络自由学习参数 <span style="font-family: monospace;">A</span>，极容易越过 <span class="lti-highlight-word">1.0</span> 的红线。
                下图展示了在多次循环（深度增加）时：红线代表稍微大于 1 (如 1.1) 造成的指数级爆炸，绿线代表 LTI 机制下受控的平稳收敛。
            </div>
            <div style="position: relative; width: 100%; height: 280px; margin-top: 20px;">
                <canvas id="lti-stability-chart"></canvas>
            </div>
        </div>
    </div>

    <!-- 步骤 3：模拟器 (精细优化输入输出说明) -->
    <div id="lti-step-3" class="lti-step-content">
        <div class="lti-info-panel">
            <h3 class="lti-info-title">
                <span style="color:var(--lti-highlight); margin-right:4px;">03</span> ZOH 离散化：不可逾越的数学壁垒
            </h3>
            <div class="lti-text-block">
                为了在代码底层彻底杜绝 A ≥ 1，作者将网络视为连续动力系统，并通过<strong>零阶保持器 (ZOH)</strong> 推导出了一个精妙的数学映射。它将神经网络生成的任意数值，强制锁死在安全区：<br>
                <div style="text-align: center; margin-top: 12px; font-weight: bold; color: var(--lti-text-strong); font-size: 20px; font-family: 'Cambria Math', serif;">
                    A = exp(-exp(γ + α))
                </div>
            </div>

            <!-- 输入区域 -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px;">
                <div class="lti-slider-group">
                    <div class="lti-slider-header">
                        <span>📥 输入 1：连续时间步长 (γ)</span>
                        <span class="lti-val-tag" id="lti-val-gamma">0.0</span>
                    </div>
                    <p style="font-size: 12px; color: var(--lti-text-secondary); margin-bottom: 12px; height: 50px;">
                        <b>物理含义</b>：代表连续系统中单次循环跨越的“虚拟时间长短”。对应代码 <code>log_dt</code>，可被优化器推向任意极大/极小值。
                    </p>
                    <input type="range" id="lti-slider-gamma" min="-8" max="8" step="0.1" value="0">
                </div>

                <div class="lti-slider-group">
                    <div class="lti-slider-header">
                        <span>📥 输入 2：状态幅值参数 (α)</span>
                        <span class="lti-val-tag" id="lti-val-alpha">0.0</span>
                    </div>
                    <p style="font-size: 12px; color: var(--lti-text-secondary); margin-bottom: 12px; height: 50px;">
                        <b>物理含义</b>：代表连续动力系统的“本征衰减率”。对应代码 <code>log_A</code>，同样是不受任何边界约束的自由实数。
                    </p>
                    <input type="range" id="lti-slider-alpha" min="-8" max="8" step="0.1" value="0">
                </div>
            </div>

            <!-- 输出区域 -->
            <div style="text-align: center; background: #f1f5f9; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; position: relative;">
                <div style="font-size: 16px; font-weight: 800; color: var(--lti-highlight-dark); margin-bottom: 8px;">
                    📤 最终输出：离散衰减因子 A
                </div>
                <div style="font-size: 13px; color: var(--lti-text-secondary); margin-bottom: 16px;">
                    <b>物理含义</b>：这是实际乘在隐状态 <span style="font-family: monospace; font-weight: bold;">h<sub>t</sub></span> 上的保留系数。<br>
                    亲自拖动上方滑块验证：无论输入被推向多离谱的数值，<br>双重指数映射 <code>exp(-exp(x))</code> 都能保证其输出<strong>绝对无法达到或超越 1.0 崩溃红线</strong>。
                </div>
                
                <div style="font-size: 40px; font-weight: 800; font-family: monospace; color: var(--lti-highlight-dark);" id="lti-val-A">
                    0.367879
                </div>
                
                <div class="lti-progress-track">
                    <div class="lti-progress-fill lti-updated-param" id="lti-bar-A" style="width: 36.78%;"></div>
                    <div class="lti-limit-line" title="致命边界 A=1"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--lti-text-secondary); margin-top: 8px; font-weight: 600;">
                    <span>0.0 (信号彻底清零)</span>
                    <span>1.0 (崩溃红线，绝对无法到达)</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Dock 分页导航 -->
    <div class="lti-dock-container">
        <div class="lti-dock">
            <button class="lti-dock-btn" id="lti-btn-prev" onclick="ltiChangeStep(-1)" disabled>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
            </button>
            <div class="lti-dock-steps">
                <button class="lti-dot lti-active" id="lti-dot-0" onclick="ltiGoToStep(0)">0</button>
                <button class="lti-dot" id="lti-dot-1" onclick="ltiGoToStep(1)">1</button>
                <button class="lti-dot" id="lti-dot-2" onclick="ltiGoToStep(2)">2</button>
                <button class="lti-dot" id="lti-dot-3" onclick="ltiGoToStep(3)">3</button>
            </div>
            <button class="lti-dock-btn" id="lti-btn-next" onclick="ltiChangeStep(1)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </button>
        </div>
    </div>

    <script>
        (function() {
            // --- 1. 分页导航逻辑 ---
            let ltiCurrentStep = 0;
            const ltiTotalSteps = 4;
            let ltiChartInitialized = false;

            window.ltiGoToStep = function(step) {
                if (step < 0 || step >= ltiTotalSteps) return;
                ltiCurrentStep = step;

                // 更新 Dock 状态
                document.querySelectorAll('.lti-dot').forEach(d => d.classList.remove('lti-active'));
                document.getElementById(`lti-dot-${step}`).classList.add('lti-active');

                // 更新按钮可用性
                document.getElementById('lti-btn-prev').disabled = (step === 0);
                document.getElementById('lti-btn-next').disabled = (step === ltiTotalSteps - 1);

                // 切换面板展示
                document.querySelectorAll('.lti-step-content').forEach(c => c.classList.remove('lti-active'));
                document.getElementById(`lti-step-${step}`).classList.add('lti-active');

                // 延迟初始化或重绘图表，以防 display:none 导致画布尺寸为 0
                if (step === 2) {
                    if (!ltiChartInitialized) {
                        initStabilityChart();
                        ltiChartInitialized = true;
                    }
                }
            };

            window.ltiChangeStep = function(delta) {
                ltiGoToStep(ltiCurrentStep + delta);
            };

            // --- 2. 公式交互块逻辑 ---
            const chunks = document.querySelectorAll('.lti-demo-wrapper .lti-chunk');
            const descs = document.querySelectorAll('.lti-demo-wrapper .lti-deep-dive');

            chunks.forEach(chunk => {
                chunk.addEventListener('click', () => {
                    chunks.forEach(c => {
                        c.classList.remove('lti-receiving');
                        c.classList.add('lti-dimmed');
                    });
                    descs.forEach(d => d.classList.remove('lti-active'));
                    
                    chunk.classList.remove('lti-dimmed');
                    chunk.classList.add('lti-receiving');
                    
                    const targetId = chunk.getAttribute('data-target');
                    document.getElementById(targetId).classList.add('lti-active');
                });
            });

            // --- 3. ZOH 模拟器逻辑 ---
            const sGamma = document.getElementById('lti-slider-gamma');
            const sAlpha = document.getElementById('lti-slider-alpha');
            const vGamma = document.getElementById('lti-val-gamma');
            const vAlpha = document.getElementById('lti-val-alpha');
            const vA = document.getElementById('lti-val-A');
            const barA = document.getElementById('lti-bar-A');

            function updateSimulator() {
                const gamma = parseFloat(sGamma.value);
                const alpha = parseFloat(sAlpha.value);
                
                vGamma.textContent = (gamma > 0 ? '+' : '') + gamma.toFixed(1);
                vAlpha.textContent = (alpha > 0 ? '+' : '') + alpha.toFixed(1);
                
                let x = gamma + alpha;
                // 代码实现中的保护截断
                x = Math.max(-20, Math.min(20, x)); 
                
                const A = Math.exp(-Math.exp(x));
                vA.textContent = A.toFixed(6);
                barA.style.width = (A * 100) + '%';
                
                // 视觉警示：接近红线时变色
                if (A > 0.9) {
                    barA.classList.remove('lti-updated-param');
                    barA.classList.add('lti-danger-param');
                } else {
                    barA.classList.remove('lti-danger-param');
                    barA.classList.add('lti-updated-param');
                }
            }

            sGamma.addEventListener('input', updateSimulator);
            sAlpha.addEventListener('input', updateSimulator);
            updateSimulator();

            // --- 4. Chart.js 图表初始化 ---
            function initStabilityChart() {
                const initChartInterval = setInterval(() => {
                    if (typeof Chart !== 'undefined') {
                        clearInterval(initChartInterval);
                        const ctx = document.getElementById('lti-stability-chart').getContext('2d');
                        const steps = Array.from({length: 20}, (_, i) => i + 1);
                        
                        new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: steps,
                                datasets: [
                                    {
                                        label: '不受约束的危险累加 (A=1.10)',
                                        data: steps.map(t => Math.pow(1.10, t)),
                                        borderColor: '#ef4444', 
                                        backgroundColor: 'rgba(239, 68, 68, 0.05)',
                                        borderWidth: 2,
                                        fill: true,
                                        tension: 0.4
                                    },
                                    {
                                        label: 'LTI 受控平滑衰减 (A=0.90)',
                                        data: steps.map(t => Math.pow(0.90, t)),
                                        borderColor: '#16a34a',
                                        backgroundColor: 'rgba(22, 163, 74, 0.15)',
                                        borderWidth: 3,
                                        fill: true,
                                        tension: 0.4
                                    }
                                ]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: { position: 'top', labels: { color: '#334155', font: {weight: '600'} } },
                                    tooltip: { mode: 'index', intersect: false }
                                },
                                scales: {
                                    x: { title: { display: true, text: '模型循环深度 (Loop t)' }, grid: { display: false } },
                                    y: { title: { display: true, text: '隐状态数值幅度' }, beginAtZero: true }
                                }
                            }
                        });
                    }
                }, 100);
            }

        })();
    </script>
</div>


## 参考文献
[1] Research, Sandy. 《Parcae: Doing More with Fewer Parameters using Stable Looped Models》. 见于 2026年7月8日. https://sandyresearch.github.io/parcae/.
