<div class="rb-demo-wrapper" id="rb-act-root" data-step="0">
    <style>
        /* =========================================
           1. 绝对沙盒化 CSS 隔离
           2. 视觉色彩规范：Deepspeed Zero2 浅色系
           ========================================= */
        .rb-demo-wrapper {
            --rb-bg: #f8fafc;
            --rb-panel-bg: #ffffff;
            --rb-text-primary: #334155;
            --rb-text-strong: #0f172a;
            --rb-text-secondary: #64748b;
            --rb-border: #cbd5e1;
            
            --rb-blue: #3b82f6;     /* p 概率 */
            --rb-green: #16a34a;    /* remainder 余量 */
            --rb-amber: #f59e0b;    /* cond 越界 */
            --rb-red: #ef4444;      /* final 最终 */
            --rb-purple: #8b5cf6;   /* mask 活跃掩码 */

            font-family: "Segoe UI", Tahoma, Geneva, Verdana, system-ui, sans-serif;
            background-color: var(--rb-bg);
            color: var(--rb-text-primary);
            width: 100%;
            margin: 0 auto;
            padding: 24px;
            box-sizing: border-box;
            line-height: 1.6;
            border-radius: 12px;
            position: relative;
        }

        .rb-demo-wrapper * { box-sizing: border-box; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); }

        /* --- 标题区域 --- */
        .rb-demo-wrapper .rb-header-title { text-align: center; font-size: 24px; font-weight: 800; color: var(--rb-text-strong); margin-bottom: 8px; }
        .rb-demo-wrapper .rb-header-desc { text-align: center; color: var(--rb-text-secondary); font-size: 14px; margin-bottom: 24px; }

        /* --- 面板与日志 --- */
        .rb-demo-wrapper .rb-panel {
            background: var(--rb-panel-bg); padding: 20px; border-radius: 12px;
            border: 1px solid var(--rb-border); border-left: 5px solid var(--rb-blue);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px;
        }
        .rb-demo-wrapper .rb-action-log {
            background: #eff6ff; color: #1d4ed8; padding: 12px 16px; border-radius: 8px;
            font-size: 14px; font-weight: 700; border: 1px dashed #93c5fd;
            display: flex; align-items: flex-start; gap: 10px; margin-bottom: 16px;
        }

        /* --- 按钮 --- */
        .rb-demo-wrapper .rb-controls { display: flex; justify-content: center; gap: 16px; margin-top: 24px; padding-top: 24px; border-top: 1px solid var(--rb-border); }
        .rb-demo-wrapper .rb-btn {
            display: flex; align-items: center; justify-content: center; gap: 6px; padding: 10px 20px; font-size: 14px; font-weight: 700; 
            border-radius: 8px; cursor: pointer; border: none;
        }
        .rb-demo-wrapper .rb-btn-primary { background: var(--rb-blue); color: white; box-shadow: 0 4px 6px rgba(59,130,246,0.3); }
        .rb-demo-wrapper .rb-btn-primary:hover { background: #1d4ed8; transform: translateY(-2px); box-shadow: 0 6px 12px rgba(59,130,246,0.4); }
        .rb-demo-wrapper .rb-btn-secondary { background: #f8fafc; color: var(--rb-text-primary); border: 1px solid var(--rb-border); }
        .rb-demo-wrapper .rb-btn-secondary:hover { background: #e2e8f0; transform: translateY(-2px); }

        /* --- 布局 --- */
        .rb-demo-wrapper .rb-layout { display: flex; flex-wrap: wrap; gap: 20px; align-items: stretch; }
        
        /* --- 左侧代码块 --- */
        .rb-demo-wrapper .rb-code-block {
            flex: 1 1 300px; background: #1e293b; color: #e2e8f0; padding: 16px; border-radius: 10px;
            font-family: 'Fira Code', 'Consolas', monospace; font-size: 13px; line-height: 1.6;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.3); display: flex; flex-direction: column;
        }
        .rb-demo-wrapper .rb-cl { padding: 4px 8px; border-radius: 6px; border-left: 3px solid transparent; display: flex; white-space: nowrap; opacity: 0.5; }
        .rb-demo-wrapper .rb-line-num { color: #64748b; width: 24px; flex-shrink: 0; user-select: none; }
        .rb-demo-wrapper .rb-c-cmt { color: #94a3b8; font-style: italic; }
        .rb-demo-wrapper .rb-c-kw { color: #c678dd; } 
        .rb-demo-wrapper .rb-c-fn { color: #61afef; } 
        .rb-demo-wrapper .rb-c-num { color: #d19a66; } 

        /* 代码行状态绑定 */
        .rb-demo-wrapper[data-step="1"] .rb-cl.step-1,
        .rb-demo-wrapper[data-step="2"] .rb-cl.step-2,
        .rb-demo-wrapper[data-step="3"] .rb-cl.step-3,
        .rb-demo-wrapper[data-step="4"] .rb-cl.step-4,
        .rb-demo-wrapper[data-step="5"] .rb-cl.step-5 {
            background: rgba(59, 130, 246, 0.35); border-left-color: var(--rb-blue); color: #fff; opacity: 1; font-weight: bold;
        }

        /* --- 右侧张量区 --- */
        .rb-demo-wrapper .rb-tensor-area { 
            flex: 1.5 1 450px; background: #f8fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; 
            display: flex; flex-direction: column; gap: 16px; overflow-x: auto;
        }

        /* 顶部全局寄存器 */
        .rb-demo-wrapper .rb-reg-box {
            background: #fff; padding: 12px; border-radius: 8px; border: 1px dashed var(--rb-border); margin-bottom: 8px;
        }
        .rb-demo-wrapper .rb-reg-header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px; }
        .rb-demo-wrapper .rb-token-header { display: flex; gap: 8px; }
        .rb-demo-wrapper .rb-token-id { width: 48px; text-align: center; font-size: 11px; font-weight: 800; color: #64748b; background: #e2e8f0; border-radius: 4px; padding: 2px 0;}

        /* 张量行基础样式 */
        .rb-demo-wrapper .rb-t-row {
            display: flex; align-items: center; justify-content: space-between; 
            opacity: 0.2; filter: grayscale(100%); transform: translateY(5px);
        }
        .rb-demo-wrapper .rb-t-label { 
            font-weight: 800; font-size: 13px; width: 120px; text-align: right; 
            font-family: monospace; color: var(--rb-text-primary); line-height: 1.2; padding-right: 12px;
        }
        .rb-demo-wrapper .rb-t-label span { font-size: 10px; font-weight: normal; color: var(--rb-text-secondary); display: block; }
        
        .rb-demo-wrapper .rb-tensor { display: flex; gap: 8px; }
        .rb-demo-wrapper .rb-cell {
            width: 48px; height: 32px; display: flex; align-items: center; justify-content: center;
            border-radius: 6px; font-family: monospace; font-weight: 700; font-size: 13px;
            border: 2px solid var(--rb-border); background: #fff; color: var(--rb-text-secondary);
        }

        /* 步骤可见性控制 */
        .rb-demo-wrapper[data-step="1"] .rb-t-row.step-1,
        .rb-demo-wrapper[data-step="2"] .rb-t-row.step-2,
        .rb-demo-wrapper[data-step="3"] .rb-t-row.step-3,
        .rb-demo-wrapper[data-step="4"] .rb-t-row.step-4 {
            opacity: 1; filter: grayscale(0%); transform: translateY(0);
        }
        
        /* 历史步骤保持可见但略微变暗 */
        .rb-demo-wrapper[data-step="2"] .rb-t-row.step-1,
        .rb-demo-wrapper[data-step="3"] .rb-t-row.step-1, .rb-demo-wrapper[data-step="3"] .rb-t-row.step-2,
        .rb-demo-wrapper[data-step="4"] .rb-t-row.step-1, .rb-demo-wrapper[data-step="4"] .rb-t-row.step-2, .rb-demo-wrapper[data-step="4"] .rb-t-row.step-3,
        .rb-demo-wrapper[data-step="5"] .rb-t-row {
            opacity: 0.6; filter: grayscale(20%); transform: translateY(0);
        }

        /* 步骤 1: 呈现数据 */
        .rb-demo-wrapper[data-step="1"] .rb-cell.is-p, .rb-demo-wrapper[data-step="2"] .rb-cell.is-p { border-color: var(--rb-blue); color: var(--rb-blue); background: #eff6ff; }
        .rb-demo-wrapper[data-step="1"] .rb-cell.is-sr, .rb-demo-wrapper[data-step="2"] .rb-cell.is-sr { border-color: var(--rb-purple); color: var(--rb-purple); background: #faf5ff; }

        /* 步骤 2: 呈现数据 */
        .rb-demo-wrapper[data-step="2"] .rb-cell.is-rem { border-color: var(--rb-green); color: var(--rb-green); background: #f0fdf4; }
        .rb-demo-wrapper[data-step="2"] .rb-cell.is-cond { border-color: var(--rb-amber); color: #b45309; background: #fffbeb; }

        /* 步骤 3: 核心 Torch.Where 共振发光路由 (NO SVG) */
        .rb-demo-wrapper[data-step="3"] .glow-rem { 
            border-color: var(--rb-green); color: var(--rb-green); background: #f0fdf4;
            box-shadow: 0 0 15px rgba(22, 163, 74, 0.5); transform: scale(1.1); z-index: 10;
        }
        .rb-demo-wrapper[data-step="3"] .glow-p { 
            border-color: var(--rb-blue); color: var(--rb-blue); background: #eff6ff;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5); transform: scale(1.1); z-index: 10;
        }
        /* 揭晓原始 weight */
        .rb-demo-wrapper[data-step="3"] .val-hide-s3 { opacity: 0; }
        
        /* 步骤 4: 乘以 still_running 掩码，抹杀死 Token */
        .rb-demo-wrapper[data-step="4"] .rb-row-final { border: 2px dashed var(--rb-red); padding: 8px; border-radius: 8px; background: #fff0f2; }
        .rb-demo-wrapper[data-step="4"] .rb-cell.is-final { border-color: var(--rb-red); color: #b91c1c; font-weight: 900; background: #fff; }
        .rb-demo-wrapper[data-step="4"] .kill-token { opacity: 0.2; text-decoration: line-through; }
        .rb-demo-wrapper[data-step="4"] .val-hide-s4 { opacity: 0; }
        
        /* 步骤 5: 更新寄存器特效 */
        .rb-demo-wrapper[data-step="5"] .rb-row-final { opacity: 0.5; border: 2px dashed var(--rb-red); padding: 8px; border-radius: 8px; }
        .rb-demo-wrapper[data-step="5"] .rb-cell.is-final { border-color: var(--rb-red); color: #b91c1c; font-weight: 900; }
        .rb-demo-wrapper[data-step="5"] .update-glow {
            border-color: var(--rb-green) !important; color: var(--rb-green) !important; background: #f0fdf4 !important;
            box-shadow: 0 0 15px rgba(22, 163, 74, 0.5); font-weight: 900; transform: scale(1.1);
        }

        /* =========================================
           新增：学习笔记专属样式 (Markdown 风格)
           ========================================= */
        .rb-demo-wrapper .rb-notes-container {
            margin-top: 32px;
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid var(--rb-border);
            padding: 32px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
            font-size: 14.5px;
            color: var(--rb-text-primary);
        }
        .rb-demo-wrapper .rb-notes-header {
            font-size: 20px;
            font-weight: 800;
            color: var(--rb-text-strong);
            display: flex;
            align-items: center;
            gap: 10px;
            padding-bottom: 16px;
            border-bottom: 2px solid #f1f5f9;
            margin-bottom: 24px;
        }
        .rb-demo-wrapper .rb-notes-section { margin-bottom: 28px; }
        .rb-demo-wrapper .rb-notes-title {
            font-size: 16px; font-weight: 700; color: var(--rb-blue);
            margin-bottom: 12px; display: flex; align-items: center; gap: 6px;
        }
        .rb-demo-wrapper .rb-notes-title::before {
            content: ''; display: inline-block; width: 6px; height: 16px; background: var(--rb-blue); border-radius: 3px;
        }
        .rb-demo-wrapper .rb-notes-p { margin-bottom: 12px; line-height: 1.7; text-align: justify; }
        .rb-demo-wrapper .rb-tag {
            background: #f1f5f9; border: 1px solid var(--rb-border); padding: 2px 6px;
            border-radius: 4px; font-family: 'Fira Code', monospace; font-size: 12px; font-weight: 700; color: var(--rb-text-strong);
        }
        .rb-demo-wrapper .rb-highlight-box {
            background: #fffbeb; border-left: 4px solid var(--rb-amber); padding: 12px 16px;
            border-radius: 0 8px 8px 0; color: #92400e; margin: 16px 0;
        }
        .rb-demo-wrapper .rb-math-box {
            background: #f8fafc; border: 1px solid var(--rb-border); padding: 16px;
            border-radius: 8px; text-align: center; font-family: 'Cambria Math', serif; font-size: 16px;
            margin: 16px 0; color: var(--rb-text-strong); font-weight: bold;
        }

    </style>

    <div class="rb-header-title" id="rb-top-anchor">ACT 核心：自适应算力张量流</div>
    <div class="rb-header-desc">摒弃繁杂连线，用「色彩共振」直击 <code>torch.where()</code> 的张量切割原理</div>

    <div class="rb-panel">
        
        <!-- 实时日志 -->
        <div class="rb-action-log" id="rb-log-box">
            <svg style="width:22px; height:22px; min-width:22px; margin-top:2px;" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            <div id="rb-log-text"></div>
        </div>

        <div class="rb-layout">
            
            <!-- 左侧：代码区 -->
            <div class="rb-code-block">
                <div class="rb-cl step-1"><div class="rb-line-num">1</div><span class="rb-c-cmt"># 1. 预测当前停机概率 & 标记存活</span></div>
                <div class="rb-cl step-1"><div class="rb-line-num">2</div>p = torch.<span class="rb-c-fn">sigmoid</span>(self.halt(h))</div>
                <div class="rb-cl step-1"><div class="rb-line-num">3</div>still_running = ~halted</div>
                
                <div class="rb-cl"><div class="rb-line-num">4</div></div>
                
                <div class="rb-cl step-2"><div class="rb-line-num">5</div><span class="rb-c-cmt"># 2. 计算剩余安全额度 & 判断越界</span></div>
                <div class="rb-cl step-2"><div class="rb-line-num">6</div>remainder = <span class="rb-c-num">1.0</span> - cumulative_p</div>
                <div class="rb-cl step-2"><div class="rb-line-num">7</div>cond = cumulative_p + p >= <span class="rb-c-num">0.99</span></div>
                
                <div class="rb-cl"><div class="rb-line-num">8</div></div>
                
                <div class="rb-cl step-3"><div class="rb-line-num">9</div><span class="rb-c-cmt"># 3. 【核心】张量路由拼接组装</span></div>
                <div class="rb-cl step-3"><div class="rb-line-num">10</div>weight_raw = torch.<span class="rb-c-fn">where</span>(cond, remainder, p)</div>
                
                <div class="rb-cl"><div class="rb-line-num">11</div></div>

                <div class="rb-cl step-4"><div class="rb-line-num">12</div><span class="rb-c-cmt"># 4. 彻底抹杀已死 Token 的权重</span></div>
                <div class="rb-cl step-4"><div class="rb-line-num">13</div>weight = weight_raw * still_running</div>
                
                <div class="rb-cl"><div class="rb-line-num">14</div></div>

                <div class="rb-cl step-5"><div class="rb-line-num">15</div><span class="rb-c-cmt"># 5. 特征融合与内存状态更新</span></div>
                <div class="rb-cl step-5"><div class="rb-line-num">16</div>h_out += weight * h</div>
                <div class="rb-cl step-5"><div class="rb-line-num">17</div>cumulative_p += p * still_running</div>
                <div class="rb-cl step-5"><div class="rb-line-num">18</div>halted = halted | (cumulative_p >= <span class="rb-c-num">0.99</span>)</div>
            </div>

            <!-- 右侧：张量数据流 -->
            <div class="rb-tensor-area">
                
                <!-- 全局寄存器 (内存) -->
                <div class="rb-reg-box">
                    <div class="rb-reg-header">
                        <span style="font-size:12px; font-weight:800;">Global Memory (跨 Loop 共享)</span>
                        <div class="rb-token-header">
                            <div class="rb-token-id">T0</div><div class="rb-token-id">T1</div><div class="rb-token-id">T2</div><div class="rb-token-id">T3</div>
                        </div>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:6px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div class="rb-t-label">cumulative_p</div>
                            <div class="rb-tensor">
                                <div class="rb-cell" id="v-cp-0">0.85</div>
                                <div class="rb-cell" id="v-cp-1">1.00</div>
                                <div class="rb-cell" id="v-cp-2">0.40</div>
                                <div class="rb-cell" id="v-cp-3">0.95</div>
                            </div>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div class="rb-t-label">halted</div>
                            <div class="rb-tensor">
                                <div class="rb-cell" id="v-ht-0">F</div>
                                <div class="rb-cell" id="v-ht-1">T</div>
                                <div class="rb-cell" id="v-ht-2">F</div>
                                <div class="rb-cell" id="v-ht-3">F</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Step 1: P & SR -->
                <div class="rb-t-row step-1">
                    <div class="rb-t-label">p <span>停机概率(本轮)</span></div>
                    <div class="rb-tensor">
                        <div class="rb-cell is-p glow-p">0.20</div>
                        <div class="rb-cell is-p">0.50</div>
                        <div class="rb-cell is-p glow-p">0.30</div>
                        <div class="rb-cell is-p">0.10</div>
                    </div>
                </div>
                <div class="rb-t-row step-1" style="margin-top:-8px;">
                    <div class="rb-t-label">still_running <span>(1=活, 0=死)</span></div>
                    <div class="rb-tensor">
                        <div class="rb-cell is-sr">1</div>
                        <div class="rb-cell is-sr">0</div>
                        <div class="rb-cell is-sr">1</div>
                        <div class="rb-cell is-sr">1</div>
                    </div>
                </div>

                <!-- Step 2: Remainder & Cond -->
                <div class="rb-t-row step-2">
                    <div class="rb-t-label">remainder <span>1.0 - cum_p</span></div>
                    <div class="rb-tensor">
                        <div class="rb-cell is-rem glow-rem">0.15</div>
                        <div class="rb-cell is-rem glow-rem">0.00</div>
                        <div class="rb-cell is-rem">0.60</div>
                        <div class="rb-cell is-rem glow-rem">0.05</div>
                    </div>
                </div>
                <div class="rb-t-row step-2" style="margin-top:-8px;">
                    <div class="rb-t-label">cond (Mask) <span>cum_p+p ≥ 0.99</span></div>
                    <div class="rb-tensor">
                        <div class="rb-cell is-cond">T</div>
                        <div class="rb-cell is-cond">T</div>
                        <div class="rb-cell">F</div>
                        <div class="rb-cell is-cond">T</div>
                    </div>
                </div>

                <!-- Step 3: Where Raw Weight -->
                <div class="rb-t-row step-3">
                    <div class="rb-t-label">weight_raw <span>torch.where组装</span></div>
                    <div class="rb-tensor">
                        <div class="rb-cell glow-rem"><span class="val-hide-s3">0.15</span></div>
                        <div class="rb-cell glow-rem kill-token"><span class="val-hide-s3">0.00</span></div>
                        <div class="rb-cell glow-p"><span class="val-hide-s3">0.30</span></div>
                        <div class="rb-cell glow-rem"><span class="val-hide-s3">0.05</span></div>
                    </div>
                </div>

                <!-- Step 4: Final Weight -->
                <div class="rb-t-row step-4 rb-row-final" style="margin-top: 5px;">
                    <div class="rb-t-label" style="color:var(--rb-red);">weight <span>最终用于更新的乘数</span></div>
                    <div class="rb-tensor">
                        <div class="rb-cell is-final"><span class="val-hide-s4">0.15</span></div>
                        <div class="rb-cell is-final kill-token"><span class="val-hide-s4">0.00</span></div>
                        <div class="rb-cell is-final"><span class="val-hide-s4">0.30</span></div>
                        <div class="rb-cell is-final"><span class="val-hide-s4">0.05</span></div>
                    </div>
                </div>

            </div>
        </div>

        <!-- 底部控制按钮 -->
        <div class="rb-controls">
            <button class="rb-btn rb-btn-secondary" onclick="rbReset()">🔄 重置状态</button>
            <button class="rb-btn rb-btn-primary" id="rb-btn-next" onclick="rbNext()">执行第一步：概率与掩码 ➔</button>
        </div>

    </div>

    <!-- ==========================================
         新增：深度学习笔记区 (Study Notes)
         ========================================== -->
    <div class="rb-notes-container">
        <div class="rb-notes-header">
            <svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path></svg>
            深度学习笔记：ACT 与 Remainder Trick 原理拆解
        </div>

        <div class="rb-notes-section">
            <div class="rb-notes-title">1. 什么是 ACT (自适应计算时间)？</div>
            <p class="rb-notes-p">
                传统的 Transformer 对待所有 Token 都是“众生平等”的：无论是极其简单的定冠词 "The"，还是需要深思熟虑的逻辑词 "Therefore"，网络都会让它们跑完所有的网络层。
            </p>
            <p class="rb-notes-p">
                在 <b>Looped Transformers（循环架构）</b>中，数据在同一层里反复迭代。ACT 机制赋予了模型<b>“懂的都懂，不懂再想”</b>的能力。它允许模型在同一个 Batch 内部，让简单的 Token 循环 1~2 次就“早退（Halting）”，而让复杂的 Token 持续思考直到达到最大循环次数（深度外推）。
            </p>
        </div>

        <div class="rb-notes-section">
            <div class="rb-notes-title">2. 核心变量的物理含义</div>
            <p class="rb-notes-p">
                • <span class="rb-tag">p</span> (Halting Probability)：经过 Sigmoid 激活的标量 (0~1)。它代表模型在当前这一轮循环中，认为自己<b>“已经想清楚了，可以停止思考”</b>的信心值。
            </p>
            <p class="rb-notes-p">
                • <span class="rb-tag">cumulative_p</span> (累积概率)：模型在之前的循环轮次中，输出的 <code>p</code> 值的总和。它被存储在全局寄存器中，跨越多个 Loop 传递。
            </p>
            <p class="rb-notes-p">
                • <span class="rb-tag">halted</span> (停机标记)：布尔值。当某个 Token 的 <code>cumulative_p >= 0.99</code> 时，该 Token 被宣告死亡（思考结束），不再参与后续特征的融合。
            </p>
        </div>

        <div class="rb-notes-section">
            <div class="rb-notes-title">3. 核心考点：为什么必须使用 Remainder Trick？</div>
            <div class="rb-highlight-box">
                <b>💡 思考一个溢出灾难：</b><br>
                假设某个复杂词，第 1 次循环给出 p=0.4，第 2 次给出 p=0.45，此时 <code>cumulative_p = 0.85</code>。<br>
                到了第 3 次循环，模型突然顿悟，给出了 p=0.50。如果直接相加，总概率会变成 <b>0.85 + 0.50 = 1.35</b>。
            </div>
            <p class="rb-notes-p">
                如果总概率变成了 1.35，这会导致最终融合输出的隐状态 <span class="rb-tag">h_out</span> 的尺度被放大了 1.35 倍。这不仅破坏了神经网络原本特征的统计分布（均值和方差），还会彻底阻断正常梯度的回传，导致模型训练崩溃。
            </p>
            <p class="rb-notes-p">
                <b>解决方案 (剩余概率补齐)：</b>我们必须强制保证所有循环步提取出来的特征权重之和<b>严格等于 1.0</b>。
            </p>
            <div class="rb-math-box">
                remainder = 1.0 - cumulative_p<br>
                weight = torch.where( cumulative_p + p ≥ 0.99, remainder, p )
            </div>
            <p class="rb-notes-p">
                正如上方交互矩阵中 <b>Step 3</b> 所示：一旦检测到加上当前的 <code>p</code> 会导致越界（条件为 True），我们就<b>抛弃网络吐出的 <code>p</code></b>，强行使用到达满分 1.0 所缺的最后那一点点缝隙（即 <span class="rb-tag">remainder</span>）作为这一步的权重。
            </p>
        </div>

        <div class="rb-notes-section">
            <div class="rb-notes-title">4. 抹杀幽灵数据：still_running 掩码</div>
            <p class="rb-notes-p">
                由于 GPU 矩阵并行计算的特性（SIMD），即使某个 Token（如矩阵中的 T1）早在上一轮就已经 <code>halted=True</code>，底层的 Transformer 块依然会对它进行无效的矩阵乘法，甚至会吐出一个无意义的概率 <code>p = 0.50</code>。
            </p>
            <p class="rb-notes-p">
                如果不加以拦截，这个幽灵 0.50 就会混入最终结果。因此，在 <b>Step 4</b> 中，我们必须将组装好的权重乘以 <span class="rb-tag">still_running</span> (即 <code>~halted</code> 的反转掩码)。这会将死 Token 的权重强行归零（0.00），彻底将其从最终的隐状态 <code>h_out</code> 的加权融合中踢出。
            </p>
        </div>

    </div>

    <script>
        (function() {
            let currentStep = 0;
            const root = document.getElementById('rb-act-root');
            const btn = document.getElementById('rb-btn-next');
            
            // 全局寄存器元素
            const cp0 = document.getElementById('v-cp-0'), cp1 = document.getElementById('v-cp-1'), cp2 = document.getElementById('v-cp-2'), cp3 = document.getElementById('v-cp-3');
            const ht0 = document.getElementById('v-ht-0'), ht1 = document.getElementById('v-ht-1'), ht2 = document.getElementById('v-ht-2'), ht3 = document.getElementById('v-ht-3');

            const logs = [
                {
                    msg: "<b>初始化完毕。</b> 场景：Batch Size = 4。在进入本次循环前，T1 的累积概率已满 (halted=True)，其余 Token 仍在思考。请点击下方执行按钮。",
                    btn: "执行第一步：预测与掩码 ➔",
                    color: "#eff6ff", border: "#93c5fd", text: "#1d4ed8"
                },
                {
                    msg: "<b>Step 1: 预测与掩码。</b> 模型输出本轮停机概率 <code>p</code>。同时计算出活跃标记 <code>still_running</code>。注意 T1 虽已死，但矩阵仍算出了无效的 p=0.50，需稍后清理。",
                    btn: "执行第二步：边界判定 ➔",
                    color: "#eff6ff", border: "#93c5fd", text: "#1d4ed8"
                },
                {
                    msg: "<b>Step 2: 边界判定。</b> 为防止概率总和超载，计算到达 1.0 的安全额度 <code>remainder</code>。随后判断如果加上本轮 p 是否越界。T0、T1、T3 判定为越界 (True)。",
                    btn: "执行第三步：Torch.Where 路由 ➔",
                    color: "#eff6ff", border: "#93c5fd", text: "#1d4ed8"
                },
                {
                    msg: "<b>Step 3: 张量色彩路由 (Torch.Where)。</b> 核心操作！请看同色发光区：T0/T1/T3 (黄色 True 越界) 从绿色的 <code>remainder</code> 处截取数值；T2 (灰色 False 未越界) 从蓝色的 <code>p</code> 处取值。完美组装出初始权重！",
                    btn: "执行第四步：死 Token 阻断 ➔",
                    color: "#dcfce7", border: "#86efac", text: "#15803d" // Success Green
                },
                {
                    msg: "<b>Step 4: 死 Token 阻断。</b> 刚才组装的矩阵中包含了已死的 T1 (0.00)。将矩阵乘以 <code>still_running</code>，T1 的权重被彻底抹杀 (变暗)。最终完美更新权重诞生！",
                    btn: "执行收尾：更新寄存器 ➔",
                    color: "#eff6ff", border: "#93c5fd", text: "#1d4ed8"
                },
                {
                    msg: "<b>Step 5: 状态更新。</b> 最终权重被用于累加 <code>h_out</code>。上方内存区被更新：T0 和 T3 的累加概率精准停在 1.00，它们的 <code>halted</code> 变为 True！下一轮它们将和 T1 一样被屏蔽。",
                    btn: "🔄 演示结束，点击重置",
                    color: "#fffbeb", border: "#fde68a", text: "#b45309" // Warn Amber
                }
            ];

            function applyLog(step) {
                const logData = logs[step];
                const logBox = document.getElementById('rb-log-box');
                document.getElementById('rb-log-text').innerHTML = logData.msg;
                logBox.style.backgroundColor = logData.color;
                logBox.style.borderColor = logData.border;
                logBox.style.color = logData.text;
                
                btn.innerHTML = logData.btn;
                if(step === 5) {
                    btn.classList.replace('rb-btn-primary', 'rb-btn-secondary');
                } else {
                    btn.classList.replace('rb-btn-secondary', 'rb-btn-primary');
                }
            }

            window.rbNext = function() {
                if(currentStep >= 5) {
                    rbReset();
                    return;
                }
                currentStep++;
                root.setAttribute('data-step', currentStep);
                applyLog(currentStep);
                
                // 平滑滚回顶部，方便对照代码
                document.getElementById('rb-top-anchor').scrollIntoView({ behavior: 'smooth', block: 'start' });

                // Step 5 专属 DOM 变动 (寄存器闪烁更新)
                if (currentStep === 5) {
                    cp0.innerText = "1.00"; cp0.classList.add('update-glow');
                    cp2.innerText = "0.70"; cp2.classList.add('update-glow');
                    cp3.innerText = "1.00"; cp3.classList.add('update-glow');
                    ht0.innerText = "T"; ht0.classList.add('update-glow');
                    ht3.innerText = "T"; ht3.classList.add('update-glow');
                }
            };

            window.rbReset = function() {
                currentStep = 0;
                root.setAttribute('data-step', 0);
                applyLog(0);
                document.getElementById('rb-top-anchor').scrollIntoView({ behavior: 'smooth', block: 'start' });

                // 还原寄存器
                cp0.innerText = "0.85"; cp0.classList.remove('update-glow');
                cp2.innerText = "0.40"; cp2.classList.remove('update-glow');
                cp3.innerText = "0.95"; cp3.classList.remove('update-glow');
                ht0.innerText = "F"; ht0.classList.remove('update-glow');
                ht3.innerText = "F"; ht3.classList.remove('update-glow');
            };

            // 初始化日志
            applyLog(0);
        })();
    </script>
</div>