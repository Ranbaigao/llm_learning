# 前端可视化组件 UI 设计规范 (Deepspeed 浅色风格)

为了保证项目中所有的独立 HTML 交互组件、图解演示（例如放置在 `.assets/` 或其他文档目录中的独立 HTML 页面）在视觉上高度一致，特制定此 UI 设计规范。该规范提取自高质量参考文件 `Deepspeed_zero2.html`，以“科技清新、数据可读”为主要基调。

## 1. 核心色彩体系 (CSS Variables)

在编写新的 HTML 组件时，请优先使用（或硬编码对应颜色值）以下 CSS 变量体系，避免使用高纯度的大红大绿，或沉闷的纯黑纯白。

```css
:root {
  /* 页面基础层 */
  --bg-color: #f8fafc;       /* 整体背景 (Tailwind slate-50)，代替旧的深色背景 */
  --text-color: #334155;     /* 正文文字 (Tailwind slate-700) */
  --title-color: #0f172a;    /* 标题文字 (Tailwind slate-900) */
  
  /* 容器与面板 */
  --panel-bg: #ffffff;       /* 浮窗、卡片、控制面板背景 */
  --border-color: #cbd5e1;   /* 柔和的边框色 (Tailwind slate-300) */
  --panel-shadow: 0 4px 6px rgba(0,0,0,0.05); /* 卡片投影 */
  
  /* 强调色与点缀色 (Accent) */
  --accent-primary: #3b82f6; /* 主强调色，科技蓝 (Tailwind blue-500) */
  --accent-text: #1e40af;    /* 强调文本色 (Tailwind blue-800) */
  --accent-light: #eff6ff;   /* 强调背景底色 (Tailwind blue-50) */
  
  /* 状态色 (Semantic) */
  --success: #16a34a;        /* 成功/正确 (green-600) */
  --danger: #dc2626;         /* 危险/错误 (red-600) */
  --warning: #d97706;        /* 警告/高亮 (amber-600) */
  --muted: #64748b;          /* 弱化文本/注释 (slate-500) */
}
```

## 2. 排版与字体规范

- **字体族 (Font Family)**: `font-family: 'Segoe UI', Tahoma, Geneva, Verdana, system-ui, sans-serif;`
- **行高 (Line Height)**: 正文保持 `1.6` 或 `1.7` 的行高，增加阅读舒适度。
- **标题体系**: 
  - `h1`: 居中，`color: var(--title-color)`，字号 `1.6rem` - `1.8rem`。
  - 次级标题/面板标题: `color: var(--accent-text)`，可以使用 `border-left: 4px solid var(--accent-primary);` 作为左侧修饰。

## 3. 组件化样式标准

### 3.1 容器与卡片面板
凡是属于“控制面板”、“侧边栏”、“数据展示块”的 `div` 容器，必须满足：
```css
.panel {
  background: var(--panel-bg); /* #ffffff */
  border: 1px solid var(--border-color); /* #cbd5e1 */
  border-radius: 8px; /* 或 12px */
  box-shadow: var(--panel-shadow);
  padding: 16px 20px;
}
```

### 3.2 交互按钮 (Button)
```css
button {
  background-color: var(--accent-primary);
  color: white;
  border: none;
  border-radius: 6px;
  padding: 8px 16px;
  cursor: pointer;
  transition: background 0.2s;
}
button:hover { background-color: #2563eb; /* blue-600 */ }
```

### 3.3 数据表格与矩阵格 (Matrix Cells)
对于矩阵、多维数组展示，单元格应：
- 背景色: 纯白 `#ffffff` 或非常浅的灰色 `#f1f5f9`。
- 边框: `#cbd5e1`。
- 如果需要通过颜色表示权重大小（Heatmap），请使用 `rgba(59, 130, 246, 权重)`，即蓝色调的透明度渐变，**不要**使用旧的青色或纯黑底色。

### 3.4 代码与公式 (Code & Math)
- 内联代码 `code`: 背景 `#f1f5f9`，字体颜色 `#d97706` (黄色强调)。
- 公式块/大段代码: 背景 `#f8fafc` 或 `#ffffff`，左侧增加蓝色粗边框作为强调。

## 4. 3D 渲染场景 (Three.js 规范)

如果 HTML 包含基于 Three.js 或 WebGL 的渲染画布：
1. **背景色**: 抛弃深色背景模式 (`0x0f1117`)，使用浅色背景或浅色径向渐变：
   ```css
   /* CSS 容器 */
   #canvas-container { background: radial-gradient(circle at center, #ffffff 0%, #f1f5f9 100%); }
   ```
   ```javascript
   // JS Render
   renderer.setClearColor(0xf8fafc, 1);
   ```
2. **标签与 HTML Overlay**: Canvas 上悬浮的 DOM 标签需使用高对比度的白底黑字：
   ```css
   .label-3d {
     background: rgba(255, 255, 255, 0.9);
     color: #334155;
     border: 1px solid #cbd5e1;
     box-shadow: 0 2px 4px rgba(0,0,0,0.05);
   }
   ```
3. **灯光 (Lighting)**: 浅色背景下，环境光 (`AmbientLight`) 的强度建议适当提高至 `0.8 - 1.0`，主光源 (`DirectionalLight`) 需要适当调弱以免模型过曝。

## 总结指令 (AI Agent Rule)

当你（AI）被要求**新建或重构任何前端演示页面 (.html)** 时：
1. 必须查阅本文档，严格套用这里的 `slate` 与 `blue` 系调色板。
2. 严禁自行发挥写死 `background: #000`, `color: #fff` 或类似极客暗黑风，除非用户明确要求切换暗黑模式。
3. 确保所有新加的可视化页面在 MkDocs 的亮色主题中能够完美融合。
