/**
 * 知识星图 Canvas 引擎（框架无关）
 *
 * 移植自 content/.assets/knowledge_graph.html 的内联脚本：物理布局（螺旋臂 + 星团云 +
 * 排斥力后处理）、Canvas 绘制（星空背景/节点/连线/标签/纹章）与输入交互（旋转/平移/缩放/
 * 悬停/固定/双击打开）全部封装在本类中；DOM 面板（目录树/榜单/搜索框）由 Vue 组件承接，
 * 通过构造函数传入的回调与引擎通信。
 *
 * 交互事件流：
 * - 画布双击笔记节点        → onOpenNote(node)
 * - 画布单击固定 / 取消固定 → onPinChange(nodeId | null)（侧栏据此高亮）
 * - 画布单击空白            → onClearSearch()（Vue 清空搜索框与候选列表）
 */

// ======= 数据类型 =======

export type GraphNodeType = 'root' | 'category' | 'subcategory' | 'note'

/** 后端 GET /api/graph 返回的原始节点 */
export interface GraphNodeInput {
  id: string
  name: string
  type: GraphNodeType
  depth: number
  url?: string
  size?: number
  note_count?: number
  format?: string
  mtime?: number
  pv?: number | null
}

export interface GraphLinkInput {
  source: string
  target: string
  type: string
}

export interface GraphStats {
  nodes?: number
  links?: number
  notes?: number
  jupyter?: number
  contains?: number
  references?: number
}

export interface GraphDataInput {
  nodes: GraphNodeInput[]
  links: GraphLinkInput[]
  stats?: GraphStats
}

interface ScreenPoint {
  x: number
  y: number
  z: number
  scale: number
}

/** 引擎内部使用的节点（原始字段 + 布局/渲染状态） */
export interface EngineNode extends GraphNodeInput {
  index: number
  radius: number
  branchId: string
  color: string
  branchAngle: number
  x: number
  y: number
  z: number
  tx: number
  ty: number
  tz: number
  screen: ScreenPoint | null
  pulse: number
  scale: number
  labelX: number
  labelY: number
  labelEl: HTMLDivElement | null
  hitRadius: number
}

interface EngineLink {
  index: number
  type: string
  source: EngineNode
  target: EngineNode
}

interface DragState {
  mode: 'pan' | 'rotate'
  x: number
  y: number
  rotX: number
  rotY: number
  panX: number
  panY: number
  moved: boolean
}

export interface StarMapEngineCallbacks {
  /** 双击笔记节点 / 搜索结果回车：打开笔记页 */
  onOpenNote?: (node: EngineNode) => void
  /** 固定状态变化（含取消固定）：侧栏同步高亮 */
  onPinChange?: (nodeId: string | null) => void
  /** 引擎内部清空了搜索状态（点空白 / 侧栏选中），Vue 同步清空搜索框 */
  onClearSearch?: () => void
}

export interface StarMapEngineOptions {
  canvas: HTMLCanvasElement
  labelLayer: HTMLElement
  tooltip: HTMLElement
  /** 根节点纹章图片 URL（/api/assets/root_badge.png） */
  badgeUrl: string
  callbacks?: StarMapEngineCallbacks
}

// ======= 常量（与旧版一致）=======

export const TYPE_LABEL: Record<string, string> = {
  root: 'ROOT',
  category: 'CATEGORY',
  subcategory: 'SUBCATEGORY',
  note: 'NOTE',
};

const ROOT_COLOR = '#f1f5f9'; // 旧版取 cssVar('--root-color')

// 星座色族：每个星座是一对相近色相（base → shift，色相差很小）。
// 簇内节点颜色按空间方位角在两色之间平滑插值——同一星团内颜色连续渐变，
// 不会出现相邻节点色相突变（参考真实星系：同一区域内恒星颜色平滑过渡）
const CONSTELLATION_FAMILIES: Record<string, { base: string; shift: string }> = {
  NLP: { base: '#60a5fa', shift: '#38bdf8' },
  CV: { base: '#2dd4bf', shift: '#5eead4' },
  性能优化: { base: '#fbbf24', shift: '#fcd34d' },
  代码实践: { base: '#22d3ee', shift: '#67e8f9' },
};
const FALLBACK_CONSTELLATION_COLORS = [
  '#60a5fa',
  '#4ade80',
  '#fbbf24',
  '#a78bfa',
  '#22d3ee',
  '#f472b6',
];

// 顶层目录固定展示顺序（侧边栏与星图布局一致）：NLP → CV → 性能优化 → 代码实践
const TOP_LEVEL_ORDER = ['NLP', 'CV', '性能优化', '代码实践'];

// 一级目录题名（drawNodes 中悬浮绘制在目录星正上方）：Share Tech 科技风无衬线英文 + 辉光脉动
const EMBLEM_TEXT: Record<string, string> = { CV: 'CV', NLP: 'NLP', 性能优化: 'PERF', 代码实践: 'CODE' };

// 默认俯仰角：饼状盘面接近水平的正侧方视角（1.18 偏俯视，此处压低到近水平）
const DEFAULT_ROT_X = 0.2;
const DEFAULT_ROT_Y = -0.28;
// 初始取景比适距更近一档，约等于滚轮向前滚动三次（deltaY=100 时单次 ≈ ×1.116）
const INITIAL_ZOOM_FACTOR = 2.4;

const STAR_SPRITE_RES = 256;

// ======= 纯工具函数 =======

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const normalized = hex.trim().replace('#', '');
  const value = Number.parseInt(
    normalized.length === 3
      ? normalized.split('').map(ch => ch + ch).join('')
      : normalized,
    16,
  );
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  };
}

function rgbToHex({ r, g, b }: { r: number; g: number; b: number }): string {
  return '#' + [r, g, b]
    .map(value => Math.round(clamp(value, 0, 255)).toString(16).padStart(2, '0'))
    .join('');
}

function mixHex(from: string, to: string, amount: number): string {
  const a = hexToRgb(from);
  const b = hexToRgb(to);
  const t = clamp(amount, 0, 1);
  return rgbToHex({
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
  });
}

function rgba(color: string, alpha: number): string {
  const rgb = hexToRgb(color);
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

function hashUnit(value: unknown, salt = 0): number {
  let hash = 2166136261 ^ salt;
  for (const char of String(value)) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function branchIdFor(node: Pick<GraphNodeInput, 'id'> | null | undefined): string {
  if (!node || node.id === '__root__') return '__root__';
  return String(node.id).split('/')[0] || String(node.id);
}

function fallbackConstellationColor(branchId: string): string {
  let hash = 0;
  for (const char of String(branchId)) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return FALLBACK_CONSTELLATION_COLORS[hash % FALLBACK_CONSTELLATION_COLORS.length];
}

// 星座色族查询：未知分支回退到调色板取 base，shift 由 base 混白推导（保持同族相近色相）
function familyFor(branchId: string): { base: string; shift: string } {
  if (CONSTELLATION_FAMILIES[branchId]) return CONSTELLATION_FAMILIES[branchId];
  const base = fallbackConstellationColor(branchId);
  return { base, shift: mixHex(base, '#ffffff', 0.22) };
}

// 布局前的初始配色；布局完成后由 applyClusterColors() 按空间位置重算。
// 侧边栏树/榜单的圆点颜色用的就是这个初始配色（与旧版 buildSidebar 先于 seedConstellation 一致）。
export function colorForNode(node: Pick<GraphNodeInput, 'id' | 'type'> | null | undefined): string {
  if (!node || node.type === 'root') return ROOT_COLOR;
  const family = familyFor(branchIdFor(node));
  if (node.type === 'category') return mixHex(family.base, '#ffffff', 0.34);
  if (node.type === 'subcategory') return mixHex(mixHex(family.base, family.shift, 0.4), '#ffffff', 0.18);
  return mixHex(family.base, '#ffffff', 0.6);
}

function radiusFor(node: GraphNodeInput): number {
  if (node.type === 'root') return 11;
  if (node.type === 'category') return Math.max(8, Math.min(12, (node.size ?? 0) * 0.5 || 9));
  if (node.type === 'subcategory') return Math.max(6, Math.min(9, (node.size ?? 0) * 0.44 || 7));
  return Math.max(3.2, Math.min(4.6, (node.size ?? 0) * 0.52 || 3.8));
}

/** 目录树/榜单共用的子节点排序：顶层按 TOP_LEVEL_ORDER，其余按类型再按名称 */
function orderedChildrenOf<T extends { id: string; name: string; type: string }>(
  childrenById: Map<string, T[]>,
  parentId: string,
): T[] {
  return [...(childrenById.get(parentId) || [])].sort((a, b) => {
    if (parentId === '__root__') {
      const oa = TOP_LEVEL_ORDER.indexOf(a.id);
      const ob = TOP_LEVEL_ORDER.indexOf(b.id);
      if (oa !== ob) return (oa === -1 ? TOP_LEVEL_ORDER.length : oa) - (ob === -1 ? TOP_LEVEL_ORDER.length : ob);
      return a.name.localeCompare(b.name, 'zh-Hans-CN');
    }
    const rank: Record<string, number> = { category: 0, subcategory: 1, note: 2 };
    const dr = (rank[a.type] ?? 9) - (rank[b.type] ?? 9);
    if (dr !== 0) return dr;
    return a.name.localeCompare(b.name, 'zh-Hans-CN');
  });
}

export function formatCount(value: unknown): string {
  const n = Number(value) || 0;
  return n >= 10000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

export function formatRelativeTime(ts: number): string {
  const days = Math.floor((Date.now() / 1000 - ts) / 86400);
  if (days <= 0) return '今天';
  if (days === 1) return '昨天';
  if (days < 30) return `${days} 天前`;
  if (days < 365) return `${Math.floor(days / 30)} 个月前`;
  return `${Math.floor(days / 365)} 年前`;
}

// ======= 侧栏目录树模型（由原始图谱数据构建，供 Vue 组件渲染）=======

export interface SidebarTreeNode {
  id: string
  name: string
  type: GraphNodeType
  color: string
  noteCount: number
  pv: number | null
  url: string
  children: SidebarTreeNode[]
}

export function buildSidebarTree(data: GraphDataInput): SidebarTreeNode[] {
  const byId = new Map(data.nodes.map(node => [node.id, node]));
  const childrenById = new Map<string, GraphNodeInput[]>();
  for (const node of data.nodes) childrenById.set(node.id, []);
  for (const link of data.links) {
    if (link.type !== 'contains') continue;
    const child = byId.get(link.target);
    const list = childrenById.get(link.source);
    if (child && list) list.push(child);
  }
  const build = (node: GraphNodeInput): SidebarTreeNode => ({
    id: node.id,
    name: node.name,
    type: node.type,
    color: colorForNode(node),
    noteCount: node.note_count || 0,
    pv: node.pv ?? null,
    url: node.url || '',
    children: orderedChildrenOf(childrenById, node.id).map(build),
  });
  return orderedChildrenOf(childrenById, '__root__').map(build);
}

// ======= 引擎本体 =======

export class StarMapEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private labelLayer: HTMLElement;
  private tooltip: HTMLElement;
  private callbacks: StarMapEngineCallbacks;
  private abort = new AbortController();

  private width = 0;
  private height = 0;
  private dpr = 1;
  private nodes: EngineNode[] = [];
  private links: EngineLink[] = [];
  private stars: Array<Record<string, any>> = [];
  private dustParticles: Array<Record<string, any>> = [];
  private meteors: Array<Record<string, any>> = [];
  private nextMeteorAt = 8;
  private nodeById = new Map<string, EngineNode>();
  private adjacency = new Map<string, Set<string>>();
  private childrenById = new Map<string, EngineNode[]>();
  private parentById = new Map<string, EngineNode>();
  private focusNode: EngineNode | null = null;
  private focusIds = new Set<string>();
  private focusChildIds = new Set<string>(); // 焦点的直接子节点：悬停时只给它们显示名字
  // 焦点来源：'pointer'（悬停，高亮 + 显示自身/子节点/父节点名字）/ 'pin'（星图或侧边栏选中）/ 'search'（搜索选中）
  // 完整文字标签（含父节点/引用节点）只在锁定选中（pin / search）时浮现，默认状态下星图无文字
  private focusSource: 'pointer' | 'pin' | 'search' | null = null;
  private searchMatches = new Set<string>();
  private searchQuery = '';
  private hoveredNode: EngineNode | null = null;
  private isHoveringNode = false;
  private pinnedNode: EngineNode | null = null;
  private pressedNode: EngineNode | null = null;
  private paused = false;
  private dragState: DragState | null = null;
  private frameHandle = 0;
  private lastTime = 0;
  private time = 0;
  private constellationRotation = 0;
  // 视角旋转中心（世界坐标）：默认整个星图的重心，固定节点后平滑移动到该节点
  private viewCenter = { x: 0, y: 0, z: 0 };
  // 星图重心：布局完成后由全部节点位置平均得到，作为默认旋转中心
  private graphCenter = { x: 0, y: 0, z: 0 };
  private layoutSeed = 0;
  private camera = {
    rotX: DEFAULT_ROT_X,
    rotY: DEFAULT_ROT_Y,
    zoom: 1,
    panX: 0,
    panY: 14,
    distance: 980,
  };

  // ======= 星空精灵图缓存 =======
  // 光晕 / 星云 / 暗角均为预渲染的离屏 canvas，绘制时只 drawImage，
  // 避免每帧 createRadialGradient 与 shadowBlur 带来的性能开销。
  private haloSpriteCache = new Map<string, HTMLCanvasElement>();
  private milkyWayCanvas: HTMLCanvasElement | null = null; // 静态银河带（resize / relayout 时重绘）
  private nebulaCanvas: HTMLCanvasElement | null = null;   // 静态星云层（resize / relayout 时重绘）
  private vignetteCanvas: HTMLCanvasElement | null = null; // 暗角后期层

  // 根节点纹章（drawNodes 中悬浮绘制在根星正上方，带全息光束连接）：自带透明底，此处仅预加载
  private rootBadgeImg = new Image();
  private rootBadgeReady = false;

  constructor(options: StarMapEngineOptions) {
    this.canvas = options.canvas;
    this.labelLayer = options.labelLayer;
    this.tooltip = options.tooltip;
    this.callbacks = options.callbacks || {};
    const ctx = this.canvas.getContext('2d', { alpha: true });
    if (!ctx) throw new Error('无法创建 2D 渲染上下文');
    this.ctx = ctx;

    this.rootBadgeImg.onload = () => { this.rootBadgeReady = true; };
    this.rootBadgeImg.src = options.badgeUrl;

    this.bindEvents();
  }

  // ======= 公共 API =======

  /** 装载数据并开启动画（对应旧版 buildGraph，不含 DOM 面板构建） */
  setData(data: GraphDataInput): void {
    cancelAnimationFrame(this.frameHandle);
    this.measure();

    this.nodeById = new Map();
    this.nodes = (Array.isArray(data.nodes) ? data.nodes : []).map((node, index) => {
      const next: EngineNode = {
        depth: 0,
        ...node,
        index,
        radius: radiusFor(node),
        branchId: branchIdFor(node),
        color: colorForNode(node),
        branchAngle: 0,
        x: 0,
        y: 0,
        z: 0,
        tx: 0,
        ty: 0,
        tz: 0,
        screen: null,
        pulse: Math.random() * Math.PI * 2,
        scale: 1,
        labelX: NaN,
        labelY: NaN,
        labelEl: null,
        hitRadius: 10,
      };
      this.nodeById.set(next.id, next);
      return next;
    });

    this.links = (Array.isArray(data.links) ? data.links : [])
      .map((link, index) => ({
        index,
        type: link.type,
        source: this.nodeById.get(link.source),
        target: this.nodeById.get(link.target),
      }))
      .filter((link): link is EngineLink => !!link.source && !!link.target);

    this.adjacency = new Map(this.nodes.map(node => [node.id, new Set<string>()]));
    this.childrenById = new Map(this.nodes.map(node => [node.id, [] as EngineNode[]]));
    this.parentById = new Map();
    for (const link of this.links) {
      this.adjacency.get(link.source.id)?.add(link.target.id);
      this.adjacency.get(link.target.id)?.add(link.source.id);
      if (link.type === 'contains') {
        this.childrenById.get(link.source.id)?.push(link.target);
        this.parentById.set(link.target.id, link.source);
      }
    }

    this.createGraphLabels();
    this.seedConstellation();
    this.createBackgroundStars();
    this.resetView();
    this.clearFocus();
    this.lastTime = performance.now();
    this.frameHandle = requestAnimationFrame(this.animate);
  }

  /** 容器尺寸变化：重测量、重绘背景、重新取景 */
  resize(): void {
    this.measure();
    if (!this.nodes.length) return;
    this.createBackgroundStars();
    this.resetView();
  }

  dispose(): void {
    cancelAnimationFrame(this.frameHandle);
    this.abort.abort();
    this.labelLayer.replaceChildren();
  }

  resetView(): void {
    this.camera.rotX = DEFAULT_ROT_X;
    this.camera.rotY = DEFAULT_ROT_Y;
    this.constellationRotation = 0;
    this.camera.panX = 0;
    this.camera.panY = 14;
    // 用 92 分位半径取景（相对星图重心），避免个别大分支的边缘星把整体缩放逐小；
    // 再乘 INITIAL_ZOOM_FACTOR 拉近到"滚轮前滚三次"的初始距离
    const radii = this.nodes.map(node => Math.hypot(
      node.tx - this.graphCenter.x,
      node.ty - this.graphCenter.y,
      node.tz - this.graphCenter.z,
    )).sort((a, b) => a - b);
    const percentileR = radii.length ? radii[Math.floor(0.92 * (radii.length - 1))] : 520;
    const maxR = Math.max(520, percentileR);
    this.camera.zoom = clamp(Math.min(this.width, this.height) / (maxR * 1.8) * INITIAL_ZOOM_FACTOR, 0.56, 2.0);
  }

  relayout(): void {
    this.layoutSeed += 1; // 每次重排生成一个全新的星系
    this.seedConstellation();
    this.createBackgroundStars();
    this.resetView();
  }

  setPaused(nextPaused: boolean): void {
    this.paused = nextPaused;
  }

  get isPaused(): boolean {
    return this.paused;
  }

  /** 侧栏/榜单选中：清空搜索态并固定高亮该节点（对应旧版 selectFromSidebar + pinNode） */
  pinNodeById(id: string, { syncSidebar = true }: { syncSidebar?: boolean } = {}): void {
    const node = this.nodeById.get(id);
    if (!node) return;
    if (this.searchQuery) {
      this.searchQuery = '';
      this.searchMatches = new Set();
      this.callbacks.onClearSearch?.();
    }
    this.pinNode(node, { syncSidebar });
  }

  /** 搜索框输入：更新内部匹配集，返回排序后的候选节点（对应旧版 input 事件） */
  setSearchQuery(query: string): EngineNode[] {
    this.pinnedNode = null; // 搜索接管焦点，取消固定状态
    this.searchQuery = query.trim().toLowerCase();
    const matchedNodes = this.searchQuery
      ? this.nodes.filter(node => this.searchable(node).includes(this.searchQuery))
      : [];
    this.searchMatches = this.searchQuery ? new Set(matchedNodes.map(node => node.id)) : new Set();

    if (!this.searchQuery) {
      this.clearFocus();
      return [];
    }

    const preferredResults = matchedNodes.filter(node => node.type === 'note');
    const rankedResults = (preferredResults.length ? preferredResults : matchedNodes)
      .filter(node => node.id !== '__root__')
      .sort((a, b) => {
        const byName = a.name.localeCompare(b.name, 'zh-Hans-CN');
        if (byName !== 0) return byName;
        return a.id.localeCompare(b.id, 'zh-Hans-CN');
      });

    if (rankedResults.length) {
      this.setFocus(rankedResults[0], false, 'search');
    } else {
      this.focusNode = null;
      this.focusSource = null;
      this.hoveredNode = null;
      this.focusIds = new Set();
      this.hideTooltip();
    }
    return rankedResults;
  }

  /** 搜索候选 hover / 方向键导航：仅移动焦点 */
  focusSearchNode(node: EngineNode): void {
    this.setFocus(node, false, 'search');
  }

  // ======= 布局 =======

  private measure(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(320, rect.width);
    this.height = Math.max(420, rect.height);
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.floor(this.width * this.dpr);
    this.canvas.height = Math.floor(this.height * this.dpr);
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  private orderedChildren(parentId: string): EngineNode[] {
    return orderedChildrenOf(this.childrenById, parentId);
  }

  private seedConstellation(): void {
    const root = this.nodeById.get('__root__');
    const categories = this.orderedChildren('__root__');
    const orbitScaleZ = 0.72; // 银盘在 z 方向压扁
    const DISC_FLATTEN_Y = 0.45; // 星团云在盘面法线方向压扁

    if (root) {
      root.tx = 0;
      root.ty = 0;
      root.tz = 0;
      root.branchAngle = 0;
    }

    // 以节点 id + layoutSeed 为种子的近似高斯（Box-Muller），保证布局可复现
    const gaussish = (value: unknown, salt: number): number => {
      const u = Math.max(1e-6, hashUnit(value, salt + this.layoutSeed * 131));
      const v = hashUnit(value, salt + 101 + this.layoutSeed * 131);
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
    };

    // 均匀球面方向（由哈希驱动）
    const sphereDir = (value: unknown, salt: number): { x: number; y: number; z: number } => {
      const phi = hashUnit(value, salt + this.layoutSeed * 131) * Math.PI * 2;
      const cosT = hashUnit(value, salt + 57 + this.layoutSeed * 131) * 2 - 1;
      const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
      return { x: sinT * Math.cos(phi), y: cosT, z: sinT * Math.sin(phi) };
    };

    const subtreeWeight = (node: EngineNode): number => {
      let weight = 0;
      const stack = [...(this.childrenById.get(node.id) || [])];
      while (stack.length) {
        const current = stack.pop()!;
        weight += current.type === 'note' ? 1 : 2;
        stack.push(...(this.childrenById.get(current.id) || []));
      }
      return weight;
    };

    // 子级星团云：以父节点为引力中心的三维散布，无整体朝向（消灭漏斗）
    const placeCluster = (parent: EngineNode): void => {
      const children = this.orderedChildren(parent.id);
      if (!children.length) return;

      const count = children.length;
      // 大分支更"致密"而非更"庞大"：云半径按平方根亚线性增长，
      // 避免节点多的一级目录（如 llm）占据过多空间、拉偏星图重心
      const cloudRadius = parent.type === 'category'
        ? clamp(95 + Math.sqrt(count) * 30 + Math.sqrt(subtreeWeight(parent)) * 16, 130, 235)
        : clamp(70 + Math.sqrt(count) * 26, 100, 190);

      for (const child of children) {
        const isCore = child.type !== 'note';
        // 星团核（子目录）分布在 0.35~1.0 壳层，笔记向核心聚集（幂律密度轮廓）
        const u = hashUnit(child.id, 13 + this.layoutSeed * 131);
        const shell = isCore ? 0.35 + u * 0.65 : Math.pow(u, 0.9);
        const dir = sphereDir(child.id, 29);
        const dist = shell * cloudRadius * (isCore ? 1 : 0.82);

        child.branchAngle = parent.branchAngle;
        child.tx = parent.tx + dir.x * dist;
        child.ty = parent.ty + dir.y * dist * DISC_FLATTEN_Y;
        child.tz = parent.tz + dir.z * dist * orbitScaleZ;

        if (this.orderedChildren(child.id).length) placeCluster(child);
      }
    };

    // 顶层：一级目录分布在 2~3 条对数螺旋臂上（r = r0·e^(bθ)）
    const ARM_COUNT = categories.length <= 3 ? 2 : 3;
    const ARM_B = 0.22;
    const perArm = Math.ceil(categories.length / ARM_COUNT);
    categories.forEach((node, index) => {
      const arm = index % ARM_COUNT;
      const armIndex = Math.floor(index / ARM_COUNT);
      const t = perArm <= 1 ? 0.5 : armIndex / (perArm - 1);
      const armPhase = (arm / ARM_COUNT) * Math.PI * 2 + hashUnit(`arm:${arm}`, 5 + this.layoutSeed * 131) * 0.25;
      const sweep = 1.0 + t * 1.5;
      const theta = armPhase + sweep;
      const radius = 170 * Math.exp(ARM_B * sweep) + Math.min(48, subtreeWeight(node) * 3);
      // 银盘厚度随半径略增
      const thickness = 20 + radius * 0.05;

      node.branchAngle = theta;
      node.tx = Math.cos(theta) * radius;
      node.tz = Math.sin(theta) * radius * orbitScaleZ;
      node.ty = gaussish(node.id, 41) * thickness * 0.4;
    });

    // 一级目录核心之间的排斥：保证各星系彼此分得开（在铺开星团云之前只动核心）
    const MIN_CATEGORY_DIST = 280;
    for (let iter = 0; iter < 50; iter++) {
      for (let i = 0; i < categories.length; i++) {
        for (let j = i + 1; j < categories.length; j++) {
          const a = categories[i];
          const b = categories[j];
          const dx = b.tx - a.tx;
          const dy = b.ty - a.ty;
          const dz = b.tz - a.tz;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < MIN_CATEGORY_DIST * MIN_CATEGORY_DIST) {
            const d = Math.sqrt(d2) || 0.001;
            const push = ((MIN_CATEGORY_DIST - d) / d) * 0.25;
            a.tx -= dx * push; a.ty -= dy * push; a.tz -= dz * push;
            b.tx += dx * push; b.ty += dy * push; b.tz += dz * push;
          }
        }
      }
    }

    for (const node of categories) placeCluster(node);

    // 排斥力后处理：推开初始坐标过近的节点，再加 jitter
    this.applyRepulsion();

    this.nodes.forEach((node, index) => {
      const jitter =
        node.type === 'root' ? 0 :
        node.type === 'category' ? 5 :
        node.type === 'subcategory' ? 4 :
        3.2;
      const phase = (index % 5) / 5 * Math.PI * 2;
      node.x = node.tx + Math.cos(phase) * jitter;
      node.y = node.ty + Math.sin(phase) * jitter * 0.8;
      node.z = node.tz + Math.sin(phase * 0.7) * jitter;
    });

    // 星图重心：全部节点位置的均值，作为默认旋转中心（而非根节点所在的银心原点）
    const center = { x: 0, y: 0, z: 0 };
    for (const node of this.nodes) {
      center.x += node.tx;
      center.y += node.ty;
      center.z += node.tz;
    }
    if (this.nodes.length) {
      center.x /= this.nodes.length;
      center.y /= this.nodes.length;
      center.z /= this.nodes.length;
    }
    this.graphCenter = center;
    this.viewCenter = { ...center };

    // 布局完成：按空间方位角为簇内节点重新配色（颜色随布局平滑分布）
    this.applyClusterColors();
  }

  /**
   * 排斥力后处理：在初始布局坐标 (tx,ty,tz) 基础上，迭代推开同星座内距离过近的节点。
   * 只修改 tx/ty/tz，后续 jitter 和 easeNodes 再处理 x/y/z。
   */
  private applyRepulsion(): void {
    const ITERATIONS = 100;
    // 不同类型节点对的最小 3D 期望距离
    const MIN_NOTE_NOTE = 75;
    const MIN_NOTE_SUB = 90;
    const MIN_SUB_SUB = 100;
    // 银心净空区半径：略低于旋臂内缘（~211），保证星系中心周围没有游离恒星
    const MIN_CENTER_DIST = 190;
    const nodes = this.nodes;

    for (let iter = 0; iter < ITERATIONS; iter++) {
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          // 银心净空区：根节点是"星系中心"，任何恒星都不得悬停在它旁边
          // （沿径向只推非根节点，根节点固定原点不动）
          if (a.type === 'root' || b.type === 'root') {
            const star = a.type === 'root' ? b : a;
            if (star.type === 'root') continue; // 防御：理论上不会有两个 root
            const dxr = star.tx, dyr = star.ty, dzr = star.tz;
            const dr = Math.sqrt(dxr * dxr + dyr * dyr + dzr * dzr);
            if (dr < MIN_CENTER_DIST) {
              if (dr < 1e-3) { star.tx = MIN_CENTER_DIST; continue; }
              const k = MIN_CENTER_DIST / dr;
              star.tx *= k; star.ty *= k; star.tz *= k;
            }
            continue;
          }
          // 跳过跨星座节点、以及 category 自身间的排斥（由 baseRadius 控制）
          if (a.type === 'category' && b.type === 'category') continue;
          if (a.branchId !== b.branchId) continue;

          const dx = b.tx - a.tx;
          const dy = b.ty - a.ty;
          const dz = b.tz - a.tz;
          const d2 = dx * dx + dy * dy + dz * dz;

          let minDist: number;
          if (a.type === 'note' && b.type === 'note') {
            minDist = MIN_NOTE_NOTE;
          } else if (a.type === 'note' || b.type === 'note') {
            minDist = MIN_NOTE_SUB;
          } else {
            minDist = MIN_SUB_SUB;
          }

          if (d2 < minDist * minDist) {
            const d = Math.sqrt(d2) || 0.001;
            const overlap = (minDist - d) / d;
            // subcategory 节点质量更大，移动幅度更小
            const wa = a.type === 'note' ? 0.56 : 0.24;
            const wb = b.type === 'note' ? 0.56 : 0.24;
            a.tx -= dx * overlap * wa;
            a.ty -= dy * overlap * wa;
            a.tz -= dz * overlap * wa;
            b.tx += dx * overlap * wb;
            b.ty += dy * overlap * wb;
            b.tz += dz * overlap * wb;
          }
        }
      }
    }
  }

  // 簇内配色（核心）：以节点相对星团中心的方位角的正弦为平滑参数 t 在色族内插值。
  // sin(angle) 绕簇一周处处连续（无接缝），空间上相邻的节点 t 必然接近 →
  // 颜色沿星团平滑渐变，彻底消除"红突变蓝"；笔记再向白色靠拢
  // （真实恒星以白/蓝白为主，仅带星座色微染），混白量同样随方位角平滑变化。
  private applyClusterColors(): void {
    for (const cat of this.orderedChildren('__root__')) {
      const family = familyFor(cat.branchId);
      // 类目节点也不使用纯色：混白降饱和，保留色相身份（真实星团"代表星"也是白底染色，
      // 避免相邻星团的纯色节点直接相撞，如青色旁边跳出一个纯黄）
      cat.color = mixHex(family.base, '#ffffff', 0.34);
      const stack = [...(this.childrenById.get(cat.id) || [])];
      while (stack.length) {
        const node = stack.pop()!;
        const angle = Math.atan2(node.tz - cat.tz, node.tx - cat.tx);
        const t = 0.5 + 0.5 * Math.sin(angle);
        const clusterColor = mixHex(family.base, family.shift, t);
        if (node.type === 'subcategory') {
          // 子目录同样先混白降饱和，再随深度轻微压暗保持层次
          const dark = clamp(0.04 + Math.max(0, node.depth - 1) * 0.02, 0.04, 0.12);
          node.color = mixHex(mixHex(clusterColor, '#ffffff', 0.18), '#0f172a', dark);
        } else {
          const white = 0.55 + 0.15 * Math.sin(angle * 2 + 1);
          node.color = mixHex(clusterColor, '#ffffff', white);
        }
        stack.push(...(this.childrenById.get(node.id) || []));
      }
    }
  }

  // ======= 背景 =======

  private createBackgroundStars(): void {
    this.stars = [];
    const count = Math.round(clamp((this.width * this.height) / 5500, 110, 260));
    for (let i = 0; i < count; i += 1) {
      const layer = Math.random() < 0.72 ? 0 : 1; // 0 远层 / 1 近层
      const tintRoll = Math.random();
      this.stars.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        r: layer === 0 ? 0.4 + Math.random() * 0.8 : 0.8 + Math.random() * 1.4,
        a: layer === 0 ? 0.13 + Math.random() * 0.2 : 0.24 + Math.random() * 0.3,
        p: Math.random() * Math.PI * 2,
        tw: 0.5 + Math.random() * 1.3,
        layer,
        // 真实星色分布：白 / 蓝白为主，少量暖黄与橙
        rgb: tintRoll < 0.55 ? '226, 232, 240' : tintRoll < 0.80 ? '191, 219, 254' : tintRoll < 0.93 ? '253, 230, 138' : '251, 146, 60',
        bright: Math.random() < 0.06,
      });
    }
    this.renderMilkyWayCanvas();
    this.renderNebulaCanvas();
    this.renderVignetteCanvas();
    this.createDustParticles();
  }

  // 漂浮微尘：细小介质粒子，缓慢漂移 + 闪烁，近层带视差
  private createDustParticles(): void {
    this.dustParticles = [];
    const count = Math.round(clamp((this.width * this.height) / 22000, 40, 80));
    for (let i = 0; i < count; i += 1) {
      const layer = Math.random() < 0.6 ? 0 : 1;
      this.dustParticles.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        r: 0.5 + Math.random() * 1.1,
        a: 0.05 + Math.random() * 0.12,
        vx: (Math.random() - 0.5) * 6,
        vy: (Math.random() - 0.5) * 4,
        p: Math.random() * Math.PI * 2,
        tw: 0.4 + Math.random() * 0.9,
        layer,
      });
    }
    this.meteors = [];
    this.nextMeteorAt = this.time + 6 + Math.random() * 4;
  }

  private updateDust(dt: number): void {
    const step = dt * 0.001;
    for (const d of this.dustParticles) {
      d.x += d.vx * step;
      d.y += d.vy * step;
      if (d.x < -8) d.x = this.width + 8; else if (d.x > this.width + 8) d.x = -8;
      if (d.y < -8) d.y = this.height + 8; else if (d.y > this.height + 8) d.y = -8;
    }
  }

  private drawDustParticles(): void {
    const ctx = this.ctx;
    for (const d of this.dustParticles) {
      const parallax = d.layer === 0 ? 4 : 10;
      const ox = this.camera.rotY * parallax;
      const oy = (this.camera.rotX - DEFAULT_ROT_X) * parallax * 0.7;
      const pulse = 0.6 + Math.sin(this.time * d.tw + d.p) * 0.4;
      ctx.beginPath();
      ctx.fillStyle = `rgba(191, 219, 254, ${d.a * pulse})`;
      ctx.arc(d.x + ox, d.y + oy, d.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 偶发流星：短促的渐变尾迹，6~11 秒一颗
  private updateMeteors(dt: number): void {
    const step = dt * 0.001;
    if (this.time >= this.nextMeteorAt) {
      this.nextMeteorAt = this.time + 6 + Math.random() * 5;
      this.meteors.push({
        x: Math.random() * this.width * 0.7,
        y: Math.random() * this.height * 0.3 - 20,
        vx: (Math.random() < 0.5 ? 1 : -1) * (320 + Math.random() * 220),
        vy: 120 + Math.random() * 90,
        life: 0,
        maxLife: 0.7 + Math.random() * 0.5,
      });
    }
    for (const m of this.meteors) {
      m.life += step;
      m.x += m.vx * step;
      m.y += m.vy * step;
    }
    this.meteors = this.meteors.filter(m =>
      m.life < m.maxLife && m.x > -140 && m.x < this.width + 140 && m.y < this.height + 140);
  }

  private drawMeteors(): void {
    const ctx = this.ctx;
    for (const m of this.meteors) {
      const envelope = Math.sin(Math.min(1, m.life / m.maxLife) * Math.PI);
      const tailX = m.x - m.vx * 0.09;
      const tailY = m.y - m.vy * 0.09;
      const grad = ctx.createLinearGradient(tailX, tailY, m.x, m.y);
      grad.addColorStop(0, 'rgba(191, 219, 254, 0)');
      grad.addColorStop(1, `rgba(226, 232, 240, ${0.75 * envelope})`);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(tailX, tailY);
      ctx.lineTo(m.x, m.y);
      ctx.stroke();
      ctx.fillStyle = `rgba(255, 255, 255, ${0.9 * envelope})`;
      ctx.beginPath();
      ctx.arc(m.x, m.y, 1.4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  private makeSpriteCanvas(res: number): HTMLCanvasElement {
    const sprite = document.createElement('canvas');
    sprite.width = res;
    sprite.height = res;
    return sprite;
  }

  // 紧致光晕精灵（节点微光与连线光流使用）：衰减很快，只为亮核提供一圈贴身辉光
  private haloSpriteFor(color: string): HTMLCanvasElement {
    let sprite = this.haloSpriteCache.get(color);
    if (sprite) return sprite;
    sprite = this.makeSpriteCanvas(STAR_SPRITE_RES);
    const g = sprite.getContext('2d')!;
    const half = STAR_SPRITE_RES / 2;
    const grad = g.createRadialGradient(half, half, 0, half, half, half);
    grad.addColorStop(0, rgba(color, 0.55));
    grad.addColorStop(0.42, rgba(color, 0.22));
    grad.addColorStop(1, rgba(color, 0));
    g.fillStyle = grad;
    g.fillRect(0, 0, STAR_SPRITE_RES, STAR_SPRITE_RES);
    this.haloSpriteCache.set(color, sprite);
    return sprite;
  }

  // 银河带：斜贯画面的浅淡亮带 = 弥散辉光 + 大量细密暗星 + 暗尘埃裂隙
  private renderMilkyWayCanvas(): void {
    const { width, height } = this;
    this.milkyWayCanvas = document.createElement('canvas');
    this.milkyWayCanvas.width = Math.max(1, Math.floor(width));
    this.milkyWayCanvas.height = Math.max(1, Math.floor(height));
    const g = this.milkyWayCanvas.getContext('2d')!;
    const diag = Math.hypot(width, height);
    const bandHeight = Math.min(width, height) * 0.52;

    g.save();
    g.translate(width / 2, height / 2);
    g.rotate(-0.46);

    // 弥散辉光（沿带轴垂直方向衰减，中心略偏蓝白）——压得很淡，保持背景深邃
    const glow = g.createLinearGradient(0, -bandHeight / 2, 0, bandHeight / 2);
    glow.addColorStop(0, 'rgba(147, 197, 253, 0)');
    glow.addColorStop(0.30, 'rgba(147, 197, 253, 0.02)');
    glow.addColorStop(0.50, 'rgba(226, 232, 240, 0.042)');
    glow.addColorStop(0.70, 'rgba(191, 219, 254, 0.02)');
    glow.addColorStop(1, 'rgba(147, 197, 253, 0)');
    g.fillStyle = glow;
    g.fillRect(-diag, -bandHeight / 2, diag * 2, bandHeight);

    // 细密星尘：沿带轴均匀、垂直方向近似高斯分布
    const dustCount = Math.round(clamp((width * height) / 1400, 500, 1400));
    for (let i = 0; i < dustCount; i += 1) {
      const x = (Math.random() * 2 - 1) * diag * 0.62;
      const gauss = (Math.random() + Math.random() + Math.random()) / 1.5 - 1;
      const y = gauss * bandHeight * 0.24;
      const edgeFade = 1 - Math.min(1, Math.abs(x) / (diag * 0.62));
      const roll = Math.random();
      const rgb = roll < 0.6 ? '226, 232, 240' : roll < 0.85 ? '191, 219, 254' : '253, 230, 138';
      g.fillStyle = `rgba(${rgb}, ${(0.05 + Math.random() * 0.2) * edgeFade})`;
      g.beginPath();
      g.arc(x, y, 0.3 + Math.random() * 0.8, 0, Math.PI * 2);
      g.fill();
    }

    // 暗尘埃裂隙：银河带内的吸收云，增加真实感
    for (let i = 0; i < 6; i += 1) {
      const x = (hashUnit(i, 211) * 2 - 1) * diag * 0.34;
      const y = (hashUnit(i, 307) * 2 - 1) * bandHeight * 0.14;
      const rx = 60 + hashUnit(i, 101) * 150;
      const ry = rx * (0.28 + hashUnit(i, 157) * 0.22);
      const grad = g.createRadialGradient(x, y, 0, x, y, rx);
      grad.addColorStop(0, 'rgba(2, 6, 23, 0.34)');
      grad.addColorStop(1, 'rgba(2, 6, 23, 0)');
      g.save();
      g.translate(x, y);
      g.scale(1, ry / rx);
      g.translate(-x, -y);
      g.fillStyle = grad;
      g.fillRect(x - rx, y - rx, rx * 2, rx * 2);
      g.restore();
    }

    g.restore();
  }

  // 静态星云层：深空中的彩色发射云 + 暗尘埃区，营造纵深与明暗
  private renderNebulaCanvas(): void {
    const { width, height } = this;
    this.nebulaCanvas = document.createElement('canvas');
    this.nebulaCanvas.width = Math.max(1, Math.floor(width));
    this.nebulaCanvas.height = Math.max(1, Math.floor(height));
    const g = this.nebulaCanvas.getContext('2d')!;
    const tints = ['#1e3a8a', '#4c1d95', '#134e4a', '#1d4ed8', '#701a75', '#0c4a6e'];
    const blobCount = Math.round(clamp((width * height) / 260000, 5, 9));
    for (let i = 0; i < blobCount; i += 1) {
      const x = hashUnit(i, 7) * width;
      const y = hashUnit(i, 13) * height;
      const r = 140 + hashUnit(i, 29) * Math.min(width, height) * 0.32;
      const tint = tints[i % tints.length];
      const grad = g.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, rgba(tint, 0.07 + hashUnit(i, 41) * 0.05));
      grad.addColorStop(0.55, rgba(tint, 0.028));
      grad.addColorStop(1, rgba(tint, 0));
      g.fillStyle = grad;
      g.fillRect(x - r, y - r, r * 2, r * 2);
    }
    // 暗尘埃区（阴影层）：局部压暗，衬托恒星的亮
    for (let i = 0; i < 3; i += 1) {
      const x = hashUnit(i, 71) * width;
      const y = hashUnit(i, 97) * height;
      const r = 120 + hashUnit(i, 53) * Math.min(width, height) * 0.22;
      const grad = g.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, 'rgba(1, 3, 10, 0.36)');
      grad.addColorStop(1, 'rgba(2, 6, 23, 0)');
      g.fillStyle = grad;
      g.fillRect(x - r, y - r, r * 2, r * 2);
    }
  }

  // 暗角后期层：四角轻微压暗，模拟镜头成像
  private renderVignetteCanvas(): void {
    const { width, height } = this;
    this.vignetteCanvas = document.createElement('canvas');
    this.vignetteCanvas.width = Math.max(1, Math.floor(width));
    this.vignetteCanvas.height = Math.max(1, Math.floor(height));
    const g = this.vignetteCanvas.getContext('2d')!;
    const cx = width / 2;
    const cy = height / 2;
    const outer = Math.hypot(cx, cy);
    const grad = g.createRadialGradient(cx, cy, outer * 0.42, cx, cy, outer);
    grad.addColorStop(0, 'rgba(0, 0, 0, 0)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0.52)');
    g.fillStyle = grad;
    g.fillRect(0, 0, width, height);
  }

  private applyVignettePass(): void {
    if (!this.vignetteCanvas) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.drawImage(this.vignetteCanvas, 0, 0, this.width, this.height);
    ctx.restore();
  }

  // 深度雾：z 为相机空间深度（越大越远），返回远→近的亮度系数 0.58~1
  private depthFade(z: number): number {
    const t = clamp((z / (this.camera.distance * 0.95)) * 0.5 + 0.5, 0, 1);
    return 1 - t * 0.42;
  }

  // ======= 投影与动画主循环 =======

  private projectPoint(x: number, y: number, z: number): ScreenPoint {
    // 平移坐标系：以 viewCenter 为旋转中心（固定节点后该节点即位于屏幕中心）
    x -= this.viewCenter.x;
    y -= this.viewCenter.y;
    z -= this.viewCenter.z;
    const cosConstellation = Math.cos(this.constellationRotation);
    const sinConstellation = Math.sin(this.constellationRotation);
    const rx = x * cosConstellation - z * sinConstellation;
    const rz = x * sinConstellation + z * cosConstellation;
    const cosY = Math.cos(this.camera.rotY);
    const sinY = Math.sin(this.camera.rotY);
    const cosX = Math.cos(this.camera.rotX);
    const sinX = Math.sin(this.camera.rotX);

    const x1 = rx * cosY - rz * sinY;
    const z1 = rx * sinY + rz * cosY;
    const y1 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;
    const perspective = this.camera.distance / Math.max(180, this.camera.distance + z2);

    return {
      x: this.width / 2 + this.camera.panX + x1 * perspective * this.camera.zoom,
      y: this.height / 2 + this.camera.panY + y1 * perspective * this.camera.zoom,
      z: z2,
      scale: perspective * this.camera.zoom,
    };
  }

  private updateProjection(): void {
    for (const node of this.nodes) {
      node.screen = this.projectPoint(node.x, node.y, node.z);
    }
  }

  private targetScale(node: EngineNode): number {
    if (this.focusNode) {
      if (node.id === this.focusNode.id) return 1.62;
      // 固定状态下悬停其他节点：放大预览，提示"点击可固定到这里"
      if (this.pinnedNode && this.hoveredNode && node.id === this.hoveredNode.id) return 1.48;
      if (this.focusIds.has(node.id)) return 1.17;
      return 0.86;
    }
    return this.searchMatches.has(node.id) ? 1.32 : 1;
  }

  private animate = (now: number): void => {
    const dt = Math.min(48, now - this.lastTime || 16);
    this.lastTime = now;
    if (!this.paused) {
      this.time += dt * 0.001;
      // 自动旋转：约 6 分钟一圈（0.000016 rad/ms），缓慢漂移不抢眼
      if (!this.dragState && !this.isHoveringNode && !this.pinnedNode) this.constellationRotation += dt * 0.000016;
      this.easeNodes(dt);
      this.updateDust(dt);
      this.updateMeteors(dt);
    }
    this.updateProjection();
    this.drawScene();
    if (this.focusNode) this.placeTooltipSmart(this.focusNode);
    this.frameHandle = requestAnimationFrame(this.animate);
  };

  private easeNodes(dt: number): void {
    const ease = 1 - Math.pow(0.001, dt / 1000);
    for (const node of this.nodes) {
      node.x += (node.tx - node.x) * ease * 0.82;
      node.y += (node.ty - node.y) * ease * 0.82;
      node.z += (node.tz - node.z) * ease * 0.82;
      node.scale += (this.targetScale(node) - node.scale) * 0.18;
    }
    // 视角旋转中心平滑追踪固定节点的当前位置（节点自身也在缓动，逐帧跟随）；
    // 取消固定后平滑回到星图重心
    const centerEase = ease * 0.55;
    const cx = this.pinnedNode ? this.pinnedNode.x : this.graphCenter.x;
    const cy = this.pinnedNode ? this.pinnedNode.y : this.graphCenter.y;
    const cz = this.pinnedNode ? this.pinnedNode.z : this.graphCenter.z;
    this.viewCenter.x += (cx - this.viewCenter.x) * centerEase;
    this.viewCenter.y += (cy - this.viewCenter.y) * centerEase;
    this.viewCenter.z += (cz - this.viewCenter.z) * centerEase;
  }

  private drawScene(): void {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    this.drawBackgroundStars();
    this.drawDustParticles();
    this.drawMeteors();
    this.drawLinks();
    this.drawNodes();
    this.applyVignettePass();
    this.updateLabels();
  }

  private drawBackgroundStars(): void {
    const ctx = this.ctx;
    const { width, height } = this;
    if (this.milkyWayCanvas) ctx.drawImage(this.milkyWayCanvas, 0, 0, width, height);
    if (this.nebulaCanvas) ctx.drawImage(this.nebulaCanvas, 0, 0, width, height);
    const brightHalo = this.haloSpriteFor('#93c5fd');
    for (const star of this.stars) {
      // 视差：远近两层随相机旋转产生不同幅度的漂移，营造景深
      const parallax = star.layer === 0 ? 6 : 16;
      const ox = this.camera.rotY * parallax;
      const oy = (this.camera.rotX - DEFAULT_ROT_X) * parallax * 0.7;
      const pulse = 0.68 + Math.sin(this.time * star.tw + star.p) * 0.32;
      if (star.bright) {
        // 亮星：紧致微光 + 脆亮小核（无星芒），与节点的新科技感样式一致
        const half = star.r * 3.2 * (0.9 + pulse * 0.2);
        ctx.globalAlpha = clamp(star.a * pulse * 0.9, 0, 1);
        ctx.drawImage(brightHalo, star.x + ox - half, star.y + oy - half, half * 2, half * 2);
        ctx.beginPath();
        ctx.fillStyle = `rgba(241, 245, 249, ${clamp(star.a * pulse * 1.5, 0, 1)})`;
        ctx.arc(star.x + ox, star.y + oy, star.r * 0.9, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.globalAlpha = 1;
        ctx.beginPath();
        ctx.fillStyle = `rgba(${star.rgb}, ${star.a * pulse})`;
        ctx.arc(star.x + ox, star.y + oy, star.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  private drawLinks(): void {
    const ctx = this.ctx;
    const sorted = [...this.links].sort((a, b) =>
      ((b.source.screen?.z || 0) + (b.target.screen?.z || 0)) - ((a.source.screen?.z || 0) + (a.target.screen?.z || 0)));
    for (const link of sorted) {
      const source = link.source.screen;
      const target = link.target.screen;
      if (!source || !target) continue;

      const active = !!this.focusNode && (link.source.id === this.focusNode.id || link.target.id === this.focusNode.id);
      const related = !!this.focusNode && (this.focusIds.has(link.source.id) || this.focusIds.has(link.target.id));
      const muted = !!this.focusNode && !related;
      const reference = link.type === 'reference';
      const color = reference
        ? mixHex(link.source.color || colorForNode(link.source), link.target.color || colorForNode(link.target), 0.5)
        : (link.source.color || colorForNode(link.source));
      const fade = this.depthFade((source.z + target.z) / 2);
      // 常态连线刻意压低透明度，让星空背景更干净；仅激活（聚焦相关）连线保持明亮
      const alpha = (muted ? 0.045 : active ? 0.82 : reference ? 0.21 : 0.19) * fade;

      ctx.save();
      // 底层细实线：把珠点连成一条隐约可见的通路
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.lineWidth = active ? 1.2 : 0.6;
      if (reference) ctx.setLineDash([5, 7]);
      ctx.strokeStyle = rgba(color, alpha * 0.5);
      ctx.stroke();
      ctx.setLineDash([]);
      // 珠链：圆头零长虚线形成等距能量珠点（未来科技感的点状航线）
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.setLineDash([0.1, 13]);
      ctx.lineCap = 'round';
      ctx.lineWidth = active ? 3.2 : 2.1;
      ctx.strokeStyle = rgba(color, Math.min(1, alpha * (reference ? 0.9 : 1.1)));
      ctx.stroke();
      ctx.restore();

      if (active) this.drawLinkLightFlow(source, target, color);
    }
  }

  // 聚焦时沿激活连线流动的光点（仅聚焦状态下计算，常态零开销）
  private drawLinkLightFlow(source: ScreenPoint, target: ScreenPoint, color: string): void {
    const ctx = this.ctx;
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const halo = this.haloSpriteFor(color);
    ctx.save();
    for (let i = 0; i < 2; i += 1) {
      const t = (this.time * 0.45 + i * 0.5) % 1;
      const envelope = Math.sin(t * Math.PI);
      const x = source.x + dx * t;
      const y = source.y + dy * t;
      const size = 14 + envelope * 6;
      ctx.globalAlpha = 0.65 * envelope;
      ctx.drawImage(halo, x - size / 2, y - size / 2, size, size);
      ctx.globalAlpha = 0.9 * envelope;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(x, y, 1.7, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  private drawNodes(): void {
    const ctx = this.ctx;
    const sorted = [...this.nodes].sort((a, b) => (b.screen?.z || 0) - (a.screen?.z || 0));
    for (const node of sorted) {
      const p = node.screen;
      if (!p) continue;
      const active = !!this.focusNode && this.focusIds.has(node.id);
      // 固定状态下悬停预览的节点：即便不在高亮组内也按激活态提亮
      const hoverPreview = !!(this.pinnedNode && this.hoveredNode && node.id === this.hoveredNode.id && node.id !== this.pinnedNode.id);
      const muted = !!this.focusNode && !active && !hoverPreview;
      const matched = this.searchMatches.has(node.id);
      const baseRadius = node.radius * p.scale * node.scale;
      const r = clamp(baseRadius, node.type === 'note' ? 2.4 : 4, node.type === 'root' ? 22 : 16);
      node.hitRadius = Math.max(9, r + (node.type === 'note' ? 7 : 5));

      const fade = this.depthFade(p.z);
      const alpha = (muted ? 0.18 : 1) * fade;
      const color = node.color;
      const highlighted = active || hoverPreview;

      ctx.save();

      // 根星纹章 / 一级目录题名：根星正上方悬浮全息徽记（自带透明底 + 冷白调色）；
      // CV/NLP 等一级目录正上方悬浮英文题名（Share Tech 科技无衬线风 + 科技蓝辉光脉动）。
      // 一道从星核射向徽记/题字底部的渐变光束把二者连成连体结构，
      // 随节点缩放/深度衰减/聚焦变暗，与星辰同呼吸
      if (node.type === 'root' && this.rootBadgeReady) {
        const bw = r * 4.3;
        const bh = bw * (this.rootBadgeImg.naturalHeight / this.rootBadgeImg.naturalWidth);
        const bx = p.x - bw / 2;
        const by = p.y - r - 10 - bh; // 底缘贴在刻度弧（r+6.5）上方
        // 全息投影光束：星核处最亮，向纹章方向渐隐
        const beamTop = by + bh - 3;
        const beam = ctx.createLinearGradient(p.x, p.y, p.x, beamTop);
        beam.addColorStop(0, rgba('#dbeafe', 0.85));
        beam.addColorStop(1, rgba('#dbeafe', 0.22));
        ctx.globalAlpha = clamp(alpha, 0, 1);
        ctx.strokeStyle = beam;
        ctx.lineWidth = Math.max(1.2, r * 0.09);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x, beamTop);
        ctx.stroke();
        ctx.globalAlpha = clamp(alpha * 0.95, 0, 1);
        ctx.drawImage(this.rootBadgeImg, bx, by, bw, bh);
      } else if (EMBLEM_TEXT[node.id]) {
        const fontSize = Math.max(10, r * 1.9);
        const baseline = p.y - r - 12; // 字面底缘贴在刻度弧（r+6.5）上方
        // 全息投影光束：星核处最亮，向题字方向渐隐
        const beam = ctx.createLinearGradient(p.x, p.y, p.x, baseline + 2);
        beam.addColorStop(0, rgba('#dbeafe', 0.85));
        beam.addColorStop(1, rgba('#dbeafe', 0.22));
        ctx.globalAlpha = clamp(alpha, 0, 1);
        ctx.strokeStyle = beam;
        ctx.lineWidth = Math.max(1.2, r * 0.09);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x, baseline + 2);
        ctx.stroke();
        // 双层辉光：外层宽幅低亮铺底，内层窄边高亮勾形，脉动模拟全息呼吸
        ctx.font = `${fontSize}px "Share Tech", "Segoe UI", sans-serif`;
        try { (ctx as any).letterSpacing = `${(fontSize * 0.08).toFixed(1)}px`; } catch { /* 旧内核无 letterSpacing */ }
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        const pulse = 0.75 + 0.25 * Math.sin(this.time * 2.1 + node.index * 1.7);
        ctx.globalAlpha = clamp(alpha * 0.5, 0, 1);
        ctx.fillStyle = '#dbeafe';
        ctx.shadowColor = rgba('#38bdf8', 0.55 * pulse);
        ctx.shadowBlur = fontSize * 1.1 * pulse;
        ctx.fillText(EMBLEM_TEXT[node.id], p.x, baseline);
        ctx.globalAlpha = clamp(alpha * 0.95, 0, 1);
        ctx.fillStyle = '#eaf2ff';
        ctx.shadowColor = rgba('#bae6fd', 0.9 * pulse);
        ctx.shadowBlur = fontSize * 0.35 * pulse;
        ctx.fillText(EMBLEM_TEXT[node.id], p.x, baseline);
        ctx.shadowBlur = 0;
        try { (ctx as any).letterSpacing = '0px'; } catch { /* 同上 */ }
      }

      // ===== 科技感节点标记：亮核为主，线条装饰只给重要节点 =====
      // 笔记 = 纯粹亮点（无环无光晕，避免密集圆圈）；子目录 = 亮核 + 贴身微光；
      // 目录/根节点 = 亮核 + 微光 + 细线圆环 + 缓转刻度弧
      const isMajor = node.type !== 'note';

      // 1) 贴身微光：范围很小、衰减很快，只为亮核提一抹辉光
      if (isMajor) {
        const halo = this.haloSpriteFor(color);
        const haloHalf = r * 2.1 * (highlighted ? 1.6 : 1);
        ctx.globalAlpha = clamp(alpha * (highlighted ? 0.55 : 0.3), 0, 1);
        ctx.drawImage(halo, p.x - haloHalf, p.y - haloHalf, haloHalf * 2, haloHalf * 2);
      }

      // 2) 亮核：偏白的实心小圆点
      ctx.globalAlpha = clamp(alpha, 0, 1);
      ctx.fillStyle = mixHex(color, '#ffffff', 0.58);
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(1.2, r * 0.4), 0, Math.PI * 2);
      ctx.fill();

      // 3) 主圆环：仅目录/根节点的细线轨道环
      if (isMajor) {
        ctx.globalAlpha = clamp(alpha * (highlighted ? 0.95 : 0.68), 0, 1);
        ctx.strokeStyle = color;
        ctx.lineWidth = highlighted ? 1.4 : 1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 2.8, 0, Math.PI * 2);
        ctx.stroke();
      }

      // 4) 刻度弧：仅根节点/一级目录外圈缓慢旋转的断续弧，交替旋向
      if (node.type === 'root' || node.type === 'category') {
        ctx.globalAlpha = clamp(alpha * 0.42, 0, 1);
        ctx.lineWidth = 1;
        ctx.setLineDash([2.5, 6]);
        ctx.lineDashOffset = -this.time * 7 * (node.index % 2 === 0 ? 1 : -1);
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 6.5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.globalAlpha = alpha;

      if (node.format === 'jupyter') {
        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        ctx.arc(p.x, p.y, r + 5.5, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(color, active ? 0.78 : 0.48);
        ctx.lineWidth = active ? 2.1 : 1.4;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 悬停预览标记：固定状态下鼠标悬停的节点，外圈亮色细环提示可点击
      if (hoverPreview) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 6.5, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(mixHex(color, '#ffffff', 0.55), 0.8);
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }

      // 固定标记：被点击锁定的焦点星外圈绘制缓慢旋转的虚线环
      if (this.pinnedNode && node.id === this.pinnedNode.id) {
        ctx.beginPath();
        ctx.setLineDash([7, 6]);
        ctx.lineDashOffset = -this.time * 16;
        ctx.arc(p.x, p.y, r + 9.5, 0, Math.PI * 2);
        ctx.strokeStyle = rgba('#93c5fd', 0.9);
        ctx.lineWidth = 1.8;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      if (matched && !this.focusNode) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 7, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(217, 119, 6, 0.48)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.restore();
    }
  }

  // ======= 标签 =======

  private createGraphLabels(): void {
    this.labelLayer.replaceChildren();
    for (const node of this.nodes) {
      const label = document.createElement('div');
      label.className = `graph-label graph-label--${node.type}`;
      label.textContent = node.name;
      label.style.setProperty('--label-color', this.labelColorFor(node));
      label.style.opacity = '0';
      this.labelLayer.appendChild(label);
      node.labelEl = label;
    }
  }

  private labelPriority(node: EngineNode): number {
    if (this.focusNode && node.id === this.focusNode.id) return 6;
    if (this.focusIds.has(node.id)) return 5;
    if (this.searchMatches.has(node.id)) return 5;
    if (node.type === 'root') return 4;
    if (node.type === 'category') return 3;
    if (node.type === 'subcategory') return 2;
    return 1;
  }

  private labelBaseSize(node: EngineNode): number {
    if (node.type === 'root') return 14.5;
    if (node.type === 'category') return 13;
    if (node.type === 'subcategory') return 10.2;
    return 8.8;
  }

  private labelColorFor(node: EngineNode): string {
    if (node.type === 'root' || node.type === 'category') return '#f8fafc';
    if (node.type === 'subcategory') return '#cbd5e1';
    return '#94a3b8';
  }

  private labelZoomBoost(): number {
    return clamp(0.72 + this.camera.zoom * 0.42, 0.82, 1.55);
  }

  // 文字标签的显示时机：
  // - 锁定选中（点击固定 pin / 搜索选中 search）：焦点与关联节点（子/父/引用）的名字全部浮现；
  // - 悬停（pointer，未固定）：只显示悬停节点与它的直接子节点、直接父节点的名字；
  // - 拉近视角：zoom 超过约 2.2 后所有节点名字随缩放渐入，远处节点更淡（深度雾）；
  // - 默认状态：星图完全无文字标识。
  private labelsLocked(): boolean {
    return !!this.focusNode && (!!this.pinnedNode || this.focusSource === 'search');
  }

  // 拉近视角时的基础透明度：超过旧取景上限后渐入，再乘深度雾让远处名字更淡
  private zoomLabelAlpha(node: EngineNode): number {
    const z = node.screen ? node.screen.z : 0;
    return clamp((this.camera.zoom - 2.2) / 0.9, 0, 0.9) * this.depthFade(z);
  }

  private labelAlphaFor(node: EngineNode): number {
    if (this.labelsLocked()) {
      if (node.id === this.focusNode!.id) return 1;
      if (this.focusIds.has(node.id)) return 0.92;
      return this.zoomLabelAlpha(node);
    }
    // 悬停状态：显示悬停节点本身、直接子节点与直接父节点的名字
    if (this.focusNode && this.focusSource === 'pointer') {
      if (node.id === this.focusNode.id) return 1;
      if (this.focusChildIds.has(node.id)) return 0.9;
      if (node === this.parentById.get(this.focusNode.id)) return 0.9;
    }
    return this.zoomLabelAlpha(node);
  }

  private labelFontSizeFor(node: EngineNode): number {
    const depthPenalty = Math.max(0, node.depth - 1) * 0.34;
    const focusBoost =
      this.focusNode && this.focusNode.id === node.id ? 1.32 :
      this.focusNode && this.focusIds.has(node.id) ? 1.12 :
      1;
    const size = (this.labelBaseSize(node) - depthPenalty) * this.labelZoomBoost() * focusBoost;
    if (node.type === 'root') return clamp(size, 12, 20);
    if (node.type === 'category') return clamp(size, 10.5, 18);
    if (node.type === 'subcategory') return clamp(size, 7.8, 15);
    return clamp(size, 6.8, 13.5);
  }

  private labelOffsetFor(node: EngineNode): number {
    const focusBoost =
      this.focusNode && this.focusNode.id === node.id ? 1.28 :
      this.focusNode && this.focusIds.has(node.id) ? 1.12 :
      1;
    const offset = node.radius * this.labelZoomBoost() * focusBoost + 8;
    return clamp(offset, node.type === 'note' ? 9 : 13, node.type === 'root' ? 34 : 26);
  }

  private updateLabels(): void {
    for (const node of this.nodes) {
      const label = node.labelEl;
      if (!label) continue;
      const p = node.screen;
      if (!p) {
        label.style.opacity = '0';
        continue;
      }

      const fontSize = this.labelFontSizeFor(node);
      const targetX = p.x;
      const targetY = p.y + this.labelOffsetFor(node);
      const hasLabelPosition = Number.isFinite(node.labelX) && Number.isFinite(node.labelY);
      const distance = hasLabelPosition ? Math.hypot(targetX - node.labelX, targetY - node.labelY) : Infinity;
      const follow = this.dragState ? 0.62 : 0.24;
      if (!hasLabelPosition || distance > 120) {
        node.labelX = targetX;
        node.labelY = targetY;
      } else {
        node.labelX += (targetX - node.labelX) * follow;
        node.labelY += (targetY - node.labelY) * follow;
      }

      label.style.fontSize = `${fontSize.toFixed(2)}px`;
      label.style.fontWeight = node.type === 'note' ? '500' : '700';
      label.style.opacity = this.labelAlphaFor(node).toFixed(3);
      const isPrimaryFocus = !!this.focusNode && this.focusNode.id === node.id;
      const isFocusChild = !!this.focusNode && this.focusIds.has(node.id) && !isPrimaryFocus;
      label.style.textShadow = isPrimaryFocus
        ? '0 0 16px rgba(147, 197, 253, 0.60), 0 1px 0 rgba(2, 6, 23, 0.95), 0 -1px 0 rgba(2, 6, 23, 0.90), 1px 0 0 rgba(2, 6, 23, 0.90), -1px 0 0 rgba(2, 6, 23, 0.90)'
        : isFocusChild
          ? '0 0 12px rgba(147, 197, 253, 0.38), 0 1px 0 rgba(2, 6, 23, 0.92), 0 -1px 0 rgba(2, 6, 23, 0.88), 1px 0 0 rgba(2, 6, 23, 0.88), -1px 0 0 rgba(2, 6, 23, 0.88)'
          : '';
      label.style.zIndex = String(10 + this.labelPriority(node));
      label.style.transform = `translate3d(${node.labelX.toFixed(2)}px, ${node.labelY.toFixed(2)}px, 0) translateX(-50%)`;
    }
  }

  // ======= 命中测试与焦点 =======

  private hitTest(clientX: number, clientY: number): EngineNode | null {
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const sorted = [...this.nodes].sort((a, b) => (a.screen?.z || 0) - (b.screen?.z || 0));
    for (const node of sorted) {
      const p = node.screen;
      if (!p) continue;
      const dist = Math.hypot(x - p.x, y - p.y);
      if (dist <= (node.hitRadius || 10)) return node;
    }
    return null;
  }

  private setFocus(node: EngineNode, fromPointer = false, source: 'pointer' | 'pin' | 'search' | null = null): void {
    this.focusNode = node;
    this.focusSource = source || (fromPointer ? 'pointer' : 'pin');
    if (fromPointer) {
      this.hoveredNode = node;
      this.isHoveringNode = true;
    }
    const ids = new Set([node.id, ...(this.adjacency.get(node.id) || [])]);
    const children = this.childrenById.get(node.id) || [];
    for (const child of children) ids.add(child.id);
    const parent = this.parentById.get(node.id);
    if (parent) ids.add(parent.id);
    this.focusIds = ids;
    this.focusChildIds = new Set(children.map(child => child.id));
    this.showTooltip(node);
  }

  private clearFocus(): void {
    this.pinnedNode = null;
    this.focusNode = null;
    this.focusSource = null;
    this.hoveredNode = null;
    this.isHoveringNode = false;
    this.focusIds = new Set();
    this.focusChildIds = new Set();
    this.searchMatches = this.searchQuery
      ? new Set(this.nodes.filter(node => this.searchable(node).includes(this.searchQuery)).map(node => node.id))
      : new Set();
    this.hideTooltip();
    this.callbacks.onPinChange?.(null);
  }

  // 完全复位：清空搜索词、搜索匹配与焦点/固定状态，
  // 回到所有节点均未选中的初始状态（点击星图空白处时调用）
  private resetAllSelection(): void {
    this.pinnedNode = null;
    this.searchQuery = '';
    this.searchMatches = new Set();
    this.callbacks.onClearSearch?.();
    this.clearFocus();
  }

  // 点击节点：把当前高亮"钉"在该节点上，鼠标移开也不会丢失，
  // 方便继续点选它的子节点；点击其他节点则切换固定对象。
  // syncSidebar=false 时不同步笔记目录（信息栏定位用）
  private pinNode(node: EngineNode, { syncSidebar = true }: { syncSidebar?: boolean } = {}): void {
    this.pinnedNode = node;
    this.setFocus(node, false, 'pin');
    if (syncSidebar) this.callbacks.onPinChange?.(node.id);
  }

  private searchable(node: EngineNode): string {
    return `${node.name} ${node.id}`.toLowerCase();
  }

  // ======= 提示浮层 =======

  private escapeHtml(value: unknown): string {
    return String(value).replace(/[&<>"']/g, char => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[char] as string));
  }

  private showTooltip(node: EngineNode): void {
    const color = node.color || colorForNode(node);
    const typeLabel = node.format === 'jupyter' ? 'JUPYTER' : (TYPE_LABEL[node.type] || node.type);
    const isPinned = this.pinnedNode && this.pinnedNode.id === node.id;
    const openHint = node.type === 'note' ? ' · 双击跳转笔记' : '';
    const hint = isPinned
      ? `已固定 · 点击子节点继续固定 · 点击空白处取消${openHint}`
      : `点击固定高亮${openHint}`;
    this.tooltip.innerHTML = `
      <div class="tip-type" style="background:${color}2e;color:${color}">${typeLabel}</div>
      <div class="tip-title">${this.escapeHtml(node.name)}</div>
      <div class="tip-path">/${this.escapeHtml(node.id)}</div>
      <div class="tip-hint">${hint}</div>
    `;
    this.tooltip.classList.add('visible');
    this.placeTooltipSmart(node);
  }

  private hideTooltip(): void {
    this.tooltip.classList.remove('visible');
  }

  private protectedRectsFor(node: EngineNode): Array<{ left: number; top: number; right: number; bottom: number; weight: number }> {
    const protectedNodes = [
      node,
      ...(this.childrenById.get(node.id) || []),
      ...(this.adjacency.get(node.id) ? [...this.adjacency.get(node.id)!].map(id => this.nodeById.get(id)).filter(Boolean) as EngineNode[] : []),
    ];
    const rects = [];
    for (const item of protectedNodes) {
      const p = item.screen;
      if (!p) continue;
      const labelWidth = Math.min(220, Math.max(58, item.name.length * 8));
      const r = item.hitRadius || 10;
      rects.push({
        left: p.x - Math.max(r + 8, labelWidth / 2),
        top: p.y - r - 10,
        right: p.x + Math.max(r + 8, labelWidth / 2),
        bottom: p.y + r + 34,
        weight: item === node ? 5 : this.parentById.get(item.id) === node ? 4 : 2,
      });
    }
    return rects;
  }

  private placeTooltipSmart(node: EngineNode): void {
    const tooltip = this.tooltip;
    if (!tooltip.classList.contains('visible') || !node.screen) return;

    const { width, height } = this;
    const tipWidth = tooltip.offsetWidth || 220;
    const tipHeight = tooltip.offsetHeight || 84;
    const p = node.screen;
    const gap = Math.max(20, (node.hitRadius || 12) + 14);
    const margin = 12;
    const children = this.childrenById.get(node.id) || [];
    const neighborScreens = children.map(child => child.screen).filter(Boolean) as ScreenPoint[];
    const avoidVector = neighborScreens.length
      ? neighborScreens.reduce((acc, point) => ({ x: acc.x + point.x - p.x, y: acc.y + point.y - p.y }), { x: 0, y: 0 })
      : { x: 0, y: -1 };
    const avoidLength = Math.hypot(avoidVector.x, avoidVector.y) || 1;
    const avoid = { x: avoidVector.x / avoidLength, y: avoidVector.y / avoidLength };

    const directions = [
      { x: 0, y: -1, anchor: 'bottom' },
      { x: 1, y: -0.35, anchor: 'left' },
      { x: -1, y: -0.35, anchor: 'right' },
      { x: 0, y: 1, anchor: 'top' },
      { x: 1, y: 0.85, anchor: 'left' },
      { x: -1, y: 0.85, anchor: 'right' },
      { x: 1, y: -1, anchor: 'left' },
      { x: -1, y: -1, anchor: 'right' },
    ];
    const protectedRects = this.protectedRectsFor(node);
    let best: { rect: { left: number; top: number; right: number; bottom: number }; score: number } | null = null;

    for (const dir of directions) {
      const length = Math.hypot(dir.x, dir.y) || 1;
      const nx = dir.x / length;
      const ny = dir.y / length;
      const cx = p.x + nx * gap;
      const cy = p.y + ny * gap;
      let rect: { left: number; top: number; right: number; bottom: number };
      if (dir.anchor === 'left') {
        rect = { left: cx, top: cy - tipHeight / 2, right: cx + tipWidth, bottom: cy + tipHeight / 2 };
      } else if (dir.anchor === 'right') {
        rect = { left: cx - tipWidth, top: cy - tipHeight / 2, right: cx, bottom: cy + tipHeight / 2 };
      } else if (dir.anchor === 'top') {
        rect = { left: cx - tipWidth / 2, top: cy, right: cx + tipWidth / 2, bottom: cy + tipHeight };
      } else {
        rect = { left: cx - tipWidth / 2, top: cy - tipHeight, right: cx + tipWidth / 2, bottom: cy };
      }
      rect = this.clampRect(rect, margin);

      let score = 0;
      for (const protectedRect of protectedRects) {
        score += this.intersectArea(rect, protectedRect) * protectedRect.weight;
      }
      const sameDirectionPenalty = Math.max(0, nx * avoid.x + ny * avoid.y) * 1800;
      const edgePenalty =
        Math.max(0, margin - rect.left) +
        Math.max(0, margin - rect.top) +
        Math.max(0, rect.right - width + margin) +
        Math.max(0, rect.bottom - height + margin);
      score += sameDirectionPenalty + edgePenalty * 120;

      if (!best || score < best.score) best = { rect, score };
    }

    const rect = best?.rect || { left: p.x + gap, top: p.y - tipHeight / 2 };
    tooltip.style.left = `${Math.round(rect.left)}px`;
    tooltip.style.top = `${Math.round(rect.top)}px`;
  }

  private intersectArea(a: { left: number; top: number; right: number; bottom: number }, b: { left: number; top: number; right: number; bottom: number }): number {
    const x = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
    const y = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
    return x * y;
  }

  private clampRect(rect: { left: number; top: number; right: number; bottom: number }, margin: number) {
    const w = rect.right - rect.left;
    const h = rect.bottom - rect.top;
    const left = clamp(rect.left, margin, this.width - w - margin);
    const top = clamp(rect.top, margin, this.height - h - margin);
    return { left, top, right: left + w, bottom: top + h };
  }

  // ======= 输入事件 =======

  private bindEvents(): void {
    const canvas = this.canvas;
    const signal = this.abort.signal;

    canvas.addEventListener('pointerdown', (event: PointerEvent) => {
      if (event.button === 2) event.preventDefault();
      const panDrag = event.button === 2;
      const node = panDrag ? null : this.hitTest(event.clientX, event.clientY);
      this.pressedNode = node;
      this.dragState = {
        mode: panDrag ? 'pan' : 'rotate',
        x: event.clientX,
        y: event.clientY,
        rotX: this.camera.rotX,
        rotY: this.camera.rotY,
        panX: this.camera.panX,
        panY: this.camera.panY,
        moved: false,
      };
      canvas.classList.add('is-dragging');
      // 合成事件（测试脚本）的 pointerId 不活跃，setPointerCapture 会抛错，静默兜底
      try { canvas.setPointerCapture?.(event.pointerId); } catch { /* noop */ }
    }, { signal });

    canvas.addEventListener('pointermove', (event: PointerEvent) => {
      if (this.dragState) {
        const dx = event.clientX - this.dragState.x;
        const dy = event.clientY - this.dragState.y;
        if (Math.hypot(dx, dy) > 3) this.dragState.moved = true;
        if (this.dragState.mode === 'pan' || event.shiftKey) {
          this.camera.panX = this.dragState.panX + dx;
          this.camera.panY = this.dragState.panY + dy;
        } else {
          this.camera.rotY = this.dragState.rotY + dx * 0.0052;
          this.camera.rotX = clamp(this.dragState.rotX + dy * 0.0042, -0.45, 1.42);
        }
        if (this.focusNode) this.placeTooltipSmart(this.focusNode);
        return;
      }

      const node = this.hitTest(event.clientX, event.clientY);
      canvas.style.cursor = node ? 'pointer' : 'grab';

      // 固定状态下：高亮锁定在 pinnedNode 上，鼠标移动不再改变焦点，
      // 仅记录悬停目标，供点击（切换固定对象）与双击（跳转）使用
      if (this.pinnedNode) {
        this.hoveredNode = node;
        this.isHoveringNode = !!node;
        return;
      }

      // 搜索选中状态下：焦点锁定在搜索结果上，悬停仅作预览，不接管焦点，
      // 避免已浮现的文字标签因鼠标滑过而消失
      if (this.searchQuery) {
        this.hoveredNode = node;
        this.isHoveringNode = !!node;
        return;
      }

      if (node !== this.hoveredNode) {
        if (node) {
          this.setFocus(node, true);
        } else {
          this.hoveredNode = null;
          this.isHoveringNode = false;
          if (!this.searchQuery) this.clearFocus();
        }
      }
    }, { signal });

    const endPointer = (event: PointerEvent): void => {
      try { canvas.releasePointerCapture?.(event.pointerId); } catch { /* noop */ }
      canvas.classList.remove('is-dragging');
      if (this.dragState && this.dragState.mode === 'rotate' && !this.dragState.moved) {
        if (this.pressedNode) {
          // 单击节点：固定当前高亮（点击其他节点则切换固定对象），不再跳转页面
          this.pinNode(this.pressedNode);
        } else if (this.pinnedNode || this.focusNode || this.searchQuery) {
          // 单击星空空白处：完全复位（取消固定、清空搜索与焦点），恢复所有节点未选中状态
          this.resetAllSelection();
        }
      }
      this.dragState = null;
      this.pressedNode = null;
    };
    canvas.addEventListener('pointerup', endPointer, { signal });
    canvas.addEventListener('pointercancel', endPointer, { signal });

    canvas.addEventListener('contextmenu', event => event.preventDefault(), { signal });

    canvas.addEventListener('dblclick', (event: MouseEvent) => {
      // 双击节点：跳转到对应笔记（单击已改为固定高亮）
      const node = this.hitTest(event.clientX, event.clientY);
      if (node) this.openNode(node);
    }, { signal });

    canvas.addEventListener('mouseleave', () => {
      this.hoveredNode = null;
      this.isHoveringNode = false;
      if (!this.dragState && !this.searchQuery && !this.pinnedNode) this.clearFocus();
    }, { signal });

    canvas.addEventListener('wheel', (event: WheelEvent) => {
      event.preventDefault();
      const delta = Math.exp(-event.deltaY * 0.0011);
      this.camera.zoom = clamp(this.camera.zoom * delta, 0.34, 6);
      if (this.focusNode) this.placeTooltipSmart(this.focusNode);
    }, { passive: false, signal });
  }

  private openNode(node: EngineNode): void {
    // 只有笔记节点可跳转；目录节点在星图/侧边栏中定位
    if (!node || node.type !== 'note' || !node.url) return;
    this.callbacks.onOpenNote?.(node);
  }
}

// ======= 榜单数据（最新更新 / 浏览热度，供 Vue 组件渲染）=======

export interface WidgetNote extends GraphNodeInput {
  color: string
}

/** 最新更新 TOP N：mtime 相同（如 CI 里 git 时间缺失）时按名字排序，保证顺序稳定 */
export function latestNotes(data: GraphDataInput, count = 5): WidgetNote[] {
  return data.nodes
    .filter(node => node.type === 'note' && node.mtime)
    .sort((a, b) => (b.mtime! - a.mtime!) || a.name.localeCompare(b.name, 'zh'))
    .slice(0, count)
    .map(node => ({ ...node, color: colorForNode(node) }));
}

/** 浏览热度 TOP N：pv 为 MySQL 真实浏览量，随 /api/graph 返回 */
export function hotNotes(data: GraphDataInput, count = 5): WidgetNote[] {
  return data.nodes
    .filter(node => node.type === 'note' && node.pv != null)
    .sort((a, b) => (b.pv! - a.pv!))
    .slice(0, count)
    .map(node => ({ ...node, color: colorForNode(node) }));
}

/** 站点总浏览量：全部笔记 pv 之和（旧版来自构建期烘焙的 site_pv，新后端无此字段） */
export function totalPageviews(data: GraphDataInput): number {
  return data.nodes.reduce((sum, node) => sum + (node.type === 'note' ? Number(node.pv) || 0 : 0), 0);
}
