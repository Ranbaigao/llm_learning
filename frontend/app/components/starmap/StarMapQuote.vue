<script setup lang="ts">
/**
 * 星图签名：顶部居中的随机名言，纤细行楷 + 动态科技蓝边缘光，纯装饰不拦截画布交互。
 * 移植自旧版 knowledge_graph.html：名言文案一个字都不能改
 * （long-cang-quote.woff2 是按这批文案的字符集子集化的，改字需重新子集化）。
 */

// 星图签名：每次加载从名句库随机抽一句，以行楷泛光呈现于星图下方
const SIGNATURE_QUOTES = [
  '星辰不是彼岸，而是航程本身——我把每一次出发，都当作对未知的一次求婚。',
  '地图的尽头，正是探险开始的地方；已知的边界越清晰，未知的召唤越灼热。',
  '我以好奇为舟，以时间为海，渡向一切尚未被命名的黎明。',
  '未来从不抵达，它只是永远在前方俯身，等一个不肯停步的人。',
  '凝视深渊的人终成深渊，而凝视远方的人，把自己走成了路。',
  '所谓成长，是不断亲手拆毁昨日的殿堂，用废墟的砖石砌一座更高的瞭望塔。',
  '在熵增的宇宙里，探索是我对无序最优雅的反抗。',
  '我不收藏答案，只收集更锋利的问题——答案会过时，问题永远年轻。',
  '黑暗并非光的缺席，而是尚未被注视的辽阔；我愿做那个举火的人。',
  '向深空要坐标，向时间要答案，向自己要远方。',
  '不恋旧日的余烬，只向未至的破晓，投掷永恒的野望。',
  '边界即是请柬；于认知的无垠暗海中，以理性凿刻不灭的星图。',
  '越过时间的事件视界，在思维的穹顶之上，缔造尚未命名的纪元。',
  '循逻辑之经纬，涉虚无之渊薮，向无限之维执着跃迁。',
  '以行行符码为阶，叩问真理的奇点；目光所及，皆是明日文明的轮廓。',
  '凡目力所不及之深渊，皆为探索者尚未加冕的疆土。',
  '向远方永恒坍缩，向未来无限逼近，我们是刺破长夜的秩序之光。',
  '执理性与浪漫之刻刀，在荒芜的现实彼端，雕琢下一座繁茂的星系。',
  '历史在身后凝固成碑，而行者的脉搏，始终与未来的潮汐同频共振。',
  '不问归途与止境，惟愿以孤绝之姿，奔赴一场与终极未知的宏大重逢。',
]

// 随机抽名言 + 按标点切短句（句内 nowrap，句间 <wbr> 断点），onMounted 中进行，
// 避免 SSR 与客户端随机结果不一致导致 hydration mismatch
const segments = ref<string[]>([])

onMounted(() => {
  const quote = SIGNATURE_QUOTES[Math.floor(Math.random() * SIGNATURE_QUOTES.length)]
  segments.value = quote.match(/.+?(?:[，。；、！？：…]|——|$)/g) || [quote]
})
</script>

<template>
  <div class="quote-panel" aria-hidden="true">
    <p class="quote-text"><template v-for="(seg, i) in segments" :key="i"><span class="qseg">{{ seg }}</span><wbr></template></p>
  </div>
</template>

<style scoped>
.quote-panel {
  position: absolute;
  z-index: 4;
  left: 50%;
  top: 24px;
  /* 避开左上搜索框与右上图例（分别约 330px / 190px 宽） */
  width: min(880px, calc(100vw - 760px));
  transform: translateX(-50%);
  pointer-events: none;
  user-select: none;
  text-align: center;
  /* 远层晕光：作用于整段文字的渲染结果，跟随实际字形与换行，无副本错位问题 */
  filter: drop-shadow(0 0 14px rgba(37, 99, 235, 0.55));
  animation: quote-edge-far 5.3s ease-in-out infinite;
}

.quote-text {
  margin: 0;
  font-family: "Long Cang", "Zhi Mang Xing", "Ma Shan Zheng", "STXingkai", "华文行楷", "KaiTi", "楷体", cursive;
  font-size: 28px;
  letter-spacing: 0.12em;
  line-height: 1.8;
  text-wrap: balance;
  color: #eaf2ff;
  text-shadow:
    0 0 3px rgba(224, 242, 254, 0.85),
    0 0 12px rgba(125, 211, 252, 0.50),
    0 0 30px rgba(56, 189, 248, 0.35);
  /* 近层边缘光：drop-shadow 沿字形轮廓发光 */
  filter: drop-shadow(0 0 4px rgba(56, 189, 248, 0.9));
  animation:
    quote-fade-in 1.8s cubic-bezier(0.22, 1, 0.36, 1) 0.35s both,
    quote-core-pulse 4.6s ease-in-out 2.2s infinite,
    quote-edge-near 3.4s ease-in-out infinite;
}

/* 按标点切分出的短句：句内禁止换行，句间由 <wbr> 提供断点，保证只在标点处换行 */
.qseg { white-space: nowrap; }

@keyframes quote-fade-in {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* 主文字核心辉光脉动 */
@keyframes quote-core-pulse {
  0%, 100% {
    text-shadow:
      0 0 3px rgba(224, 242, 254, 0.85),
      0 0 12px rgba(125, 211, 252, 0.50),
      0 0 30px rgba(56, 189, 248, 0.35);
  }
  50% {
    text-shadow:
      0 0 4px rgba(224, 242, 254, 1),
      0 0 18px rgba(125, 211, 252, 0.72),
      0 0 44px rgba(56, 189, 248, 0.50);
  }
}

/* 近层边缘光：快速小幅脉动 */
@keyframes quote-edge-near {
  0%, 100% { filter: drop-shadow(0 0 4px rgba(56, 189, 248, 0.9)); }
  50% { filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.55)); }
}

/* 远层晕光：慢速大幅呼吸，与近层周期错开形成干涉感 */
@keyframes quote-edge-far {
  0%, 100% { filter: drop-shadow(0 0 14px rgba(37, 99, 235, 0.55)); }
  50% { filter: drop-shadow(0 0 24px rgba(37, 99, 235, 0.32)); }
}

@media (max-width: 1500px) {
  .quote-text { font-size: 22px; letter-spacing: 0.1em; }
}

/* 中窄屏下顶部中央会被搜索框/图例挤占，直接隐藏签名 */
@media (max-width: 1199px) {
  .quote-panel { display: none; }
}

@media (prefers-reduced-motion: reduce) {
  .quote-panel {
    animation: none;
    filter: drop-shadow(0 0 14px rgba(37, 99, 235, 0.45));
  }
  .quote-text {
    animation: quote-fade-in 0.6s ease-out both;
    filter: drop-shadow(0 0 5px rgba(56, 189, 248, 0.7));
  }
}
</style>
