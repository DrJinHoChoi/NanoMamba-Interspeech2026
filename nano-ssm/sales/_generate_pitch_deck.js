const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_16x9';
pptx.author = 'NC-SSM Project';
pptx.title = 'NC-SSM Pitch Deck';

// ── Color Palette ──
const BG_DARK   = '0a0a0f';
const BG_CARD   = '12121a';
const CYAN      = '00e5ff';
const PURPLE    = '7c4dff';
const GREEN     = '00e676';
const WHITE     = 'ffffff';
const GRAY      = '8a8a9a';
const DIM       = '555566';

// ── Helpers ──
function darkBg(slide) {
  slide.background = { color: BG_DARK };
}

function accentBar(slide, { x = 0, y = 0, w = 0.08, h = 1.2, color = CYAN } = {}) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color } });
}

function topGradientBar(slide) {
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: '100%', h: 0.04, fill: { color: CYAN } });
}

function bottomBar(slide) {
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 7.2, w: '100%', h: 0.04, fill: { color: PURPLE } });
}

function slideNumber(slide, num) {
  slide.addText(String(num).padStart(2, '0'), {
    x: 12.1, y: 6.9, w: 0.7, h: 0.4,
    fontSize: 10, color: DIM, align: 'right', fontFace: 'Consolas'
  });
}

function sectionTitle(slide, title, subtitle, num) {
  darkBg(slide);
  topGradientBar(slide);
  bottomBar(slide);
  accentBar(slide, { x: 0.6, y: 1.1, w: 0.06, h: 0.55, color: CYAN });
  slide.addText(title, {
    x: 0.85, y: 1.0, w: 11, h: 0.7,
    fontSize: 28, bold: true, color: WHITE, fontFace: 'Segoe UI'
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.85, y: 1.65, w: 11, h: 0.4,
      fontSize: 14, color: GRAY, fontFace: 'Segoe UI'
    });
  }
  slideNumber(slide, num);
}

function metricBox(slide, { x, y, w = 2.4, h = 1.1, value, label, accent = CYAN }) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: BG_CARD }, rectRadius: 0.08 });
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h: 0.035, fill: { color: accent }, rectRadius: 0.02 });
  slide.addText(value, {
    x, y: y + 0.15, w, h: 0.55,
    fontSize: 22, bold: true, color: accent, align: 'center', fontFace: 'Consolas'
  });
  slide.addText(label, {
    x, y: y + 0.65, w, h: 0.35,
    fontSize: 10, color: GRAY, align: 'center', fontFace: 'Segoe UI'
  });
}

function bulletList(slide, items, { x = 0.85, y = 2.3, w = 11, fontSize = 13, lineSpacing = 28 } = {}) {
  const textRows = items.map(item => {
    if (typeof item === 'string') {
      return [
        { text: '\u2022  ', options: { color: CYAN, fontSize, bold: true } },
        { text: item, options: { color: WHITE, fontSize } }
      ];
    }
    // item is { highlight, rest }
    return [
      { text: '\u2022  ', options: { color: CYAN, fontSize, bold: true } },
      { text: item.highlight, options: { color: CYAN, fontSize, bold: true } },
      { text: item.rest, options: { color: WHITE, fontSize } }
    ];
  });
  // Flatten into a single text body with line breaks
  const body = [];
  textRows.forEach((row, i) => {
    row.forEach(seg => body.push(seg));
    if (i < textRows.length - 1) body.push({ text: '\n', options: { fontSize: 6 } });
  });
  slide.addText(body, {
    x, y, w, h: 4.5,
    valign: 'top', fontFace: 'Segoe UI', lineSpacingMultiple: 1.35,
    paraSpaceAfter: 6
  });
}

// ════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ════════════════════════════════════════════════════════════════════
(() => {
  const slide = pptx.addSlide();
  darkBg(slide);

  // Large decorative shapes
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: '100%', h: 0.05, fill: { color: CYAN } });
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 7.15, w: '100%', h: 0.35, fill: { color: BG_CARD } });

  // Decorative accent rectangles
  slide.addShape(pptx.ShapeType.rect, { x: 0.6, y: 2.0, w: 0.08, h: 1.8, fill: { color: CYAN } });
  slide.addShape(pptx.ShapeType.rect, { x: 11.5, y: 1.5, w: 1.2, h: 0.06, fill: { color: PURPLE } });
  slide.addShape(pptx.ShapeType.rect, { x: 11.0, y: 4.5, w: 1.7, h: 0.06, fill: { color: GREEN } });

  slide.addText('NC-SSM', {
    x: 0.9, y: 1.6, w: 10, h: 1.2,
    fontSize: 54, bold: true, color: CYAN, fontFace: 'Segoe UI'
  });
  slide.addText('Voice AI on a $5 Chip', {
    x: 0.9, y: 2.7, w: 10, h: 0.7,
    fontSize: 26, color: WHITE, fontFace: 'Segoe UI'
  });
  slide.addText('Noise-Conditioned State Space Model\nfor Edge Keyword Spotting', {
    x: 0.9, y: 3.5, w: 10, h: 0.9,
    fontSize: 16, color: GRAY, fontFace: 'Segoe UI', lineSpacingMultiple: 1.4
  });
  slide.addText([
    { text: 'Interspeech 2026', options: { color: CYAN } },
    { text: '  |  ', options: { color: DIM } },
    { text: 'CES 2027', options: { color: PURPLE } },
    { text: '  |  ', options: { color: DIM } },
    { text: 'Patent Pending', options: { color: GREEN } }
  ], {
    x: 0.9, y: 4.8, w: 10, h: 0.4,
    fontSize: 12, fontFace: 'Segoe UI'
  });
  slideNumber(slide, 1);
})();

// ════════════════════════════════════════════════════════════════════
// SLIDE 2 — Problem
// ════════════════════════════════════════════════════════════════════
(() => {
  const slide = pptx.addSlide();
  sectionTitle(slide, 'The Problem', 'Voice AI is broken at the edge', 2);

  bulletList(slide, [
    { highlight: 'Expensive Hardware: ', rest: 'Current voice AI requires powerful processors, driving up BOM cost and power budget' },
    { highlight: 'Noise Failure: ', rest: 'Models trained in clean conditions collapse in real-world noise (factory, car, kitchen)' },
    { highlight: 'High Latency: ', rest: 'DS-CNN-S: 24K params, 53ms latency -- too slow for real-time interaction on MCUs' },
    { highlight: 'Power Hungry: ', rest: 'Always-on listening drains batteries -- CNNs process every frame even during silence' },
    { highlight: 'No Streaming: ', rest: 'Conventional models need the full 1-second window before detection begins' }
  ], { y: 2.3 });

  // Problem stat boxes
  metricBox(slide, { x: 0.85, y: 5.3, value: '53ms', label: 'CNN Latency', accent: 'ff5252' });
  metricBox(slide, { x: 3.45, y: 5.3, value: '24K', label: 'DS-CNN-S Params', accent: 'ff5252' });
  metricBox(slide, { x: 6.05, y: 5.3, value: '196KB', label: 'CNN Feature Maps', accent: 'ff5252' });
  metricBox(slide, { x: 8.65, y: 5.3, value: '66.3%', label: 'CNN @ 0dB SNR', accent: 'ff5252' });
})();

// ════════════════════════════════════════════════════════════════════
// SLIDE 3 — Solution
// ════════════════════════════════════════════════════════════════════
(() => {
  const slide = pptx.addSlide();
  sectionTitle(slide, 'The Solution', 'Structural noise robustness -- no denoising module needed', 3);

  bulletList(slide, [
    { highlight: 'SSM Architecture: ', rest: 'State Space Models process sequences with O(1) memory per step -- ideal for streaming on MCUs' },
    { highlight: 'Noise-Conditioning: ', rest: '5 mechanisms that adapt model dynamics in real-time based on per-band SNR estimates' },
    { highlight: 'Zero Overhead: ', rest: 'No separate denoising module, no extra MACs -- noise robustness is built into the architecture' },
    { highlight: 'Silence Efficiency: ', rest: 'SSM state update costs 0 MACs during silence, enabling 10x battery life improvement' },
    { highlight: 'Streaming Native: ', rest: '~350ms detection for short keywords vs 1053ms for CNN -- instant wake-word response' }
  ], { y: 2.3 });

  // Solution stat boxes
  metricBox(slide, { x: 0.85, y: 5.3, value: '7.1ms', label: 'NC-SSM Latency', accent: CYAN });
  metricBox(slide, { x: 3.45, y: 5.3, value: '7,443', label: 'Parameters', accent: CYAN });
  metricBox(slide, { x: 6.05, y: 5.3, value: '720B', label: 'Hidden State', accent: GREEN });
  metricBox(slide, { x: 8.65, y: 5.3, value: '77.7%', label: 'Accuracy @ 0dB', accent: GREEN });
})();

// ════════════════════════════════════════════════════════════════════
// SLIDE 4 — Technology / Architecture
// ════════════════════════════════════════════════════════════════════
(() => {
  const slide = pptx.addSlide();
  sectionTitle(slide, 'Technology', 'NC-SSM Architecture: End-to-End Noise-Conditioned Pipeline', 4);

  // Pipeline flow boxes
  const pipeline = [
    { label: '1s Audio', color: DIM },
    { label: 'STFT', color: DIM },
    { label: '40-band\nMel', color: DIM },
    { label: 'SNR\nEstimation', color: PURPLE },
    { label: 'DualPCEN', color: PURPLE },
    { label: 'SSM\nBlock x2', color: CYAN },
    { label: 'Classifier', color: GREEN },
  ];
  const boxW = 1.45;
  const gap = 0.15;
  const startX = 0.85;
  const pipeY = 2.4;
  pipeline.forEach((item, i) => {
    const bx = startX + i * (boxW + gap);
    slide.addShape(pptx.ShapeType.rect, {
      x: bx, y: pipeY, w: boxW, h: 0.85,
      fill: { color: BG_CARD }, line: { color: item.color, width: 1.5 }, rectRadius: 0.06
    });
    slide.addText(item.label, {
      x: bx, y: pipeY, w: boxW, h: 0.85,
      fontSize: 11, bold: true, color: item.color, align: 'center', valign: 'middle', fontFace: 'Consolas'
    });
    // Arrow between boxes
    if (i < pipeline.length - 1) {
      slide.addText('\u25B6', {
        x: bx + boxW - 0.02, y: pipeY + 0.15, w: gap + 0.04, h: 0.55,
        fontSize: 10, color: DIM, align: 'center', valign: 'middle'
      });
    }
  });

  // Noise-conditioning mechanisms
  slide.addText('5 Noise-Conditioning Mechanisms', {
    x: 0.85, y: 3.7, w: 11, h: 0.45,
    fontSize: 16, bold: true, color: PURPLE, fontFace: 'Segoe UI'
  });

  const mechanisms = [
    { name: 'Sub-band\nSelectivity', desc: 'Per-frequency\nattention gating' },
    { name: 'Delta Floor\nConditioning', desc: 'Adaptive noise\nfloor estimation' },
    { name: 'B-base\nConditioning', desc: 'Dynamic SSM\nmatrix scaling' },
    { name: 'Learned\nSpectral Gate', desc: 'SNR-driven\nfeature masking' },
    { name: 'Spectral\nSubtraction', desc: 'Noise spectrum\nremoval' }
  ];
  const mechW = 2.1;
  const mechGap = 0.15;
  const mechY = 4.3;
  mechanisms.forEach((m, i) => {
    const mx = 0.85 + i * (mechW + mechGap);
    slide.addShape(pptx.ShapeType.rect, {
      x: mx, y: mechY, w: mechW, h: 1.5,
      fill: { color: BG_CARD }, rectRadius: 0.06
    });
    slide.addShape(pptx.ShapeType.rect, {
      x: mx, y: mechY, w: mechW, h: 0.035, fill: { color: PURPLE }, rectRadius: 0.02
    });
    slide.addText(m.name, {
      x: mx, y: mechY + 0.1, w: mechW, h: 0.65,
      fontSize: 11, bold: true, color: CYAN, align: 'center', valign: 'middle', fontFace: 'Segoe UI'
    });
    slide.addText(m.desc, {
      x: mx, y: mechY + 0.8, w: mechW, h: 0.55,
      fontSize: 9, color: GRAY, align: 'center', valign: 'middle', fontFace: 'Segoe UI'
    });
  });
})();

// ════════════════════════════════════════════════════════════════════
// SLIDE 5 — Performance
// ════════════════════════════════════════════════════════════════════
(() => {
  const slide = pptx.addSlide();
  sectionTitle(slide, 'Performance', 'Benchmarks on Google Speech Commands v2 (12-class)', 5);

  // Table header
  const tableY = 2.5;
  const cols = ['Model', 'Params', 'MACs', 'Latency', 'Accuracy', '0dB SNR'];
  const colW = [2.8, 1.5, 1.5, 1.4, 1.4, 1.5];
  let cx = 0.85;
  cols.forEach((col, i) => {
    slide.addShape(pptx.ShapeType.rect, {
      x: cx, y: tableY, w: colW[i], h: 0.45, fill: { color: BG_CARD }
    });
    slide.addText(col, {
      x: cx, y: tableY, w: colW[i], h: 0.45,
      fontSize: 10, bold: true, color: CYAN, align: 'center', valign: 'middle', fontFace: 'Consolas'
    });
    cx += colW[i] + 0.05;
  });

  const rows = [
    { data: ['DS-CNN-S (baseline)', '24,000', '5.4M', '53ms', '95.4%', '66.3%'], accent: 'ff5252' },
    { data: ['NC-SSM (ours)', '7,443', '0.86M', '7.1ms', '95.3%', '77.7%'], accent: CYAN },
    { data: ['NC-SSM-20K (ours)', '20,000', '2.44M', '10.4ms', '96.4%', '--'], accent: GREEN }
  ];
  rows.forEach((row, ri) => {
    const ry = tableY + 0.5 + ri * 0.55;
    let rx = 0.85;
    row.data.forEach((val, ci) => {
      slide.addShape(pptx.ShapeType.rect, {
        x: rx, y: ry, w: colW[ci], h: 0.48,
        fill: { color: ri === 0 ? '15151f' : '0e0e18' },
        line: { color: '222233', width: 0.5 }
      });
      slide.addText(val, {
        x: rx, y: ry, w: colW[ci], h: 0.48,
        fontSize: 11, color: ci === 0 ? (ri === 0 ? GRAY : row.accent) : WHITE,
        bold: ci === 0, align: 'center', valign: 'middle', fontFace: 'Consolas'
      });
      rx += colW[ci] + 0.05;
    });
  });

  // Key takeaway boxes
  metricBox(slide, { x: 0.85, y: 5.2, w: 3.2, value: '5x less latency', label: 'vs DS-CNN-S at matched accuracy', accent: CYAN });
  metricBox(slide, { x: 4.3, y: 5.2, w: 3.2, value: '10x fewer MACs', label: 'NC-SSM-20K vs DS-CNN-S', accent: GREEN });
  metricBox(slide, { x: 7.75, y: 5.2, w: 3.2, value: '+11.4% @ 0dB', label: 'Noise robustness (clean-trained)', accent: PURPLE });
})();

// ════════════════════════════════════════════════════════════════════
// SLIDE 6 — Competitive Advantage
// ════════════════════════════════════════════════════════════════════
(() => {
  const slide = pptx.addSlide();
  sectionTitle(slide, 'Competitive Advantage', 'Why NC-SSM wins at the edge', 6);

  // Advantage cards - 2x2 grid
  const cards = [
    { title: 'Proven Noise Robustness', body: 'Only KWS model with formal structural\nnoise robustness proof', icon: '\u2714', accent: CYAN },
    { title: 'Instant Streaming', body: '~350ms detection for short words\nvs 1053ms for CNN (3x faster)', icon: '\u26A1', accent: GREEN },
    { title: '720 Bytes State', body: '720 bytes hidden state vs 196KB\nCNN feature maps (272x smaller)', icon: '\u2B21', accent: PURPLE },
    { title: 'Zero Silence Cost', body: '0 MACs during silence periods\n= 10x battery life improvement', icon: '\u2600', accent: CYAN }
  ];
  const cardW = 5.1;
  const cardH = 1.6;
  cards.forEach((card, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const cx = 0.85 + col * (cardW + 0.3);
    const cy = 2.4 + row * (cardH + 0.25);

    slide.addShape(pptx.ShapeType.rect, {
      x: cx, y: cy, w: cardW, h: cardH,
      fill: { color: BG_CARD }, rectRadius: 0.08
    });
    slide.addShape(pptx.ShapeType.rect, {
      x: cx, y: cy, w: 0.05, h: cardH, fill: { color: card.accent }, rectRadius: 0.03
    });
    slide.addText(card.title, {
      x: cx + 0.25, y: cy + 0.15, w: cardW - 0.4, h: 0.4,
      fontSize: 15, bold: true, color: card.accent, fontFace: 'Segoe UI'
    });
    slide.addText(card.body, {
      x: cx + 0.25, y: cy + 0.6, w: cardW - 0.4, h: 0.85,
      fontSize: 12, color: GRAY, fontFace: 'Segoe UI', lineSpacingMultiple: 1.3
    });
  });
})();

// ════════════════════════════════════════════════════════════════════
// SLIDE 7 — Business Model
// ════════════════════════════════════════════════════════════════════
(() => {
  const slide = pptx.addSlide();
  sectionTitle(slide, 'Business Model', 'Multiple revenue streams from a single core technology', 7);

  const streams = [
    { title: 'IP Licensing', price: '$0.01-0.05/chip', desc: 'Per-chip royalty for\nNC-SSM core IP', accent: CYAN },
    { title: 'Nano AI SDK', price: '$500-5K/mo', desc: 'SaaS platform for\ncustom model training', accent: PURPLE },
    { title: 'Custom Wake Word', price: '$10K-50K/project', desc: 'Turnkey custom keyword\ndetection solutions', accent: GREEN },
    { title: 'Edge AI Module', price: 'BOM $4.80 / Sell $15-30', desc: 'Ready-to-integrate\nhardware module', accent: CYAN }
  ];
  const sw = 2.55;
  const sh = 2.6;
  streams.forEach((s, i) => {
    const sx = 0.85 + i * (sw + 0.2);
    const sy = 2.4;
    slide.addShape(pptx.ShapeType.rect, {
      x: sx, y: sy, w: sw, h: sh,
      fill: { color: BG_CARD }, rectRadius: 0.08
    });
    slide.addShape(pptx.ShapeType.rect, {
      x: sx, y: sy, w: sw, h: 0.04, fill: { color: s.accent }, rectRadius: 0.02
    });
    slide.addText(s.title, {
      x: sx, y: sy + 0.2, w: sw, h: 0.4,
      fontSize: 14, bold: true, color: s.accent, align: 'center', fontFace: 'Segoe UI'
    });
    slide.addText(s.price, {
      x: sx, y: sy + 0.7, w: sw, h: 0.45,
      fontSize: 16, bold: true, color: WHITE, align: 'center', fontFace: 'Consolas'
    });
    slide.addShape(pptx.ShapeType.rect, {
      x: sx + 0.4, y: sy + 1.25, w: sw - 0.8, h: 0.01, fill: { color: DIM }
    });
    slide.addText(s.desc, {
      x: sx, y: sy + 1.4, w: sw, h: 0.9,
      fontSize: 11, color: GRAY, align: 'center', fontFace: 'Segoe UI', lineSpacingMultiple: 1.3
    });
  });
})();

// ════════════════════════════════════════════════════════════════════
// SLIDE 8 — Market
// ════════════════════════════════════════════════════════════════════
(() => {
  const slide = pptx.addSlide();
  sectionTitle(slide, 'Market Opportunity', 'Riding the edge AI wave', 8);

  // Market size boxes
  metricBox(slide, { x: 0.85, y: 2.5, w: 3.5, h: 1.3, value: '$26.8B', label: 'Global Voice Recognition (2025)', accent: CYAN });
  // Arrow
  slide.addText('\u27A1', {
    x: 4.5, y: 2.8, w: 0.8, h: 0.7,
    fontSize: 28, color: DIM, align: 'center', valign: 'middle'
  });
  metricBox(slide, { x: 5.5, y: 2.5, w: 3.5, h: 1.3, value: '$50B+', label: 'Projected Market (2030)', accent: GREEN });

  slide.addText('Always-on edge AI: fastest growing segment', {
    x: 9.3, y: 2.7, w: 3.2, h: 0.9,
    fontSize: 13, bold: true, color: PURPLE, fontFace: 'Segoe UI', align: 'center'
  });

  // Target verticals
  slide.addText('Target Verticals', {
    x: 0.85, y: 4.2, w: 11, h: 0.4,
    fontSize: 16, bold: true, color: WHITE, fontFace: 'Segoe UI'
  });

  const verticals = [
    { name: 'IoT / Smart Home', desc: 'Voice-controlled\nappliances & hubs' },
    { name: 'Automotive', desc: 'In-cabin voice\ncommands' },
    { name: 'Wearables', desc: 'Watches, earbuds,\nhealth devices' },
    { name: 'Industrial', desc: 'Factory floor,\nhands-free control' },
    { name: 'Consumer\nElectronics', desc: 'TVs, remotes,\nspeakers' }
  ];
  const vw = 2.0;
  verticals.forEach((v, i) => {
    const vx = 0.85 + i * (vw + 0.2);
    const vy = 4.8;
    slide.addShape(pptx.ShapeType.rect, {
      x: vx, y: vy, w: vw, h: 1.5,
      fill: { color: BG_CARD }, rectRadius: 0.06
    });
    slide.addShape(pptx.ShapeType.rect, {
      x: vx, y: vy, w: vw, h: 0.035, fill: { color: CYAN }, rectRadius: 0.02
    });
    slide.addText(v.name, {
      x: vx, y: vy + 0.12, w: vw, h: 0.55,
      fontSize: 12, bold: true, color: CYAN, align: 'center', valign: 'middle', fontFace: 'Segoe UI'
    });
    slide.addText(v.desc, {
      x: vx, y: vy + 0.75, w: vw, h: 0.6,
      fontSize: 10, color: GRAY, align: 'center', valign: 'middle', fontFace: 'Segoe UI'
    });
  });
})();

// ════════════════════════════════════════════════════════════════════
// SLIDE 9 — Traction
// ════════════════════════════════════════════════════════════════════
(() => {
  const slide = pptx.addSlide();
  sectionTitle(slide, 'Traction & Milestones', 'From research to production-ready', 9);

  const milestones = [
    { status: 'DONE', text: 'Interspeech 2026 paper submitted', accent: GREEN },
    { status: 'DONE', text: 'CES 2027 demo ready (live inference)', accent: GREEN },
    { status: 'DONE', text: 'US + KR patent applications filed', accent: GREEN },
    { status: 'DONE', text: 'Working C SDK for ARM Cortex-M7', accent: GREEN },
    { status: 'DONE', text: 'FPGA implementation complete', accent: GREEN },
    { status: 'LIVE', text: 'Open-source community on GitHub', accent: CYAN }
  ];
  milestones.forEach((m, i) => {
    const my = 2.4 + i * 0.65;
    // Status badge
    slide.addShape(pptx.ShapeType.rect, {
      x: 0.85, y: my, w: 0.7, h: 0.45,
      fill: { color: BG_CARD }, rectRadius: 0.04
    });
    slide.addText(m.status, {
      x: 0.85, y: my, w: 0.7, h: 0.45,
      fontSize: 9, bold: true, color: m.accent, align: 'center', valign: 'middle', fontFace: 'Consolas'
    });
    // Milestone text
    slide.addText(m.text, {
      x: 1.7, y: my, w: 9, h: 0.45,
      fontSize: 14, color: WHITE, valign: 'middle', fontFace: 'Segoe UI'
    });
    // Connecting line
    if (i < milestones.length - 1) {
      slide.addShape(pptx.ShapeType.rect, {
        x: 1.18, y: my + 0.45, w: 0.02, h: 0.2, fill: { color: DIM }
      });
    }
  });

  // Right side: key IP box
  slide.addShape(pptx.ShapeType.rect, {
    x: 8.5, y: 2.4, w: 3.8, h: 3.8,
    fill: { color: BG_CARD }, rectRadius: 0.08
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 8.5, y: 2.4, w: 3.8, h: 0.04, fill: { color: PURPLE }
  });
  slide.addText('IP Portfolio', {
    x: 8.5, y: 2.6, w: 3.8, h: 0.4,
    fontSize: 14, bold: true, color: PURPLE, align: 'center', fontFace: 'Segoe UI'
  });
  const ipItems = [
    'NC-SSM Architecture',
    'DualPCEN Front-end',
    '5 Noise-Conditioning\nMechanisms',
    'Streaming SSM\nInference Method',
    'Silence-Aware\nPower Management'
  ];
  ipItems.forEach((item, i) => {
    slide.addText([
      { text: '\u25C6  ', options: { color: PURPLE, fontSize: 11 } },
      { text: item, options: { color: GRAY, fontSize: 11 } }
    ], {
      x: 8.8, y: 3.15 + i * 0.55, w: 3.2, h: 0.5,
      fontFace: 'Segoe UI', valign: 'middle'
    });
  });
})();

// ════════════════════════════════════════════════════════════════════
// SLIDE 10 — Contact / CTA
// ════════════════════════════════════════════════════════════════════
(() => {
  const slide = pptx.addSlide();
  darkBg(slide);

  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: '100%', h: 0.05, fill: { color: CYAN } });
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 7.15, w: '100%', h: 0.35, fill: { color: BG_CARD } });

  // Decorative
  slide.addShape(pptx.ShapeType.rect, { x: 0.6, y: 2.2, w: 0.08, h: 1.0, fill: { color: CYAN } });
  slide.addShape(pptx.ShapeType.rect, { x: 11.5, y: 2.0, w: 1.2, h: 0.06, fill: { color: PURPLE } });

  slide.addText('Deploy Noise-Robust\nVoice AI Today', {
    x: 0.9, y: 1.8, w: 10, h: 1.5,
    fontSize: 40, bold: true, color: WHITE, fontFace: 'Segoe UI', lineSpacingMultiple: 1.2
  });

  slide.addText('NC-SSM: 7,443 params. 7.1ms latency. Runs on a $5 chip.', {
    x: 0.9, y: 3.4, w: 10, h: 0.5,
    fontSize: 16, color: CYAN, fontFace: 'Consolas'
  });

  // Contact card
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.9, y: 4.3, w: 5.5, h: 2.2,
    fill: { color: BG_CARD }, rectRadius: 0.08
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.9, y: 4.3, w: 5.5, h: 0.04, fill: { color: CYAN }
  });

  slide.addText('Dr. Jin Ho Choi', {
    x: 1.15, y: 4.5, w: 5, h: 0.45,
    fontSize: 18, bold: true, color: WHITE, fontFace: 'Segoe UI'
  });
  slide.addText('NC-SSM Project Lead', {
    x: 1.15, y: 4.95, w: 5, h: 0.35,
    fontSize: 12, color: CYAN, fontFace: 'Segoe UI'
  });

  const contactLines = [
    'github.com/nc-ssm',
    'Interspeech 2026 | CES 2027',
    'US + KR Patent Pending'
  ];
  contactLines.forEach((line, i) => {
    slide.addText(line, {
      x: 1.15, y: 5.4 + i * 0.32, w: 5, h: 0.3,
      fontSize: 11, color: GRAY, fontFace: 'Segoe UI'
    });
  });

  // Right side CTA boxes
  const ctaItems = [
    { text: 'Schedule a Demo', accent: CYAN },
    { text: 'License the IP', accent: PURPLE },
    { text: 'Try the SDK', accent: GREEN }
  ];
  ctaItems.forEach((cta, i) => {
    const cy = 4.3 + i * 0.75;
    slide.addShape(pptx.ShapeType.rect, {
      x: 7.0, y: cy, w: 4.5, h: 0.6,
      fill: { color: BG_CARD }, line: { color: cta.accent, width: 1.5 }, rectRadius: 0.06
    });
    slide.addText(cta.text, {
      x: 7.0, y: cy, w: 4.5, h: 0.6,
      fontSize: 14, bold: true, color: cta.accent, align: 'center', valign: 'middle', fontFace: 'Segoe UI'
    });
  });

  slideNumber(slide, 10);
})();

// ── Write file ──
const outPath = 'C:/Users/jinho/Downloads/NanoMamba-Interspeech2026/nano-ssm/sales/NC-SSM_Pitch_Deck.pptx';
pptx.writeFile({ fileName: outPath })
  .then(() => console.log('SUCCESS: ' + outPath))
  .catch(err => { console.error('FAILED:', err); process.exit(1); });
