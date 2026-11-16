import { el, slider, canvas, theme, onTheme } from './ui.mjs';

const fx = 1000, screen = 1000; // 画素単位の焦点距離と画面幅

// 原点のカメラから見た1σ楕円の輪郭。接線の傾きx/zを二つ返す。
// カメラが楕円の内側にあるときと、楕円がカメラの面（z = 0）に届くときは、写る範囲に端がないので null。
export function silhouette(x0, z0, sx, sz) {
  const k = x0 * x0 / (sx * sx) + z0 * z0 / (sz * sz) - 1;
  if (k <= 0 || z0 <= sz) return null;
  const a = x0 * x0 / sx ** 4 - k / (sx * sx);
  const b = 2 * x0 * z0 / (sx * sx * sz * sz);
  const c = z0 * z0 / sz ** 4 - k / (sz * sz);
  const root = Math.sqrt(Math.max(0, b * b - 4 * a * c));
  return [(-b - root) / (2 * a), (-b + root) / (2 * a)].sort((p, q) => p - q);
}

export function mount(root) {
  const view = canvas(640, 400, '上から見たカメラとGaussian、その下に画面上の広がり。上段は線形近似、下段は実際の1σ輪郭。');
  const depth = slider('奥行き z', { min: 0.5, max: 6, step: 0.05, value: 2, format: v => `${v.toFixed(2)} m` });
  const side = slider('横位置 x', { min: -1.2, max: 1.2, step: 0.05, value: 0, format: v => `${v.toFixed(2)} m` });
  const sx = slider('横の広がり σx', { min: 0.02, max: 0.5, step: 0.01, value: 0.1, format: v => `${v.toFixed(2)} m` });
  const sz = slider('奥行きの広がり σz', { min: 0.02, max: 0.8, step: 0.01, value: 0.1, format: v => `${v.toFixed(2)} m` });
  const readout = el('pre', { className: 'lab-readout' });
  const scale = 88, ox = 56, oy = 140; // 上面図: 右がz、上がx
  const stripY = 340, stripX = 16, stripW = 480;

  function draw() {
    const c = theme(), g = view.context;
    const z = depth.value, x = side.value;
    const mean = fx * x / z;
    const sigma = Math.sqrt((fx / z) ** 2 * sx.value ** 2 + (fx * x / z ** 2) ** 2 * sz.value ** 2);
    const edges = silhouette(x, z, sx.value, sz.value);
    g.fillStyle = c.paper;
    g.fillRect(0, 0, view.width, view.height);
    g.font = '13px sans-serif';
    g.fillStyle = c.muted;
    g.fillText('上から見た配置', 12, 20);
    g.fillText('画面（幅1000画素、fx = 1000画素）', 12, 300);

    g.strokeStyle = c.line;
    g.lineWidth = 1;
    g.setLineDash([4, 4]);
    g.beginPath(); g.moveTo(ox, oy); g.lineTo(630, oy); g.stroke();
    g.setLineDash([]);
    for (const m of [-0.5, 0.5]) { // 画角の端
      g.beginPath(); g.moveTo(ox, oy); g.lineTo(ox + 6.5 * scale, oy - m * 6.5 * scale); g.stroke();
    }
    g.fillStyle = c.ink;
    g.beginPath(); g.moveTo(ox, oy); g.lineTo(ox - 22, oy - 12); g.lineTo(ox - 22, oy + 12); g.fill();

    g.fillStyle = c.accent + '55';
    g.strokeStyle = c.accent;
    g.beginPath(); g.ellipse(ox + z * scale, oy - x * scale, sz.value * scale, sx.value * scale, 0, 0, 2 * Math.PI);
    g.fill(); g.stroke();
    if (edges) {
      g.strokeStyle = c.warm;
      for (const m of edges) { g.beginPath(); g.moveTo(ox, oy); g.lineTo(ox + 6.5 * scale, oy - m * 6.5 * scale); g.stroke(); }
    }

    const toStrip = u => stripX + (u + screen / 2) / screen * stripW;
    g.strokeStyle = c.line;
    g.strokeRect(stripX, stripY - 22, stripW, 44);
    g.save();
    g.beginPath(); g.rect(stripX, stripY - 22, stripW, 44); g.clip();
    g.fillStyle = c.accent + 'cc';
    g.fillRect(toStrip(mean - sigma), stripY - 22, sigma * 2 / screen * stripW, 22);
    if (edges) {
      g.fillStyle = c.warm + '88';
      g.fillRect(toStrip(fx * edges[0]), stripY, (edges[1] - edges[0]) * fx / screen * stripW, 22);
    }
    g.restore();
    g.fillStyle = c.accent; g.fillRect(stripX + stripW + 8, stripY - 16, 10, 10);
    g.fillStyle = c.warm; g.fillRect(stripX + stripW + 8, stripY + 6, 10, 10);
    g.fillStyle = c.ink; g.fillText('線形近似 ±1σ', stripX + stripW + 24, stripY - 6); g.fillText('実際の輪郭', stripX + stripW + 24, stripY + 16);

    const inside = (x / sx.value) ** 2 + (z / sz.value) ** 2 <= 1;
    const actual = edges ? `${(fx * edges[0]).toFixed(1)} 〜 ${(fx * edges[1]).toFixed(1)}` : inside ? 'カメラが楕円の内側' : '楕円がカメラの面（z = 0）に届くので、範囲に端がない';
    readout.textContent = `画面上の中心 u = fx·x/z = ${mean.toFixed(1)} 画素\n線形近似の標準偏差 ${sigma.toFixed(1)} 画素 → 1σの範囲 ${(mean - sigma).toFixed(1)} 〜 ${(mean + sigma).toFixed(1)}\n実際の1σ輪郭が写る範囲 ${actual}`;
  }

  root.append(el('div', { className: 'lab-frame' }, view.node), el('div', { className: 'controls' }, depth.root, side.root, sx.root, sz.root), readout);
  [depth, side, sx, sz].forEach(control => control.input.addEventListener('input', draw));
  onTheme(draw);
  draw();
}
