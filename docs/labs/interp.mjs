import { el, slider, choice, canvas, theme, onTheme } from './ui.mjs';

// 物体は弧を描いて動く。これが「正解」で、撮影したフレームの時刻でだけ観測できる。
export const truth = t => [140 + 360 * t, 230 - 560 * t * (1 - t)];

// 時刻tをはさむ二つの撮影フレームと、その間での進み具合を返す。
export function between(frames, t) {
  const s = t * (frames - 1), i = Math.min(frames - 2, Math.floor(s));
  return { before: i / (frames - 1), after: (i + 1) / (frames - 1), weight: s - i };
}

export function mount(root) {
  const view = canvas(640, 300, '弧を描いて動く物体を、撮影フレームの間の時刻で描画した図。破線の円が正解の位置。');
  const time = slider('時刻 t', { min: 0, max: 1, step: 0.005, value: 0.5, format: v => v.toFixed(2) });
  const frames = slider('撮影フレームの数', { min: 2, max: 9, value: 2 });
  const mode = choice('間の描画方法', { blend: '前後のフレームの粒を混ぜる', move: '同じ粒を直線で動かす' });
  const play = el('button', { type: 'button' }, '再生');
  const readout = el('pre', { className: 'lab-readout' });
  const offsets = [[0, 0], [15, 0], [-15, 0], [7.5, 13], [-7.5, 13], [7.5, -13], [-7.5, -13]];
  let timer = null;

  function draw() {
    const c = theme(), g = view.context, { before, after, weight } = between(frames.value, time.value);
    const a = truth(before), b = truth(after), now = truth(time.value);
    const blob = ([x, y], alpha) => offsets.forEach(([dx, dy]) => {
      const fill = g.createRadialGradient(x + dx, y + dy, 0, x + dx, y + dy, 27);
      for (let r = 0; r <= 3; r += 0.5) fill.addColorStop(r / 3, c.ink + Math.round(150 * alpha * (Math.exp(-0.5 * r * r) - Math.exp(-4.5))).toString(16).padStart(2, '0'));
      g.fillStyle = fill;
      g.fillRect(x + dx - 27, y + dy - 27, 54, 54);
    });
    g.fillStyle = c.paper;
    g.fillRect(0, 0, view.width, view.height);
    g.strokeStyle = c.line;
    g.setLineDash([4, 5]);
    g.beginPath();
    for (let i = 0; i <= 60; i += 1) { const [x, y] = truth(i / 60); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
    g.stroke();
    g.setLineDash([]);
    g.fillStyle = c.muted;
    for (let i = 0; i < frames.value; i += 1) { const [x, y] = truth(i / (frames.value - 1)); g.beginPath(); g.arc(x, y, 3, 0, 2 * Math.PI); g.fill(); }
    const moved = [a[0] + (b[0] - a[0]) * weight, a[1] + (b[1] - a[1]) * weight];
    if (mode.value === 'blend') { blob(a, 1 - weight); blob(b, weight); } else blob(moved, 1);
    g.strokeStyle = c.accent;
    g.lineWidth = 2.5;
    g.setLineDash([6, 4]);
    g.beginPath(); g.arc(now[0], now[1], 30, 0, 2 * Math.PI); g.stroke();
    g.setLineDash([]);
    g.lineWidth = 1;
    const gap = Math.hypot(b[0] - a[0], b[1] - a[1]), miss = Math.hypot(moved[0] - now[0], moved[1] - now[1]);
    readout.textContent = mode.value === 'blend'
      ? `${gap.toFixed(0)}画素離れた二つの像が、${(1 - weight).toFixed(2)} と ${weight.toFixed(2)} の濃さで重なっている`
      : `像は一つ。正解の位置（破線）から ${miss.toFixed(0)}画素ずれている`;
  }

  play.addEventListener('click', () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = '再生'; return; }
    play.textContent = '停止';
    timer = setInterval(() => { time.value = time.value >= 1 ? 0 : Math.min(1, time.value + 0.01); draw(); }, 40);
  });
  root.append(el('div', { className: 'lab-frame' }, view.node), el('div', { className: 'controls' }, time.root, frames.root, mode.root, el('div', { className: 'lab-actions' }, play)), readout);
  [time, frames, mode].forEach(control => control.input.addEventListener('input', draw));
  onTheme(draw);
  draw();
}
