import { el, slider, canvas, theme, onTheme } from './ui.mjs';

// 時刻tで切った断面。中心は傾き(速度)で動き、重みは寿命で減る。
export function slice({ mx, mt, velocity, life }, t) {
  return { center: mx + velocity * (t - mt), weight: Math.exp(-0.5 * ((t - mt) / life) ** 2) };
}

export function mount(root) {
  const view = canvas(640, 380, '横軸が時刻、縦軸が位置の平面に置いた時空間Gaussian。縦線の時刻で切った断面を下の帯に表示する。');
  const time = slider('時刻 t', { min: 0, max: 1, step: 0.01, value: 0.3, format: v => v.toFixed(2) });
  const velocity = slider('速度 Σxt / Σtt', { min: -1, max: 1, step: 0.05, value: 0.6, format: v => v.toFixed(2) });
  const life = slider('時間方向の広がり √Σtt', { min: 0.05, max: 0.6, step: 0.01, value: 0.2, format: v => v.toFixed(2) });
  const play = el('button', { type: 'button' }, '再生');
  const readout = el('pre', { className: 'lab-readout' });
  const blob = { mx: 0.5, mt: 0.5, spread: 0.06 };
  const plot = { x: 56, y: 24, w: 560, h: 240 }, strip = { y: 316, h: 40 };
  let timer = null;

  function draw() {
    const c = theme(), g = view.context;
    const model = { ...blob, velocity: velocity.value, life: life.value };
    const now = slice(model, time.value);
    const px = t => plot.x + t * plot.w, py = x => plot.y + (1 - x) * plot.h;
    g.fillStyle = c.paper;
    g.fillRect(0, 0, view.width, view.height);
    g.strokeStyle = c.line;
    g.lineWidth = 1;
    g.strokeRect(plot.x, plot.y, plot.w, plot.h);
    g.font = '13px sans-serif';
    g.fillStyle = c.muted;
    g.fillText('位置 x', 8, plot.y + 14);
    g.fillText('時刻 t →', plot.x + plot.w - 60, plot.y + plot.h + 18);
    g.fillText('時刻 t の断面', 8, strip.y - 8);

    g.save();
    g.beginPath(); g.rect(plot.x, plot.y, plot.w, plot.h); g.clip();
    // (t, x)平面の輪郭: tを±kσtで動かし、xは中心線±空間の広がり
    for (const k of [2, 1]) {
      g.beginPath();
      for (let i = 0; i <= 80; i += 1) {
        const a = i / 80 * 2 * Math.PI;
        const t = blob.mt + k * life.value * Math.cos(a);
        const x = blob.mx + velocity.value * (t - blob.mt) + k * blob.spread * Math.sin(a);
        if (i === 0) g.moveTo(px(t), py(x)); else g.lineTo(px(t), py(x));
      }
      g.fillStyle = c.accent + (k === 1 ? '55' : '22');
      g.strokeStyle = c.muted;
      g.fill(); g.stroke();
    }
    g.setLineDash([5, 4]);
    g.beginPath(); g.moveTo(px(0), py(slice(model, 0).center)); g.lineTo(px(1), py(slice(model, 1).center)); g.stroke();
    g.setLineDash([]);
    g.restore();
    g.strokeStyle = c.warm;
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(px(time.value), plot.y); g.lineTo(px(time.value), plot.y + plot.h); g.stroke();
    g.fillStyle = c.warm;
    g.beginPath(); g.arc(px(time.value), py(now.center), 5, 0, 2 * Math.PI); g.fill();

    g.strokeStyle = c.line;
    g.lineWidth = 1;
    g.strokeRect(plot.x, strip.y, plot.w, strip.h);
    g.save();
    g.beginPath(); g.rect(plot.x, strip.y, plot.w, strip.h); g.clip();
    const cx = plot.x + now.center * plot.w, radius = blob.spread * plot.w * 3;
    const fill = g.createRadialGradient(cx, strip.y + strip.h / 2, 0, cx, strip.y + strip.h / 2, radius);
    for (let r = 0; r <= 3; r += 0.25) fill.addColorStop(r / 3, c.accent + Math.round(255 * now.weight * Math.exp(-0.5 * r * r)).toString(16).padStart(2, '0'));
    g.fillStyle = fill;
    g.fillRect(plot.x, strip.y, plot.w, strip.h);
    g.restore();

    readout.textContent = `中心 μx(t) = ${blob.mx.toFixed(2)} + ${velocity.value.toFixed(2)} × (${time.value.toFixed(2)} − ${blob.mt.toFixed(2)}) = ${now.center.toFixed(3)}\n時刻 t での濃さの倍率 = ${now.weight.toFixed(3)}`;
  }

  play.addEventListener('click', () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = '再生'; return; }
    play.textContent = '停止';
    timer = setInterval(() => { time.value = time.value >= 1 ? 0 : Math.min(1, time.value + 0.01); draw(); }, 40);
  });
  root.append(el('div', { className: 'lab-frame' }, view.node), el('div', { className: 'controls' }, time.root, velocity.root, life.root, el('div', { className: 'lab-actions' }, play)), readout);
  [time, velocity, life].forEach(control => control.input.addEventListener('input', draw));
  onTheme(draw);
  draw();
}
