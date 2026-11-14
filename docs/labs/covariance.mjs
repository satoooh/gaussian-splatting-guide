import { el, slider, canvas, theme, onTheme, arrow } from './ui.mjs';

export function mount(root) {
  const view = canvas(640, 320, 'スケールと回転で決まる楕円。矢印は二つの主軸、線は1σと2σの輪郭。');
  const sx = slider('スケール sx', { min: 0.4, max: 3, step: 0.1, value: 3, format: v => v.toFixed(1) });
  const sy = slider('スケール sy', { min: 0.4, max: 3, step: 0.1, value: 1, format: v => v.toFixed(1) });
  const angle = slider('回転', { min: -90, max: 90, value: 30, format: v => `${v}°` });
  const readout = el('pre', { className: 'lab-readout' });
  const unit = 24;

  function draw() {
    const c = theme(), g = view.context;
    const a = angle.value * Math.PI / 180, cos = Math.cos(a), sin = Math.sin(a);
    const xx = cos * cos * sx.value ** 2 + sin * sin * sy.value ** 2;
    const xy = cos * sin * (sx.value ** 2 - sy.value ** 2);
    const yy = sin * sin * sx.value ** 2 + cos * cos * sy.value ** 2;
    g.fillStyle = c.paper;
    g.fillRect(0, 0, view.width, view.height);
    g.strokeStyle = c.line;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, 160); g.lineTo(640, 160); g.moveTo(320, 0); g.lineTo(320, 320);
    g.stroke();
    g.save();
    g.translate(320, 160);
    g.rotate(-a); // 画面のyは下向きなので、数学の向きに合わせて反転する
    g.scale(sx.value * unit, sy.value * unit);
    const fill = g.createRadialGradient(0, 0, 0, 0, 0, 3);
    for (let r = 0; r <= 3; r += 0.25) fill.addColorStop(r / 3, c.accent + Math.round(200 * Math.exp(-0.5 * r * r)).toString(16).padStart(2, '0'));
    g.fillStyle = fill;
    g.beginPath(); g.arc(0, 0, 3, 0, 2 * Math.PI); g.fill();
    g.restore();
    g.strokeStyle = c.ink;
    for (const k of [1, 2]) {
      g.setLineDash(k === 1 ? [] : [5, 4]);
      g.beginPath(); g.ellipse(320, 160, sx.value * unit * k, sy.value * unit * k, -a, 0, 2 * Math.PI); g.stroke();
    }
    g.setLineDash([]);
    g.strokeStyle = c.ink;
    g.lineWidth = 2;
    arrow(g, 320, 160, 320 + cos * sx.value * unit, 160 - sin * sx.value * unit);
    arrow(g, 320, 160, 320 - sin * sy.value * unit, 160 - cos * sy.value * unit);
    const f = v => v.toFixed(2).padStart(6);
    readout.textContent = `Σ = [${f(xx)} ${f(xy)} ]   主軸の長さ（固有値の平方根） ${Math.max(sx.value, sy.value).toFixed(1)} と ${Math.min(sx.value, sy.value).toFixed(1)}\n    [${f(xy)} ${f(yy)} ]   実線は1σ、破線は2σ`;
  }

  root.append(el('div', { className: 'lab-frame' }, view.node), el('div', { className: 'controls' }, sx.root, sy.root, angle.root), readout);
  [sx, sy, angle].forEach(control => control.input.addEventListener('input', draw));
  onTheme(draw);
  draw();
}
