import { gaussianWeight, composite } from '../lib/gaussian.mjs';
import { el, slider, choice, pixels, theme, onTheme, ACCENT_RGB, SECOND_RGB } from './ui.mjs';

const colors = { a: ACCENT_RGB, b: SECOND_RGB };

export function mount(root) {
  const view = pixels(640, 320, '色の違う二つのGaussian AとBの合成。中央の十字は値を読み取る画素。');
  const sx = slider('Aの横の標準偏差', { min: 15, max: 85, value: 55 });
  const sy = slider('Aの縦の標準偏差', { min: 15, max: 75, value: 28 });
  const angle = slider('Aの回転', { min: -90, max: 90, value: 20, format: v => `${v}°` });
  const alpha = slider('Aのピーク不透明度', { min: 0, max: 1, step: 0.05, value: 0.5, format: v => v.toFixed(2) });
  const beta = slider('Bのピーク不透明度', { min: 0, max: 1, step: 0.05, value: 0.5, format: v => v.toFixed(2) });
  const front = choice('前後関係', { a: 'Aが手前', b: 'Bが手前' });
  const readout = el('pre', { className: 'lab-readout' });
  readout.setAttribute('aria-live', 'polite');
  const rgb = new Float32Array(640 * 320 * 3);

  function draw() {
    const c = theme();
    const a = { cx: 275, cy: 160, sx: sx.value, sy: sy.value, angle: angle.value * Math.PI / 180 };
    const b = { cx: 360, cy: 160, sx: 52, sy: 42, angle: -20 * Math.PI / 180 };
    const samples = (x, y) => {
      const list = [
        { name: 'A', rgb: colors.a, alpha: alpha.value * gaussianWeight(x, y, a) },
        { name: 'B', rgb: colors.b, alpha: beta.value * gaussianWeight(x, y, b) },
      ];
      return front.value === 'b' ? list.reverse() : list;
    };
    for (let y = 0; y < 320; y += 1) {
      for (let x = 0; x < 640; x += 1) rgb.set(composite(samples(x + 0.5, y + 0.5), c.background).rgb, (y * 640 + x) * 3);
    }
    view.put(rgb);
    const g = view.context;
    g.strokeStyle = c.ink;
    g.lineWidth = 1.5;
    g.beginPath();
    g.moveTo(314, 160); g.lineTo(326, 160); g.moveTo(320, 154); g.lineTo(320, 166);
    g.stroke();
    const [first, second] = samples(320, 160);
    const center = composite([first, second], c.background);
    const w1 = first.alpha, w2 = (1 - first.alpha) * second.alpha, w3 = 1 - center.alpha;
    readout.textContent = `十字の画素の配分  手前${first.name} ${w1.toFixed(3)} ＋ 奥${second.name} ${w2.toFixed(3)} ＋ 背景 ${w3.toFixed(3)} ＝ 1\n合成不透明度 ${center.alpha.toFixed(3)} ／ RGB ${center.rgb.map(v => v.toFixed(3)).join(', ')}`;
  }

  root.append(el('div', { className: 'lab-frame' }, view.node), el('div', { className: 'controls' }, sx.root, sy.root, angle.root, alpha.root, beta.root, front.root), readout);
  [sx, sy, angle, alpha, beta, front].forEach(control => control.input.addEventListener('input', draw));
  onTheme(draw);
  draw();
}
