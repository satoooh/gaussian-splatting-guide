import { el, slider, pixels, theme, onTheme, ACCENT_RGB } from './ui.mjs';

// 3DGSの実装と同じ並びの実球面調和関数（次数3まで、16個）。
export function basis(x, y, z, out) {
  const xx = x * x, yy = y * y, zz = z * z;
  out[0] = 0.28209479177387814;
  out[1] = -0.4886025119029199 * y; out[2] = 0.4886025119029199 * z; out[3] = -0.4886025119029199 * x;
  out[4] = 1.0925484305920792 * x * y; out[5] = -1.0925484305920792 * y * z;
  out[6] = 0.31539156525252005 * (2 * zz - xx - yy);
  out[7] = -1.0925484305920792 * x * z; out[8] = 0.5462742152960396 * (xx - yy);
  out[9] = -0.5900435899266435 * y * (3 * xx - yy); out[10] = 2.890611442640554 * x * y * z;
  out[11] = -0.4570457994644658 * y * (4 * zz - xx - yy); out[12] = 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy);
  out[13] = -0.4570457994644658 * x * (4 * zz - xx - yy); out[14] = 1.445305721320277 * z * (xx - yy);
  out[15] = -0.5900435899266435 * x * (xx - 3 * yy);
  return out;
}

// 見る方向dに対する「本当の色」。差し色の拡散面に、一方向だけ強い白いハイライトを足す。
export function appearance(x, y, z) {
  const diffuse = 0.35 + 0.65 * Math.max(0, 0.5 * x + 0.5 * y + 0.707 * z);
  const highlight = Math.max(0, 0.6 * x - 0.3 * y + 0.742 * z) ** 20;
  return ACCENT_RGB.map(v => v * diffuse + highlight);
}

// 球面上の積分で係数を求める。c = ∫ f(d) Y(d) dd
export function project(samples = 4000) {
  const coefficients = new Float64Array(48), y = new Float64Array(16);
  for (let i = 0; i < samples; i += 1) {
    const z = 1 - 2 * (i + 0.5) / samples, r = Math.sqrt(1 - z * z), phi = i * 2.399963229728653;
    const d = [r * Math.cos(phi), r * Math.sin(phi), z];
    const rgb = appearance(...d);
    basis(...d, y);
    for (let k = 0; k < 16; k += 1) for (let c = 0; c < 3; c += 1) coefficients[k * 3 + c] += rgb[c] * y[k] * 4 * Math.PI / samples;
  }
  return coefficients;
}

export function mount(root) {
  const size = 200;
  const truth = pixels(size, size, '方向ごとの本来の色を球に描いた図');
  const approx = pixels(size, size, '球面調和関数で近似した色を球に描いた図');
  const degree = slider('次数 L', { min: 0, max: 3, value: 3 });
  const yaw = slider('見る向き', { min: -180, max: 180, value: 0, format: v => `${v}°` });
  const readout = el('pre', { className: 'lab-readout' });
  const coefficients = project();
  const a = new Float32Array(size * size * 3), b = new Float32Array(size * size * 3), y = new Float64Array(16);

  function draw() {
    const cos = Math.cos(yaw.value * Math.PI / 180), sin = Math.sin(yaw.value * Math.PI / 180);
    const count = (degree.value + 1) ** 2;
    const paper = theme().background;
    for (let p = 0; p < size * size; p += 1) { a.set(paper, p * 3); b.set(paper, p * 3); }
    let center = [0, 0, 0];
    for (let py = 0; py < size; py += 1) {
      for (let px = 0; px < size; px += 1) {
        const sx = (px + 0.5) / size * 2 - 1, sy = 1 - (py + 0.5) / size * 2;
        const rr = sx * sx + sy * sy;
        if (rr > 0.96) continue;
        const sz = Math.sqrt(1 - rr);
        const d = [cos * sx + sin * sz, sy, -sin * sx + cos * sz];
        const p = (py * size + px) * 3;
        a.set(appearance(...d), p);
        basis(...d, y);
        const rgb = [0, 0, 0];
        for (let k = 0; k < count; k += 1) for (let c = 0; c < 3; c += 1) rgb[c] += coefficients[k * 3 + c] * y[k];
        b.set(rgb, p);
        if (px === size / 2 && py === size / 2) center = rgb;
      }
    }
    truth.put(a);
    approx.put(b);
    readout.textContent = `係数の数 (L+1)² × RGB = ${count} × 3 = ${count * 3}\n正面から見た色 RGB ${center.map(v => Math.min(1, Math.max(0, v)).toFixed(2)).join(', ')}`;
  }

  root.append(
    el('div', { className: 'lab-pair' }, el('figure', {}, truth.node, el('figcaption', {}, '本来の見え方')), el('figure', {}, approx.node, el('figcaption', {}, 'SHで表した見え方'))),
    el('div', { className: 'controls' }, degree.root, yaw.root), readout,
  );
  [degree, yaw].forEach(control => control.input.addEventListener('input', draw));
  onTheme(draw);
  draw();
}
