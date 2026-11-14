import { el, slider, choice, toggle, pixels, theme, onTheme, ACCENT_RGB } from './ui.mjs';
import { seededRandom } from '../lib/fit.mjs';

const W = 320, H = 240, FOCAL = 300;
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = v => { const n = Math.hypot(...v); return v.map(value => value / n); };
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// 面に沿った薄い円盤状のGaussian。共分散 Σ = R S Sᵀ Rᵀ を先に作っておく。
function disc(center, normal, radius, rgb, opacity = 0.9) {
  const n = unit(normal);
  const a = unit(cross(n, Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0])), b = cross(n, a);
  const axes = [a, b, n], scales = [radius, radius, radius * 0.15];
  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let k = 0; k < 3; k += 1) for (let i = 0; i < 3; i += 1) for (let j = 0; j < 3; j += 1) cov[i][j] += axes[k][i] * axes[k][j] * scales[k] ** 2;
  return { center, cov, rgb, opacity };
}

// 縞のある球と、それを囲む薄い輪。輪は球の手前と奥を通るので前後関係が目立つ。
export function scene() {
  const list = [], tilt = 0.45, light = unit([0.5, 0.8, 0.4]);
  const rotate = ([x, y, z]) => [x * Math.cos(tilt) - y * Math.sin(tilt), x * Math.sin(tilt) + y * Math.cos(tilt), z];
  const count = 420;
  for (let i = 0; i < count; i += 1) {
    const y = 1 - 2 * (i + 0.5) / count, r = Math.sqrt(1 - y * y), phi = i * 2.399963229728653;
    const n = rotate([r * Math.cos(phi), y, r * Math.sin(phi)]);
    const band = Math.floor((y + 1) * 2.5) % 2;
    const shade = 0.45 + 0.55 * Math.max(0, dot(n, light));
    const base = band ? [0.84, 0.84, 0.82] : [0.52, 0.52, 0.52];
    list.push(disc(n.map(v => v * 0.7), n, 0.07, base.map(v => v * shade)));
  }
  for (const [radius, tone] of [[1.05, 0.95], [1.2, 0.8], [1.35, 0.65]]) {
    const steps = Math.round(radius * 60);
    for (let i = 0; i < steps; i += 1) {
      const a = i / steps * 2 * Math.PI;
      list.push(disc(rotate([radius * Math.cos(a), 0, radius * Math.sin(a)]), rotate([0, 1, 0]), 0.075, ACCENT_RGB.map(v => v * tone), 0.8));
    }
  }
  const random = seededRandom(5); // 保存順は前後と無関係にしておく
  for (let i = list.length - 1; i > 0; i -= 1) { const j = Math.floor(random() * (i + 1)); [list[i], list[j]] = [list[j], list[i]]; }
  return list;
}

export function camera(yaw, pitch, distance = 3.4) {
  const eye = [distance * Math.sin(yaw) * Math.cos(pitch), distance * Math.sin(pitch), distance * Math.cos(yaw) * Math.cos(pitch)];
  const forward = unit(eye.map(v => -v)), right = unit(cross(forward, [0, 1, 0])), up = cross(right, forward);
  return { eye, rows: [right, up.map(v => -v), forward] }; // 画像のyは下向き
}

// 第4章の式: Σscreen ≈ J W Σ Wᵀ Jᵀ。戻り値は画面上の中心と2×2共分散。
export function project(gaussian, view, focal = FOCAL) {
  const offset = gaussian.center.map((v, i) => v - view.eye[i]);
  const [x, y, z] = view.rows.map(row => dot(row, offset));
  if (z < 0.2) return null;
  const J = [[focal / z, 0, -focal * x / (z * z)], [0, focal / z, -focal * y / (z * z)]];
  const M = J.map(j => [0, 1, 2].map(k => j[0] * view.rows[0][k] + j[1] * view.rows[1][k] + j[2] * view.rows[2][k])); // J W
  const S = (p, q) => { let sum = 0; for (let i = 0; i < 3; i += 1) for (let j = 0; j < 3; j += 1) sum += M[p][i] * gaussian.cov[i][j] * M[q][j]; return sum; };
  return { u: focal * x / z, v: focal * y / z, depth: z, a: S(0, 0), b: S(0, 1), c: S(1, 1) };
}

// 投影した粒を手前から順に合成する。本文の第4章の計算そのもの。
export function rasterize(gaussians, cam, { width = W, height = H, focal = FOCAL, mode = 'splat', sorting = true, background = [1, 1, 1], rgb = new Float32Array(width * height * 3) } = {}) {
  const trans = new Float32Array(width * height).fill(1);
  const splats = gaussians.map(g => ({ g, p: project(g, cam, focal) })).filter(item => item.p);
  if (sorting) splats.sort((m, n) => m.p.depth - n.p.depth);
  rgb.fill(0);
  let work = 0;
  for (const { g, p } of splats) {
    const a = p.a + 0.3, c = p.c + 0.3, b = p.b; // 3DGSと同じく、画素より細い楕円を少し太らせる
    const det = a * c - b * b;
    const cx = width / 2 + p.u, cy = height / 2 + p.v;
    const reach = mode === 'point' ? 1 : 3 * Math.sqrt(Math.max(a, c));
    const x0 = Math.max(0, Math.floor(cx - reach)), x1 = Math.min(width - 1, Math.ceil(cx + reach));
    const y0 = Math.max(0, Math.floor(cy - reach)), y1 = Math.min(height - 1, Math.ceil(cy + reach));
    for (let y = y0; y <= y1; y += 1) {
      for (let x = x0; x <= x1; x += 1) {
        const dx = x + 0.5 - cx, dy = y + 0.5 - cy;
        const q = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det;
        const alpha = mode === 'point' ? 1 : mode === 'solid' ? (q < 1 ? 1 : 0) : (q < 9 ? Math.min(0.99, g.opacity * Math.exp(-0.5 * q)) : 0);
        const i = y * width + x;
        if (alpha < 1 / 255 || trans[i] < 0.004) continue;
        work += 1;
        const weight = trans[i] * alpha;
        rgb[i * 3] += weight * g.rgb[0]; rgb[i * 3 + 1] += weight * g.rgb[1]; rgb[i * 3 + 2] += weight * g.rgb[2];
        trans[i] *= 1 - alpha;
      }
    }
  }
  // backgroundがnullなら背景を足さず、透過率をそのまま返す（図で透明な切り抜きにするため）
  if (background) for (let i = 0; i < width * height; i += 1) for (let k = 0; k < 3; k += 1) rgb[i * 3 + k] += trans[i] * background[k];
  return { rgb, trans, splats, count: splats.length, work };
}

export function mount(root) {
  const view = pixels(W, H, '縞のある球と輪を表す約600個のGaussianを描画した画像。ドラッグで視点が動く。');
  view.node.style.touchAction = 'none';
  view.node.style.cursor = 'grab';
  const yaw = slider('視点を回す', { min: -180, max: 180, value: 35, format: v => `${v}°` });
  const pitch = slider('視点の高さ', { min: -60, max: 60, value: 18, format: v => `${v}°` });
  const mode = choice('描画方法', { splat: 'スプラット', solid: '不透明な楕円', point: '中心点だけ' });
  const sorting = toggle('深度ソート', true);
  const readout = el('pre', { className: 'lab-readout' });
  const gaussians = scene(), rgb = new Float32Array(W * H * 3);

  function draw() {
    const cam = camera(yaw.value * Math.PI / 180, pitch.value * Math.PI / 180);
    const result = rasterize(gaussians, cam, { mode: mode.value, sorting: sorting.value, background: theme().background, rgb });
    view.put(rgb);
    readout.textContent = `Gaussian ${result.count}個 ／ 画素への合成 ${result.work.toLocaleString('ja-JP')}回 ／ ${W}×${H}画素`;
  }

  let drag = null;
  view.node.addEventListener('pointerdown', event => { drag = { x: event.clientX, y: event.clientY, yaw: yaw.value, pitch: pitch.value }; view.node.setPointerCapture(event.pointerId); });
  view.node.addEventListener('pointermove', event => {
    if (!drag) return;
    yaw.value = ((drag.yaw - (event.clientX - drag.x) * 0.6 + 540) % 360) - 180;
    pitch.value = Math.max(-60, Math.min(60, drag.pitch + (event.clientY - drag.y) * 0.4));
    draw();
  });
  for (const type of ['pointerup', 'pointercancel']) view.node.addEventListener(type, () => { drag = null; });
  root.append(el('div', { className: 'lab-frame' }, view.node));
  if (!root.hasAttribute('data-compact')) root.append(el('div', { className: 'controls' }, yaw.root, pitch.root, mode.root, el('div', { className: 'lab-actions' }, sorting.root)), readout);
  [yaw, pitch, mode, sorting].forEach(control => control.input.addEventListener('input', draw));
  onTheme(draw);
  draw();
}
