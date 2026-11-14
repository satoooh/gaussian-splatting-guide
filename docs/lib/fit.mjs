import { photo, PHOTO_SIZE } from './photo.mjs';

// 2D Gaussianで画像を再現する小さな学習器。3DGSの「描く→比べる→勾配で直す→個数を変える」を投影なしで再現する。
export const STRIDE = 9; // x, y, log sx, log sy, 回転, 不透明度のlogit, r, g, b
const ALPHA_MIN = 1 / 255;
const ALPHA_MAX = 0.99;
const CUTOFF = 9; // 3σより外は計算しない

export function createModel({ width, height, capacity = 1024, background = [1, 1, 1] }) {
  const size = capacity * STRIDE;
  return {
    width, height, capacity, background, count: 0, iteration: 0,
    params: new Float32Array(size), grads: new Float32Array(size),
    m: new Float32Array(size), v: new Float32Array(size), age: new Uint32Array(capacity),
    pull: new Float32Array(capacity * 4), // 位置勾配の合計x, y、大きさの合計、観測回数
    color: new Float32Array(width * height * 3), trans: new Float32Array(width * height),
    dColor: new Float32Array(width * height * 3), behind: new Float32Array(width * height * 3),
  };
}

export function addGaussian(model, { x, y, sx, sy, angle = 0, opacity = 0.5, rgb }) {
  if (model.count >= model.capacity) return false;
  const k = model.count * STRIDE;
  model.params.set([x, y, Math.log(sx), Math.log(sy), angle, Math.log(opacity / (1 - opacity)), ...rgb], k);
  resetState(model, model.count);
  model.count += 1;
  return true;
}

function resetState(model, index) {
  const k = index * STRIDE;
  model.m.fill(0, k, k + STRIDE);
  model.v.fill(0, k, k + STRIDE);
  model.age[index] = 0;
  model.pull.fill(0, index * 4, index * 4 + 4);
}

export function gaussianAt(model, index) {
  const p = model.params, k = index * STRIDE;
  return {
    x: p[k], y: p[k + 1], sx: Math.exp(p[k + 2]), sy: Math.exp(p[k + 3]), angle: p[k + 4],
    opacity: 1 / (1 + Math.exp(-p[k + 5])), rgb: [p[k + 6], p[k + 7], p[k + 8]],
  };
}

// 手前(index 0)から順に合成する。scaleを変えると同じ表現を別の解像度で描ける。
export function render(model, scale = 1, color = model.color, trans = model.trans) {
  const { params, count, background } = model;
  const W = Math.round(model.width * scale), H = Math.round(model.height * scale);
  color.fill(0);
  trans.fill(1);
  for (let i = 0; i < count; i += 1) {
    const k = i * STRIDE;
    const cx = params[k] * scale, cy = params[k + 1] * scale;
    const sx = Math.exp(params[k + 2]) * scale, sy = Math.exp(params[k + 3]) * scale;
    const cos = Math.cos(params[k + 4]), sin = Math.sin(params[k + 4]);
    const o = 1 / (1 + Math.exp(-params[k + 5]));
    const r = params[k + 6], g = params[k + 7], b = params[k + 8];
    const reach = 3 * Math.max(sx, sy);
    const x0 = Math.max(0, Math.floor(cx - reach)), x1 = Math.min(W - 1, Math.ceil(cx + reach));
    const y0 = Math.max(0, Math.floor(cy - reach)), y1 = Math.min(H - 1, Math.ceil(cy + reach));
    for (let y = y0; y <= y1; y += 1) {
      const dy = y + 0.5 - cy;
      for (let x = x0; x <= x1; x += 1) {
        const dx = x + 0.5 - cx;
        const u = (cos * dx + sin * dy) / sx, v = (-sin * dx + cos * dy) / sy;
        const q = u * u + v * v;
        if (q > CUTOFF) continue;
        let a = o * Math.exp(-0.5 * q);
        if (a < ALPHA_MIN) continue;
        if (a > ALPHA_MAX) a = ALPHA_MAX;
        const p = y * W + x, weight = trans[p] * a;
        color[p * 3] += weight * r;
        color[p * 3 + 1] += weight * g;
        color[p * 3 + 2] += weight * b;
        trans[p] *= 1 - a;
      }
    }
  }
  for (let p = 0; p < W * H; p += 1) {
    for (let c = 0; c < 3; c += 1) color[p * 3 + c] += trans[p] * background[c];
  }
  return color;
}

// 画素差の絶対値の平均(L1)と、全パラメータへの勾配を求める。戻り値は損失。PSNR用の二乗誤差はmodel.mseに残す。
export function lossAndGradients(model, target) {
  const { params, grads, count, width: W, height: H, color, trans, dColor, behind, background } = model;
  render(model);
  const total = W * H * 3;
  let loss = 0, squared = 0;
  for (let j = 0; j < total; j += 1) {
    const diff = color[j] - target[j];
    loss += Math.abs(diff);
    squared += diff * diff;
    dColor[j] = Math.sign(diff) / total;
  }
  model.mse = squared / total;
  for (let p = 0; p < W * H; p += 1) {
    for (let c = 0; c < 3; c += 1) behind[p * 3 + c] = trans[p] * background[c];
  }
  grads.fill(0, 0, count * STRIDE);
  // 奥から手前へ戻りながら、各Gaussianの手前の透過率と奥の色を復元する。
  for (let i = count - 1; i >= 0; i -= 1) {
    const k = i * STRIDE;
    const cx = params[k], cy = params[k + 1];
    const sx = Math.exp(params[k + 2]), sy = Math.exp(params[k + 3]);
    const cos = Math.cos(params[k + 4]), sin = Math.sin(params[k + 4]);
    const o = 1 / (1 + Math.exp(-params[k + 5]));
    const r = params[k + 6], g = params[k + 7], b = params[k + 8];
    const reach = 3 * Math.max(sx, sy);
    const x0 = Math.max(0, Math.floor(cx - reach)), x1 = Math.min(W - 1, Math.ceil(cx + reach));
    const y0 = Math.max(0, Math.floor(cy - reach)), y1 = Math.min(H - 1, Math.ceil(cy + reach));
    let gx = 0, gy = 0, gsx = 0, gsy = 0, gth = 0, go = 0, gr = 0, gg = 0, gb = 0;
    for (let y = y0; y <= y1; y += 1) {
      const dy = y + 0.5 - cy;
      for (let x = x0; x <= x1; x += 1) {
        const dx = x + 0.5 - cx;
        const u = (cos * dx + sin * dy) / sx, v = (-sin * dx + cos * dy) / sy;
        const q = u * u + v * v;
        if (q > CUTOFF) continue;
        const weight = Math.exp(-0.5 * q);
        let a = o * weight;
        if (a < ALPHA_MIN) continue;
        const clamped = a > ALPHA_MAX;
        if (clamped) a = ALPHA_MAX;
        const p = y * W + x, p3 = p * 3;
        const t = trans[p] / (1 - a); // このGaussianの手前まで透けていた割合
        trans[p] = t;
        const d0 = dColor[p3], d1 = dColor[p3 + 1], d2 = dColor[p3 + 2];
        gr += d0 * t * a; gg += d1 * t * a; gb += d2 * t * a;
        if (!clamped) {
          const dAlpha = d0 * (t * r - behind[p3] / (1 - a)) + d1 * (t * g - behind[p3 + 1] / (1 - a)) + d2 * (t * b - behind[p3 + 2] / (1 - a));
          go += dAlpha * weight;
          const dq = -0.5 * weight * o * dAlpha;
          const du = 2 * u * dq, dv = 2 * v * dq;
          gx += du * (-cos / sx) + dv * (sin / sy);
          gy += du * (-sin / sx) + dv * (-cos / sy);
          gsx += du * -u;
          gsy += dv * -v;
          gth += du * (v * sy / sx) + dv * (-u * sx / sy);
        }
        behind[p3] += t * a * r; behind[p3 + 1] += t * a * g; behind[p3 + 2] += t * a * b;
      }
    }
    grads[k] = gx; grads[k + 1] = gy; grads[k + 2] = gsx; grads[k + 3] = gsy; grads[k + 4] = gth;
    grads[k + 5] = go * o * (1 - o);
    grads[k + 6] = gr; grads[k + 7] = gg; grads[k + 8] = gb;
  }
  return loss / total;
}

const RATES = [0.25, 0.25, 0.03, 0.03, 0.03, 0.05, 0.02, 0.02, 0.02];

// 一反復: 描画、損失、勾配、Adamによる更新。位置勾配は密度制御のために蓄積する。
export function step(model, target) {
  const loss = lossAndGradients(model, target);
  const { params, grads, m, v, age, pull, count } = model;
  const pixels = model.width * model.height;
  for (let i = 0; i < count; i += 1) {
    const k = i * STRIDE;
    age[i] += 1;
    const c1 = 1 - 0.9 ** age[i], c2 = 1 - 0.999 ** age[i];
    for (let j = 0; j < STRIDE; j += 1) {
      const gradient = grads[k + j];
      m[k + j] = 0.9 * m[k + j] + 0.1 * gradient;
      v[k + j] = 0.999 * v[k + j] + 0.001 * gradient * gradient;
      params[k + j] -= RATES[j] * (m[k + j] / c1) / (Math.sqrt(v[k + j] / c2) + 1e-12);
    }
    params[k + 2] = Math.min(params[k + 2], Math.log(model.width));
    params[k + 3] = Math.min(params[k + 3], Math.log(model.width));
    pull[i * 4] += grads[k] * pixels;
    pull[i * 4 + 1] += grads[k + 1] * pixels;
    pull[i * 4 + 2] += Math.hypot(grads[k], grads[k + 1]) * pixels;
    pull[i * 4 + 3] += 1;
  }
  model.iteration += 1;
  return loss;
}

export function positionPull(model, index) {
  const n = model.pull[index * 4 + 3];
  return n ? model.pull[index * 4 + 2] / n : 0;
}

// 位置勾配が大きいGaussianを、小さければ複製、大きければ分割する。不透明度が低いものと画面の外へ出たものは削除する。
export function densify(model, random, { threshold = 0.6, splitSize = 3, minOpacity = 0.03 } = {}) {
  const result = { cloned: 0, split: 0, pruned: 0 };
  const { params } = model;
  const before = model.count;
  for (let i = 0; i < before; i += 1) {
    const k = i * STRIDE;
    const strong = positionPull(model, i) > threshold;
    const px = model.pull[i * 4], py = model.pull[i * 4 + 1];
    model.pull.fill(0, i * 4, i * 4 + 4);
    if (!strong || model.count >= model.capacity) continue;
    const g = gaussianAt(model, i);
    const copy = model.count * STRIDE;
    params.copyWithin(copy, k, k + STRIDE);
    resetState(model, model.count);
    model.count += 1;
    if (Math.max(g.sx, g.sy) > splitSize) {
      for (const base of [k, copy]) {
        const a = normal(random) * g.sx, b = normal(random) * g.sy;
        params[base] = g.x + Math.cos(g.angle) * a - Math.sin(g.angle) * b;
        params[base + 1] = g.y + Math.sin(g.angle) * a + Math.cos(g.angle) * b;
        params[base + 2] -= Math.log(1.6);
        params[base + 3] -= Math.log(1.6);
      }
      resetState(model, i);
      result.split += 1;
    } else {
      const length = Math.hypot(px, py) || 1;
      params[copy] -= px / length * Math.min(g.sx, g.sy);
      params[copy + 1] -= py / length * Math.min(g.sx, g.sy);
      result.cloned += 1;
    }
  }
  let kept = 0;
  for (let i = 0; i < model.count; i += 1) {
    const g = gaussianAt(model, i);
    const outside = g.x < -g.sx || g.y < -g.sy || g.x > model.width + g.sx || g.y > model.height + g.sy;
    if (g.opacity < minOpacity || outside) { result.pruned += 1; continue; }
    if (kept !== i) {
      for (const array of [params, model.m, model.v]) array.copyWithin(kept * STRIDE, i * STRIDE, (i + 1) * STRIDE);
      model.age[kept] = model.age[i];
      model.pull.copyWithin(kept * 4, i * 4, i * 4 + 4);
    }
    kept += 1;
  }
  model.count = kept;
  return result;
}

export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(random) {
  return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

// SfMの疎な点群の代わりに、ランダムな位置へその場所の色を持つGaussianを置く。
export function initialize(model, target, random, count = 24) {
  model.count = 0;
  model.iteration = 0;
  for (let i = 0; i < count; i += 1) {
    const x = random() * model.width, y = random() * model.height;
    const p = (Math.min(model.height - 1, Math.floor(y)) * model.width + Math.min(model.width - 1, Math.floor(x))) * 3;
    addGaussian(model, { x, y, sx: model.width / 10, sy: model.width / 10, rgb: [target[p], target[p + 1], target[p + 2]] });
  }
}

export const psnr = mse => -10 * Math.log10(Math.max(mse, 1e-10));

// 式で定義した教師画像。著作物を同梱せず、Nodeのテストでも同じ画像を使える。
const mix = (a, b, t) => a.map((value, i) => value + (b[i] - value) * t);
const smooth = (edge, x) => Math.min(1, Math.max(0, 0.5 - (x - edge) * 40));
export const targets = {
  photo: { label: '実写の料理', pixel(u, v) {
    const x = Math.min(PHOTO_SIZE - 1, Math.floor(u * PHOTO_SIZE)), y = Math.min(PHOTO_SIZE - 1, Math.floor(v * PHOTO_SIZE)), at = (y * PHOTO_SIZE + x) * 3;
    return [photo[at] / 255, photo[at + 1] / 255, photo[at + 2] / 255];
  } },
  sunset: { label: '夕焼けと山', pixel(u, v) {
    let c = mix([0.16, 0.2, 0.5], [0.98, 0.62, 0.32], Math.min(1, v / 0.62));
    c = mix(c, [1, 0.93, 0.6], smooth(0.11, Math.hypot(u - 0.68, v - 0.36)));
    c = mix(c, [0.28, 0.2, 0.4], smooth(0, 0.42 + 0.5 * Math.abs(u - 0.3) - v));
    c = mix(c, [0.12, 0.1, 0.24], smooth(0, 0.56 + 0.35 * Math.abs(u - 0.78) - v));
    return v > 0.8 ? mix(c, [0.2, 0.28, 0.5], 0.75) : c;
  } },
  fruit: { label: '光沢のある球', pixel(u, v) {
    const d = Math.hypot(u - 0.5, v - 0.52);
    const shade = Math.max(0, 1 - Math.hypot(u - 0.4, v - 0.4) * 1.6);
    let c = mix([0.94, 0.94, 0.9], [0.72, 0.72, 0.7], Math.min(1, Math.max(0, (v - 0.75) * 4)));
    c = mix(c, mix([0.5, 0.05, 0.08], [1, 0.42, 0.3], shade), smooth(0.3, d));
    return mix(c, [1, 1, 1], smooth(0.035, Math.hypot(u - 0.4, v - 0.38)));
  } },
  stripes: { label: '細い縞と文字', pixel(u, v) {
    if (v < 0.5) return Math.floor(u * (4 + Math.floor(u * 4) * 6)) % 2 ? [0.1, 0.1, 0.12] : [0.96, 0.96, 0.94];
    const bar = Math.abs(u - 0.3) < 0.035 || (Math.abs(v - 0.62) < 0.03 && Math.abs(u - 0.3) < 0.14);
    const ring = Math.abs(Math.hypot(u - 0.68, v - 0.74) - 0.14) < 0.03;
    return bar || ring ? [0, 0.09, 0.76] : [0.96, 0.96, 0.94];
  } },
};

export function targetImage(name, width, height) {
  const image = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) image.set(targets[name].pixel((x + 0.5) / width, (y + 0.5) / height), (y * width + x) * 3);
  }
  return image;
}
