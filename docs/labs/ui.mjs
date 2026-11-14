export function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

// note用の静止画は、本文が語る状態を撮る。?回転=60 のように、操作部品の名前で最初の値を渡せる。
const given = new URLSearchParams(globalThis.location?.search ?? '');

export function slider(label, { min, max, step = 1, value, format = v => v }) {
  const output = el('output');
  const input = el('input', { type: 'range', min, max, step, value: given.get(label) ?? value });
  const sync = () => { output.textContent = format(Number(input.value)); };
  input.addEventListener('input', sync);
  sync();
  return {
    root: el('label', {}, el('span', {}, label, ' ', output), input), input,
    get value() { return Number(input.value); },
    set value(next) { input.value = next; sync(); },
  };
}

export function choice(label, options, value = Object.keys(options)[0]) {
  const input = el('select', {}, ...Object.entries(options).map(([value, text]) => el('option', { value }, text)));
  input.value = given.get(label) ?? value;
  return { root: el('label', {}, label, input), input, get value() { return input.value; } };
}

export function toggle(label, checked = false) {
  const input = el('input', { type: 'checkbox', checked: given.has(label) ? given.get(label) === '1' : checked });
  return { root: el('label', { className: 'lab-toggle' }, input, el('span', {}, label)), input, get value() { return input.checked; } };
}

// 論理サイズで描けるcanvas。高密度ディスプレイでは内部解像度だけ上げる。
export function canvas(width, height, label) {
  const node = el('canvas', { width, height });
  node.setAttribute('role', 'img');
  node.setAttribute('aria-label', label);
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  node.width = width * ratio;
  node.height = height * ratio;
  const context = node.getContext('2d');
  context.scale(ratio, ratio);
  return { node, context, width, height };
}

// 画素を直接書くcanvas。拡大しても画素の境界が見える。
export function pixels(width, height, label) {
  const node = el('canvas', { width, height });
  node.setAttribute('role', 'img');
  node.setAttribute('aria-label', label);
  const context = node.getContext('2d');
  const image = context.createImageData(width, height);
  image.data.fill(255);
  return { node, context, image, width, height, put(rgb) {
    for (let p = 0; p < width * height; p += 1) {
      for (let c = 0; c < 3; c += 1) image.data[p * 4 + c] = Math.round(Math.min(1, Math.max(0, rgb[p * 3 + c])) * 255);
    }
    context.putImageData(image, 0, 0);
  } };
}

// 図と実験に共通の色。グレースケールで描き、注目させたい一点だけに差し色（黄色）を使う。
// 差し色はここで一つ選ぶ。rgbは実験の中で粒そのものを塗る色。
const ACCENTS = {
  blue: { light: '#2f5bea', dark: '#86a0ff', rgb: [0.18, 0.36, 0.92] },
  yellow: { light: '#f0b400', dark: '#ffd24a', rgb: [0.96, 0.72, 0.04] },
};
const ACCENT = ACCENTS.yellow;
export const ACCENT_RGB = ACCENT.rgb;
export const SECOND_RGB = (ACCENT === ACCENTS.blue ? ACCENTS.yellow : ACCENTS.blue).rgb;
export const LIGHT = { paper: '#f2f2ef', card: '#ffffff', ink: '#1e1e1e', muted: '#7b7b7b', line: '#d0d0cc', soft: '#a8a8a6', accent: ACCENT.light, background: [0.949, 0.949, 0.937] };
export const DARK = { paper: '#222327', card: '#2c2d32', ink: '#efefef', muted: '#a1a1a1', line: '#45464c', soft: '#77787d', accent: ACCENT.dark, background: [0.133, 0.137, 0.153] };
const dark = matchMedia('(prefers-color-scheme: dark)');
// warmは二つ目の系列の色。色相を増やさず、濃いグレーで描き分ける。
export const theme = () => { const c = dark.matches ? DARK : LIGHT; return { ...c, warm: c.ink }; };
export const onTheme = draw => dark.addEventListener('change', draw);

export function arrow(context, x1, y1, x2, y2) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.lineTo(x2 - 9 * Math.cos(angle - 0.4), y2 - 9 * Math.sin(angle - 0.4));
  context.moveTo(x2, y2);
  context.lineTo(x2 - 9 * Math.cos(angle + 0.4), y2 - 9 * Math.sin(angle + 0.4));
  context.stroke();
}
