import { createModel, initialize, step, lossAndGradients, densify, render, gaussianAt, seededRandom, targetImage, targets, psnr } from '../lib/fit.mjs';
import { el, choice, toggle, pixels, canvas, theme, onTheme } from './ui.mjs';

const SIZE = 96, ZOOM = 2, LIMIT = 3000, GROW_UNTIL = 2000, GROW_EVERY = 100;

export function mount(root) {
  const truth = pixels(SIZE, SIZE, '教師画像');
  const output = pixels(SIZE * ZOOM, SIZE * ZOOM, '現在のGaussianを描画した画像');
  const error = pixels(SIZE, SIZE, '描画と教師画像の差。明るいほど差が大きい。');
  const chart = canvas(640, 110, 'PSNRとGaussian数の推移');
  for (const view of [truth, error]) view.node.classList.add('lab-pixels');
  const subject = choice('題材', { ...Object.fromEntries(Object.entries(targets).map(([id, item]) => [id, item.label])), file: '自分の画像…' });
  const capacity = choice('粒の上限', { 40: '40個', 200: '200個', 600: '600個' }, '600');
  const outlines = toggle('楕円を表示', true);
  const growth = toggle('密度制御', true);
  const run = el('button', { type: 'submit' }, '学習する');
  const reset = el('button', { type: 'button' }, 'やり直す');
  const file = el('input', { type: 'file', accept: 'image/*', hidden: true });
  const readout = el('pre', { className: 'lab-readout' });
  const big = new Float32Array(SIZE * SIZE * ZOOM * ZOOM * 3), bigTrans = new Float32Array(SIZE * SIZE * ZOOM * ZOOM);
  const difference = new Float32Array(SIZE * SIZE * 3);
  let model, target, random, history, events, lastEvent, frame = 0, custom = null;

  function start() {
    cancelAnimationFrame(frame);
    frame = 0;
    target = subject.value === 'file' && custom ? custom : targetImage(subject.value === 'file' ? 'photo' : subject.value, SIZE, SIZE);
    model = createModel({ width: SIZE, height: SIZE, capacity: Number(capacity.value), background: [0.5, 0.5, 0.5] });
    random = seededRandom(11);
    initialize(model, target, random, 24);
    history = [];
    events = [];
    lastEvent = '初期状態: ランダムな位置に24個。色だけ教師画像から取った。';
    truth.put(target);
    lossAndGradients(model, target); // 反復0のPSNRを出すために損失だけを求める。粒はまだ動かさない
    run.textContent = '学習する';
    paint();
  }

  function paint() {
    const c = theme();
    output.put(render(model, ZOOM, big, bigTrans));
    if (outlines.value) {
      const g = output.context;
      g.lineWidth = 1;
      g.strokeStyle = 'rgba(255,255,255,.5)';
      for (let i = 0; i < model.count; i += 1) {
        const s = gaussianAt(model, i);
        g.beginPath();
        g.ellipse(s.x * ZOOM, s.y * ZOOM, s.sx * ZOOM, s.sy * ZOOM, s.angle, 0, 2 * Math.PI);
        g.stroke();
      }
    }
    render(model);
    for (let j = 0; j < difference.length; j += 3) {
      const d = (Math.abs(model.color[j] - target[j]) + Math.abs(model.color[j + 1] - target[j + 1]) + Math.abs(model.color[j + 2] - target[j + 2])) * 2;
      difference[j] = difference[j + 1] = difference[j + 2] = d;
    }
    error.put(difference);

    const g = chart.context, w = chart.width, h = chart.height;
    g.fillStyle = c.paper;
    g.fillRect(0, 0, w, h);
    g.strokeStyle = c.line;
    g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, w - 1, h - 1);
    const x = iteration => iteration / LIMIT * w;
    g.strokeStyle = c.line;
    for (const iteration of events) { g.beginPath(); g.moveTo(x(iteration), h - 8); g.lineTo(x(iteration), h); g.stroke(); }
    const line = (key, max, color) => {
      g.strokeStyle = color;
      g.lineWidth = 2.5;
      g.beginPath();
      history.forEach((point, i) => { const y = h - 6 - point[key] / max * (h - 12); if (i) g.lineTo(x(point.iteration), y); else g.moveTo(x(point.iteration), y); });
      g.stroke();
    };
    line('psnr', 50, c.accent);
    line('count', model.capacity, c.warm);
    g.font = '13px sans-serif';
    g.fillStyle = c.accent; g.fillRect(8, 9, 10, 10);
    g.fillStyle = c.warm; g.fillRect(72, 9, 10, 10);
    g.fillStyle = c.ink; g.fillText('PSNR', 23, 18); g.fillText('Gaussian数', 87, 18);
    readout.textContent = `反復 ${model.iteration} ／ Gaussian ${model.count}個 ／ PSNR ${psnr(model.mse).toFixed(1)} dB\n${lastEvent}`;
  }

  function advance() {
    step(model, target);
    if (model.iteration % 10 === 0) history.push({ iteration: model.iteration, psnr: psnr(model.mse), count: model.count });
    if (growth.value && model.iteration % GROW_EVERY === 0 && model.iteration <= GROW_UNTIL) {
      const change = densify(model, random);
      events.push(model.iteration);
      lastEvent = `反復${model.iteration}の密度制御: 分割${change.split}・複製${change.cloned}・削除${change.pruned}` + (model.count >= model.capacity ? '（上限に到達）' : '');
    }
  }

  function loop() {
    const began = performance.now();
    while (performance.now() - began < 10 && model.iteration < LIMIT) advance();
    paint();
    if (model.iteration >= LIMIT) { frame = 0; run.textContent = '完了'; return; }
    frame = requestAnimationFrame(loop);
  }

  run.addEventListener('click', () => {
    if (frame) { cancelAnimationFrame(frame); frame = 0; run.textContent = '再開する'; return; }
    if (model.iteration >= LIMIT) start();
    run.textContent = '一時停止';
    frame = requestAnimationFrame(loop);
  });
  reset.addEventListener('click', start);
  capacity.input.addEventListener('change', start);
  subject.input.addEventListener('change', () => { if (subject.value === 'file') file.click(); else start(); });
  file.addEventListener('change', async () => {
    if (!file.files[0]) return;
    const bitmap = await createImageBitmap(file.files[0]);
    const scratch = el('canvas', { width: SIZE, height: SIZE }).getContext('2d');
    const side = Math.min(bitmap.width, bitmap.height);
    scratch.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, SIZE, SIZE);
    const data = scratch.getImageData(0, 0, SIZE, SIZE).data;
    custom = new Float32Array(SIZE * SIZE * 3);
    for (let p = 0; p < SIZE * SIZE; p += 1) for (let c = 0; c < 3; c += 1) custom[p * 3 + c] = data[p * 4 + c] / 255;
    start();
  });
  outlines.input.addEventListener('change', paint);
  onTheme(paint);
  // note用のGIFのコマを撮るときは、?反復=300 のように渡した回数まで進めた状態を最初に表示する
  const ahead = Math.min(LIMIT, Number(new URLSearchParams(location.search).get('反復')) || 0);

  const figure = (view, caption) => el('figure', {}, view.node, el('figcaption', {}, caption));
  root.append(
    el('div', { className: 'lab-trio' }, figure(truth, '教師画像'), figure(output, '現在の描画'), figure(error, '差（明るいほど大きい）')),
    el('div', { className: 'lab-frame' }, chart.node),
    el('div', { className: 'lab-actions' }, run, reset, outlines.root, growth.root),
    el('div', { className: 'controls' }, subject.root, capacity.root, file),
    readout,
  );
  start();
  if (ahead) { while (model.iteration < ahead) advance(); paint(); }
}
