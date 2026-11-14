import { el, slider, choice, theme, onTheme } from './ui.mjs';

// .splat形式を読む。1粒32バイト: 位置(float32×3)、スケール(float32×3)、色と不透明度(uint8×4)、回転(uint8×4, w x y z)。
// 第2章の式 Σ = R S Sᵀ Rᵀ で、粒ごとの共分散を先に作っておく。
export function parseSplat(buffer) {
  const count = Math.floor(buffer.byteLength / 32), f = new Float32Array(buffer, 0, count * 8), u = new Uint8Array(buffer);
  const pos = new Float32Array(count * 3), cov = new Float32Array(count * 6), rgba = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    pos.set(f.subarray(i * 8, i * 8 + 3), i * 3);
    for (let k = 0; k < 4; k += 1) rgba[i * 4 + k] = u[i * 32 + 24 + k] / 255;
    let w = u[i * 32 + 28] - 128, x = u[i * 32 + 29] - 128, y = u[i * 32 + 30] - 128, z = u[i * 32 + 31] - 128;
    const norm = Math.hypot(w, x, y, z) || 1;
    w /= norm; x /= norm; y /= norm; z /= norm;
    const R = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
    const s = [f[i * 8 + 3] ** 2, f[i * 8 + 4] ** 2, f[i * 8 + 5] ** 2];
    let n = 0;
    for (let a = 0; a < 3; a += 1) for (let b = a; b < 3; b += 1) cov[i * 6 + n++] = R[a * 3] * R[b * 3] * s[0] + R[a * 3 + 1] * R[b * 3 + 1] * s[1] + R[a * 3 + 2] * R[b * 3 + 2] * s[2];
  }
  // 視点を回す中心と距離を、粒の散らばりから決める
  const center = [0, 1, 2].map(k => { let sum = 0; for (let i = 0; i < count; i += 1) sum += pos[i * 3 + k]; return sum / count; });
  let spread = 0;
  for (let i = 0; i < count; i += 1) spread += (pos[i * 3] - center[0]) ** 2 + (pos[i * 3 + 1] - center[1]) ** 2 + (pos[i * 3 + 2] - center[2]) ** 2;
  return { count, pos, cov, rgba, center, radius: Math.sqrt(spread / count), work: { u: new Float32Array(count), v: new Float32Array(count), a: new Float32Array(count), b: new Float32Array(count), c: new Float32Array(count), depth: new Float32Array(count), order: new Uint32Array(count) } };
}

// 中心を見ながら周回するカメラを作る。上方向はシーンごとに違うので、scene.up で与える（既定はy軸の負の向き）。
export function orbit(scene, yaw, pitch, distance = 5) {
  const unit = v => { const n = Math.hypot(...v); return v.map(value => value / n); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const up = unit(scene.up ?? [0, -1, 0]), e1 = unit(cross(up, Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1])), e2 = cross(up, e1), d = scene.radius * distance;
  const eye = scene.center.map((c, k) => c + d * (Math.cos(pitch) * (Math.sin(yaw) * e1[k] + Math.cos(yaw) * e2[k]) + Math.sin(pitch) * up[k]));
  const forward = unit(scene.center.map((v, k) => v - eye[k])), right = unit(cross(forward, up)), camUp = cross(right, forward);
  return { eye, rows: [right, camUp.map(v => -v), forward] };
}

// 第4章の計算そのまま: 投影して、奥行きで並べ、手前から合成する。粒が11万個あっても手順は変わらない。
export function drawScene(scene, cam, rgb, trans, width, height, focal, background, sorting = true) {
  const { count, pos, cov, rgba, work } = scene, [r0, r1, r2] = cam.rows, eye = cam.eye;
  const limX = 0.65 * width / focal, limY = 0.65 * height / focal;
  let near = Infinity, far = -Infinity;
  for (let i = 0; i < count; i += 1) {
    const tx = pos[i * 3] - eye[0], ty = pos[i * 3 + 1] - eye[1], tz = pos[i * 3 + 2] - eye[2];
    const x = r0[0] * tx + r0[1] * ty + r0[2] * tz, y = r1[0] * tx + r1[1] * ty + r1[2] * tz, z = r2[0] * tx + r2[1] * ty + r2[2] * tz;
    if (z < 0.05) { work.depth[i] = -1; continue; }
    // 元の3DGSの実装と同じく、視野の1.3倍より外の粒はヤコビ行列の x/z, y/z を端の値で止める。止めないと、カメラのすぐ脇の粒が画面を覆う。
    const fz = focal / z, jx = -fz * Math.max(-limX, Math.min(limX, x / z)), jy = -fz * Math.max(-limY, Math.min(limY, y / z));
    const m0 = fz * r0[0] + jx * r2[0], m1 = fz * r0[1] + jx * r2[1], m2 = fz * r0[2] + jx * r2[2];
    const n0 = fz * r1[0] + jy * r2[0], n1 = fz * r1[1] + jy * r2[1], n2 = fz * r1[2] + jy * r2[2];
    const k = i * 6, xx = cov[k], xy = cov[k + 1], xz = cov[k + 2], yy = cov[k + 3], yz = cov[k + 4], zz = cov[k + 5];
    const s0 = xx * m0 + xy * m1 + xz * m2, s1 = xy * m0 + yy * m1 + yz * m2, s2 = xz * m0 + yz * m1 + zz * m2;
    work.a[i] = m0 * s0 + m1 * s1 + m2 * s2 + 0.3;
    work.b[i] = n0 * s0 + n1 * s1 + n2 * s2;
    work.c[i] = n0 * (xx * n0 + xy * n1 + xz * n2) + n1 * (xy * n0 + yy * n1 + yz * n2) + n2 * (xz * n0 + yz * n1 + zz * n2) + 0.3;
    work.u[i] = width / 2 + fz * x; work.v[i] = height / 2 + fz * y; work.depth[i] = z;
    if (z < near) near = z;
    if (z > far) far = z;
  }
  // 奥行きを65536段に分けて数え上げる並べ替え。粒が多くても速い。
  const order = work.order, buckets = new Uint32Array(65537), scale = 65535 / Math.max(far - near, 1e-6);
  let visible = 0;
  if (sorting) {
    for (let i = 0; i < count; i += 1) if (work.depth[i] > 0) buckets[((work.depth[i] - near) * scale | 0) + 1] += 1;
    for (let b = 1; b <= 65536; b += 1) buckets[b] += buckets[b - 1];
    for (let i = 0; i < count; i += 1) if (work.depth[i] > 0) { order[buckets[(work.depth[i] - near) * scale | 0]++] = i; visible += 1; }
  } else {
    for (let i = 0; i < count; i += 1) if (work.depth[i] > 0) order[visible++] = i;
  }
  rgb.fill(0);
  trans.fill(1);
  for (let n = 0; n < visible; n += 1) {
    const i = order[n], a = work.a[i], b = work.b[i], c = work.c[i], det = a * c - b * b, opacity = rgba[i * 4 + 3];
    if (det <= 0 || opacity < 1 / 255) continue;
    const cx = work.u[i], cy = work.v[i], reach = 3 * Math.sqrt(Math.max(a, c));
    const x0 = Math.max(0, Math.floor(cx - reach)), x1 = Math.min(width - 1, Math.ceil(cx + reach));
    const y0 = Math.max(0, Math.floor(cy - reach)), y1 = Math.min(height - 1, Math.ceil(cy + reach));
    const ia = c / det, ib = -b / det, ic = a / det, r = rgba[i * 4], g = rgba[i * 4 + 1], bl = rgba[i * 4 + 2];
    for (let y = y0; y <= y1; y += 1) {
      const dy = y + 0.5 - cy;
      for (let x = x0; x <= x1; x += 1) {
        const p = y * width + x, t = trans[p];
        if (t < 0.004) continue;
        const dx = x + 0.5 - cx, q = ia * dx * dx + 2 * ib * dx * dy + ic * dy * dy;
        if (q > 9) continue;
        let alpha = opacity * Math.exp(-0.5 * q);
        if (alpha < 1 / 255) continue;
        if (alpha > 0.99) alpha = 0.99;
        const weight = t * alpha;
        rgb[p * 3] += weight * r; rgb[p * 3 + 1] += weight * g; rgb[p * 3 + 2] += weight * bl;
        trans[p] = t * (1 - alpha);
      }
    }
  }
  for (let p = 0; p < width * height; p += 1) for (let k = 0; k < 3; k += 1) rgb[p * 3 + k] += trans[p] * background[k];
  return visible;
}

// 同梱している実写のシーン。view は周回の中心、広がり、上方向で、省くと粒の散らばりから決める。
export const SCENES = {
  plant: { label: '観葉植物（約11万粒）', url: new URL('../splats/plant.splat', import.meta.url).href, yaw: 35, pitch: 18, distance: 5, credit: ['Houseplant', 'https://marcelpadilla.com/Gaussian_Splat_Object_Dataset/', 'Marcel Padilla', 'CC BY 4.0', 'https://creativecommons.org/licenses/by/4.0/', '元データから描画。'] },
  bowl: { label: '料理（約16万粒、この連載で学習）', url: new URL('../splats/bowl.splat', import.meta.url).href, yaw: 23, pitch: 52, distance: 2.2, view: { center: [0.2, 2.37, 2.6], radius: 1.76, up: [-0.073, -0.775, -0.628] }, credit: ['chili-salmon-bowl', 'https://huggingface.co/datasets/promc/reconstruction-scenes', 'Charin Rungchaowarat', 'CC0 1.0', 'https://creativecommons.org/publicdomain/zero/1.0/', '写真51枚からBrushで学習。'] },
};

// GPUでの描画。計算は上の drawScene と同じで、投影と重みの計算をシェーダに移し、並べ替えだけをCPUで行う。
const VERTEX = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D u_data; uniform vec3 u_r0, u_r1, u_r2, u_eye; uniform float u_focal; uniform vec2 u_viewport;
in uint a_index; out vec2 v_pos; out vec4 v_color;
vec4 fetch(uint k) { uint i = a_index * 4u + k; return texelFetch(u_data, ivec2(int(i % 1024u), int(i / 1024u)), 0); }
void main() {
  vec4 c = fetch(0u), color = fetch(1u), s0 = fetch(2u), s1 = fetch(3u);
  vec3 t = c.xyz - u_eye; float x = dot(u_r0, t), y = dot(u_r1, t), z = dot(u_r2, t);
  if (z < 0.05) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float fz = u_focal / z;
  vec2 lim = 0.65 * u_viewport / u_focal, q = clamp(vec2(x, y) / z, -lim, lim); // CPU側と同じ、視野端での打ち切り
  vec3 m = fz * u_r0 - fz * q.x * u_r2, n = fz * u_r1 - fz * q.y * u_r2; // J W の二行
  mat3 S = mat3(s0.x, s0.y, s0.z, s0.y, s1.x, s1.y, s0.z, s1.y, s1.z);
  vec3 Sm = S * m, Sn = S * n;
  float a = dot(m, Sm) + 0.3, b = dot(n, Sm), d = dot(n, Sn) + 0.3;              // 画面上の共分散
  float mid = 0.5 * (a + d), rad = length(vec2(0.5 * (a - d), b));
  float l1 = mid + rad, l2 = max(mid - rad, 0.05);                               // 固有値 = 主軸方向の分散
  vec2 axis = normalize(vec2(b, l1 - a) + vec2(1e-6, 0.0));
  vec2 major = min(3.0 * sqrt(l1), 1024.0) * axis, minor = min(3.0 * sqrt(l2), 1024.0) * vec2(-axis.y, axis.x);
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, float(gl_VertexID >> 1) * 2.0 - 1.0);
  vec2 px = vec2(fz * x, fz * y) + corner.x * major + corner.y * minor;
  gl_Position = vec4(px.x / (0.5 * u_viewport.x), -px.y / (0.5 * u_viewport.y), 0.0, 1.0);
  v_pos = corner * 3.0; v_color = vec4(color.rgb, c.w);
}`;
const FRAGMENT = `#version 300 es
precision highp float; in vec2 v_pos; in vec4 v_color; out vec4 o;
void main() { float q = dot(v_pos, v_pos); if (q > 9.0) discard; float alpha = min(0.99, v_color.a * exp(-0.5 * q)); o = vec4(v_color.rgb * alpha, alpha); }`;

function gpuView(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false });
  if (!gl) return null;
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, VERTEX], [gl.FRAGMENT_SHADER, FRAGMENT]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source); gl.compileShader(shader); gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;
  gl.useProgram(program);
  const uniform = name => gl.getUniformLocation(program, name), orderBuffer = gl.createBuffer(), texture = gl.createTexture(), location = gl.getAttribLocation(program, 'a_index');
  gl.bindBuffer(gl.ARRAY_BUFFER, orderBuffer);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribIPointer(location, 1, gl.UNSIGNED_INT, 0, 0);
  gl.vertexAttribDivisor(location, 1);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); // 奥から順に重ねる。手前から透過率を掛けていく計算と同じ結果になる
  let scene = null, order = null, depth = null;
  return {
    label: 'GPUで描画',
    setScene(next) {
      scene = next; order = new Uint32Array(scene.count); depth = new Float32Array(scene.count);
      const rows = Math.ceil(scene.count * 4 / 1024), data = new Float32Array(rows * 1024 * 4);
      for (let i = 0; i < scene.count; i += 1) {
        data.set([scene.pos[i * 3], scene.pos[i * 3 + 1], scene.pos[i * 3 + 2], scene.rgba[i * 4 + 3], scene.rgba[i * 4], scene.rgba[i * 4 + 1], scene.rgba[i * 4 + 2], 0, scene.cov[i * 6], scene.cov[i * 6 + 1], scene.cov[i * 6 + 2], 0, scene.cov[i * 6 + 3], scene.cov[i * 6 + 4], scene.cov[i * 6 + 5], 0], i * 16);
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1024, rows, 0, gl.RGBA, gl.FLOAT, data);
    },
    render(cam, background) {
      const { count, pos } = scene, [r0, r1, r2] = cam.rows, eye = cam.eye;
      let near = Infinity, far = -Infinity;
      for (let i = 0; i < count; i += 1) {
        const z = r2[0] * (pos[i * 3] - eye[0]) + r2[1] * (pos[i * 3 + 1] - eye[1]) + r2[2] * (pos[i * 3 + 2] - eye[2]);
        depth[i] = z;
        if (z < near) near = z;
        if (z > far) far = z;
      }
      const buckets = new Uint32Array(65537), scale = 65535 / Math.max(far - near, 1e-6);
      for (let i = 0; i < count; i += 1) buckets[65535 - ((depth[i] - near) * scale | 0) + 1] += 1; // 奥が先
      for (let b = 1; b <= 65536; b += 1) buckets[b] += buckets[b - 1];
      for (let i = 0; i < count; i += 1) order[buckets[65535 - ((depth[i] - near) * scale | 0)]++] = i;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(...background, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.bufferData(gl.ARRAY_BUFFER, order, gl.DYNAMIC_DRAW);
      gl.uniform3fv(uniform('u_r0'), r0); gl.uniform3fv(uniform('u_r1'), r1); gl.uniform3fv(uniform('u_r2'), r2); gl.uniform3fv(uniform('u_eye'), eye);
      gl.uniform1f(uniform('u_focal'), canvas.width * 0.9);
      gl.uniform2f(uniform('u_viewport'), canvas.width, canvas.height);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
    },
  };
}

// WebGL2が使えない環境では、同じ計算をCPUで行う。
function cpuView(canvas) {
  const width = canvas.width / 2, height = canvas.height / 2, context = canvas.getContext('2d'), small = el('canvas', { width, height });
  const rgb = new Float32Array(width * height * 3), trans = new Float32Array(width * height), image = small.getContext('2d').createImageData(width, height);
  let scene = null;
  return {
    label: 'CPUで描画',
    setScene(next) { scene = next; },
    render(cam, background) {
      drawScene(scene, cam, rgb, trans, width, height, width * 0.9, background);
      for (let p = 0; p < width * height; p += 1) { for (let c = 0; c < 3; c += 1) image.data[p * 4 + c] = Math.round(Math.min(1, Math.max(0, rgb[p * 3 + c])) * 255); image.data[p * 4 + 3] = 255; }
      small.getContext('2d').putImageData(image, 0, 0);
      context.drawImage(small, 0, 0, canvas.width, canvas.height);
    },
  };
}

export function mount(root) {
  const canvas = el('canvas', { width: 960, height: 720 });
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', '実際に撮影した写真から作ったGaussianを描画した画像。ドラッグで視点が動く。');
  canvas.style.cssText = 'display:block;width:100%;height:auto;touch-action:none;cursor:grab';
  const active = gpuView(canvas) ?? cpuView(canvas), target = canvas;
  const yaw = slider('視点を回す', { min: -180, max: 180, value: 35, format: v => `${v}°` });
  const pitch = slider('視点の高さ', { min: -10, max: 80, value: 18, format: v => `${v}°` });
  const subject = choice('題材', Object.fromEntries(Object.entries(SCENES).map(([id, item]) => [id, item.label])));
  const readout = el('pre', { className: 'lab-readout' }, '読み込み中…');
  const credit = el('p', { className: 'fine' });
  let scene = null, distance = 5;

  function draw() {
    if (!scene) return;
    active.render(orbit(scene, yaw.value * Math.PI / 180, pitch.value * Math.PI / 180, distance), theme().background);
    readout.textContent = `Gaussian ${scene.count.toLocaleString('ja-JP')}個 ／ ${active.label}`;
  }

  let drag = null;
  target.addEventListener('pointerdown', event => { drag = { x: event.clientX, y: event.clientY, yaw: yaw.value, pitch: pitch.value }; target.setPointerCapture(event.pointerId); });
  target.addEventListener('pointermove', event => {
    if (!drag) return;
    yaw.value = ((drag.yaw - (event.clientX - drag.x) * 0.6 + 540) % 360) - 180;
    pitch.value = Math.max(-10, Math.min(80, drag.pitch + (event.clientY - drag.y) * 0.4));
    draw();
  });
  for (const type of ['pointerup', 'pointercancel']) target.addEventListener(type, () => { drag = null; });
  root.append(el('div', { className: 'lab-frame' }, target), credit);
  if (!root.hasAttribute('data-compact')) root.append(el('div', { className: 'controls' }, yaw.root, pitch.root, subject.root), readout);
  [yaw, pitch].forEach(control => control.input.addEventListener('input', draw));
  onTheme(draw);

  function load() {
    const item = SCENES[subject.value], [name, page, author, license, terms, note] = item.credit;
    scene = null;
    readout.textContent = '読み込み中…';
    credit.replaceChildren('「', el('a', { href: page }, name), `」${author} / `, el('a', { href: terms }, license), `。${note}`);
    fetch(item.url).then(response => {
      if (!response.ok) throw new Error(`データを読み込めません（${response.status}）`);
      return response.arrayBuffer();
    }).then(buffer => {
      scene = Object.assign(parseSplat(buffer), item.view);
      active.setScene(scene);
      yaw.value = item.yaw; pitch.value = item.pitch; distance = item.distance;
      draw();
    }).catch(error => { readout.textContent = error.message; });
  }
  subject.input.addEventListener('change', load);
  const preset = root.dataset.scene || new URLSearchParams(location.search).get('scene'); // 本文に埋め込むときは data-scene で題材を指定する
  if (SCENES[preset]) subject.input.value = preset;
  load();
}
