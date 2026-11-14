export function gaussianWeight(x, y, { cx, cy, sx, sy, angle }) {
  if (!(sx > 0 && sy > 0)) throw new Error('Gaussianのスケールは正である必要があります。');
  const dx = x - cx;
  const dy = y - cy;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const u = (cosine * dx + sine * dy) / sx;
  const v = (-sine * dx + cosine * dy) / sy;
  return Math.exp(-0.5 * (u * u + v * v));
}

export function composite(samples, background = [1, 1, 1]) {
  const color = [0, 0, 0];
  let transmittance = 1;
  for (const { rgb, alpha } of samples) {
    if (alpha < 0 || alpha > 1) throw new Error('不透明度は0以上1以下である必要があります。');
    for (let channel = 0; channel < 3; channel += 1) {
      color[channel] += transmittance * alpha * rgb[channel];
    }
    transmittance *= 1 - alpha;
  }
  return {
    rgb: color.map((value, index) => value + transmittance * background[index]),
    alpha: 1 - transmittance,
  };
}
