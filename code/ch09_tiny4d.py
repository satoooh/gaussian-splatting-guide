import argparse
import json
import math
import random
from pathlib import Path

WIDTH, HEIGHT = 48, 32
ITERATIONS, BATCH_SIZE = 600, 384
TRAIN_TIMES, EVAL_TIMES = list(range(0, 31, 2)), list(range(1, 30, 2))
BACKGROUND = (0.12, 0.12, 0.12)
COLORS = [(0.95, 0.8, 0.2), (0.85, 0.85, 0.85), (0.6, 0.6, 0.6)]
VELOCITIES = [(0.7, 0.1), (0.0, 0.0), (0.0, 0.0)]
TEMPORAL_CENTER, TEMPORAL_SIGMA = 15, 3.5
RATES = (0.08, 0.08, 0.08, 0.025, 0.025, 0.012, 0.012, 0.06)


def teacher(x, y, t):
    vx, vy = VELOCITIES[0]
    if (x - 10 - vx * t) ** 2 + (y - 7 - vy * t) ** 2 < 4 ** 2:
        return COLORS[0]
    if (x - 36) ** 2 + (y - 23) ** 2 < 4 ** 2:
        weight = math.exp(-0.5 * ((t - TEMPORAL_CENTER) / TEMPORAL_SIGMA) ** 2)
        return tuple(weight * c + (1 - weight) * b for c, b in zip(COLORS[1], BACKGROUND))
    if 5 < x < 17 and 23 < y < 27:
        return COLORS[2]
    return BACKGROUND


# 一粒は [μx, μy, μt, log σ, log σt, vx, vy, 不透明度のロジット]。
# Σ = [[σ²+vx²Σtt, vx vy Σtt, vxΣtt], [vx vy Σtt, σ²+vy²Σtt, vyΣtt], [vxΣtt, vyΣtt, Σtt]]。
# σとσtをexpで正に保つので、共分散は常に正定値になる。
def activate(p):
    variance, tt = math.exp(2 * p[3]), math.exp(2 * p[4])
    return variance, tt, p[5] * tt, p[6] * tt, 1 / (1 + math.exp(-p[7]))


# 時刻で切ると、中心は μx + Σxt/Σtt (t−μt)、分散は σ²になる。
def evaluate(splats, samples):
    values = [activate(p) for p in splats]
    gradients = [[0.0] * 8 for p in splats]
    squared, rendered, scale = 0.0, [], 1 / (3 * len(samples))
    for x, y, t, target in samples:
        result, remaining, used = [0.0] * 3, 1.0, []
        for i, (p, (variance, tt, xt, yt, opacity)) in enumerate(zip(splats, values)):
            dt = t - p[2]
            dx, dy = x - p[0] - xt / tt * dt, y - p[1] - yt / tt * dt
            space, time = (dx * dx + dy * dy) / variance, dt * dt / tt
            alpha = opacity * math.exp(-0.5 * (space + time))
            used.append((i, alpha, remaining, dx, dy, dt, space, time))
            for k in range(3):
                result[k] += remaining * alpha * COLORS[i][k]
            remaining *= 1 - alpha
        result = [result[k] + remaining * BACKGROUND[k] for k in range(3)]
        rendered.append(result)
        error = [result[k] - target[k] for k in range(3)]
        squared += sum(e * e for e in error) * scale
        behind = list(BACKGROUND)
        for i, alpha, remaining, dx, dy, dt, space, time in reversed(used):
            p, (variance, tt, xt, yt, opacity) = splats[i], values[i]
            pull = sum(2 * scale * error[k] * remaining * (COLORS[i][k] - behind[k]) for k in range(3)) * alpha
            # μtを変えると、時間の重みと断面の中心の両方が変わる。
            derivatives = [dx / variance, dy / variance,
                           dt / tt - (dx * xt + dy * yt) / (tt * variance), space, time,
                           dx * dt / variance, dy * dt / variance, 1 - opacity]
            for j in range(8):
                gradients[i][j] += pull * derivatives[j]
            behind = [alpha * COLORS[i][k] + (1 - alpha) * behind[k] for k in range(3)]
    return squared, -10 * math.log10(max(squared, 1e-15)), rendered, gradients


def check_gradients():
    splats = [[13.2, 12.4, 14, math.log(4), math.log(6), 0.3, -0.2, 0.4],
              [15.3, 14.1, 12, math.log(3), math.log(5), -0.1, 0.2, -0.3],
              [11.4, 16.2, 13, math.log(5), math.log(8), 0.2, 0.1, 0.1]]
    samples = [(x, y, t, (0.4, 0.3, 0.2)) for x, y, t in [(12.1, 11.3, 9), (16.2, 14.3, 17)]]
    analytic, worst, h = evaluate(splats, samples)[3], 0.0, 1e-5
    for i, p in enumerate(splats):
        for j in range(8):
            original = p[j]
            p[j] = original + h
            plus = evaluate(splats, samples)[0]
            p[j] = original - h
            minus = evaluate(splats, samples)[0]
            p[j] = original
            worst = max(worst, abs((plus - minus) / (2 * h) - analytic[i][j]))
    assert worst < 1e-7, worst
    print(f"勾配の検査: 24個の値で一致（最大の差 {worst:.1e}）")


def adam(p, g, state, moving):
    first, second = state[0], state[1]
    state[2] += 1
    for j in range(8):
        if not moving and j in (5, 6):
            continue
        first[j] = 0.9 * first[j] + 0.1 * g[j]
        second[j] = 0.999 * second[j] + 0.001 * g[j] ** 2
        step = (first[j] / (1 - 0.9 ** state[2])) / (math.sqrt(second[j] / (1 - 0.999 ** state[2])) + 1e-8)
        p[j] -= RATES[j] * (1 - 0.9 * state[2] / ITERATIONS) * step


def samples_at(times):
    return [(x + 0.5, y + 0.5, t, teacher(x + 0.5, y + 0.5, t))
            for t in times for y in range(HEIGHT) for x in range(WIDTH)]


def train(training, evaluation, moving):
    rng = random.Random(0)
    splats = [[x + rng.uniform(-2, 2), y + rng.uniform(-2, 2), 12, math.log(3), math.log(12), 0, 0, 0]
              for x, y in [(20, 9), (34, 21), (12, 24)]]
    states = [[[0.0] * 8, [0.0] * 8, 0] for p in splats]
    for iteration in range(ITERATIONS):
        gradients = evaluate(splats, rng.sample(training, BATCH_SIZE))[3]
        for p, g, state in zip(splats, gradients, states):
            adam(p, g, state, moving)
    loss, psnr, rendered, _ = evaluate(splats, evaluation)
    print(f"{'時空間の相関を学習' if moving else 'Σxt = Σyt = 0に固定'}: 評価PSNR {psnr:.2f} dB")
    particles = []
    for i, p in enumerate(splats):
        variance, tt, xt, yt, opacity = activate(p)
        particles.append(dict(mean=p[:3], sigma=math.sqrt(variance), tt=tt, xt=xt, yt=yt, opacity=opacity))
        print(f"粒{i + 1}: Σxt/Σtt = {xt / tt:.4f}, Σyt/Σtt = {yt / tt:.4f} 画素/フレーム"
              f"（教師 {VELOCITIES[i][0]:.4f}, {VELOCITIES[i][1]:.4f}）")
    p = particles[1]
    print(f"出現・消失する粒: μt = {p['mean'][2]:.4f}, √Σtt = {math.sqrt(p['tt']):.4f} フレーム"
          f"（教師 {TEMPORAL_CENTER:.4f}, {TEMPORAL_SIGMA:.4f}）")
    return dict(psnr=psnr, mse=loss, particles=particles, frames=rendered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2Dに時間を足した最小のガウシアン学習")
    parser.add_argument("--static", action="store_true", help="ΣxtとΣytを0に固定した比較だけを実行")
    args = parser.parse_args()
    check_gradients()
    print(f"{WIDTH}×{HEIGHT}画素、31時刻、学習{len(TRAIN_TIMES)}枚、評価{len(EVAL_TIMES)}枚、3粒、{ITERATIONS}反復、seed 0")
    training, evaluation = samples_at(TRAIN_TIMES), samples_at(EVAL_TIMES)
    results = {"static": train(training, evaluation, False)} if args.static else {
        "learned": train(training, evaluation, True), "static": train(training, evaluation, False)}
    data = dict(width=WIDTH, height=HEIGHT, iterations=ITERATIONS, batch_size=BATCH_SIZE, seed=0,
                train_times=TRAIN_TIMES, eval_times=EVAL_TIMES, teacher_velocities=VELOCITIES,
                temporal_center=TEMPORAL_CENTER, temporal_sigma=TEMPORAL_SIGMA,
                teacher=[s[3] for s in evaluation], results=results)
    output = Path(__file__).with_suffix(".static.json" if args.static else ".json")
    output.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{output.name} を作成しました（評価15枚の全画素と共分散）")
