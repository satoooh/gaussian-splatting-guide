import math
import random

DENSIFY = True  # False にすると、粒は8個のまま増えない
WIDTH, HEIGHT = 48, 32
BACKGROUND = (0.15, 0.15, 0.15)
RATES = (0.08, 0.08, 0.025, 0.05, 0.03, 0.03, 0.03)  # x, y, log σ, 不透明度, r, g, b の学習率


def teacher(x, y):
    if 6 < x < 42 and abs(y - 0.25 * x - 19) < 1.2:
        return (0.95, 0.8, 0.2)  # 細い線
    if (x - 14) ** 2 + (y - 12) ** 2 < 7 ** 2:
        return (0.9, 0.25, 0.15)
    if (x - 33) ** 2 + (y - 17) ** 2 < 6 ** 2:
        return (0.15, 0.55, 0.95)
    return BACKGROUND


# 一つの粒は [x, y, log σ, 不透明度のロジット, r, g, b]。
# σはexp、不透明度はシグモイドを通して使う。どう更新しても、σは正に、不透明度は0から1に収まる。
def activate(p):
    return p[0], p[1], math.exp(p[2]), 1 / (1 + math.exp(-p[3])), [min(1, max(0, c)) for c in p[4:]]


# 描画して、損失、PSNR、画像、勾配を返す。
def evaluate(splats, samples):
    values = [activate(p) for p in splats]
    gradients = [[0.0] * 7 for p in splats]
    loss, squared, rendered = 0.0, 0.0, []
    scale = 1 / (3 * len(samples))
    for x, y, target in samples:
        # 前向き：手前から重ねて、画素の色を決める
        result, remaining, used = [0.0] * 3, 1.0, []
        for i, (cx, cy, sigma, opacity, color) in enumerate(values):
            dx, dy = x - cx, y - cy
            r2 = (dx * dx + dy * dy) / sigma ** 2
            if r2 > 9:  # 3σより遠い画素には、ほとんど効かない
                continue
            alpha = opacity * math.exp(-0.5 * r2)
            used.append((i, alpha, remaining, dx, dy, r2))
            for k in range(3):
                result[k] += remaining * alpha * color[k]
            remaining *= 1 - alpha
        result = [result[k] + remaining * BACKGROUND[k] for k in range(3)]
        rendered.append(result)
        error = [result[k] - target[k] for k in range(3)]
        loss += sum(abs(e) for e in error) * scale
        squared += sum(e * e for e in error) * scale
        # 後ろ向き：奥から手前へたどって、勾配を足していく
        dL_dC = [scale * ((e > 0) - (e < 0)) for e in error]
        behind = list(BACKGROUND)
        for i, alpha, remaining, dx, dy, r2 in reversed(used):
            cx, cy, sigma, opacity, color = values[i]
            # ∂C/∂α = T (c − B)。Bはこの粒より後ろに見えている色
            dL_dalpha = sum(dL_dC[k] * remaining * (color[k] - behind[k]) for k in range(3))
            g = gradients[i]
            g[0] += dL_dalpha * alpha * dx / sigma ** 2  # ∂α/∂x
            g[1] += dL_dalpha * alpha * dy / sigma ** 2  # ∂α/∂y
            g[2] += dL_dalpha * alpha * r2               # ∂α/∂(log σ)
            g[3] += dL_dalpha * alpha * (1 - opacity)    # ∂α/∂(ロジット)
            for k in range(3):
                g[4 + k] += dL_dC[k] * remaining * alpha * (0 <= splats[i][4 + k] <= 1)  # ∂C/∂c = T α
                behind[k] = alpha * color[k] + (1 - alpha) * behind[k]
    return loss, -10 * math.log10(max(squared, 1e-15)), rendered, gradients


# 式で求めた勾配が、値を少し動かして測った傾きと合うことを確かめる。
def check_gradients():
    splats = [[12.3, 13.4, math.log(5.1), -0.3, 0.7, 0.2, 0.4],
              [16.2, 11.7, math.log(4.3), 0.4, 0.3, 0.8, 0.2],
              [14.8, 15.1, math.log(3.7), -0.6, 0.2, 0.3, 0.9]]
    samples = [(x, y, (0.95, 0.04, 0.98)) for x, y in [(12.1, 11.3), (15.6, 14.2), (17.4, 16.8)]]
    analytic = evaluate(splats, samples)[3]
    worst, h = 0.0, 1e-5
    for i, p in enumerate(splats):
        for j in range(7):
            original = p[j]
            p[j] = original + h
            plus = evaluate(splats, samples)[0]
            p[j] = original - h
            minus = evaluate(splats, samples)[0]
            p[j] = original
            worst = max(worst, abs((plus - minus) / (2 * h) - analytic[i][j]))
    assert worst < 1e-7, worst
    print(f"勾配の検査: 21個の値で一致（最大の差 {worst:.1e}）")


# Adam。勾配の向きと大きさの移動平均を使って、値ごとに歩幅を調整する。
def adam(p, g, state):
    first, second = state[0], state[1]
    state[2] += 1
    for j in range(7):
        first[j] = 0.9 * first[j] + 0.1 * g[j]
        second[j] = 0.999 * second[j] + 0.001 * g[j] ** 2
        step = (first[j] / (1 - 0.9 ** state[2])) / (math.sqrt(second[j] / (1 - 0.999 ** state[2])) + 1e-8)
        p[j] -= RATES[j] * step
    p[4:] = [min(1, max(0, c)) for c in p[4:]]


# 密度制御。薄い粒を削り、位置の勾配が大きい粒を、大きければ分割、小さければ複製する。
def densify(splats, states, pull, rng):
    fresh = lambda: [[0.0] * 7, [0.0] * 7, 0]
    kept = [(p, s, score) for p, s, score in zip(splats, states, pull) if activate(p)[3] >= 0.05]
    splats, states, room = [], [], 60 - len(kept)
    for p, state, score in kept:
        if score / 100 <= 0.00002 or room <= 0:
            splats.append(p)
            states.append(state)
            continue
        sigma, children = math.exp(p[2]), [p.copy(), p.copy()]
        if sigma > 2:  # 分割：二つとも小さくして、元の広がりの中にずらして置く
            for child in children:
                child[0] += rng.uniform(-0.7, 0.7) * sigma
                child[1] += rng.uniform(-0.7, 0.7) * sigma
                child[2] -= math.log(1.6)
        splats.extend(children)
        states.extend([state if sigma <= 2 else fresh(), fresh()])
        room -= 1
    return splats, states


check_gradients()
rng = random.Random(0)
samples = [(x + 0.5, y + 0.5, teacher(x + 0.5, y + 0.5)) for y in range(HEIGHT) for x in range(WIDTH)]
splats = [[rng.uniform(5, 43), rng.uniform(4, 28), math.log(6), 0.0, rng.random(), rng.random(), rng.random()]
          for i in range(8)]
states = [[[0.0] * 7, [0.0] * 7, 0] for p in splats]
pull = [0.0] * len(splats)  # 位置の勾配の大きさを、密度制御まで足し込む
first_psnr = evaluate(splats, samples)[1]
for iteration in range(601):
    loss, psnr, rendered, gradients = evaluate(splats, samples)
    if iteration % 50 == 0:
        print(f"反復{iteration:4d}  損失 {loss:.4f}  粒 {len(splats):2d}個  PSNR {psnr:.2f} dB")
    if iteration == 600:
        break
    for i, (p, g, state) in enumerate(zip(splats, gradients, states)):
        pull[i] += math.hypot(g[0], g[1])
        adam(p, g, state)
    if DENSIFY and (iteration + 1) % 100 == 0 and iteration < 400:
        splats, states = densify(splats, states, pull, rng)
        pull = [0.0] * len(splats)
print(f"PSNRは {first_psnr:.2f} dB から {psnr:.2f} dB になった")
with open("train.ppm", "wb") as image:
    image.write(f"P6\n{WIDTH * 2} {HEIGHT}\n255\n".encode("ascii"))
    for y in range(HEIGHT):
        row = [s[2] for s in samples[y * WIDTH:(y + 1) * WIDTH]] + rendered[y * WIDTH:(y + 1) * WIDTH]
        image.write(bytes(round(255 * min(1, max(0, value))) for rgb in row for value in rgb))
print("train.ppm を作成しました（左が教師画像、右が学習した結果）")
