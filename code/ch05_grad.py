import random


def composite(alphas, colors, background):
    result = [0.0, 0.0, 0.0]
    remaining = 1.0
    for alpha, color in zip(alphas, colors):
        for k in range(3):
            result[k] += remaining * alpha * color[k]
        remaining *= 1 - alpha
    return tuple(result[k] + remaining * background[k] for k in range(3))


# ∂C/∂α_i = T_i (c_i − B_i)。B_i は i 番目より後ろだけを重ねた色で、奥から手前へたどって求める。
def gradients(alphas, colors, background):
    transmittances, remaining = [], 1.0
    for alpha in alphas:
        transmittances.append(remaining)
        remaining *= 1 - alpha
    result, behind = [None] * len(alphas), background
    for i in reversed(range(len(alphas))):
        alpha, color = alphas[i], colors[i]
        result[i] = tuple(transmittances[i] * (color[k] - behind[k]) for k in range(3))
        behind = tuple(alpha * color[k] + (1 - alpha) * behind[k] for k in range(3))
    return result


def loss(alphas, colors, background, target):
    return sum(abs(a - b) for a, b in zip(composite(alphas, colors, background), target)) / 3


# ∂L/∂α_i = Σ_k sign(C_k − target_k) (∂C_k/∂α_i) / 3
def loss_gradient(alphas, colors, background, target):
    color = composite(alphas, colors, background)
    signs = [(a > b) - (a < b) for a, b in zip(color, target)]
    return [sum(signs[k] * gradient[k] for k in range(3)) / 3
            for gradient in gradients(alphas, colors, background)]


alphas, colors, background = [0.5, 0.5], [(1, 0, 0), (0, 0, 1)], (0, 0, 0)
target = (0.8, 0, 0.1)
assert composite(alphas, colors, background) == (0.5, 0.0, 0.25)
assert gradients(alphas, colors, background)[0] == (1.0, 0.0, -0.5)
assert abs(loss(alphas, colors, background, target) - 0.15) < 1e-12
assert loss_gradient(alphas, colors, background, target) == [-0.5, 1 / 6]
print("色:", composite(alphas, colors, background))
print("赤の ∂C/∂α:", gradients(alphas, colors, background)[0])
print("L1:", round(loss(alphas, colors, background, target), 6))
print("∂L/∂α（赤、青）:", [round(g, 6) for g in loss_gradient(alphas, colors, background, target)])

# 値を少し動かして測った傾き（数値微分）と比べる
rng, errors = random.Random(0), []
for _ in range(10):
    a = [rng.uniform(0.1, 0.9) for _ in range(4)]
    c = [tuple(rng.random() for _ in range(3)) for _ in a]
    bg, teacher = [rng.random() for _ in range(3)], [rng.random() for _ in range(3)]
    for i, exact in enumerate(loss_gradient(a, c, bg, teacher)):
        plus, minus = a[:], a[:]
        plus[i] += 1e-6
        minus[i] -= 1e-6
        numerical = (loss(plus, c, bg, teacher) - loss(minus, c, bg, teacher)) / 2e-6
        errors.append(abs(exact - numerical))
assert max(errors) < 1e-6
print(f"四つ重ねた乱数の例10通りで、式の勾配と数値微分が一致（最大の差 {max(errors):.1e}）")
