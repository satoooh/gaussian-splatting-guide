import math
import random

WIDTH, HEIGHT, FOCAL = 32, 24, 26.0
BACKGROUND = (0.04, 0.04, 0.04)
ITERATIONS, SEED = 350, 0
RATES = (0.025,) * 3 + (0.025,) * 3 + (0.018,) * 4 + (0.035,) + (0.025,) * 3
PIXELS = [(x + 0.5, y + 0.5) for y in range(HEIGHT) for x in range(WIDTH)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def rotation(q):
    norm = math.sqrt(sum(value * value for value in q))
    w, x, y, z = (value / norm for value in q)
    return [[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]]


def camera(yaw, pitch):
    # 世界→カメラ回転W。カメラ中心は−3W[2]で、世界座標の原点を向く。
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    return [[cy, 0, -sy], [-sp * sy, cp, -sp * cy], [cp * sy, sp, cp * cy]]


# 一つの粒は [x,y,z, log sx,log sy,log sz, qw,qx,qy,qz, 不透明度のロジット, r,g,b]。
def project(p, W):
    R, variances = rotation(p[6:10]), [math.exp(2 * s) for s in p[3:6]]
    # Σ = R S Sᵀ Rᵀ。Sの対角がexp(log s)である。
    cov = [[sum(R[i][k] * variances[k] * R[j][k] for k in range(3))
            for j in range(3)] for i in range(3)]
    mean = [sum(W[i][j] * p[j] for j in range(3)) + (3 if i == 2 else 0) for i in range(3)]
    x, y, z = mean
    assert z > 0, "この例の粒はすべてカメラの前に置く"
    J = [[FOCAL / z, 0, -FOCAL * x / z**2], [0, FOCAL / z, -FOCAL * y / z**2]]
    A = matmul(J, W)
    # C = J W Σ Wᵀ Jᵀ + 0.3 I。0.3の単位は画素の二乗である。
    C = matmul(matmul(A, cov), transpose(A))
    C[0][0] += 0.3
    C[1][1] += 0.3
    a, b, c = C[0][0], C[0][1], C[1][1]
    det = a * c - b * b
    conic = (c / det, -b / det, a / det)
    center = (FOCAL * x / z + WIDTH / 2, FOCAL * y / z + HEIGHT / 2)
    cache = (mean, R, variances, cov, J, A, W)
    return center, conic, 1 / (1 + math.exp(-p[10])), p[11:14], cache


def pixel(projected, order, px, py):
    result, remaining, used = [0.0] * 3, 1.0, []
    for i in order:
        center, (a, b, c), opacity, color, cache = projected[i]
        dx, dy = px - center[0], py - center[1]
        vx, vy = a * dx + b * dy, b * dx + c * dy
        alpha = opacity * math.exp(-0.5 * (dx * vx + dy * vy))
        used.append((i, alpha, remaining, vx, vy))
        for k in range(3):
            result[k] += remaining * alpha * color[k]
        remaining *= 1 - alpha
    return [result[k] + remaining * BACKGROUND[k] for k in range(3)], used


def alpha_backward(g, dalpha, alpha, vx, vy, opacity):
    # dμ₂ = dα α C⁻¹d, dC = ½ dα α (C⁻¹d)(C⁻¹d)ᵀ, du = dα α(1−o)。
    pull = dalpha * alpha
    g[0] += pull * vx
    g[1] += pull * vy
    g[2] += 0.5 * pull * vx * vx
    g[3] += 0.5 * pull * vx * vy  # 対称な2×2行列の両非対角に同じ値を置く
    g[4] += 0.5 * pull * vy * vy
    g[5] += pull * (1 - opacity)


def composite_backward(projected, used, dcolor, gradients, screen):
    behind = list(BACKGROUND)
    for i, alpha, remaining, vx, vy in reversed(used):
        center, conic, opacity, color, cache = projected[i]
        # dα_i = dC_pixel · T_i(c_i−B_i), dc_i = T_i α_i dC_pixel。
        dalpha = sum(dcolor[k] * remaining * (color[k] - behind[k]) for k in range(3))
        alpha_backward(screen[i], dalpha, alpha, vx, vy, opacity)
        for k in range(3):
            gradients[i][11 + k] += dcolor[k] * remaining * alpha
            behind[k] = alpha * color[k] + (1 - alpha) * behind[k]


def rotation_backward(q, dR):
    norm = math.sqrt(sum(value * value for value in q))
    unit = [value / norm for value in q]
    w, x, y, z = unit
    derivatives = [
        [[0, -2*z, 2*y], [2*z, 0, -2*x], [-2*y, 2*x, 0]],
        [[0, 2*y, 2*z], [2*y, -4*x, -2*w], [2*z, 2*w, -4*x]],
        [[-4*y, 2*x, 2*w], [2*x, 0, 2*z], [-2*w, 2*z, -4*y]],
        [[-4*z, -2*w, 2*x], [2*w, -4*z, 2*y], [2*x, 2*y, 0]],
    ]
    # d q̂_k = Σ_ij dR_ij (∂R_ij/∂q̂_k)。上の四行がRの各成分の微分である。
    du = [sum(dR[i][j] * derivative[i][j] for i in range(3) for j in range(3))
          for derivative in derivatives]
    radial = sum(unit[k] * du[k] for k in range(4))
    # dq = (I−q̂q̂ᵀ) d q̂ / ||q||。正規化によって消える半径方向も除く。
    return [(du[k] - unit[k] * radial) / norm for k in range(4)]


def covariance_backward(p, R, variances, G):
    local = matmul(matmul(transpose(R), G), R)
    # d(log s_k) = 2s_k²(RᵀGR)_kk, dR = 2 G R diag(s_k²)。
    dscale = [2 * variances[k] * local[k][k] for k in range(3)]
    GR = matmul(G, R)
    dR = [[2 * GR[i][j] * variances[j] for j in range(3)] for i in range(3)]
    return dscale + rotation_backward(p[6:10], dR)


def project_backward(p, cache, screen, gradient):
    mean, R, variances, cov, J, A, W = cache
    gx, gy, gxx, gxy, gyy, du = screen
    G2 = [[gxx, gxy], [gxy, gyy]]
    # G₃ = AᵀG₂A, dA = 2G₂AΣ, dJ = dA Wᵀ（A = J W）。
    G3 = matmul(matmul(transpose(A), G2), A)
    dJ = matmul(matmul(matmul(G2, A), cov), transpose(W))
    dJ = [[2 * value for value in row] for row in dJ]
    x, y, z = mean
    # dμ_camera = Jᵀdμ₂ + Σ_ij dJ_ij ∇μ J_ij。Jの位置依存も含める。
    dm = [J[0][0] * gx - FOCAL / z**2 * dJ[0][2],
          J[1][1] * gy - FOCAL / z**2 * dJ[1][2],
          J[0][2] * gx + J[1][2] * gy - FOCAL / z**2 * (dJ[0][0] + dJ[1][1])
          + 2 * FOCAL / z**3 * (x * dJ[0][2] + y * dJ[1][2])]
    # dμ_world = Wᵀ dμ_camera。カメラ姿勢は固定して学習する。
    for k in range(3):
        gradient[k] += sum(W[i][k] * dm[i] for i in range(3))
    for k, value in enumerate(covariance_backward(p, R, variances, G3), 3):
        gradient[k] += value
    gradient[10] += du


def render(splats, W, pixels=PIXELS):
    projected = [project(p, W) for p in splats]
    order = sorted(range(len(splats)), key=lambda i: projected[i][4][0][2])
    return [pixel(projected, order, x, y)[0] for x, y in pixels]


# viewsの各要素は（固定カメラW、[(画素x, 画素y, 教師RGB), ...]）。損失は全画素のMSE。
def evaluate(splats, views, backward=True):
    gradients = [[0.0] * 14 for p in splats]
    loss, scale = 0.0, 1 / (3 * sum(len(samples) for W, samples in views))
    for W, samples in views:
        projected = [project(p, W) for p in splats]
        # 奥行きの順序が変わらない区間を微分する。ソート自体の勾配は定義しない。
        order = sorted(range(len(splats)), key=lambda i: projected[i][4][0][2])
        screen = [[0.0] * 6 for p in splats]  # dμx, dμy, dCxx, dCxy(=dCyx), dCyy, du
        for x, y, target in samples:
            result, used = pixel(projected, order, x, y)
            error = [result[k] - target[k] for k in range(3)]
            loss += sum(e * e for e in error) * scale
            if backward:
                composite_backward(projected, used, [2 * scale * e for e in error], gradients, screen)
        if backward:
            for i, p in enumerate(splats):
                project_backward(p, projected[i][4], screen[i], gradients[i])
    return loss, -10 * math.log10(max(loss, 1e-15)), gradients


def scene():
    values = [((-0.42, -0.14, -0.20), (0.26, 0.12, 0.18), (1.1, 0.18, -0.25, 0.31), 1.3, (0.88, 0.23, 0.12)),
              ((0.35, -0.06, 0.10), (0.14, 0.29, 0.20), (0.9, -0.22, 0.33, 0.12), 1.7, (0.12, 0.48, 0.91)),
              ((0.03, 0.35, 0.36), (0.23, 0.11, 0.30), (1.2, 0.31, 0.14, -0.23), 1.0, (0.83, 0.74, 0.17))]
    return [list(mean) + [math.log(s) for s in scale] + list(q) + [u] + list(color)
            for mean, scale, q, u, color in values]


def check_gradients(cameras):
    splats, h = scene(), 1e-5
    # 非単位q、異方的な粒、光軸外、重なり、回転したカメラを含む画素で全成分を検査する。
    samples = [(x, y, (0.73, 0.18, 0.62)) for x, y in [(13.2, 11.3), (16.4, 13.1), (18.3, 10.7)]]
    worst, sensitivity = [0.0] * 14, [0.0] * 14
    for W in cameras:
        views = [(W, samples)]
        analytic = evaluate(splats, views)[2]
        for i, p in enumerate(splats):
            for j in range(14):
                original = p[j]
                p[j] = original + h
                plus = evaluate(splats, views, False)[0]
                p[j] = original - h
                minus = evaluate(splats, views, False)[0]
                p[j] = original
                numeric, exact = (plus - minus) / (2 * h), analytic[i][j]
                error = abs(numeric - exact)
                assert error < 1e-8 + 1e-5 * max(abs(numeric), abs(exact)), (i, j, numeric, exact)
                worst[j] = max(worst[j], error)
                sensitivity[j] = max(sensitivity[j], abs(exact))
    assert min(sensitivity) > 1e-7, sensitivity
    groups = [("位置", 0, 3), ("logスケール", 3, 6), ("クォータニオン", 6, 10), ("不透明度", 10, 11), ("RGB", 11, 14)]
    print(f"勾配の自己検査: {len(cameras) * len(splats) * 14}成分で有限差分と一致")
    print("最大絶対差: " + " / ".join(f"{name} {max(worst[a:b]):.1e}" for name, a, b in groups))
    # 恒等カメラで中心がz=2、sx=0.1ならCxx=(26/2×0.1)²+0.3=1.99。
    p = [0, 0, -1] + [math.log(s) for s in (0.1, 0.2, 0.3)] + [1, 0, 0, 0, 0, 0.5, 0.5, 0.5]
    projected = project(p, camera(0, 0))
    g = [0.0] * 14
    project_backward(p, projected[4], [0, 0, -0.2, 0, 0, 0], g)
    variance = 1 / projected[1][0]
    assert math.isclose(variance, 1.99) and math.isclose(g[3], -0.676)
    assert rotation([2, 0, 0, 0]) == rotation([1, 0, 0, 0])
    print(f"数値例: 焦点距離26 px / 奥行き2 / sx=0.1 → 2D分散x {variance:.2f} px²、∂L/∂Cxx=-0.2 → ∂L/∂log sx={g[3]:.3f}")


def adam(p, g, state, decay):
    first, second = state[0], state[1]
    state[2] += 1
    for j in range(14):
        first[j] = 0.9 * first[j] + 0.1 * g[j]
        second[j] = 0.999 * second[j] + 0.001 * g[j] ** 2
        step = (first[j] / (1 - 0.9 ** state[2])) / (math.sqrt(second[j] / (1 - 0.999 ** state[2])) + 1e-8)
        p[j] -= RATES[j] * decay * step
    p[11:] = [min(1, max(0, c)) for c in p[11:]]


def main():
    cameras = [camera(yaw, pitch) for yaw, pitch in [(-0.65, -0.2), (-0.2, 0.3), (0.3, -0.25), (0.7, 0.2), (0.1, 0.1)]]
    check_gradients(cameras[:4])
    teacher = scene()
    views = [(W, [(x, y, rgb) for (x, y), rgb in zip(PIXELS, render(teacher, W))]) for W in cameras]
    training, heldout = views[:4], views[4:]
    rng = random.Random(SEED)
    perturb = (0.16,) * 3 + (0.22,) * 3 + (0.18,) * 4 + (0.35,) + (0.12,) * 3
    splats = [[value + rng.uniform(-amount, amount) for value, amount in zip(p, perturb)] for p in teacher]
    states = [[[0.0] * 14, [0.0] * 14, 0] for p in splats]
    position_error = lambda: sum(math.dist(p[:3], t[:3]) for p, t in zip(splats, teacher)) / len(splats)
    before = (evaluate(splats, training, False)[1], evaluate(splats, heldout, False)[1], position_error())
    print(f"条件: 学習4視点 / 評価1視点 / {WIDTH}×{HEIGHT}画素 / {len(splats)}粒固定 / {ITERATIONS}反復 / seed={SEED} / MSE")
    # 教師の粒の値は更新に使わない。更新にはtrainingの画像とカメラだけを渡す。
    for iteration in range(ITERATIONS):
        loss, psnr, gradients = evaluate(splats, training)
        if iteration % 100 == 0:
            print(f"反復{iteration:4d}  MSE {loss:.8f}  学習PSNR {psnr:.2f} dB")
        decay = 0.2 ** (iteration / ITERATIONS)
        for p, g, state in zip(splats, gradients, states):
            adam(p, g, state, decay)
    after = (evaluate(splats, training, False)[1], evaluate(splats, heldout, False)[1], position_error())
    assert after[0] > before[0] and after[1] > before[1] and after[2] < before[2]
    print(f"学習視点PSNR: {before[0]:.2f} → {after[0]:.2f} dB")
    print(f"評価視点PSNR: {before[1]:.2f} → {after[1]:.2f} dB（更新には不使用）")
    print(f"平均位置誤差: {before[2]:.6f} → {after[2]:.6f}（既知の粒との対応、世界座標の単位）")


if __name__ == "__main__":
    main()
