import math


def rotation(q):
    norm = math.sqrt(sum(value * value for value in q))
    w, x, y, z = (value / norm for value in q)
    return [[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]]


# Σ = R S Sᵀ Rᵀ
def covariance(scale, q):
    R = rotation(q)
    return [[sum(R[i][k] * scale[k]**2 * R[j][k] for k in range(3))
             for j in range(3)] for i in range(3)]


# Σ_screen = J Σ Jᵀ
def project(mean, cov, fx, fy, cx, cy):
    x, y, z = mean
    J = [[fx / z, 0, -fx * x / z**2], [0, fy / z, -fy * y / z**2]]
    cov2 = [[sum(J[i][k] * cov[k][l] * J[j][l]
                 for k in range(3) for l in range(3)) for j in range(2)] for i in range(2)]
    return (fx * x / z + cx, fy * y / z + cy), cov2


# g = exp(−½ dᵀ Σ⁻¹ d)
def weight(center, cov2, px, py):
    dx, dy = px - center[0], py - center[1]
    a, b, c = cov2[0][0], cov2[0][1], cov2[1][1]
    r2 = (c * dx**2 - 2 * b * dx * dy + a * dy**2) / (a * c - b * b)
    return math.exp(-0.5 * r2)


def render(splats, width, height, fx, fy, cx, cy, background):
    projected = []
    for mean, scale, q, opacity, rgb in sorted(splats, key=lambda item: item[0][2]):
        center, cov2 = project(mean, covariance(scale, q), fx, fy, cx, cy)
        projected.append((center, cov2, opacity, rgb))
    data = bytearray()
    for y in range(height):
        for x in range(width):
            result = [0.0, 0.0, 0.0]
            remaining = 1.0
            for center, cov2, opacity, color in projected:
                alpha = opacity * weight(center, cov2, x + 0.5, y + 0.5)
                for k in range(3):
                    result[k] += remaining * alpha * color[k]
                remaining *= 1 - alpha
            rgb = [result[k] + remaining * background[k] for k in range(3)]
            data.extend(round(255 * min(1, max(0, value))) for value in rgb)
    return data


q = (1, 0, 0, 0)
cov = covariance((0.01, 0.01, 0.01), q)
stds = [math.sqrt(project((0, 0, z), cov, 1000, 1000, 0, 0)[1][0][0]) for z in (2, 1)]
assert stds == [5.0, 10.0]
center, cov2 = project((0.6, 0, 2), covariance((0.1, 0.1, 0.6), q), 1000, 1000, 0, 0)
std = math.sqrt(cov2[0][0])
assert center == (300.0, 0.0) and math.isclose(std, 102.95630140987)
rotated = covariance((0.3, 0.1, 0.1), (1, 0, 0, 1))
assert all(math.isclose(rotated[i][i], value) for i, value in enumerate((0.01, 0.09, 0.01)))
assert weight(center, cov2, *center) == 1.0
print(f"画面上の標準偏差: z=2 → {stds[0]:.2f} px, z=1 → {stds[1]:.2f} px")
print(f"光軸の外: 中心={center[0]:.2f} px, 標準偏差={std:.5f} px, 範囲={center[0]-std:.2f}〜{center[0]+std:.2f} px")
print("z軸90度回転後の共分散の対角:", tuple(round(rotated[i][i], 5) for i in range(3)))

# カメラは原点にあって+zを向く。粒はカメラ座標で与える（世界座標からの変換は省いた）。
# 青が奥、赤が手前、緑がその間にある。
splats = [
    ((0.15, 0.0, 3.0), (0.50, 0.28, 0.2), q, 0.9, (0.1, 0.3, 1)),
    ((-0.10, 0.02, 1.5), (0.10, 0.20, 0.1), (1, 0, 0, 0.25), 0.95, (1, 0.3, 0.05)),
    ((0.35, 0.15, 2.2), (0.16, 0.10, 0.2), q, 0.8, (0.1, 0.9, 0.3)),
]
width, height = 192, 128
data = render(splats, width, height, 160, 160, width / 2, height / 2, (0.04, 0.04, 0.04))
with open("splats3d.ppm", "wb") as image:
    image.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
    image.write(data)
print("検査OK / splats3d.ppm を作成しました")
