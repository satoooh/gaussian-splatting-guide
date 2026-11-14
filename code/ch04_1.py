import math


def alpha_at(splat, x, y):
    cx, cy, sx, sy, angle, depth, opacity, rgb = splat
    dx, dy = x - cx, y - cy
    c, s = math.cos(angle), math.sin(angle)
    u, v = c * dx + s * dy, -s * dx + c * dy
    r2 = (u / sx) ** 2 + (v / sy) ** 2
    return opacity * math.exp(-0.5 * r2)


def pixel(splats, x, y, background=(0, 0, 0)):
    result = [0.0, 0.0, 0.0]
    remaining = 1.0
    for splat in sorted(splats, key=lambda item: item[5]):
        alpha = alpha_at(splat, x, y)
        color = splat[7]
        for k in range(3):
            result[k] += remaining * alpha * color[k]
        remaining *= 1 - alpha
    return tuple(result[k] + remaining * background[k] for k in range(3))


red = (0, 0, 1, 1, 0, 1, 0.5, (1, 0, 0))
blue = (0, 0, 1, 1, 0, 2, 0.5, (0, 0, 1))
assert pixel([blue, red], 0, 0) == (0.5, 0.0, 0.25)
assert alpha_at(red, 0, 0) == 0.5

splats = [
    (80, 64, 32, 16, math.radians(-25), 1, 0.7, (0.1, 0.4, 1)),
    (112, 64, 24, 28, math.radians(20), 2, 0.6, (1, 0.35, 0.05)),
]
width, height = 192, 128
data = bytearray()
for y in range(height):
    for x in range(width):
        rgb = pixel(splats, x + 0.5, y + 0.5, (0.04, 0.04, 0.04))
        data.extend(round(255 * min(1, max(0, value))) for value in rgb)
with open("splats.ppm", "xb") as image:
    image.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
    image.write(data)
print("splats.ppm を作成しました")
