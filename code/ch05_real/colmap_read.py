"""COLMAPの sparse/0 にある三つのバイナリ（cameras.bin、images.bin、points3D.bin）を読む。

形式は https://colmap.github.io/format.html の「Binary File Format」に従う。
"""
import struct

import numpy as np

# カメラのモデル番号 → 内部パラメータの数。SIMPLE_RADIALは歪みの係数kを一つ持つ。
MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4)}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def rotation(q):
    # クォータニオン (QW, QX, QY, QZ) → 回転行列。COLMAPは w を先頭に置く。
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
                     [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
                     [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]])


def read_cameras(path):
    cameras = {}
    with open(path, "rb") as f:
        for _ in range(read(f, "Q")[0]):
            camera_id, model, width, height = read(f, "iiQQ")
            assert model in MODELS, f"対応していないカメラのモデル（番号{model}）"
            params = read(f, "d" * MODELS[model][1])
            fx, fy, cx, cy = params if model == 1 else (params[0], params[0], params[1], params[2])
            # kを無視する（Brushと同じ）。料理のシーンはk=−0.001で、画像の隅でも約0.03%しかずれない。
            assert model != 2 or abs(params[3]) < 0.01, "歪みの大きい写真は、先に歪みを取り除く"
            cameras[camera_id] = dict(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    return cameras


def read_images(path):
    # 各写真の外部パラメータ。X_cam = R X + t（世界座標→カメラ座標、第3章）。
    images = []
    with open(path, "rb") as f:
        for _ in range(read(f, "Q")[0]):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = read(f, "i7di")
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            f.read(read(f, "Q")[0] * 24)  # 二次元の特徴点（x, y, 点の番号）は使わない
            images.append(dict(name=name.decode(), camera_id=camera_id,
                               R=rotation((qw, qx, qy, qz)), t=np.array([tx, ty, tz])))
    return sorted(images, key=lambda image: image["name"])


def read_points(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        for _ in range(read(f, "Q")[0]):
            values = read(f, "Q3d3BdQ")
            xyz.append(values[1:4])
            rgb.append(values[4:7])
            f.read(values[8] * 8)  # この点が写った写真の一覧（写真の番号, 特徴点の番号）は使わない
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.float32) / 255


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/bowl/sparse/0"
    cameras, images = read_cameras(f"{root}/cameras.bin"), read_images(f"{root}/images.bin")
    xyz, rgb = read_points(f"{root}/points3D.bin")
    assert np.allclose(images[0]["R"] @ images[0]["R"].T, np.eye(3))
    print(cameras)
    print(len(images), "枚", images[0]["name"], "点", len(xyz))
    centers = np.array([-image["R"].T @ image["t"] for image in images])  # カメラの位置は −Rᵀt
    print("カメラの位置の平均", centers.mean(0).round(3))
