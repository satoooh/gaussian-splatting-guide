"""第4章の描画を、PyTorchのテンソル演算で一枚ぶんまとめて書いたもの。

投影 → タイルへの割り当て → 奥行きでの並べ替え → 手前からのアルファ合成、の順に進む。
逆伝播は書かない。torchの演算だけで組んであるので、autogradが勾配を求める。
"""
import math

import torch

TILE = 16  # タイルは16×16画素（元の実装と同じ）

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def rotation(q):
    # 長さを1にそろえてから回転行列にする。q = (w, x, y, z)。
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
                        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
                        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)], -1).reshape(-1, 3, 3)


def sh_color(sh, dirs, degree):
    # 球面調和関数の係数 sh (N, 16, 3) と、カメラから粒への向き dirs (N, 3) から色を求める。
    x, y, z = (v[:, None] for v in dirs.unbind(-1))
    c = SH_C0 * sh[:, 0]
    if degree > 0:
        c = c - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        c = (c + SH_C2[0] * x * y * sh[:, 4] + SH_C2[1] * y * z * sh[:, 5]
             + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6] + SH_C2[3] * x * z * sh[:, 7]
             + SH_C2[4] * (xx - yy) * sh[:, 8])
    if degree > 2:
        c = (c + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9] + SH_C3[1] * x * y * z * sh[:, 10]
             + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11] + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
             + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + SH_C3[5] * z * (xx - yy) * sh[:, 14]
             + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return (c + 0.5).clamp(min=0)


def project(g, cam):
    """粒ごとに、画面上の中心、共分散の逆行列（conic）、3σの半径、奥行きを求める。"""
    W, t = cam["R"], cam["t"]  # 世界→カメラの回転と平行移動（第3章のRとt）
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    # X_cam = R X + t
    x, y, z = (g["means"] @ W.T + t).unbind(-1)
    valid = z > 0.2  # カメラのすぐ前と後ろの粒は描画しない（元の実装と同じ0.2）
    z = torch.where(valid, z, torch.ones_like(z))
    # Σ = R S Sᵀ Rᵀ。Sの対角がexp(log s)である。
    M = rotation(g["quats"]) * g["log_scales"].exp()[:, None, :]
    cov = M @ M.transpose(1, 2)
    # Jは中心の周りで透視投影を線形にしたもの。視野の1.3倍より外の粒は、そこで止めて求める（元の実装と同じ）。
    limit_x, limit_y = 1.3 * 0.5 * cam["width"] / fx, 1.3 * 0.5 * cam["height"] / fy
    tx = (x / z).clamp(-limit_x, limit_x) * z
    ty = (y / z).clamp(-limit_y, limit_y) * z
    zero = torch.zeros_like(z)
    J = torch.stack([fx / z, zero, -fx * tx / z**2, zero, fy / z, -fy * ty / z**2], -1).reshape(-1, 2, 3)
    A = J @ W
    # C = J W Σ Wᵀ Jᵀ + 0.3 I。0.3の単位は画素の二乗である。
    C = A @ cov @ A.transpose(1, 2) + 0.3 * torch.eye(2, device=z.device)
    a, b, c = C[:, 0, 0], C[:, 0, 1], C[:, 1, 1]
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], -1)
    # 長い主軸の分散は大きいほうの固有値 (a+c)/2 + √(((a−c)/2)² + b²)。半径はその標準偏差の3倍。
    mid = 0.5 * (a + c)
    radius = (3 * (mid + (mid * mid - det).clamp(min=0.1).sqrt()).sqrt()).ceil()
    radius = torch.where(valid & (det > 0), radius, torch.zeros_like(radius))
    center = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)  # 画素の中心は (i+0.5, j+0.5)
    return center, conic, radius.detach(), z.detach()


def assign_tiles(center, radius, depth, width, height):
    """粒をタイルに割り当て、タイルごとに粒の番号を手前から並べた表 (タイル数, 最大の重なり) を返す。"""
    tiles_x, tiles_y = math.ceil(width / TILE), math.ceil(height / TILE)
    center = center.detach()
    # 3σの円を囲む正方形が掛かるタイルの範囲（元の実装のgetRect）
    x0 = ((center[:, 0] - radius) / TILE).floor().clamp(0, tiles_x).long()
    x1 = ((center[:, 0] + radius + TILE - 1) / TILE).floor().clamp(0, tiles_x).long()
    y0 = ((center[:, 1] - radius) / TILE).floor().clamp(0, tiles_y).long()
    y1 = ((center[:, 1] + radius + TILE - 1) / TILE).floor().clamp(0, tiles_y).long()
    count = torch.where(radius > 0, (x1 - x0) * (y1 - y0), torch.zeros_like(x0))
    # 重なり一つにつき一行。ids は粒の番号、k はその粒の中で何番目のタイルか。
    ids = torch.repeat_interleave(torch.arange(len(count), device=count.device), count)
    k = torch.arange(len(ids), device=ids.device) - (count.cumsum(0) - count)[ids]
    span = (x1 - x0)[ids]
    tile = (y0[ids] + k // span) * tiles_x + x0[ids] + k % span
    # キー（タイルの番号, 奥行き）の順に並べる。奥行きで並べてから、タイルの番号で安定に並べ直す。
    order = torch.argsort(depth[ids], stable=True)
    ids, tile = ids[order], tile[order]
    order = torch.argsort(tile, stable=True)
    ids, tile = ids[order], tile[order]
    # タイルごとの範囲（並びのどこからどこまでか）を求め、タイルを行とする表に詰め直す
    per_tile = torch.bincount(tile, minlength=tiles_x * tiles_y)
    rank = torch.arange(len(ids), device=ids.device) - (per_tile.cumsum(0) - per_tile)[tile]
    table = torch.full((tiles_x * tiles_y, int(per_tile.max()) if len(ids) else 1), -1,
                       dtype=torch.long, device=ids.device)
    table[tile, rank] = ids
    return table, per_tile, count > 0


def composite(center, conic, opacity, color, table, per_tile, width, height, background, chunk):
    """タイルごとに、表の粒を手前から chunk 個ずつ重ねる。(タイル, 256画素, chunk) の形でまとめて計算する。"""
    tiles_x = math.ceil(width / TILE)
    n_tiles = table.shape[0]
    tile = torch.arange(n_tiles, device=table.device)
    local = torch.arange(TILE * TILE, device=table.device)
    px = ((tile % tiles_x)[:, None] * TILE + local % TILE + 0.5)[:, :, None]
    py = ((tile // tiles_x)[:, None] * TILE + local // TILE + 0.5)[:, :, None]
    rgb = torch.zeros(n_tiles, TILE * TILE, 3, device=table.device)
    T = torch.ones(n_tiles, TILE * TILE, 1, device=table.device)
    done = torch.zeros(n_tiles, TILE * TILE, 1, dtype=torch.bool, device=table.device)
    for start in range(0, table.shape[1], chunk):
        # 粒が残っていて、打ち切っていない画素を含むタイルだけを計算する
        active = ((per_tile > start) & ~done.flatten(1).all(1)).nonzero()[:, 0]
        if len(active) == 0:
            break
        idx = table[active, start:start + chunk]
        used, idx = idx >= 0, idx.clamp(min=0)
        dx = px[active] - center[idx, 0][:, None, :]
        dy = py[active] - center[idx, 1][:, None, :]
        a, b, c = (v[:, None, :] for v in conic[idx].unbind(-1))
        # α = o · exp(−½ dᵀ C⁻¹ d)。0.99で頭打ちにし、1/255未満は飛ばす。
        power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
        alpha = (opacity[idx][:, None, :] * power.exp()).clamp(max=0.99)
        alpha = alpha * (used[:, None, :] & (power <= 0) & (alpha >= 1 / 255) & ~done[active])
        # 透過率 T が0.0001を下回る粒から先は重ねない（その画素はそこで打ち切る）
        test = T[active].detach() * torch.cumprod(1 - alpha.detach(), -1) < 1e-4
        alpha = alpha * ~test
        # C = Σ c_i α_i T_i,  T_i = Π_{j<i} (1 − α_j)
        inside = torch.cumprod(1 - alpha, -1)
        weight = alpha * T[active] * torch.cat([torch.ones_like(inside[..., :1]), inside[..., :-1]], -1)
        rgb = rgb.index_add(0, active, weight @ color[idx])
        T = T.index_copy(0, active, T[active] * inside[..., -1:])
        done = done.index_copy(0, active, done[active] | test.any(-1, keepdim=True))
    rgb = rgb + T * background
    tiles_y = n_tiles // tiles_x
    image = rgb.reshape(tiles_y, tiles_x, TILE, TILE, 3).permute(0, 2, 1, 3, 4).reshape(tiles_y * TILE, tiles_x * TILE, 3)
    return image[:height, :width]


def render(g, cam, sh_degree, chunk=256, background=None):
    """粒の値 g（dict）とカメラ cam（dict）から画像 (H, W, 3) を描画する。"""
    width, height = cam["width"], cam["height"]
    background = torch.zeros(3, device=g["means"].device) if background is None else background
    center, conic, radius, depth = project(g, cam)
    table, per_tile, visible = assign_tiles(center, radius, depth, width, height)
    campos = -cam["R"].T @ cam["t"]  # カメラの位置は −Rᵀt
    dirs = g["means"] - campos
    sh = torch.cat([g["sh0"], g["shN"]], 1)
    color = sh_color(sh, dirs / dirs.norm(dim=-1, keepdim=True), sh_degree)
    opacity = torch.sigmoid(g["opacity"])
    image = composite(center, conic, opacity, color, table, per_tile, width, height, background, chunk)
    return image, dict(center=center, visible=visible, per_tile=per_tile, radius=radius)
