"""COLMAPの結果と写真から、自分で書いた描画（render_real.py）で3DGSを学習する。

  python train_real.py --data data/bowl --out out/run --long-side 300

逆伝播はautogradに任せる。既定値は第5章の表（元の実装の値）に合わせてある。
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from colmap_read import read_cameras, read_images, read_points
from render_real import SH_C0, render, rotation

ITERATIONS = 30_000
LAMBDA_DSSIM = 0.2           # 損失 = 0.8 L1 + 0.2 (1 − SSIM)
DENSIFY_FROM, DENSIFY_UNTIL, DENSIFY_EVERY = 500, 15_000, 100
GRAD_THRESHOLD = 0.0002      # 画面上の位置の勾配の平均（元の実装と同じ、正規化した画面座標での値）
PERCENT_DENSE = 0.01         # 粒の大きさがシーンの広がりの1%以下なら複製、超えれば分割
MIN_OPACITY = 0.005
RESET_EVERY = 3_000
SH_EVERY = 1_000             # SHの次数を1,000反復ごとに一つ上げる
POSITION_LR_STEPS = 30_000   # 位置の学習率を、この反復数をかけて100分の1まで下げる


def load_views(data, long_side, device):
    cameras = read_cameras(f"{data}/sparse/0/cameras.bin")
    views = []
    for image in read_images(f"{data}/sparse/0/images.bin"):
        cam = cameras[image["camera_id"]]
        photo = Image.open(f"{data}/images/{image['name']}").convert("RGB")
        # EXIFの回転情報で縦横が入れ替わっていないことを確かめる（COLMAPの幅と高さに合わせる）
        assert photo.size == (cam["width"], cam["height"]), (photo.size, cam)
        s = long_side / max(cam["width"], cam["height"])
        width, height = round(cam["width"] * s), round(cam["height"] * s)
        photo = np.asarray(photo.resize((width, height), Image.LANCZOS), dtype=np.float32) / 255
        sx, sy = width / cam["width"], height / cam["height"]
        views.append(dict(name=image["name"], photo=torch.tensor(photo, device=device),
                          R=torch.tensor(image["R"], dtype=torch.float32, device=device),
                          t=torch.tensor(image["t"], dtype=torch.float32, device=device),
                          fx=cam["fx"] * sx, fy=cam["fy"] * sy, cx=cam["cx"] * sx, cy=cam["cy"] * sy,
                          width=width, height=height))
    return [v for i, v in enumerate(views) if i % 8], views[::8]  # 名前の順で8枚おきを評価に残す


def init_gaussians(data, sh_degree, device):
    # SfMの点一つに粒を一つ置く。大きさは近い3点までの距離の二乗の平均から、向きはなし、不透明度は0.1。
    xyz, rgb = (torch.tensor(a, device=device) for a in read_points(f"{data}/sparse/0/points3D.bin"))
    d2 = torch.cat([torch.cdist(xyz[i:i + 4096], xyz).pow(2).topk(4, largest=False).values[:, 1:].mean(1)
                    for i in range(0, len(xyz), 4096)])
    n = len(xyz)
    g = dict(means=xyz,
             log_scales=d2.clamp(min=1e-7).sqrt().log()[:, None].repeat(1, 3),
             quats=torch.tensor([1.0, 0, 0, 0], device=device).repeat(n, 1),
             opacity=torch.full((n,), math.log(0.1 / 0.9), device=device),  # シグモイドを通すと0.1
             sh0=((rgb - 0.5) / SH_C0)[:, None, :],
             shN=torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3, device=device))
    return {name: value.requires_grad_() for name, value in g.items()}


def ssim(a, b):
    # 11×11、σ=1.5のガウス窓で平均と分散を求める（元の実装と同じ）。a, b は (H, W, 3)。
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    k = torch.exp(-(torch.arange(11, device=a.device) - 5.0) ** 2 / (2 * 1.5 ** 2))
    k = k / k.sum()
    window = (k[:, None] * k[None, :]).expand(3, 1, 11, 11)
    blur = lambda z: F.conv2d(z, window, padding=5, groups=3)
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mx * my + c1) * (2 * sxy + c2) / ((mx * mx + my * my + c1) * (sxx + syy + c2))).mean()


def psnr(a, b):
    return -10 * math.log10(F.mse_loss(a.clamp(0, 1), b).item())


def rebuild(opt, g, keep, extra):
    """粒を選び直す（keep）か足す（extra）。Adamの移動平均は、残る粒は引き継ぎ、足した粒は0から始める。"""
    for group in opt.param_groups:
        name, old = group["name"], group["params"][0]
        add = extra[name] if extra else old[:0].detach()
        new = torch.cat([old.detach()[keep], add]).requires_grad_()
        state = opt.state.pop(old, None)
        if state:
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = torch.cat([state[key][keep], torch.zeros_like(add)])
            opt.state[new] = state
        group["params"][0] = g[name] = new


def densify_and_prune(opt, g, grads, extent, prune_big):
    """密度制御：位置の勾配の平均が大きい粒を、小さければ複製し、大きければ二つに分割する。薄い粒は削る。"""
    scale = g["log_scales"].detach().exp()
    selected = grads >= GRAD_THRESHOLD
    small = scale.max(1).values <= PERCENT_DENSE * extent
    clone, split = selected & small, selected & ~small
    # 複製：位置、形、不透明度、色をそのまま写し、同じ場所に置く
    extra = {name: value.detach()[clone] for name, value in g.items()}
    # 分割：元の粒のガウス分布から二つの位置を選び、スケールを1.6で割る。元の粒は消す。
    children = {name: value.detach()[split].repeat(2, *[1] * (value.dim() - 1)) for name, value in g.items()}
    s = scale[split].repeat(2, 1)
    offset = (rotation(children["quats"]) @ (torch.randn_like(s) * s)[..., None])[..., 0]
    children["means"] = children["means"] + offset
    children["log_scales"] = (s / 1.6).log()
    extra = {name: torch.cat([extra[name], children[name]]) for name in g}
    rebuild(opt, g, ~split, extra)
    # 削除：不透明度が0.005未満の粒。最初の不透明度のリセットの後は、シーンの広がりの10%より大きい粒も。
    prune = torch.sigmoid(g["opacity"].detach()) < MIN_OPACITY
    if prune_big:
        # 元の実装は画面上の半径が20画素を超える粒も削る。小さい画像では基準が変わるので、ここでは省いた。
        prune |= g["log_scales"].detach().exp().max(1).values > 0.1 * extent
    rebuild(opt, g, ~prune, None)
    return int(clone.sum()), int(split.sum()), int(prune.sum())


def reset_opacity(opt, g):
    # 不透明度を0.01以下へ下げ、その値のAdamの移動平均を0に戻す
    with torch.no_grad():
        g["opacity"].clamp_(max=math.log(0.01 / 0.99))
    for value in opt.state.get(g["opacity"], {}).values():
        if torch.is_tensor(value) and value.dim():
            value.zero_()


def evaluate(g, views, sh_degree, chunk):
    with torch.no_grad():
        renders = [render(g, v, sh_degree, chunk)[0].clamp(0, 1) for v in views]
        return (float(np.mean([psnr(r, v["photo"]) for r, v in zip(renders, views)])),
                float(np.mean([ssim(r, v["photo"]).item() for r, v in zip(renders, views)])), renders)


def save_png(image, path):
    Image.fromarray((image.detach().clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(path)


def tile_counts_png(per_tile, view, path):
    # タイルごとに重なった粒の数を濃淡にする（最大の数を白）。1タイルを16×16画素で塗る。
    tiles_x = math.ceil(view["width"] / 16)
    counts = per_tile.float().reshape(-1, tiles_x)
    gray = (counts / counts.max()).repeat_interleave(16, 0).repeat_interleave(16, 1)[:view["height"], :view["width"]]
    save_png(gray[..., None].repeat(1, 1, 3), path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/bowl")
    p.add_argument("--out", default="out/run")
    p.add_argument("--long-side", type=int, default=300)
    p.add_argument("--iters", type=int, default=ITERATIONS)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--chunk", type=int, default=256, help="タイルごとに一度に重ねる粒の数（元の実装は256）")
    p.add_argument("--no-densify", action="store_true")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--snapshots", default="0,500,2000,7000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)
    os.makedirs(args.out, exist_ok=True)

    train, test = load_views(args.data, args.long_side, args.device)
    g = init_gaussians(args.data, args.sh_degree, args.device)
    centers = torch.stack([-v["R"].T @ v["t"] for v in train + test])
    # シーンの広がり：カメラの位置の平均から最も遠いカメラまでの距離の1.1倍（元の実装と同じ）
    extent = 1.1 * (centers - centers.mean(0)).norm(dim=1).max().item()
    rates = dict(means=0.00016 * extent, log_scales=0.005, quats=0.001, opacity=0.025, sh0=0.0025, shN=0.0025 / 20)
    opt = torch.optim.Adam([dict(params=[g[n]], lr=rates[n], name=n) for n in g], eps=1e-15)
    snapshots = {int(s) for s in args.snapshots.split(",")} | {args.iters}
    shown = test[0]  # 図に使う保持視点
    print(f"条件: 学習{len(train)}枚 / 評価{len(test)}枚 / {shown['width']}×{shown['height']}画素 / "
          f"初期の粒{len(g['means'])}個 / 広がり{extent:.3f} / SH次数{args.sh_degree} / "
          f"密度制御{'なし' if args.no_densify else 'あり'} / seed={args.seed} / {args.device}")
    save_png(shown["photo"], f"{args.out}/photo-{shown['name'][:-4]}.png")

    grad_sum = torch.zeros(len(g["means"]), device=args.device)
    seen = torch.zeros_like(grad_sum)
    history, order, train_seconds = [], [], 0.0
    for step in range(args.iters + 1):
        if step % args.eval_every == 0 or step in snapshots:
            value_psnr, value_ssim, renders = evaluate(g, test, min(step // SH_EVERY, args.sh_degree), args.chunk)
            history.append(dict(step=step, gaussians=len(g["means"]), psnr=value_psnr, ssim=value_ssim,
                                train_seconds=round(train_seconds, 1)))
            print(f"反復{step:6d}  粒 {len(g['means']):7d}個  保持視点 PSNR {value_psnr:.2f} dB  SSIM {value_ssim:.4f}"
                  f"  学習 {train_seconds:7.1f} s", flush=True)
            if step in snapshots:
                save_png(renders[0], f"{args.out}/step-{step:05d}.png")
        if step == args.iters:
            break
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        # 位置の学習率を、対数の上で一定の割合ずつ下げる
        for group in opt.param_groups:
            if group["name"] == "means":
                group["lr"] = rates["means"] * 0.01 ** min(step / POSITION_LR_STEPS, 1)
        if not order:
            order = random.sample(train, len(train))
        view = order.pop()
        degree = min(step // SH_EVERY, args.sh_degree)
        image, info = render(g, view, degree, args.chunk)
        info["center"].retain_grad()
        loss = (1 - LAMBDA_DSSIM) * (image - view["photo"]).abs().mean() + LAMBDA_DSSIM * (1 - ssim(image, view["photo"]))
        loss.backward()
        done = step + 1
        if not args.no_densify and done < DENSIFY_UNTIL:
            with torch.no_grad():
                # 画面上の位置の勾配の大きさを足し込む。画素単位の勾配を、−1〜1に正規化した画面座標での値に直す。
                grad = info["center"].grad * torch.tensor([view["width"] / 2, view["height"] / 2], device=args.device)
                visible = info["visible"]
                grad_sum[visible] += grad[visible].norm(dim=-1)
                seen[visible] += 1
            if done > DENSIFY_FROM and done % DENSIFY_EVERY == 0:
                counts = densify_and_prune(opt, g, grad_sum / seen.clamp(min=1), extent, done > RESET_EVERY)
                grad_sum = torch.zeros(len(g["means"]), device=args.device)
                seen = torch.zeros_like(grad_sum)
                if done % 1000 == 0:
                    print(f"  密度制御 反復{done}: 複製{counts[0]} 分割{counts[1]} 削除{counts[2]}", flush=True)
            if done % RESET_EVERY == 0:
                reset_opacity(opt, g)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if args.device == "cuda":
            torch.cuda.synchronize()
        train_seconds += time.time() - start

    value_psnr, value_ssim, renders = evaluate(g, test, args.sh_degree, args.chunk)
    for r, v in zip(renders, test):
        save_png(r, f"{args.out}/final-{v['name'][:-4]}.png")
        save_png(v["photo"], f"{args.out}/photo-{v['name'][:-4]}.png")
    with torch.no_grad():
        info = render(g, shown, args.sh_degree, args.chunk)[1]
    tile_counts_png(info["per_tile"], shown, f"{args.out}/tile-counts.png")
    torch.save({n: v.detach().cpu() for n, v in g.items()}, f"{args.out}/gaussians.pt")
    result = dict(width=shown["width"], height=shown["height"], iterations=args.iters, gaussians=len(g["means"]),
                  psnr=value_psnr, ssim=value_ssim, train_seconds=round(train_seconds, 1),
                  sh_degree=args.sh_degree, densify=not args.no_densify, seed=args.seed,
                  tiles=len(info["per_tile"]),
                  max_per_tile=int(info["per_tile"].max()), shown_view=shown["name"],
                  peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 2**20) if args.device == "cuda" else None,
                  history=history)
    json.dump(result, open(f"{args.out}/result.json", "w"), ensure_ascii=False, indent=1)
    print(f"最終: 保持視点{len(test)}枚 PSNR {value_psnr:.2f} dB / SSIM {value_ssim:.4f} / 粒 {len(g['means'])}個 / "
          f"学習 {train_seconds:.0f} s")


if __name__ == "__main__":
    main()
