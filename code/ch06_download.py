# 料理の写真51枚とカメラ姿勢を取得する。標準ライブラリだけで動く。
# 使い方：python3 ch06_download.py [保存先のフォルダ]
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REVISION = 'bb2ec30fd17ffb65e24ab40e713517d0c65de41b'  # 連載で使った版に固定する
HOST = 'https://huggingface.co'
DATASET = 'datasets/promc/reconstruction-scenes'
SCENE = 'chili-salmon-bowl'
COLMAP = ['cameras.bin', 'images.bin', 'points3D.bin']
out = Path(sys.argv[1] if len(sys.argv) > 1 else SCENE)


def listing(folder):
    with urllib.request.urlopen(f'{HOST}/api/{DATASET}/tree/{REVISION}/{folder}') as response:
        return {item['path'].split('/')[-1]: item for item in json.load(response) if item['type'] == 'file'}


photos, top = listing(f'{SCENE}/images'), listing(SCENE)
assert len(photos) == 51, len(photos)
# 配布元では三つの .bin が料理のフォルダの直下にある。学習器が読めるように sparse/0/ へ置く
files = [(f'images/{name}', f'images/{name}', item) for name, item in sorted(photos.items())]
files += [(name, f'sparse/0/{name}', top[name]) for name in COLMAP]

total = 0
for remote, local, item in files:
    path = out / local
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + '.part')
        urllib.request.urlretrieve(f'{HOST}/{DATASET}/resolve/{REVISION}/{SCENE}/{remote}', part)
        part.replace(path)
    contents = path.read_bytes()
    assert len(contents) == item['size'], f'{path} の大きさが違う。消してからやり直す'
    if 'lfs' in item:  # 大きなファイルには、配布元の一覧にSHA-256が載っている
        assert hashlib.sha256(contents).hexdigest() == item['lfs']['oid'], f'{path} の中身が違う'
    total += len(contents)

assert total == 139_159_579, total
print(f'写真{len(photos)}枚とカメラ情報{len(COLMAP)}つを {out}/ に置いた。合計 {total / 1e6:.1f} MB')
