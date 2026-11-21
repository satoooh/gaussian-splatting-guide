# 3DGSのPLYを読み、値を変えない保存の仕方でどこまで小さくなるかを測る。標準ライブラリだけで動く。
# 使い方：python3 sizes.py bowl-30000.ply
import sys
import zlib
from array import array

raw = open(sys.argv[1], 'rb').read()
end = raw.index(b'end_header\n') + len(b'end_header\n')
header = raw[:end].decode().split('\n')
count = next(int(line.split()[2]) for line in header if line.startswith('element vertex'))
names = [line.split()[2] for line in header if line.startswith('property')]
assert all(line.split()[1] == 'float' for line in header if line.startswith('property')), '32ビット浮動小数点数だけを想定'
body = raw[end:]
width = 4 * len(names)  # 一粒のバイト数
assert len(body) == count * width, (len(body), count, width)
assert sys.byteorder == 'little' and 'format binary_little_endian 1.0' in header
values = array('f', body)  # 粒0の全属性、粒1の全属性、…の順に並んでいる


def packed(data):
    return len(zlib.compress(data, 9))


def columns(order):  # 属性ごとにまとめる。xだけ、yだけ、…の順
    out = array('f')
    for k in range(len(names)):
        out.extend(values[i * len(names) + k] for i in order)
    return out.tobytes()


def planes(data):  # さらに、4バイトの何バイト目かでまとめる。値は変えない
    return b''.join(data[k::4] for k in range(4))


def morton(i):  # 位置を各軸10ビットに丸め、ビットを交互に並べた番号。空間で近い粒は番号も近い
    code = 0
    for axis in range(3):
        column = names.index('xyz'[axis])
        cell = int((values[i * len(names) + column] - low[axis]) / (high[axis] - low[axis]) * 1023)
        for bit in range(10):
            code |= (cell >> bit & 1) << (3 * bit + axis)
    return code


xyz = [[values[i * len(names) + names.index(axis)] for i in range(count)] for axis in 'xyz']
low, high = [min(column) for column in xyz], [max(column) for column in xyz]
original = list(range(count))
nearby = sorted(original, key=morton)

rows = [
    ('そのまま', len(body)),
    ('そのまま圧縮する', packed(body)),
    ('属性ごとにまとめて圧縮する', packed(columns(original))),
    ('バイトの位ごとにまとめて圧縮する', packed(planes(columns(original)))),
    ('近い粒を隣に並べてから同じことをする', packed(planes(columns(nearby)))),
]
print(f'{count}粒、一粒{width}バイト（{len(names)}個の値）')
for label, size in rows:
    print(f'{size / 1e6:6.2f} MB  {size / count:6.1f} バイト/粒  {size / len(body):4.0%}  {label}')

# 並べ替えても、まとめ直しても、値は一つも変えていない
assert sorted(array('f', columns(nearby))) == sorted(values)
