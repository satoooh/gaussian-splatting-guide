# Gaussian Splattingを学ぶ

note連載「Gaussian Splattingを学ぶ」のサンプルコードと、ブラウザで動かせる実験。

- 動かせる実験：https://satoooh.github.io/gaussian-splatting-guide/
- サンプルコード：[code/](code/)

## 動かせる実験

- 第1章　[実写のスプラットを回す](https://satoooh.github.io/gaussian-splatting-guide/lab/real3d/)　写真から作った約11万個の粒。計算は惑星のときと変わらない。
- 第2章　[形を作る](https://satoooh.github.io/gaussian-splatting-guide/lab/covariance/)　回しても変わらない値と、変わる値を見分ける。
- 第2章　[方向で変わる色](https://satoooh.github.io/gaussian-splatting-guide/lab/sh/)　次数を下げると、ハイライトはどうなるか。
- 第4章　[画面へ投影する](https://satoooh.github.io/gaussian-splatting-guide/lab/projection/)　近づけたとき、近似と実際の輪郭はどこからずれるか。
- 第4章　[二つのスプラットを重ねる](https://satoooh.github.io/gaussian-splatting-guide/lab/compositing/)　前後を入れ替えても変わらない値はどれか。
- 第4章　[小さな3DGSを回す](https://satoooh.github.io/gaussian-splatting-guide/lab/splat3d/)　深度ソートを切ると、どの向きで破綻するか。
- 第5章　[学習を見る](https://satoooh.github.io/gaussian-splatting-guide/lab/training/)　粒が増えるたびに、どこの細部が現れるかを追う。
- 第9章　[撮影フレームの間を描画する](https://satoooh.github.io/gaussian-splatting-guide/lab/interp/)　粒を混ぜるか、動かすか。フレームを増やすと、ずれはどう減るか。
- 第9章　[4Dガウシアンを時刻で切る](https://satoooh.github.io/gaussian-splatting-guide/lab/spacetime/)　傾きが速度に、時間方向の幅が寿命になる。

実験は外部のライブラリを使わない素のJavaScriptで、`docs/` をそのまま開けば動く。手元で動かすときは `docs/` を置いたフォルダで `python3 -m http.server` を実行し、表示されたアドレスを開く。

## サンプルコード

- [code/ch04_tiny.py](code/ch04_tiny.py)　第4章「3DGSの描画の仕組み: 投影、画面上の共分散、アルファ合成、タイル」
- [code/ch05_grad.py](code/ch05_grad.py)　第5章「3DGSの学習の仕組み: 損失、勾配、Adam、密度制御」
- [code/ch06_download.py](code/ch06_download.py)　第6章「3DGSの作り方: 写真の準備、Brushでの学習、SuperSplatでの粒の削除、評価」
- [code/ch08_playcanvas.html](code/ch08_playcanvas.html)　第8章「3DGSのファイル形式と配布: PLY、圧縮、SOG、SPZ、ビューア」
- [code/ch08_sizes.py](code/ch08_sizes.py)　第8章「3DGSのファイル形式と配布: PLY、圧縮、SOG、SPZ、ビューア」
- [code/ch08_spark.html](code/ch08_spark.html)　第8章「3DGSのファイル形式と配布: PLY、圧縮、SOG、SPZ、ビューア」
- [code/ch04_tiny3d/](code/ch04_tiny3d/)　第4章「3DGSの描画の仕組み: 投影、画面上の共分散、アルファ合成、タイル」
- [code/ch05_real/](code/ch05_real/)　第5章「3DGSの学習の仕組み: 損失、勾配、Adam、密度制御」
- [code/ch05_train/](code/ch05_train/)　第5章「3DGSの学習の仕組み: 損失、勾配、Adam、密度制御」
- [code/ch05_train3d/](code/ch05_train3d/)　第5章「3DGSの学習の仕組み: 損失、勾配、Adam、密度制御」
- [code/ch09_tiny4d/](code/ch09_tiny4d/)　第9章「4D Gaussian Splatting（4DGS）の基礎: 時間を表す四つの方式、学習の実験、試し方」

Pythonのコードは標準ライブラリだけで動く。ただし `code/ch05_real/` は、PyTorch、NumPy、Pillowを使う。HTMLの例は、`python3 -m http.server` で配って開く。公開している版は https://satoooh.github.io/gaussian-splatting-guide/examples/ の下にある。

## ライセンス

コードと実験は [MIT License](LICENSE)。実験に同梱した実写のシーンは次のデータから作った。

- [Houseplant](https://marcelpadilla.com/Gaussian_Splat_Object_Dataset/)（Marcel Padilla、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)）
- [chili-salmon-bowl](https://huggingface.co/datasets/promc/reconstruction-scenes)（Charin Rungchaowarat、[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)）
