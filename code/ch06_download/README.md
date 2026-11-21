# 第6章 download.py：料理の写真51枚とカメラ姿勢を取得する

第6章で学習に使う[Reconstruction Scenes](https://huggingface.co/datasets/promc/reconstruction-scenes)のchili-salmon-bowl（CC0）を取得するスクリプトです。Pythonの標準ライブラリだけで動きます。後半には、本文から外したBrushとCOLMAPのコマンドをまとめました。

## 取得する

```sh
python3 download.py
```

保存先のフォルダを引数で渡せます。省略すると `chili-salmon-bowl/` に置きます。成功すると、最後に次の一行が表示されます。

```text
写真51枚とカメラ情報3つを chili-salmon-bowl/ に置いた。合計 139.2 MB
```

- `REVISION` で、連載で使った版に固定して取得します。
- 写真51枚とCOLMAPの三つの.binファイルの大きさを、配布元の一覧と照らし合わせます。大きなファイルはSHA-256も照らし合わせます。
- 配布元では三つの.binファイルが料理のフォルダの直下にあります。学習器が読めるように `sparse/0/` へ置きます。
- 途中で止まったときは、もう一度実行すれば取得済みのファイルを飛ばして残りを取得します。
- 「大きさが違う」「中身が違う」と出たときは、表示されたファイルを消してから実行し直します。

配布元のフォルダには `rigs.bin` と `frames.bin` もあります。どちらも新しいCOLMAPが書き出すファイルですが、Brush 0.3.0の学習には三つの.binファイルがあれば足ります。このスクリプトも取得しません。自分でCOLMAPを実行したときは、出力フォルダのファイルを足したり削ったりせず、そのまま学習器に渡します。

## 料理の写真の歪みを補正しなくてよい理由

`cameras.bin` のカメラモデルはSIMPLE_RADIALで、歪みを係数一つで表しています。焦点距離を $f$、係数を $k$、画像の中心からの距離を焦点距離で割った値を $r$ とすると、歪みによる投影のずれは $f|k|r^3$ 画素です。このデータでは $f$ が約4918画素、$k$ が約−0.001、画像の隅での $r$ が約0.52です。ずれは隅でも元の解像度で約0.7画素、Brushが読み込む長辺1920画素では約0.3画素にとどまります。

## 学習中の評価値をログに残す

`--eval-every 7000` を付けると、7000反復ごとに評価用の7枚で評価値を測ります。反復30000は7000の倍数ではありませんが、最後の反復でも評価されます。評価値は、端末の進行表示の下に「Eval iter 7000: PSNR 24.4…, ssim 0.85…」と表示されます。出力をファイルへ書き出すと進行表示が省かれるので、コマンドの先頭に `RUST_LOG=brush_cli=info` を付けます。

## 保存したPLYの評価だけやり直す

本文の手順2の表にある反復ごとのPSNRとSSIMは、次の方法で測りました。学習をやり直さずに、保存したPLYを粒の初期値として読み込ませ、一反復だけ進めます。

1. 測りたいPLYをコピーし、`init.ply` という名前で学習データのフォルダ（`chili-salmon-bowl/`）に置きます。Brushは、学習データのフォルダの下にPLYが一つだけあると、それを粒の初期値にします。
2. 反復7000のPLYなら、次のコマンドで評価用の7枚を測ります。

```bash
RUST_LOG=brush_cli=info ./brush_app chili-salmon-bowl \
  --start-iter 6999 \
  --total-steps 7000 \
  --max-splats 200000 \
  --eval-split-every 8 \
  --eval-every 7000 \
  --mean-noise-weight 0 \
  --lr-mean 1e-30 --lr-mean-end 1e-30 \
  --lr-coeffs-dc 1e-30 --lr-opac 1e-30 \
  --lr-scale 1e-30 --lr-scale-end 1e-30 \
  --lr-rotation 1e-30 \
  --export-path eval-7000 \
  --export-name frozen.ply
```

3. 測り終えたら `init.ply` を削除します。残したままだと、次の学習もこのPLYから始まります。

学習率を極めて小さくし、粒に加えるノイズも止めているので、この一反復で粒の位置、形、色、不透明度は変わりません。連載の検証でも、変わらないことを確かめました。

## COLMAPの画面から進める

本文の三つのコマンドと同じ工程は、COLMAPの画面からも進められます。

| 工程 | メニュー |
| --- | --- |
| プロジェクトを作る | File > New project |
| 特徴点を取り出す | Processing > Extract features |
| 写真の間で対応を探す | Processing > Feature matching |
| カメラ姿勢と点群を求める | Reconstruction > Start |
| 結果を保存する | File > Export model |

## 歪みを補正したときの出力

- 料理の写真では、PINHOLEの指定を外して既定のSIMPLE_RADIALで試しても、51枚すべてが登録され、学習の結果も落ちませんでした。PINHOLEとSIMPLE_RADIALは別のフォルダに分けて実行しました。
- `colmap image_undistorter` の出力先には、補正した画像が入った `images` と、カメラ姿勢が入った `sparse` が作られます。カメラモデルはPINHOLEに書き換えられ、画像の大きさも変わります。料理の写真をSIMPLE_RADIALで処理した結果に実行すると、3060×4080画素の写真が3055×4074画素になりました。
- この `sparse` には、番号のフォルダを挟まずに三つの.binファイルが直接入ります。Brush 0.3.0はこの構成も読み込めます。`sparse/0` を必要とする学習器へ渡すときは、`sparse` の中に `0` というフォルダを作って三つのファイルを移します。
