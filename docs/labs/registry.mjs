// 実験の一覧。章のMarkdownからは [表示名](/lab/ID/) の一行で埋め込める。
export const labs = [
  { id: 'real3d', title: '実写のスプラットを回す', prompt: '写真から作った約11万個の粒。計算は惑星のときと変わらない。', chapter: '01-overview' },
  { id: 'training', title: '学習を見る', prompt: '粒が増えるたびに、どこの細部が現れるかを追う。', chapter: '05-learning' },
  { id: 'covariance', title: '形を作る', prompt: '回しても変わらない値と、変わる値を見分ける。', chapter: '02-gaussians' },
  { id: 'sh', title: '方向で変わる色', prompt: '次数を下げると、ハイライトはどうなるか。', chapter: '02-gaussians' },
  { id: 'projection', title: '画面へ投影する', prompt: '近づけたとき、近似と実際の輪郭はどこからずれるか。', chapter: '04-rendering' },
  { id: 'compositing', title: '二つのスプラットを重ねる', prompt: '前後を入れ替えても変わらない値はどれか。', chapter: '04-rendering' },
  { id: 'splat3d', title: '小さな3DGSを回す', prompt: '深度ソートを切ると、どの向きで破綻するか。', chapter: '04-rendering' },
  { id: 'interp', title: '撮影フレームの間を描画する', prompt: '粒を混ぜるか、動かすか。フレームを増やすと、ずれはどう減るか。', chapter: '09-dynamic' },
  { id: 'spacetime', title: '時空間のガウシアンを切る', prompt: '傾きが速度に、時間方向の幅が寿命になる。', chapter: '09-dynamic' },
];
