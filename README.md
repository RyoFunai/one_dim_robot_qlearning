# One Dim Robot Q-Learning

**One Dim Robot Q-Learning** は、1次元環境でロボットがゴールに向かう最適な行動をQ学習で学び、シミュレーションを実行するプログラムです。

## 特徴
- **環境**: 1次元の整数座標上で、ロボットが左右に移動
- **状態**: (ロボット位置, ゴール位置) のペア
- **Q学習**: ε-greedy 方策を用いて最適行動を学習
- **可視化**: Matplotlib のアニメーションで動作を表示

## 実行環境
- Ubuntu 22.04 LTS

## 必要なライブラリ
- Python 3.10.12
- NumPy 1.24.3
- Matplotlib 3.9.1

## 実行
```
python3 one_dim_robot_qlearning.py
```
学習が終わると、以下のようなアニメーションが出力されます。
ランダムに生成されるゴールをロボットが追いかけます。
[![](https://img.youtube.com/vi/eABil7OW-uQ/0.jpg)](https://www.youtube.com/watch?v=eABil7OW-uQ)
