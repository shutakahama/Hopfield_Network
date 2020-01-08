# Hopfield_Network
ここでは全結合型ニューラルネットワークであるホップフィールドネットワークを実装した．  

## ホップフィールドネットワークとは
ホップフィールドネットワークとは、全結合型のニューラルネットワークである．  
深層学習で主流になっているフィードフォーワードネットワークは，ニューロンのまとまりである層を積み重ねた構造をしており，各層から次の層への一方向の結合しか存在しない．  
それに対しホップフィールドネットワークは任意の2つのニューロン全てについて両方向の結合が存在している．  

<img src="https://github.com/shutakahama/Hopfield_Network/blob/master/fig/hopfield.png" width="300">

ホップフィールドネットワークの特徴として，特定のパターンを記憶できる事が挙げられる．
いくつかのパターンを入力として学習させておくと，ノイズを含んだパターンを入力した際に元のパターンを想起する事ができる．  
具体的には、記憶するパターンを𝑥^1, x^2, ... , x^QのQ個とする．xはニューロン数の次元(Nとする)を持っており，各ニューロンにその値を記憶させる．  
これらに対して重み行列Wを

<img src="https://github.com/shutakahama/Hopfield_Network/blob/master/fig/formula1.png" width="300">

と定める。Wはこの場合N*N次元の行列となる。この計算でネットワークの学習は終わりである．  

その後ノイズの入った入力から元のパターンを想起する．  
記憶パターンQ個のうち一つにある割合でノイズを加えたものをテスト入力xとする．時刻tでの更新は

<img src="https://github.com/shutakahama/Hopfield_Network/blob/master/fig/formula2.png" width="300">

で行う．  
sgn関数の性質によりxの要素は1, 0, -1のいずれかの値をとる。また定義上w_ij = w_ji となる.さらに自己結合を考慮しない場合はw = 0とする．   
この更新を十分行って平衡状態に達したら終了である．更新を繰り返すことで，ノイズを含んだ入力であっても記憶したパターンに収束していくことが知られている．

更新の方法には同期型、非同期型の2つがある。同期型は全てのニューロンを同時に更新し、非同期型は一つずつニューロンを更新していく．
ホップフィールドネットワークは基本的に非同期型である．  

平衡状態の判定はニューロンの値に変化がなくなった時としても良いが，より厳密にはネットワークエネルギーを使う。エネルギーは

<img src="https://github.com/shutakahama/Hopfield_Network/blob/master/fig/formula3.png" width="300">

で表せる．このエネルギーはリアプノフ性があり、xを更新すると単調減少ため，Eが変化しなくなったところが平衡点と言える．  


## 実装
ここでは5*5画素の画像を最大6枚記憶させる．よってトレーニングデータとして6*25次元のベクトルx_trainを用意する．各要素の値は1か-1のいづれかである．  
記憶パターンは以下の6つであり，左から記憶パターン 1, 2, ..., 6 と呼ぶことにする．  

<img src="https://github.com/shutakahama/Hopfield_Network/blob/master/fig/train.png" width="600">

runを実行するとまずfit()関数によって学習が行われる．fit()は25*25次元の重み行列Wを返す．  
テストデータはtest_make()で生成する．トレーニングデータ行列の中からパターンを一つランダムに取り出し、それにノイズを加えたものをテストデータとする．  
ノイズは指定した確率で各ニューロンの値に−１をかけることである．  
記憶パターン1についてrate = 0.1でノイズを与えた場合の結果は例えば以下のようになる.

<img src="https://github.com/shutakahama/Hopfield_Network/blob/master/fig/test_sample.png" width="300">

パターンの想起は predict_syn(), predict_asyn()によって行う.  
前者は同期更新で全てのニューロンを同時に更新する．収束の判定はエネルギー関数を使った．
後者は非同期更新で，ランダムにニューロンを１つ選びその値を更新することを繰り返す．1つずつニューロンを更新するとエネルギー変化で収束を判断するのが難しいため、ここでは単純に更新を十分な数繰り返した．  
想起性能はsimilarity()で評価する。トレーニングデータのうち想起結果と最も差が小さかったパターンをsimidx，その差をsimとして出力する。差は値の違うマス目の数である．  
この試行を複数回行ってその平均を取る．また差が0だった個数を正答率としてカウントする．  
結果はplotで描画可能．  
  
## 実行
### データセット
訓練データはコード内にに２５次元のデータが6個分定義してある．データは自由に書き換えることができるが，描画の関係上要素の数は平方数(25や36など)であることが望ましい．

### 実行環境
```
python = 3.7
numpy = 1.17
matplotlib = 3.1
```

### 実行
```
python3 hopfield_network.py
```
類似度sim, 正答率accが表示される．必要に応じて描画関数を入れる．  

### オプションの設定

| 変数 | 説明 |
----|---- 
| train_num | 記憶パターンの数．サンプルデータを使った際は最大で6． |
| loop_update | 想起の最大回数．テスト時にこの回数だけ値を更新する． |
| loop_test | テストの試行回数． |
| noise | テスト時に混ぜるノイズの割合（0~1) |
| test_idx | 訓練データのうちテストデータとして使う記憶パターンのインデックス |
| syn | Trueで同期更新，Falseで非同期更新 |

## 結果

記憶パターンの数とノイズの割合をそれぞれ変化させた時の誤差simと正答率accの値は以下のようになった．  
この値はloop_test=1000，非同期更新での値である．  
なお一つの表の枠内に( 誤差 / 正答率 )として表示している．  
記憶パターンの個数が増えるにつれ，またノイズの割合が増えるにつれ正答率が下がっていくことがわかる．  

<img src="https://github.com/shutakahama/Hopfield_Network/blob/master/fig/result_table.png" width="600">
