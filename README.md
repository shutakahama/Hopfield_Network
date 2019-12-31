# Hopfield_Network
（書きかけの項目です）
ここでは全結合型ニューラルネットワークであるホップフィールドネットワークを実装した．  

## ホップフィールドネットワークとは
ホップフィールドネットワークとは、全結合型のニューラルネットワークである。  
深層学習で主流になっているフィードフォーワードネットワークは，ニューロンのまとまりである層を積み重ねた構造をしており，各層から次の層への一方向の結合しか存在しない．  
それに対しホップフィールドネットワークは任意の2つのニューロン全てについて両方向の結合が存在している．  

（図）

ホップフィールドネットワークの特徴として，特定のパターンを記憶できる事が挙げられる．
いくつかのパターンを入力として学習させておくと，ノイズを含んだパターンを入力した際に元のパターンを想起する事ができる．  
具体的には、記憶するパターンを𝑥^1, x^2, ... , x^QのQ個とする．xはニューロン数の次元(Nとする)を持っており，各ニューロンにその値を記憶させる．  
これらに対して重み行列Wを

（図）

と定める。Wはこの場合N*N次元の行列となる。この計算でネットワークの学習は終わりである．  

その後ノイズの入った入力から元のパターンを想起する．  
記憶パターンQ個のうち一つにある割合でノイズを加えたものをテスト入力xとする．時刻tでの更新は

(図)

で行う．  
sgn関数の性質によりxの要素は1, 0, -1のいずれかの値をとる。また定義上w_ij = w_ji となる.さらに自己結合を考慮しない場合はw = 0とする．   
この更新を十分行って平衡状態に達したら終了である．更新を繰り返すことで，ノイズを含んだ入力であっても記憶したパターンに収束していくことが知られている．

更新の方法には同期型、非同期型の2つがある。同期型は全てのニューロンを同時に更新し、非同期型は一つずつニューロンを更新していく．
ホップフィールドネットワークは基本的に非同期型である．  


平衡状態の判定はニューロンの値に変化がなくなった時としても良いが，より厳密にはネットワークエネルギーを使う。エネルギーは

（図）

で表せる．このエネルギーはリアプノフ性があり、xを更新すると単調減少ため，Eが変化しなくなったところが平衡点と言える．  


## 実装
ここでは5*5画素の画像を最大6枚記憶させる．よってトレーニングデータとして6*25次元のベクトルx_trainを用意する．各要素の値は1か-1のいづれかである．  
記憶パターンは以下の6つであり，左から記憶パターン 1, 2, ..., 6 と呼ぶことにする．  

(図)

runを実行するとまずfit()関数によって学習が行割れる．fit()は25*25次元の重み行列Wを返す．  
テストデータはtest_make()で生成する．トレーニングデータ行列の中からパターンを一つランダムに取り出し、それにノイズを加えたものをテストデータとする．  
ノイズは指定した確率で各ニューロンの値に−１をかけることである．  
記憶パターン1についてrate = 0.1でノイズを与えた場合の結果は例えば以下のようになる.

（図）

パターンの想起は predict_syn(), predict_asyn()によって行う.  
前者は同期更新で全てのニューロンを同時に更新する．収束の判定はエネルギー関数を使った．
後者は非同期更新で，ランダムにニューロンを１つ選びその値を更新することを繰り返す．1つずつニューロンを更新するとエネルギー変化で収束を判断するのが難しいため、ここでは単純に更新を十分な数繰り返した．  
想起性能はsimilarity()で評価する。トレーニングデータのうち想起結果と最も差が小さかったパターンをsimidx，その差をsimとして出力する。差は値の違うマス目の数である．  
この試行を複数回行ってその平均を取る．また差が0だった個数を正答率としてカウントする．  
結果はplotで描画可能．  
  
## 実行
データのありか  
各変数の説明  
ライブラリ  
実行コマンド

## 結果
