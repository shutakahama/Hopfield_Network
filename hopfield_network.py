# -*- coding: utf-8 -*-
import argparse
import copy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *
from tqdm import tqdm


class Hopfield:
    def __init__(self, train_data, args):
        self.train_data = train_data[:, :args.train_num]
        self.train_num = args.train_num
        self.data_size = int(np.sqrt(train_data.shape[0]))
        self.noise = args.noise
        self.loop_update = args.loop_update
        self.loop_test = args.loop_test
        self.syn = args.syn
        self.W = np.zeros((self.data_size*self.data_size, self.data_size*self.data_size))

        # トレーニングデータのプロット
        self.plot(self.train_data.T, 'train_data')

    # エネルギー計算の関数
    def energy(self, x):
        # V = -1/2 * ΣΣ w*x*x
        return -0.5*np.dot(x.T, np.dot(self.W, x))

    # 類似度（距離）計算
    def distance(self, x):
        # 対象データと訓練データの差を計算
        x = x.reshape(self.data_size * self.data_size, 1)
        dis = np.sum(np.abs(self.train_data - x), axis=0) / 2

        # 最小の距離とそのインデックスを取り出す
        sim = np.min(dis)
        simidx = np.argmin(dis)
        # print('train model = {0}, distance = {1}'.format(simidx, sim))
        return sim, simidx

    # テストデータの作成
    def test_make(self, test_idx):
        x_test = copy.deepcopy(self.train_data[:, test_idx])
        # 確率rateで符号を反転させる
        flip = choice([1, -1], self.data_size * self.data_size, p=[1 - self.noise, self.noise])
        x_test = x_test * flip

        return x_test

    # トレーニングデータによる学習
    def fit(self):
        self.W = np.dot(self.train_data, self.train_data.T)/self.train_num  # W = Σ x*x^T
        for i in range(self.data_size * self.data_size):
            self.W[i, i] = 0  # 対角成分0

        return self.W

    # テストデータを使って想起(非同期更新)
    def predict_asyn(self, test_data):
        for _ in range(self.loop_update):
            num = randint(self.data_size*self.data_size)
            test_data[num] = np.sign(np.dot(self.W[num], test_data))

        return test_data

    # 同期更新
    def predict_syn(self, test_data):
        e_old = float("inf")
        for _ in range(self.loop_update):
            # テストデータの更新とエネルギーの計算
            test_data = np.sign(np.dot(self.W, test_data))
            e_new = self.energy(test_data)

            # エネルギーが変化しなくなったら打ち切り
            if e_new == e_old:
                break

            e_old = e_new

        return test_data

    # 描画
    def plot(self, data, name='example'):
        for i in range(len(data)):
            train_graph = data[i].reshape(self.data_size, self.data_size)
            plt.subplot(1, len(data), i + 1)
            plt.imshow(train_graph, cmap=cm.Greys_r, interpolation='nearest')
            plt.title(f'data{i+1}')

        plt.show()
        # plt.savefig('fig/{}.png'.format(name))

    def run(self, test_idx):
        dis = 0  # 正解と異なるマスの数
        acc = 0  # 正解率

        # 訓練データから重み行列の計算
        self.W = self.fit()

        for l in tqdm(range(self.loop_test)):
            # テストデータの作成
            test_data = self.test_make(test_idx)
            # self.plot(test_data.reshape(1, -1), 'test_{:04d}_data'.format(l))

            # テストデータからの想起
            test_predict = self.predict_syn(test_data) if self.syn else self.predict_asyn(test_data)
            # self.plot(test_data.reshape(1, -1), 'test_{:04d}_after'.format(l))

            # 正答率，距離の計算
            _dis, _ = self.distance(test_predict)
            dis += _dis
            if _dis == 0:
                acc += 1

            # print(_dis, acc)

        dis /= self.loop_test
        acc /= float(self.loop_test)
        print("distance = {0}".format(dis))
        print("accuracy = {0}".format(acc))


def main():
    parser = argparse.ArgumentParser(description='hopfield')
    parser.add_argument('--loop_update', type=int, default=300)  # 想起の最大回数
    parser.add_argument('--loop_test', type=int, default=100)  # テスト回数
    parser.add_argument('--noise', type=float, default=0.2)  # ノイズの割合（0~1)
    parser.add_argument('--syn', type=bool, default=False)  # 同期更新，非同期更新
    parser.add_argument('--train_data', type=str, default="data/train_data.npy")  # 訓練データのパス
    parser.add_argument('--train_num', type=int, default=4)  # 記憶パターンの数
    parser.add_argument('--test_idx', type=int, default=0)  # テストデータとする記憶パターンの番号
    args = parser.parse_args()

    # トレーニングデータ読み込み(6*25)
    train_data = np.load(args.train_data).astype(np.float32)
    train_data = train_data.T

    hop = Hopfield(train_data, args)
    hop.run(args.test_idx)


if __name__ == '__main__':
    main()
