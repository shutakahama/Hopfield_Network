# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.random import *
import copy
import argparse


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
        # self.plot(self.train_data.T, 'train_data')

    # 描画
    def plot(self, data, name='example'):
        data_num = data.shape[0]
        
        for i in range(data_num):
            train_graph = data[i].reshape(self.data_size, self.data_size)
            plt.subplot(1, data_num, i+1)
            plt.imshow(train_graph, cmap=cm.Greys_r, interpolation='nearest')
            plt.title('data{0}'.format(i+1))

        # plt.show()
        plt.savefig('fig/{}.png'.format(name))

    # テストデータの作成
    def test_make(self, test_idx):
        x_test = copy.deepcopy(self.train_data[:,test_idx])
        # 確率rateで符号を反転させる
        for k in range(self.data_size * self.data_size):
            if rand() < self.noise: 
                x_test[k] *= -1
        
        return x_test

    # トレーニングデータによる学習
    def fit(self):
        self.W = np.dot(self.train_data, self.train_data.T)  # W = Σ x*x^T
        for i in range(self.data_size * self.data_size):
            self.W[i, i] = 0  # 対角成分0

        self.W /= self.train_num

        return self.W

    # エネルギー計算の関数
    def energy(self, x):
        # V = -1/2 * ΣΣ w*x*x
        return -0.5*np.dot(x.T, np.dot(self.W, x))

    # テストデータを使って想起(非同期更新)
    def predict_asyn(self, test_data):
        for ii in range(self.loop_update):
            num = randint(25)
            a = np.dot(self.W[num], test_data)
            test_data[num] = np.sign(a)

        return test_data

    # 同期更新
    def predict_syn(self, test_data):
        for ii in range(self.loop_update):
            e_old = self.energy(test_data)
            a = np.dot(self.W, test_data)
            test_data = np.sign(a)

            # エネルギーが変化しなくなったら打ち切り
            if self.energy(test_data) == e_old:
                return test_data

        return test_data

    # 類似度（距離）計算
    def distance(self, x):
        x = x.reshape(self.data_size * self.data_size, 1)
        dis = np.abs(self.train_data - x)  # 対象データと訓練データの差を計算
        dis = np.sum(dis, axis=0)
        dis /= 2
        sim = np.min(dis)  # 最小の距離とそのインデックスを取り出す
        simidx = np.argmin(dis)
        # print('train model = {0}, distance = {1}'.format(simidx, sim))
        return sim, simidx

    def run(self, test_idx):
        self.W = self.fit()
        dis = 0  # 正解と異なるマスの数
        acc = 0  # 正解率

        for l in range(self.loop_test):
            test_data = self.test_make(test_idx)
            # テストデータのプロット
            # self.plot(test_data.reshape(1, -1), 'test_{:04d}_data'.format(l))

            # 更新の方法
            if self.syn:
                print('t')
                test_predict = self.predict_syn(test_data)
            else:
                test_predict = self.predict_asyn(test_data)

            # 変更後のテストデータのプロット
            # self.plot(test_data.reshape(1, -1), 'test_{:04d}_after'.format(l))

            _dis, _ = self.distance(test_predict)
            dis += _dis
            if _dis == 0:
                acc += 1

            # print(_dis, acc)

        dis /= self.loop_test
        acc /= float(self.loop_test)
        print("sim = {0}".format(dis))
        print("accuracy = {0}".format(acc))


def main():
    parser = argparse.ArgumentParser(description='hopfield')
    parser.add_argument('--train_num', type=int, default=4)  # 記憶パターンの数
    parser.add_argument('--loop_update', type=int, default=300)  # 想起の最大回数
    parser.add_argument('--loop_test', type=int, default=100)  # テスト回数
    parser.add_argument('--noise', type=float, default=0.2)  # ノイズの割合（0~1)
    parser.add_argument('--test_idx', type=int, default=0)  # テストデータとする記憶パターンの番号
    parser.add_argument('--syn', type=bool, default=False)  # 同期更新，非同期更新
    args = parser.parse_args()

    # トレーニングデータ生成(6*25)
    train_data = [[-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1],
                  [1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1],
                  [1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1],
                  [1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1],
                  [-1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1],
                  [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1]]

    train_data = np.array(train_data, dtype=np.float32)
    train_data = train_data.T

    hop = Hopfield(train_data, args)
    hop.run(args.test_idx)


if __name__ == '__main__':
    main()
