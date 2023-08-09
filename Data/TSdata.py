import random

import numpy as np
import pandas as pd


class TSLoader(object):
    """load the time series data"""
    def __init__(self, path, split, split_n, seed):
        data = np.array(pd.read_csv(path)).astype("float32")

        self.data = data
        self.seed = seed
        self.split = split
        self.split_n = split_n

    def get_data(self):
        label0 = self.data[self.data[:, 0] == 0]
        label1 = self.data[self.data[:, 0] == 1]
        label2 = self.data[self.data[:, 0] == 2]
        number0 = label0.shape[0]
        number1 = label1.shape[0]
        number2 = label2.shape[0]

        np.random.seed(self.seed)
        label0_dis = np.random.permutation(label0)
        label1_dis = np.random.permutation(label1)
        label2_dis = np.random.permutation(label2)

        train0 = label0_dis[0:int(number0 * self.split)]
        val0 = label0_dis[int(number0 * self.split):int(number0 * self.split_n)]
        #test0 = label0_dis[int(number0 * split_n):int(number0)]
        test0 = label0_dis[int(number0 * self.split_n):int(number0)]

        train1 = label1_dis[0:int(round((number1 * self.split)))]
        val1 = label1_dis[int(number1 * self.split):int(number1 * self.split_n)]
        #test1 = label1_dis[int(round(number1 * split_n)):int(round(number1))]
        test1 = label1_dis[int(round(number1 * self.split_n)):int(round(number1))]

        train2 = label2_dis[0:int(number2 * self.split)]
        val2 = label2_dis[int(number2 * self.split):int(number2 * self.split_n)]
        #test2 = label2_dis[int(number2 * split_n):int(number2)]
        test2 = label2_dis[int(round(number2 * self.split_n)):int(round(number2))]

        train = np.vstack((train0, train1, train2))
        val = np.vstack((val0, val1, val2))
        test = np.vstack((test0, test1, test2))

        Y_train, Y_val ,Y_test = train[:, 0], val[:, 0], test[:, 0]
        X_train, X_val ,X_test = train[:, 1:], val[:, 1:], test[:, 1:]

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def return_data(self):
        X_train, Y_train, X_val, Y_val, X_test, Y_test = self.get_data()

        l = int(X_train.shape[1]/2)

        X_train = X_train.reshape(-1, 2, l).transpose(0, 2, 1)

        X_val = X_val.reshape(-1,2,l).transpose(0, 2, 1)

        X_test = X_test.reshape(-1, 2, l).transpose(0, 2, 1)

        train_dataset = []
        for i in range(X_train.shape[0]):
            train_dataset.append((X_train[i], int(Y_train[i])))

        val_dataset = []
        for i in range(X_val.shape[0]):
            val_dataset.append((X_val[i], int(Y_val[i])))

        test_dataset = []
        for i in range(X_test.shape[0]):
            test_dataset.append((X_test[i], int(Y_test[i])))

        return train_dataset, val_dataset, test_dataset
