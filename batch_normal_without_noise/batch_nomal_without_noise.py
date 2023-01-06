import copy
import random
import numpy as np
import time
import sklearn.pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib
import os
import tensorflow as tf


from data_set_file import get_batch_normal_dataset

x, y = get_batch_normal_dataset()


degree = 2
poly = PolynomialFeatures(degree, include_bias=False)
w = np.random.random(degree).reshape((-1, 1))
b = np.zeros((1, 1))
plt.ion()
learning_rate = 0.3
# learning_rate = 0.00001



batch_size = 50
batch_pointer = 0
sample_count = len(x)
shu_in = np.random.choice(np.arange(sample_count), size=sample_count, replace=False)
# x = x[shu_in]
# y = y[shu_in]
poly.fit(x)
for _ in range(1000):
    if batch_size + batch_pointer > sample_count:
        x_batch = np.vstack((x[batch_pointer:], x[:batch_pointer + batch_size - sample_count]))
        y_batch = np.vstack((y[batch_pointer:], y[:batch_pointer + batch_size - sample_count]))
        batch_pointer = batch_pointer + batch_size - sample_count
    else:
        x_batch = x[batch_pointer: batch_pointer + batch_size]
        y_batch = y[batch_pointer: batch_pointer + batch_size]
        batch_pointer += batch_size
    x_batch = poly.transform(x_batch)
    std_x = StandardScaler()
    std_x.fit(x_batch)
    std_y = StandardScaler()
    std_y.fit(y_batch)
    shuffle_indexes = np.random.choice(np.arange(batch_size), size=batch_size, replace=False)
    x_batch = x_batch[shuffle_indexes, :]
    x_batch = std_x.transform(x_batch)
    y_batch = y_batch[shuffle_indexes]
    y_batch = std_y.transform(y_batch)

    # 从parameter中抽取出来 标准化 参数
    # 这里暂时只给了一个维度 实际上应该是 多个维度的
    b = (b - std_y.mean_ + np.dot(w.T, std_x.mean_.reshape(w.shape))) / np.sqrt(std_y.var_.reshape(b.shape))

    w = w * (np.sqrt(std_x.var_.reshape(w.shape)) / np.sqrt(std_y.var_.reshape(b.shape)))

    for __ in range(batch_size):
        x_train = x_batch[__, :].reshape((1, -1))
        y_train = y_batch[__, :].reshape((1, -1))
        plt.cla()
        plt.scatter(x, y, color='purple')
        x_for_pred = np.linspace(np.min(x), np.max(x), 100).reshape((-1, 1))
        x_for_pred = poly.fit_transform(x_for_pred)
        x_for_pred = std_x.transform(x_for_pred)
        pred = np.dot(x_for_pred, w) + b
        plt.plot(np.linspace(np.min(x), np.max(x), 100).reshape((-1, 1)),
                 std_y.inverse_transform(pred),
                 color='red')

        prediction = np.dot(x_train, w) + b
        residual_error = prediction - y_train

        dl_dw = learning_rate * (1 / batch_size) * np.dot(x_train.T, residual_error)
        dl_db = learning_rate * (1 / batch_size) * residual_error

        w = w - dl_dw
        b = b - dl_db
        plt.pause(0.001)

    b = b * np.sqrt(std_y.var_.reshape(b.shape)) + std_y.mean_.reshape(b.shape) \
        - np.sqrt(std_y.var_) * np.dot(w.T, (std_x.mean_.reshape(w.shape) / np.sqrt(std_x.var_.reshape(w.shape))))

    w = w * (np.sqrt(std_y.var_.reshape(b.shape)) / np.sqrt(std_x.var_.reshape(w.shape)))

    print('iteration:', _, 'w', w, 'b', b)
