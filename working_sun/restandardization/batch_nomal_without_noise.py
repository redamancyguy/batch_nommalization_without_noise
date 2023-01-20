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
from my_utils_for_study import StandMyself, re_standardization
from my_utils_for_study import get_batch_normal_dataset
from my_utils_for_study import get_mean_and_variance

plt.ion()

x, y = get_batch_normal_dataset()
np.random.seed(123)
degree = 2
poly = PolynomialFeatures(degree, include_bias=False)
w = np.random.random(degree).reshape((-1, 1))
b = np.zeros((1, 1))

learning_rate = 0.01
number_of_samples_in_a_batch = 5
batch_size = 5
batch_pointer = 0
sample_quantity = len(x)
# did not random disrupt the data x and y. There is a synthetic data ordered,then we make a bad batch

# shu_in = np.random.choice(np.arange(sample_quantity), size=sample_quantity, replace=False)
# x = x[shu_in]
# y = y[shu_in]
user_self_made_standardScalar = True
poly.fit(x)





for _ in range(10000):
    # draw a batch from all samples
    if batch_size + batch_pointer > sample_quantity:
        x_batch = np.vstack((x[batch_pointer:], x[:batch_pointer + batch_size - sample_quantity]))
        y_batch = np.vstack((y[batch_pointer:], y[:batch_pointer + batch_size - sample_quantity]))
        batch_pointer = batch_pointer + batch_size - sample_quantity
    else:
        x_batch = x[batch_pointer: batch_pointer + batch_size]
        y_batch = y[batch_pointer: batch_pointer + batch_size]
        batch_pointer += batch_size
    x_batch = poly.transform(x_batch)
    # a standardScalar for this new batch
    std_x = StandMyself()
    std_x.fit(x_batch)
    std_y = StandMyself()
    std_y.fit(y_batch)

    user_self_made_standardScalar = True
    # 手动计算方差和均值
    if user_self_made_standardScalar:
        mean_x, var_x = get_mean_and_variance(x_batch)
        mean_y, var_y = get_mean_and_variance(y_batch)
        #
        # std_x.mean_ = mean_x
        # std_x.var_ = var_x
        # std_y.mean_ = mean_y
        # std_y.var_ = var_y

        # std_x.mean_ = (np.max(x_batch, axis=0) + np.min(x_batch, axis=0)) / 2
        # std_x.var_ = (np.max(x_batch, axis=0) - np.min(x_batch, axis=0)) / 2
        # std_y.mean_ = (np.max(y_batch, axis=0) + np.min(y_batch, axis=0)) / 2
        # std_y.var_ = (np.max(y_batch, axis=0) - np.min(y_batch, axis=0)) / 2
        #
        # std_x.mean_ *= 1.1
        # std_x.var_ *= 10
        # std_y.mean_ *= 1
        # std_y.var_ *= 1000
        #
        # std_x.mean_ = mean_x
        # std_x.var_ = var_x
        # std_y.mean_ = mean_y
        # std_y.var_ = var_y

    shuffle_indexes = np.random.choice(np.arange(batch_size), size=batch_size, replace=False)
    x_batch = x_batch[shuffle_indexes, :]
    x_batch = std_x.transform(x_batch)
    y_batch = y_batch[shuffle_indexes]
    y_batch = std_y.transform(y_batch)

    # 从parameter中抽取出来 标准化 参数
    # scrape the influence from the model about the new parameter for standardization
    w,b = re_standardization.scrape_out(w,b,std_x,std_y)

    for __ in range(batch_size):
        plt.cla()
        x_train = x_batch[__, :].reshape((1, -1))
        y_train = y_batch[__, :].reshape((1, -1))
        prediction = np.dot(x_train, w) + b
        residual_error = prediction - y_train

        dl_dw = learning_rate * np.dot(x_train.T, residual_error)
        dl_db = learning_rate * residual_error
        w = w - dl_dw
        b = b - dl_db

        plt.scatter(x, y, color='purple')
        x_for_prediction = np.linspace(np.min(x), np.max(x), 100).reshape((-1, 1))
        x_for_prediction = poly.fit_transform(x_for_prediction)
        x_for_prediction = std_x.transform(x_for_prediction)
        pred = np.dot(x_for_prediction, w) + b
        plt.plot(np.linspace(np.min(x), np.max(x), 100).reshape((-1, 1)),
                 std_y.inverse_transform(pred),
                 color='red')

        plt.pause(1e-10)

        if __ == batch_size - 1:
            pass
    # integrate parameters into the model
    w,b = re_standardization.integrate_in(w,b,std_x,std_y)

    print('iteration:', _, 'w', w.flatten(), 'b', b)
