
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
from test2 import StandMyself
from data_set_file import get_batch_normal_dataset

x, y = get_batch_normal_dataset(True)
plt.ion()

for i in range(100):
    plt.cla()
    samples = np.random.choice(np.arange(len(x)), 6)
    x_ = x[samples]
    y_ = y[samples]
    lng = LinearRegression()
    poly = PolynomialFeatures(2,include_bias=True)
    lng.fit(poly.fit_transform(x_),y_)

    plt.scatter(x,y,color='yellow')
    plt.scatter(x_,y_,color= 'blue')
    x_show = np.linspace(np.min(x),np.max(x),100).reshape((-1,1))
    plt.plot(x_show,lng.predict(poly.fit_transform(x_show)),color='red')
    plt.pause(1)