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

degree = 2
poly = PolynomialFeatures(degree, include_bias=True)

from data_set_file import get_batch_normal_dataset

x, y = get_batch_normal_dataset()

lng = LinearRegression(fit_intercept=False)

lng.fit(poly.fit_transform(x), y)
plt.scatter(x, y, color='purple')
plt.plot(x, lng.predict(poly.fit_transform(x)), color='red')

print(y - lng.predict(poly.fit_transform(x)))

plt.show()
