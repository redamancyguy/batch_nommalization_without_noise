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


a = np.random.rand(100)
print(a)
mean = 0
for i in range(100):
    mean += a[i]
var = 0
for i in range(100):
    var += (a[i] ** 2)
mean /= 100
var /= 100
var -= mean * mean
print(mean,var)
print(np.mean(a),np.var(a))


