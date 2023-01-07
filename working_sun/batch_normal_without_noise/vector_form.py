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

a = np.array([1, 2, 3, 4]).reshape((1, -1))

b = np.array([2,4,6,8]).reshape(a.shape)

print(a * b)

print(a / b)
