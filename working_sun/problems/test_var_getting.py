import copy
import random
import stat
import numpy as np
import time
import sklearn.pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt




def get_mean_and_variance(data_samples):
    if len(data_samples) < 10:
        return np.mean(data_samples,axis=0), np.var(data_samples,axis=0)
    samples = np.random.choice(np.arange(len(data_samples)), 10)
    samples = data_samples[samples]
    samples = np.sort(samples,axis=0)
    return data_samples[int(len(data_samples) / 2)], ((np.max(samples,axis=0) - np.min(samples,axis=0)) * 0.3249) ** 2


if __name__ == '__main__':
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    for _ in range(300):
        a = np.random.random(100000)
        a.sort()
        print('==============================')
        # print('a.mean', np.mean(a), a[int(len(a) / 2)])

        b = np.random.choice(a, 10)
        c = np.random.choice(a, 7)

        count1 += (a.mean() - b.mean()) ** 2
        count2 += (a.mean() - a[int(len(a) / 2)]) ** 2

        count3 += (a.var() - b.var()) ** 2
        count4 += (a.var() - ((np.max(b) - np.min(b)) * 0.3249) ** 2) ** 2

        count5 += (a.var() - c.var()) ** 2
        count6 += (a.var() - ((np.max(c) - np.min(c)) * 0.3698) ** 2) ** 2
        print('b.极差', ((np.max(b) - np.min(b)) * 0.3249) ** 2, 'a.var()', a.var(), b.var())
        print(get_mean_and_variance(a))
        __,cc = get_mean_and_variance(a)
        count7 += (a.var() - cc) ** 2

    print(count1)
    print(count2)
    print(count3)
    print(count4)
    print(count5)
    print(count6)
    print(count7)
