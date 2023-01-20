import numpy as np
import time

start = time.time_ns()
count = 0
from sklearn.preprocessing import PolynomialFeatures
for i in range(10000):
    a = np.random.randn(100,100)
    b = np.random.randn(100,100)
    c = np.dot(a,b)

print((time.time_ns() - start) / 1000)
print(count)

