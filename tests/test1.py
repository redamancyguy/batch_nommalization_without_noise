ss = ''' for(Block* i = block_link_64.head;i!=block_link_64.tail;){
            Block * temp = i->next;
            std::free(i);
            i = temp;
        }
        std::free(block_link_64.head);'''

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

x = np.linspace(0, 100, 100).reshape((-1, 1))
y = x ** 2 + 30 * x + 100
x_ = PolynomialFeatures(2, include_bias=False).fit_transform(x)
theta = np.random.rand(2, 1)
print('raw_X',x_)

print('==================')
lng = LinearRegression()

lng.fit(x_, y)

plt.scatter(x, y)
plt.plot(x, lng.predict(x_), color='red')
plt.show()








