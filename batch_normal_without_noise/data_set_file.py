
import numpy as np
def get_batch_normal_dataset():
    x_bias = -1
    y_bias = 10000
    x = np.linspace(-100, 100, 1000).reshape((-1, 1)) + x_bias
    y = 100 * x ** 2 + 3000 * x + y_bias
    # y += np.random.random(1000).reshape(x.shape) * (np.max(y) - np.min(y)) * 0.1
    return x,y