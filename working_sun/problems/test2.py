import numpy as np
from sklearn.preprocessing import StandardScaler

class StandMyself:
    def __init__(self):
        self.var_ = None
        self.mean_ = None

    def fit(self, x):
        self.mean_ = np.mean(x, axis=0)
        self.var_ = np.var(x, axis=0)

    def transform(self, x):
        return (x - self.mean_) / np.sqrt(self.var_)

    def inverse_transform(self, x):
        return x * np.sqrt(self.var_) + self.mean_


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4]).reshape(-1, 1)
    ss = StandardScaler()
    ss.fit(a)
    ss.mean_ = 2.5
    ss.var_ = 1.25
    print(ss.transform(a))
    print(a.var())
    print(ss.var_)