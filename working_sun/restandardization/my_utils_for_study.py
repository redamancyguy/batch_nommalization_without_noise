import numpy as np


def get_batch_normal_dataset(add_noise = False):
    x_bias = -15
    x = np.linspace(-100, 100, 1000).reshape((-1, 1)) + x_bias
    y = 100 * x ** 2 + 3000 * x

    y += np.max(np.abs(y)) / 2
    if add_noise:
        y += np.random.random(1000).reshape(x.shape) * (np.max(y) - np.min(y)) * 0.05
    return x, y


def get_mean_and_variance(data_samples):
    if len(data_samples) < 10:
        return np.mean(data_samples, axis=0), np.var(data_samples, axis=0)
    samples = np.random.choice(np.arange(len(data_samples)), 10)
    samples = data_samples[samples]
    samples = np.sort(samples, axis=0)
    return data_samples[int(len(data_samples) / 2)], ((np.max(samples, axis=0) - np.min(samples, axis=0)) * 0.3249) ** 2


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


class re_standardization:
    @staticmethod
    def integrate_in(w, b, std_x, std_y):
        b = b * np.sqrt(std_y.var_.reshape(1)) + std_y.mean_.reshape(1) \
            - np.sqrt(std_y.var_) * np.dot(w.T, (std_x.mean_.reshape(w.shape) / np.sqrt(std_x.var_.reshape(w.shape))))

        w = w / np.sqrt(std_x.var_.reshape(w.shape)) * np.sqrt(std_y.var_.reshape(1))
        return w, b

    @staticmethod
    def scrape_out(w, b, std_x, std_y):
        b = (b + np.dot(std_x.mean_, w) - std_y.mean_) / np.sqrt(std_y.var_.reshape(1))

        w = w * np.sqrt(std_x.var_.reshape(w.shape)) / np.sqrt(std_y.var_.reshape(1))
        return w, b
