import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
def plot_line(x, y, hist=False):
    if hist:
        sns.barplot(x=x, y=y)
    else:
        sns.lineplot(x=x, y=y)

    plt.show()


def plot_sample(sample, kind='kde'):
    sns.displot(sample, kind=kind)

    plt.show()


def time_f(f, t='f'):
    start = time.time()
    x = f()
    end = time.time()
    print(t, end - start)
    return x
