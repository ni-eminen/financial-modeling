import numpy as np
from numpy import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
class Operator:
    def __init__(self, prior=None, historical_performance=None):
        """
        :param historical_performance: Historical sales, shape (n_sales, n_features)
        :param assumption: Assumed distribution of sales
        """
        if historical_performance is None:
            historical_performance = []

        # If no prior we assume standard normal prior
        prior = random.normal(size=1000) if prior is None else prior

        self.historical_performance = historical_performance
        self.prior = norm(loc=np.mean(prior), scale=np.std(prior))

        self.posterior = self.prior



    def sample(self, samples):
        sample = self.posterior.rvs(size=samples)
        return sample

def plot_sample(sample):
    sns.displot(sample, kind='kde')

    plt.show()

prior = norm.rvs(1, 10, size=1000)
operator = Operator(prior=prior)

sample = operator.sample(1000)

plot_sample(sample)