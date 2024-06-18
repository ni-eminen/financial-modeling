import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, binom
import itertools
from scipy.stats import gaussian_kde


class SalesOperator:
    def __init__(self, sales, deals):
        """
        :param historical_performance: Historical sales, shape (n_sales, n_features)
        :param assumption: Assumed distribution of sales
        """
        self.sales = sales
        self.deals = deals

        self.deals_samples, self.sales_samples = self.sample(1000)


    def sample(self, samples):
        deals = self.deals.rvs(30, .2, size=samples)
        sales = self.sales.rvs(4, scale=500, size=samples)

        return deals, sales

    def convolution(self, quantities, operation):
        z = itertools.product(quantities)
        if operation == '+':
            z = [a + b for (a, b) in z]
        elif operation == '-':
            z = [a - b for (a, b) in z]
        elif operation == '*':
            z = [a * b for (a, b) in z]

        return z

    def model_pdf(self, model_distribution):
        model_density = gaussian_kde((np.array(model_distribution) - 3000) / 2)
        model_values = np.linspace(min(model_distribution), max(model_distribution), 1000)
        model_pdf = model_density(model_values)

        return model_density, model_values, model_pdf

    def cdf_quantity(self, amt, model_density, quantity):
        probability = np.trapz(y=model_density(np.linspace(amt, max(np.array(quantity)), 1000)),
                               x=np.linspace(amt, max(np.array(quantity)), 1000))

        return probability

    def p_profit(self, amt):
        model_density, model_values, model_pdf = self.model_pdf(self.convolution([self.deals, self.sales], '*'))
        probability = self.cdf_quantity()

def plot_sample(sample, kind='kde'):
    sns.displot(sample, kind=kind)

    plt.show()


operator = SalesOperator(sales=gamma, deals=binom)
p_profit = operator.p_profit(3000)
