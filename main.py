import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, binom
import itertools
from scipy.stats import gaussian_kde
import time


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
        a, b = quantities
        z = itertools.product(a, b)
        if operation == '+':
            z = [a + b for (a, b) in z]
        elif operation == '-':
            z = [a - b for (a, b) in z]
        elif operation == '*':
            z = [a * b for (a, b) in z]

        return np.array(z)

    def model_pdf(self, model_distribution):
        model_density = gaussian_kde(model_distribution)
        model_values = np.linspace(min(model_distribution), max(model_distribution), 1000)
        model_pdf = model_density(model_values)

        return model_density, model_values, model_pdf

    def quantity_cdf(self, amt, model_density, quantity):
        probability = np.trapz(y=model_density(np.linspace(amt, max(np.array(quantity)), 1000)),
                               x=np.linspace(amt, max(np.array(quantity)), 1000))

        return probability

    def p_profit(self, amt, distribution):
        model_density, model_values, model_pdf = self.model_pdf((np.array(distribution) - 3000) / 2)
        probability = self.quantity_cdf(amt, model_density, distribution)

        return probability

def plot_sample(sample, kind='kde'):
    sns.displot(sample, kind=kind)

    plt.show()

def time_f(f, t='f'):
    start = time.time()
    x = f()
    end = time.time()
    print(t, end - start)
    return x

operator = time_f(lambda: SalesOperator(sales=gamma, deals=binom), 'creating operator')
income = time_f(lambda: operator.convolution((operator.deals_samples, operator.sales_samples), '*'), 'creating income')

p_profit = time_f(lambda: operator.p_profit(3000, (income - 3000) / 2), 'creating p_profit')
print(p_profit)
