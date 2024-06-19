import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, binom
import itertools
from scipy.stats import gaussian_kde
import time


class Operator:
    def __init__(self, quantities: list[str], samplers: dict):
        """
        :param quantities: A dictionary of type dict[string, scipy distribution]
        """
        self.quantities = {quantity: {'samples': [], 'interpolation': None, 'sampler': None} for quantity in quantities}
        for quantity in samplers.keys():
            self.quantities[quantity]['sampler'] = samplers[quantity]

    def sample(self, quantities: list[str], sample_n: int):
        for q in quantities:
            self.quantities[q]['samples'] = self.quantities[q]['sampler'].rvs(sample_n)

        return self.quantity_samples

    def convolution(self, quantities: tuple[str], operation: str):
        """
        :param quantities: Two distributions that are convoluted into one according to some operation
        :param operation: Type of operation: '+', '-', '*'
        :return: A convolved distribution
        """
        for q in quantities:
            if self.quantity_samples[q]['samples'] == []:
                self.sample([q], 1000)

        a, b = self.quantity_samples[quantities[0]], self.quantity_samples[quantities[1]]

        z = itertools.product(a, b)
        if operation == '+':
            z = [a + b for (a, b) in z]
        elif operation == '-':
            z = [a - b for (a, b) in z]
        elif operation == '*':
            z = [a * b for (a, b) in z]

        z = np.array(z)

        return z

    def model_pdf(self, distribution: list):
        model_density = gaussian_kde(distribution)
        model_values = np.linspace(min(distribution), max(distribution), 1000)
        model_pdf = model_density(model_values)

        return model_density, model_pdf

    # def quantity_cdf(self, amt, model_density, quantity):
    #     probability = np.trapz(y=model_density(np.linspace(amt, max(np.array(quantity)), 1000)),
    #                            x=np.linspace(amt, max(np.array(quantity)), 1000))
    #
    #     return probability

    def distribution_cdf(self, amt, distribution):
        interpolated = self.distribution_interpolation(distribution)
        val = self.interpolated_integral(amt, np.inf, interpolated)
        tot = self.interpolated_integral(-np.inf, np.inf, interpolated)

        return val / tot

    def distribution_interpolation(self, distribution):
        return scipy.interpolate.interp1d(distribution)

    def interpolated_integral(self, a, b, f):
        result, error = scipy.integrate.quad(f, a, b)
        return result



def plot_sample(sample, kind='kde'):
    sns.displot(sample, kind=kind)

    plt.show()


def time_f(f, t='f'):
    start = time.time()
    x = f()
    end = time.time()
    print(t, end - start)
    return x


quantities = ['sales', 'deals']
operator = Operator(quantities=quantities, samplers={'sales': gamma, 'deals': binom})
operator.sample(['sales', 'deals'], sample_n=1000)
income = operator.convolution(quantities=('sales', 'deals'), operation='*')
p_profit = operator.distribution_cdf(3000, income)

print(p_profit)
