import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, binom
import itertools
from scipy.stats import gaussian_kde
import time

from Sampler import Sampler


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
            self.quantities[q]['samples'] = self.quantities[q]['sampler'].rvs(size=sample_n)

        return self.quantities

    def convolution(self, quantities: tuple[str], operation: str):
        """
        :param quantities: Two distributions that are convoluted into one according to some operation
        :param operation: Type of operation: '+', '-', '*'
        :return: A convolved distribution
        """
        for q in quantities:
            if len(self.quantities[q]['samples']) == 0:
                self.sample([q], 1000)

        a, b = self.quantities[quantities[0]]['samples'], self.quantities[quantities[1]]['samples']

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

    def distribution_cdf(self, amt, distribution, ceil=10000):
        interpolated = self.distribution_interpolation(distribution)
        val = self.interpolated_integral(amt, ceil, interpolated)
        tot = self.interpolated_integral(0, ceil, interpolated)

        return val / tot

    # TODO: Here the x is wrong, the y's did not come from those x's, they were sampled and we don't know the
    # TODO: x's at this point. We can find out the x's with the inverse of the cdf, PPF
    def distribution_interpolation(self, distribution):
        return scipy.interpolate.interp1d(x=list(range(len(distribution))), y=distribution, fill_value="extrapolate")

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
operator = Operator(quantities=quantities, samplers={'sales': Sampler(distribution=gamma, pos_args=(4,), params={'scale': 500}),
                                                     'deals': Sampler(distribution=binom, params={'n': 30, 'p': .2})})
operator.sample(['sales', 'deals'], sample_n=1000)
income = operator.convolution(quantities=('sales', 'deals'), operation='*')
p_profit = operator.distribution_cdf(3000, income, ceil=10**3)

print(p_profit)
