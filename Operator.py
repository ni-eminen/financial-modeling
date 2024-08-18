import numpy as np

from Distribution import Distribution, ConvolutionDistribution
from helpers import plot_line


class Operator:
    def __init__(self, name):
        """
        :param quantities: A dictionary of type dict[string, scipy distribution]
        """
        self.quantities = {}
        self.name = name

    def create_quantity(self, name, pdf, cdf, sample, kwargs, domain_type):
        if name not in self.quantities.keys():
            self.quantities[name] = {}

        self.quantities[name] = Distribution(name=name, pdf=pdf, cdf=cdf, sample=sample, kwargs=kwargs,
                                             domain_type=domain_type)

    def create_convolution(self, conv_name, quantity1: Distribution, quantity2: Distribution, operation='*'):
        if conv_name not in self.quantities.keys():
            self.quantities[conv_name] = {}

        new_quantity = ConvolutionDistribution(dist1=quantity1, dist2=quantity2, conv_operation=operation)

        self.quantities[conv_name] = new_quantity


    def visualize_quantity(self, f, quantity):
        a, b = np.min(quantity.samples), np.max(quantity.samples)
        if quantity.domain_type == 'discrete':
            x = list(range(a, b))
        else:
            x = list(np.linspace(a, b, 10000))
        y = [f(x_) for x_ in x]
        plot_line(x=x, y=y, hist=True if quantity.domain_type == 'discrete' else False)

