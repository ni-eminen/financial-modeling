import numpy as np
import scipy.integrate as integrate
import scipy.stats
from scipy.integrate import quad
from scipy.stats import rv_histogram
import itertools
from scipy.interpolate import CubicSpline


class Distribution:
    def __init__(self, name, pdf, cdf, sample, kwargs, domain_type):
        self.name = name
        self.pdf_f = pdf
        self.cdf_f = cdf
        self.kwargs = kwargs
        self.sample_f = sample
        self.samples = self.sample(1000)
        self.domain_type = domain_type

    def pdf(self, x):
        return self.pdf_f(x, **self.kwargs)

    def cdf(self, x):
        return self.cdf_f(x, **self.kwargs)

    def sample(self, n):
        samples = []
        for _ in range(n):
            samples.append(self.sample_f(**self.kwargs))
        return np.array(samples)


class ConvolutionDistribution:
    def __init__(self, dist1, dist2, conv_operation='*'):
        self.dist1 = dist1
        self.dist2 = dist2
        self.conv_operation = conv_operation

        if dist1.domain_type == 'continuous' or dist2.domain_type == 'continuous':
            self.domain_type = 'continuous'
        else:
            self.domain_type = 'discrete'

        d1_samples = self.dist1.sample(1000)
        d2_samples = self.dist2.sample(1000)
        cartesian = itertools.product(d1_samples, d2_samples)

        if self.conv_operation == '*':
            self.samples = [a * b for a, b in cartesian]
        elif self.conv_operation == '+':
            self.samples = [a + b for a, b in cartesian]
        elif self.conv_operation == '-':
            self.samples = [a - b for a, b in cartesian]
        elif self.conv_operation == '/':
            self.samples = [a / b for a, b in cartesian if b != 0]

        hist = np.histogram(self.samples, bins=500) # TODO: amt of bins, granularity
        self.rv_hist = rv_histogram(hist)

    def pdf(self, x):
        return self.rv_hist.pdf(x)

    def cdf(self, x):
        return self.rv_hist.cdf(x)

    def sample(self, n):
        return self.rv_hist.rvs(size=n)
