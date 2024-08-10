import numpy as np
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.interpolate import CubicSpline


class Distribution:
    def __init__(self, name, pdf, cdf, sample, kwargs, domain_type):
        self.name = name
        self.pdf_f = pdf
        self.cdf_f = cdf
        self.kwargs = kwargs
        self.sample_f = sample
        self.domain_type = domain_type

    def pdf(self, x):
        return self.pdf_f(x, **self.kwargs)

    def cdf(self, x):
        return self.cdf_f(x, **self.kwargs)

    def sample(self):
        return self.sample_f(self.kwargs)


class ConvolutionDistributionDiscreteContinuous:
    def __init__(self, pmf_d, pdf_c, cdf_c, domain_d):
        self.pmf_d = pmf_d
        self.pdf_c = pdf_c
        self.domain_d = domain_d
        self.cdf_c = cdf_c

    def pdf(self, z):
        return np.sum([self.pmf_d(x) * self.pdf_c(z / (x + 1e-6)) * (1 / np.abs(x + 1e-6))
                       for x in self.domain_d])

    def cdf(self, z):
        return np.sum([self.pmf_d(x) * self.cdf_c(z / (x + 1e-6)) for x in self.domain_d])

    def sample(self):
        return 1


class ConvolutionDistributionDiscrete:
    def __init__(self, pmf_d1, pmf_d2, cdf_d1, domain_d2):
        self.pmf_d1 = pmf_d1
        self.pmf_d2 = pmf_d2
        self.domain_d2 = domain_d2
        self.cdf_d1 = cdf_d1

    def pdf(self, z):
        return np.sum([self.pmf_d1(x) * self.pmf_d2(z / x) * (1 / np.abs(x))
                       for x in self.domain_d2 if x != 0])

    def cdf(self, z):
        return np.sum([self.pmf_d1(x) * self.cdf_d1(z / x)
                       for x in self.domain_d2 if x != 0])

    def sample(self):
        return 1


class ConvolutionDistributionContinuous:
    def __init__(self, pdf_d1, pdf_d2, cdf_c1):#, domain: (float, float)):
        self.pdf_c1 = pdf_d1
        self.pdf_c2 = pdf_d2
        self.cdf_c1 = cdf_c1

        # interpolate_domain = np.linspace(domain[0], domain[1])
        # self.i_pdf = CubicSpline(x=np.linspace(domain[0], domain[1]), y=)

    def pdf_integrand(self, x, z):
        if x < 1e-4:
            return 0
        return self.pdf_c1(x) * self.pdf_c2(z / x) * (1 / np.abs(x))

    def pdf(self, z, points_amt=21):
        points = np.linspace(1e-8, z, points_amt)
        a_points = points[0:points_amt - 1]
        b_points = points[1:points_amt]

        result = np.sum([quad(self.pdf_integrand, a, b, epsabs=1e-3, args=(z,))[0] for a, b in zip(a_points, b_points)])

        return result


    def cdf(self, z):
        result = integrate.quad(self.pdf, 0, z)[0]

        return result

    def sample(self):
        return 1
