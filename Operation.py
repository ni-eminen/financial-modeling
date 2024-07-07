from abc import abstractmethod
import numpy as np


class Operation:
    @abstractmethod
    def run_operation(self):
        """
        :return: pdf and cdf of the convolution
        """
        pass


class Sum(Operation):
    def __init__(self, dist, fixed_num):
        self.dist = dist
        self.fixed_num = fixed_num

    def run_operation(self):
        pdf = lambda x: self.dist['pdf'](x + self.fixed_num)
        cdf = lambda x: self.dist['cdf'](x + self.fixed_num)

        return pdf, cdf


class Scale(Operation):
    def __init__(self, scalar, pdf, cdf):
        self.scalar = scalar
        self.pdf = pdf
        self.cdf = cdf

    def run_operation(self):
        revert = 1 / self.scalar
        pdf = lambda x: pdf(revert * x)
        cdf = lambda x: pdf(revert * x)

        return pdf, cdf


class ContinuousDiscreteProduct(Operation):
    def __init__(self, discrete_dist, continuous_dist, d_domain):
        self.cdist = discrete_dist
        self.ddist = continuous_dist
        self.d_domain = d_domain

    def pdf_product_convolution_discrete_continuous(self, z, pmf_d, pdf_c, domain_d):
        return np.sum([pmf_d(x) * pdf_c(z / (x + 1e-6)) * (1 / np.abs(x + 1e-6)) for x in domain_d])

    def cdf_product_convolution_discrete_continuous(self, z, pmf_d, cdf_c, domain_d):
        return np.sum([pmf_d(x) * cdf_c(z / (x + 1e-6)) for x in domain_d])

    def run_operation(self):
        """
        :return: pdf and cdf of the convolution
        """
        c_quantity_pdf = self.cdist['pdf']
        d_quantity_pmf = self.ddist['pdf']
        c_quantity_cdf = self.cdist['cdf']

        new_quantity_pdf = lambda z: self.pdf_product_convolution_discrete_continuous(z, d_quantity_pmf,
                                                                                      c_quantity_pdf, self.d_domain)
        new_quantity_cdf = lambda z: self.cdf_product_convolution_discrete_continuous(z, d_quantity_pmf,
                                                                                      c_quantity_cdf, self.d_domain)

        return new_quantity_pdf, new_quantity_cdf
