import numpy as np

from Distribution import Distribution, ConvolutionDistributionDiscreteContinuous, ConvolutionDistributionDiscrete, \
    ConvolutionDistributionContinuous
from helpers import plot_line


class Operator:
    def __init__(self):
        """
        :param quantities: A dictionary of type dict[string, scipy distribution]
        """
        self.quantities = {}

    def create_quantity(self, name, pdf, cdf, sample, kwargs, domain_type):
        if name not in self.quantities.keys():
            self.quantities[name] = {}

        self.quantities[name] = Distribution(name=name, pdf=pdf, cdf=cdf, sample=sample,
                                             kwargs=kwargs, domain_type=domain_type)

    def create_dc_convolution(self, conv_name, c_quantity: Distribution, d_quantity: Distribution,
                              operation, domain_d):
        if conv_name not in self.quantities.keys():
            self.quantities[conv_name] = {}

        c_quantity_pdf = c_quantity.pdf
        d_quantity_pmf = d_quantity.pdf
        c_quantity_cdf = c_quantity.cdf

        new_quantity = ConvolutionDistributionDiscreteContinuous(pmf_d=d_quantity_pmf, pdf_c=c_quantity_pdf,
                                                                 cdf_c=c_quantity_cdf, domain_d=domain_d)

        self.quantities[conv_name] = new_quantity

    def create_dd_convolution(self, conv_name, d_quantity1: Distribution, d_quantity2: Distribution,
                              operation, domain_d2):
        if conv_name not in self.quantities.keys():
            self.quantities[conv_name] = {}

        c_quantity_pdf = d_quantity1.pdf
        d_quantity_pmf = d_quantity2.pdf
        c_quantity_cdf = d_quantity1.cdf

        new_quantity = ConvolutionDistributionDiscrete(pmf_d1=d_quantity_pmf, pmf_d2=c_quantity_pdf,
                                                       cdf_d1=c_quantity_cdf, domain_d2=domain_d2)

        self.quantities[conv_name] = new_quantity

    def create_cc_convolution(self, conv_name, c_quantity1: Distribution, c_quantity2: Distribution, operation):
        if conv_name not in self.quantities.keys():
            self.quantities[conv_name] = {}

        c_quantity_pdf = c_quantity1.pdf
        d_quantity_pmf = c_quantity2.pdf
        c_quantity_cdf = c_quantity1.cdf

        new_quantity = ConvolutionDistributionContinuous(pdf_d1=d_quantity_pmf, pdf_d2=c_quantity_pdf,
                                                         cdf_c1=c_quantity_cdf)

        self.quantities[conv_name] = new_quantity

    def fixed_scale_convolution(self, scalar, quantity):
        revert = 1 / scalar
        pdf = lambda x: self.quantities[quantity]['pdf'](revert * x)
        cdf = lambda x: self.quantities[quantity]['cdf'](revert * x)

        return pdf, cdf

    def fixed_sum_convolution(self, fixed_num, quantity):
        pdf = lambda x: self.quantities[quantity]['pdf'](x + fixed_num)
        cdf = lambda x: self.quantities[quantity]['cdf'](x + fixed_num)

        return pdf, cdf

    def visualize_quantity(self, f, a, b, discrete='n'):
        if discrete == 'n':
            x = np.linspace(a, b)
        else:
            x = list(range(int(a), int(b)))

        y = [f(x_) for x_ in x]
        plot_line(x=x, y=y)

