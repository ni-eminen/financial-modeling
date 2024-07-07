import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, binom
import time

from Operation import ContinuousDiscreteProduct


class Operator:
    def __init__(self):
        """
        :param quantities: A dictionary of type dict[string, scipy distribution]
        """
        self.quantities = {}

    def create_quantity(self, name, pdf, cdf):
        if name not in self.quantities.keys():
            self.quantities[name] = {}

        self.quantities[name]['pdf'] = pdf
        self.quantities[name]['cdf'] = cdf

    def create_dc_convolution(self, conv_name, c_quantity, d_quantity, operation, domain_d):
        if conv_name not in self.quantities.keys():
            self.quantities[conv_name] = {}

        c_quantity_pdf = self.quantities[c_quantity]['pdf']
        d_quantity_pmf = self.quantities[d_quantity]['pdf']
        c_quantity_cdf = self.quantities[c_quantity]['cdf']

        new_quantity_pdf = lambda z: pdf_product_convolution_discrete_continuous(z, d_quantity_pmf,
                                                                                 c_quantity_pdf, domain_d)
        new_quantity_cdf = lambda z: cdf_product_convolution_discrete_continuous(z, d_quantity_pmf,
                                                                                 c_quantity_cdf, domain_d)

        self.quantities[conv_name]['pdf'] = new_quantity_pdf
        self.quantities[conv_name]['cdf'] = new_quantity_cdf

    def fixed_scale_convolution(self, scalar, quantity):
        revert = 1 / scalar
        pdf = lambda x: self.quantities[quantity]['pdf'](revert * x)
        cdf = lambda x: self.quantities[quantity]['cdf'](revert * x)

        return pdf, cdf

    def fixed_sum_convolution(self, fixed_num, quantity):
        pdf = lambda x: self.quantities[quantity]['pdf'](x + fixed_num)
        cdf = lambda x: self.quantities[quantity]['cdf'](x + fixed_num)

        return pdf, cdf

def pdf_product_convolution_discrete_continuous(z, pmf_d, pdf_c, domain_d):
    return np.sum([pmf_d(x) * pdf_c(z / (x + 1e-6)) * (1/np.abs(x + 1e-6)) for x in domain_d])

def cdf_product_convolution_discrete_continuous(z, pmf_d, cdf_c, domain_d):
    return np.sum([pmf_d(x) * cdf_c(z / (x + 1e-6)) for x in domain_d])

def plot_sample(sample, kind='kde'):
    sns.displot(sample, kind=kind)

    plt.show()


def time_f(f, t='f'):
    start = time.time()
    x = f()
    end = time.time()
    print(t, end - start)
    return x

def prior_to_commission(commission, after_commission):
    """
    Gives you the real value of which commission percent has been taken as commission to reach
    after_commission net value.
    :param commission: real in domain [0, 1]
    :param after_commission: real from which commission percentage has been deducted
    :return: the original value from which commission has not been deducted
    """
    after_commission_percentage = 1 - commission
    before_commission_value = 1 / after_commission_percentage

    return before_commission_value * after_commission

# gamma params
a = 10
b = 2000

# binom params
n = 30
p = .2

quantities = ['sales', 'deals']
operator = Operator()
operator.create_quantity(name='sales', pdf=lambda x: gamma.pdf(x, a=a, scale=b), cdf=lambda x: gamma.cdf(x, a=a, scale=b))
operator.create_quantity(name='n_sales', pdf=lambda x: binom.pmf(x, n=n, p=p), cdf=lambda x: binom.cdf(x, n=n, p=p))

operator.create_dc_convolution(conv_name='income', c_quantity='sales', d_quantity='n_sales', operation='*',
                               domain_d=range(30))

p_profit = operator.quantities['income']['cdf'](100_000)
print(p_profit)

x = np.linspace(0, 500000)
y = [operator.quantities['income']['cdf'](z=x_) for x_ in x]

commission = .2
fixed_costs = 40_000
profit = 20_000
# profit = income - commission - fixed costs
# commission = .2 * sales income
p_20_000_profit = operator.quantities['income']['cdf'](prior_to_commission(commission=.2,
                                                                           after_commission=(profit + fixed_costs)))

# now we want to be able to create objectives (goals that we want to model, the really important quantities
# for the team)

# operator.create_objective ???