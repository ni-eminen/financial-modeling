import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, binom
import time
from Operator import Operator

print('program started')
def pdf_product_convolution_discrete_continuous(z, pmf_d, pdf_c, domain_d):
    return np.sum([pmf_d(x) * pdf_c(z / (x + 1e-6)) * (1 / np.abs(x + 1e-6)) for x in domain_d])


def cdf_product_convolution_discrete_continuous(z, pmf_d, cdf_c, domain_d):
    return np.sum([pmf_d(x) * cdf_c(z / (x + 1e-6)) for x in domain_d])


# gamma params
a = 10
b = 20

# binom params
n = 30
p = .2
operator = Operator()
operator.create_quantity('s1', gamma.pdf, cdf=gamma.cdf, sample=gamma.rvs, kwargs={'a': a, 'scale': b},
                         domain_type='c')
operator.create_quantity('s2', gamma.pdf, cdf=gamma.cdf, sample=gamma.rvs, kwargs={'a': a, 'scale': b},
                         domain_type='c')
operator.create_quantity('n_sales', binom.pmf, cdf=binom.cdf, sample=binom.rvs, kwargs={'n': 30, 'p': .2},
                         domain_type='d')
operator.create_cc_convolution('s3', operator.quantities['s1'], operator.quantities['s2'], '*')

operator.visualize_quantity(operator.quantities['s3'].cdf, 0, 100, )
operator.quantities['s3'].cdf(100_000)

print('Start by creating quantities')
while True:
    print('create quantity: q')
    print('create a convolution: c')
    print('visualize: v')
    print('inference: i')

    inp = input('> ')

    if inp == 'q':
        quantity_name = input('quantity name > ')
        quantity_model = input('quantity model (gamma: g, binomial: b) > ')
        if quantity_model == 'b':
            model = binom
            args = ['n', 'p']
        else:
            model = gamma
            args = ['a', 'scale']

        args_dict = {}
        print('select parameters')
        for arg in args:
            val = float(input(f'{arg}: '))
            args_dict[arg] = val

        if quantity_model == 'b':
            operator.create_quantity(name=quantity_name, pdf=binom.pmf, cdf=binom.cdf,
                                     sample=binom.rvs, kwargs=args_dict, domain_type='d')
        if quantity_model == 'g':
            operator.create_quantity(name=quantity_name, pdf=gamma.pdf, cdf=gamma.cdf,
                                     sample=gamma.rvs, kwargs=args_dict, domain_type='c')

        print(f'new quantity created: {quantity_name}')
        print()

    if inp == 'c':
        print(f'available quantities: {list(operator.quantities.keys())}')
        q1 = input('quantity 1 > ')
        q2 = input('quantity 2 > ')
        operation = input('operation (*, +, -, /) > ')
        name = input('convolution name > ')
        d1, d2 = operator.quantities[q1].domain_type, operator.quantities[q2].domain_type

        if d1 == 'c' and d2 == 'c':
            operator.create_cc_convolution(conv_name=name, c_quantity1=operator.quantities[q1],
                                           c_quantity2=operator.quantities[q2], operation=operation)
        elif d1 == 'c' and d2 == 'd':
            operator.create_dc_convolution(conv_name=name, c_quantity=operator.quantities[q1],
                                           d_quantity=operator.quantities[q2], operation=operation,
                                           domain_d=range(30))
        elif d1 == 'd' and d2 == 'd':
            operator.create_dd_convolution(conv_name=name, d_quantity1=operator.quantities[q1],
                                           d_quantity2=operator.quantities[q2],
                                           operation=operation, domain_d2=range(30))
        else:
            operator.create_dc_convolution()

    if inp == 'i':
        print(f'available quantities: {list(operator.quantities.keys())}')
        quantity = input('quantity > ')
        f = input('pdf / cdf > ')
        x = float(input('x > '))

        if f == 'pdf':
            print(operator.quantities[quantity].pdf(x))
        elif f == 'cdf':
            print(operator.quantities[quantity].cdf(x))

    if inp == 'v':
        q = input('quantity to visualize > ')
        f = input('pdf / cdf > ')
        a = float(input('range low end > '))
        b = float(input('range high end > '))
        discrete = input('discrete? (y / n) > ')

        if f == 'pdf':
            operator.visualize_quantity(operator.quantities[q].pdf, a, b, discrete)
        elif f == 'cdf':
            operator.visualize_quantity(operator.quantities[q].cdf, a, b, discrete)

# now we want to be able to create objectives (goals that we want to model, the really important quantities
# for the team)
