import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, binom
import time
from Operator import Operator

print('program started')

# gamma params
a = 10
b = 2000

# binom params
n = 30
p = .2
operator = Operator()
# operator.create_quantity('s1', gamma.pdf, cdf=gamma.cdf, sample=gamma.rvs, kwargs={'a': a, 'scale': b},
#                          domain_type='continuous')
# operator.create_quantity('s2', gamma.pdf, cdf=gamma.cdf, sample=gamma.rvs, kwargs={'a': a, 'scale': b},
#                          domain_type='continuous')
# operator.create_quantity('n_sales', binom.pmf, cdf=binom.cdf, sample=binom.rvs, kwargs={'n': 30, 'p': .2},
#                          domain_type='discrete')
# operator.create_convolution('s3', operator.quantities['s1'], operator.quantities['s2'], '*')
# operator.create_convolution('s4', operator.quantities['s2'], operator.quantities['s3'], '*')



print('Start by creating quantities')
while True:
    print('create quantity: q')
    print('create a convolution: c')
    print('visualize: v')
    print('inference: i')
    print('list quantities: l')

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
            args_dict['n'] = int(args_dict['n'])
            operator.create_quantity(name=quantity_name, pdf=binom.pmf, cdf=binom.cdf,
                                     sample=binom.rvs, kwargs=args_dict, domain_type='discrete')
        if quantity_model == 'g':
            operator.create_quantity(name=quantity_name, pdf=gamma.pdf, cdf=gamma.cdf,
                                     sample=gamma.rvs, kwargs=args_dict, domain_type='continuous')

        print(f'new quantity created: {quantity_name}')
        print()

    if inp == 'c':
        print(f'available quantities: {list(operator.quantities.keys())}')
        q1 = input('quantity 1 > ')
        q2 = input('quantity 2 > ')
        operation = input('operation (*, +, -, /) > ')
        name = input('convolution name > ')

        operator.create_convolution(conv_name=name, quantity1=operator.quantities[q1],
                                    quantity2=operator.quantities[q2], operation=operation)

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

        if f == 'pdf':
            operator.visualize_quantity(operator.quantities[q].pdf, quantity=operator.quantities[q])
        elif f == 'cdf':
            operator.visualize_quantity(operator.quantities[q].cdf, operator.quantities[q])

    if inp == 'l':
        print(list(operator.quantities.keys()))