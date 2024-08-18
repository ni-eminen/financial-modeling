import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, binom
import time

from Context import Context
from Operator import Operator

def create_global_convolution(ctx):
    print(f'choose operators: ')
    print([o.name for o in ctx.operators])
    op1name = input('operator 1 > ')
    op2name = input('operator 2 > ')

    op1 = ctx.get_operator(op1name)
    op2 = ctx.get_operator(op2name)
    print(f'operator 1 quantities: ')
    print(op1.quantities)
    print()
    print(f'operator 2 quantities:')
    print(op2.quantities)
    print()

    print('choose quantities:')
    q1name = input('quantity 1 > ')
    q2name = input('quantity 2 > ')
    operation = input('operation (*, +, -, /) > ')

    name = input('convolution name > ')
    ctx.create_convolution(conv_name=name, quantity1=op1.quantities[q1name], quantity2=op2.quantities[q2name],
                           operation=operation)
    print(f'global convolution created')
    print()

def create_convolution(operator):
    print(f'available quantities: {list(operator.quantities.keys())}')
    q1 = input('quantity 1 > ')
    q2 = input('quantity 2 > ')
    operation = input('operation (*, +, -, /) > ')
    name = input('convolution name > ')

    operator.create_convolution(conv_name=name, quantity1=operator.quantities[q1],
                                quantity2=operator.quantities[q2], operation=operation)
    print()


def choose_operator(ctx):
    print([o.name for o in ctx.operators])
    inp = input('Choose an operator (name) > ')
    found = False
    operator = None

    if inp == 'global':
        operator = ctx
        return operator

    for o in ctx.operators:
        if o.name == inp:
            found = True
            operator = o
            print()

    if found == False:
        print(f'no operator with name {inp}')
        print('operators:')
        print([o.name for o in ctx.operators])
        print()

    return operator


def create_quantity(operator: Operator):
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


print('program started')

# gamma params
a = 10
b = 2000

# binom params
n = 30
p = .2
# operator.create_quantity('s1', gamma.pdf, cdf=gamma.cdf, sample=gamma.rvs, kwargs={'a': a, 'scale': b},
#                          domain_type='continuous')
# operator.create_quantity('s2', gamma.pdf, cdf=gamma.cdf, sample=gamma.rvs, kwargs={'a': a, 'scale': b},
#                          domain_type='continuous')
# operator.create_quantity('n_sales', binom.pmf, cdf=binom.cdf, sample=binom.rvs, kwargs={'n': 30, 'p': .2},
#                          domain_type='discrete')
# operator.create_convolution('s3', operator.quantities['s1'], operator.quantities['s2'], '*')
# operator.create_convolution('s4', operator.quantities['s2'], operator.quantities['s3'], '*')
operator_selected = False
ctx = Context(name='context')
print('Start by creating quantities')
while True:
    print('create operator: o')
    print('choose operator: oo')
    print('create quantity: q')
    print('create a convolution: c')
    print('create a global quantity: gq')
    print('create a global convolution: gc')
    print('visualize: v')
    print('inference: i')
    print('list quantities: l')

    inp = input('> ')
    if inp in 'qcvil' and not operator_selected:
        print('no operator chosen')
        print()
        continue
    if inp == 'o':
        inp = input('operator name > ')
        ctx.operators.append(Operator(name=inp))
        print(f'operator added: {inp}')
        print()

    if inp == 'oo':
        operator = choose_operator(ctx)
        operator_selected = True

    if inp == 'q':
        create_quantity(operator)

    if inp == 'c':
        create_convolution(operator=operator)

    if inp == 'i':
        print(f'available quantities: {list(operator.quantities.keys())}')
        quantity = input('quantity > ')
        f = input('pdf / cdf > ')
        x = float(input('x > '))

        if f == 'pdf':
            print(operator.quantities[quantity].pdf(x))
        elif f == 'cdf':
            print(operator.quantities[quantity].cdf(x))
        print()

    if inp == 'v':
        print(f'quantities available for {operator.name}')
        q = input('quantity to visualize > ')
        f = input('pdf / cdf > ')

        if f == 'pdf':
            operator.visualize_quantity(operator.quantities[q].pdf, quantity=operator.quantities[q])
        elif f == 'cdf':
            operator.visualize_quantity(operator.quantities[q].cdf, operator.quantities[q])
        print()

    if inp == 'l':
        print(list(operator.quantities.keys()))
        print()

    if inp == 'gq':
        create_quantity(ctx)

    if inp == 'gc':
        create_global_convolution(ctx)
