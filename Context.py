import numpy as np
import scipy
from typing import override

from .Distribution import ConvolutionDistribution
from .Operator import Operator


class Context(Operator):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.quantities = {}
        self.operators = []

    def print_quantities(self):
        for o in self.operators:
            print(f'{o.name} quantities')
            print(f'{[q.name for q in o.quantities]}')

    def get_operator(self, name):
        operator = None
        for o in self.operators:
            if o.name == name:
                operator = o

        return operator

    def create_convolution(self, conv_name, quantity1, quantity2, operation):
        if conv_name not in self.quantities.keys():
            self.quantities[conv_name] = {}

        new_quantity = ConvolutionDistribution(name=conv_name, dist1=quantity1, dist2=quantity2,
                                               conv_operation=operation, parent=self.name)

        self.quantities[conv_name] = new_quantity

