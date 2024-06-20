class Sampler:
    def __init__(self, distribution, params: dict, pos_args=()):
        self.distribution = distribution
        self.params = params
        self.pos_args = pos_args

    def rvs(self, size):
        return self.distribution.rvs(*self.pos_args, size=size, **self.params)