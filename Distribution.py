import numpy as np
from scipy.stats import rv_histogram, multinomial
import itertools


class Distribution:
    def __init__(self, name, pdf, cdf, sample, kwargs, domain_type, parent):
        self.name = name
        self.domain_type = domain_type
        self.pdf_f = pdf
        self.cdf_f = cdf
        self.kwargs = kwargs
        self.sample_f = sample
        self.domain_max = 0
        self.domain_min = 100
        self.samples = self.sample(1000)
        self.update_domain()
        self.parent = parent
        self.type = "generic"

    def pdf(self, x):
        return self.pdf_f(x, **self.kwargs)

    def cdf(self, x):
        return self.cdf_f(x, **self.kwargs)

    def sample(self, n):
        samples = []
        print(self.kwargs)
        for _ in range(n):
            samples.append(self.sample_f(**self.kwargs))

        return list(samples)

    def update_params(self, kwargs):
        self.kwargs = kwargs
        self.samples = self.sample(1000)
        self.update_domain()

    def update_domain(self):
        a, b = np.min(self.samples), np.max(self.samples)
        self.domain_min = a
        self.domain_max = b

    def generate_image(self, update_samples=False):
        # if update_samples:
        #     self.samples = self.sample(1000)
        #     self.update_domain()
        #
        # pdf_samples = [self.pdf(xi) for xi in self.x]
        # pdf_samples = np.array(pdf_samples) / np.sum(pdf_samples)
        # cdf_samples = [self.cdf(xi) for xi in self.x]
        #
        # image = {
        #     "name": self.name,
        #     "operator": self.parent,
        #     "samples": self.samples,  # Ensure this is a list
        #     "pdf_samples": {
        #         "x": self.x,
        #         "y": list(pdf_samples)  # Convert numpy array to list
        #     },
        #     "cdf_samples": {
        #         "x": self.x,
        #         "y": list(cdf_samples)  # Ensure CDF samples are lists
        #     },
        #     "categories": [],
        #     "domain_type": self.domain_type,
        #     "params": self.kwargs
        # }

        if self.domain_type == 'discrete':
            x = list(range(self.domain_min, self.domain_max + 1))  # Corrected for discrete domain
        elif self.domain_type == 'continuous':
            x = list(np.linspace(self.domain_min, self.domain_max, 1000))

        pdf_samples = [self.pdf(xi) for xi in x]
        pdf_samples = np.array(pdf_samples) / np.sum(pdf_samples)
        cdf_samples = [self.cdf(xi) for xi in x]

        to_return = {
            "name": self.name,
            "type": self.type,
            "operator": self.parent,
            "samples": self.samples,  # Ensure this is a list
            "pdf_samples": {
                "x": x,
                "y": list(pdf_samples)  # Convert numpy array to list
            },
            "cdf_samples": {
                "x": x,
                "y": list(cdf_samples)  # Ensure CDF samples are lists
            },
            "categories": [],
            "domain_type": self.domain_type,
            "params": self.kwargs
        }

        return to_return


class multinomialDistribution:
    def __init__(self, name, sample, kwargs, domain_type, categories, parent):
        self.type = "multinomial"
        self.categories = categories
        self.parent = parent
        self.domain_type = domain_type
        self.x = []
        self.values = kwargs.pop('values')
        self.values = [float(value) for value in self.values]
        self.name = name
        self.pdf_f = multinomial.pmf
        self.kwargs = kwargs
        self.sample_f = sample
        self.samples = self.sample(1000)
        self.update_domain()

    def pdf(self, x):
        x_arr = np.zeros(len(self.kwargs['p']))
        x_arr[x] = 1
        return self.pdf_f(x_arr, n=1, **self.kwargs)

    # TODO: Multinomial cdf inference
    def cdf(self, x):
        return 1

    def sample(self, n):
        samples = []
        for _ in range(n):
            samples.append(self.sample_f(n=1, **self.kwargs))

        return [self.values[np.argmax(sample)] for sample in samples]

    def update_params(self, kwargs):
        self.kwargs = kwargs
        self.samples = self.sample(1000)
        self.update_domain()

    def update_domain(self):
        self.x = list(range(len(self.categories)))

    def generate_image(self, update_samples=False):
        if update_samples:
            self.samples = self.sample(1000)

        pdf_samples = [self.pdf(xi) for xi in self.x]
        pdf_samples = np.array(pdf_samples) / np.sum(pdf_samples)
        cdf_samples = [self.cdf(xi) for xi in self.x]

        image = {
            "name": self.name,
            "type": self.type,
            "operator": self.parent,
            "samples": self.samples,  # Ensure this is a list
            "pdf_samples": {
                "x": self.x,
                "y": list(pdf_samples)  # Convert numpy array to list
            },
            "cdf_samples": {
                "x": self.x,
                "y": list(cdf_samples)  # Ensure CDF samples are lists
            },
            "categories": self.categories,
            "domain_type": self.domain_type
        }

        return image


class ConvolutionDistribution:
    def __init__(self, name, dist1, dist2, parent, conv_operation, categories=None):
        self.categories = categories
        self.parent = parent
        self.name = name
        self.dist1 = dist1
        self.dist2 = dist2
        self.conv_operation = conv_operation
        self.domain_type = 'continuous'
        self.type = "convolution"

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

        hist = np.histogram(self.samples, bins=500)  # TODO: amt of bins, granularity
        self.rv_hist = rv_histogram(hist)

    def pdf(self, x):
        return self.rv_hist.pdf(x)

    def cdf(self, x):
        return self.rv_hist.cdf(x)

    def sample(self, n):
        return self.rv_hist.rvs(size=n)

    def update_params(self, kwargs):
        # update params doesn't work with convolutional distributions
        # we have to keep track of the distributions and params that make up the convolution and update those
        # create the convolution again, sample it and create the new image
        self.kwargs = kwargs
        self.samples = self.sample(1000)

    def update_domain(self):
        a, b = np.min(self.samples), np.max(self.samples)
        self.domain_min = a
        self.domain_max = b

        if self.domain_type == 'discrete':
            self.x = list(range(self.domain_min, self.domain_min + 1))  # Corrected for discrete domain
        elif self.domain_type == 'continuous':
            self.x = list(np.linspace(self.domain_min, self.domain_min, 1000))

    def generate_image(self, update_samples=False):
        if update_samples:
            self.samples = self.sample(1000)

        if self.categories == None:
            self.categories = []

        pdf_samples = [self.pdf(xi) for xi in self.x]
        pdf_samples = np.array(pdf_samples) / np.sum(pdf_samples)
        cdf_samples = [self.cdf(xi) for xi in self.x]

        image = {
            "name": self.name,
            "type": self.type,
            "operator": self.parent,
            "samples": self.samples,  # Ensure this is a list
            "pdf_samples": {
                "x": self.x,
                "y": list(pdf_samples)  # Convert numpy array to list
            },
            "cdf_samples": {
                "x": self.x,
                "y": list(cdf_samples)  # Ensure CDF samples are lists
            },
            "categories": self.categories,
            "domain_type": self.domain_type
        }

        return image
