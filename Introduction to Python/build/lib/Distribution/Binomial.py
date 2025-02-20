import math
import matplotlib.pyplot as plt
from .Distribution import Distribution


class Binomial(Distribution):

    def __init__(self, prob=.5, size=20):

        self.n = size
        self.p = prob

        Distribution.__init__(self, self.calculate_mean(), self.calculate_stdev())

    def calculate_mean(self):
        self.mean = self.p * self.n
        return self.mean

    def calculate_stdev(self):

        self.stdev = math.sqrt(self.n * self.p * (1 - self.p))

        return self.stdev

    def replace_stats_with_data(self):
        self.n = len(self.data)
        self.p = 1.0 * sum(self.data) / len(self.data)
        self.mean = self.calculate_mean()
        self.stdev = self.calculate_stdev()

        return self.p, self.n

    def plot_bar(self):
        plt.bar(x=['0', '1'], height=[(1 - self.p) * self.n, self.p * self.n])
        plt.title('Bar Chart of Data')
        plt.xlabel('outcome')
        plt.ylabel('count')

    def pdf(self, k):
        a = math.factorial(self.n) / (math.factorial(k) * (math.factorial(self.n - k)))
        b = (self.p ** k) * (1 - self.p) ** (self.n - k)

        return a * b

    def plot_bar_pdf(self):
        x = []
        y = []

        # calculate the x values to visualize
        for i in range(self.n + 1):
            x.append(i)
            y.append(self.pdf(i))

        # make the plots
        plt.bar(x, y)
        plt.title('Distribution of Outcomes')
        plt.ylabel('Probability')
        plt.xlabel('Outcome')

        plt.show()

        return x, y

    def __add__(self, other):
        try:
            assert self.p == other.p, 'p values are not equal'
        except AssertionError as error:
            raise

        result = Binomial()
        result.n = self.n + other.n
        result.p = self.p
        result.calculate_mean()
        result.calculate_stdev()

        return result

    def __repr__(self):
        return "mean {}, standard deviation {}, p {}, n {}". \
            format(self.mean, self.stdev, self.p, self.n)
