from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from scipy import linalg

style.use('ggplot')


class RidgeRegression(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def standardize(self, data):
        return (data - np.mean(data)) / (np.max(data) - np.min(data))

    def hypothesis(self, theta, x):
        h = theta[0]
        for i in np.arange(1, len(theta)):
            h += theta[i] * x ** i
        return h

    def computeCost(self, x, y, theta):
        m = len(y)
        h = self.hypothesis(theta, x)
        residuals = h - y

        return (1 / (2 * m)) * np.sum(residuals ** 2), residuals

    def fit(self, order=1, bias_importance_parameter=0.1):

        d = {}
        d['x' + str(0)] = np.ones([1, len(self.x)])[0]
        for i in np.arange(1, order + 1):
            d['x' + str(i)] = self.x ** (i)

        d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        X = np.column_stack(d.values())
        self.Xtr = np.matmul(np.transpose(X), X)
        self.array_size = len(self.Xtr)
        theta = np.matmul(
            np.matmul(
                linalg.pinv(np.matmul(np.transpose(X), X) + bias_importance_parameter * np.identity(self.array_size)),
                np.transpose(X)),
            self.y)
        self.theta = theta
        return self.theta

    def plot_predictedPolyLine(self, test_set_x_values, test_set_y_values):

        plt.figure()
        plt.scatter(test_set_x_values, test_set_y_values, s=30, c='b')
        line = self.theta[0]  # y-intercept
        label_holder = []
        label_holder.append('%.*f' % (2, self.theta[0]))
        for i in np.arange(1, len(self.theta)):
            line += self.theta[i] * test_set_x_values ** i
            label_holder.append(' + ' + '%.*f' % (2, self.theta[i]) + r'$x^' + str(i) + '$')

        plt.plot(test_set_x_values, line, label=''.join(label_holder))
        plt.title('Polynomial Fit: Order ' + str(len(self.theta) - 1))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.show()
        return line

    def predict(self, test_set_x_values):
        line = self.theta[0]  # y-intercept
        for i in np.arange(1, len(self.theta)):
            line += self.theta[i] * test_set_x_values ** i
        return line
