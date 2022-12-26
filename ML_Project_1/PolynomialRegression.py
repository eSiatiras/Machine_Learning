from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from scipy import linalg

style.use('ggplot')


class PolynomialRegression(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def hypothesis(self, theta, x):
        h = theta[0]
        for i in np.arange(1, len(theta)):
            h += theta[i] * x ** i
        return h

    def computeCost(self, x, y, theta):
        m = len(y)
        h = self.hypothesis(theta, x)
        errors = h - y

        return (1 / (2 * m)) * np.sum(errors ** 2), errors

    def fit(self, method='normal_equation', order=1, tol=10 ** -3, numIters=20, learningRate=0.01):

        d = {}
        d['x' + str(0)] = np.ones([1, len(self.x)])[0]
        for i in np.arange(1, order + 1):
            d['x' + str(i)] = self.x ** (i)

        d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        X = np.column_stack(d.values())

        theta = np.matmul(np.matmul(linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), self.y)

        self.method = method
        self.theta = theta

        return self.theta

    def plot_predictedPolyLine(self, test_set_x_values, test_set_y_values,true_model=0,plot=0):
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.figure()
        plt.scatter(test_set_x_values, test_set_y_values, s=20, color='#0000CC',alpha=0.5,zorder=1)
        line = self.theta[0]  # y-intercept
        label_holder = []
        label_holder.append('%.*f' % (2, self.theta[0]))
        for i in np.arange(1, len(self.theta)):
            line += self.theta[i] * test_set_x_values ** i
            label_holder.append(' + ' + '%.*f' % (2, self.theta[i]) + r'$x^' + str(i) + '$')

        plt.plot(test_set_x_values, line, label=''.join(label_holder),zorder=2,c='r')
        if plot>0 :
            plt.plot(test_set_x_values, true_model, label='true function', zorder=2, c='g')
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
