import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')


def generatePolyPoints(start, stop, num_points, coeff, mu=0, sigma=0, noise=1, plot=1):
    x_pts = np.arange(start, stop, (stop - start) / num_points)
    line = coeff[0]

    for i in np.arange(1, len(coeff)):
        line += coeff[i] * x_pts ** i

    if noise > 0:
        y_pts = np.random.normal(mu, sigma, len(x_pts)) + line
    else:
        y_pts = line

    if plot == 1:  # Plot option
        plt.figure()
        plt.scatter(x_pts, y_pts)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return x_pts, y_pts, line
