
from numpy.linalg import norm

from PolynomialRegression import PolynomialRegression as Pregression
from functions import print_data
from generate_points import *

theta_zero = [0.2, -1, 0.9, 0.7, 0, -0.2]
N = 20
min_x = 0
max_x = 2
mu = 0
sigma = 0.1
polynomial_order =2
x_true, y_true ,true_function = generatePolyPoints(min_x, max_x, N, theta_zero ,mu, sigma, plot=0)
fig, ax = plt.subplots()
ax.set(title='Polynomial Fit: Order '+ str(polynomial_order))
experiments = 100
theta = np.zeros((experiments, polynomial_order+1))
y_pred = np.zeros((experiments, 20))
variance = np.zeros((experiments, 1))
mean_vector = np.zeros(N)
for i in range(0, 100):
    x_pts, y_pts,line = generatePolyPoints(min_x, max_x, N, theta_zero, mu, sigma, plot=0)
    PR = Pregression(x_pts, y_pts)
    theta[i] = PR.fit(method='normal_equation', order=polynomial_order)
    ax.plot(x_pts, PR.hypothesis(theta[i], x_pts), color='gray', alpha=0.2)
    # variance[i] = np.mean(PR.hypothesis(theta[i], x_pts)-np.mean(PR.hypothesis(theta[i],x_pts))**2)+sigma
    y_pred[i] = PR.predict(x_pts)
    mean_vector += y_pred[i]
mean_vector /= experiments
bias_2 = norm(mean_vector - y_true) / y_true.size
variance = 0
for p_y in y_pred:
    variance += norm(mean_vector - p_y)
variance /= y_true.size * experiments
error = variance + bias_2

print_data('squared_bias', bias_2)
print_data('variance', variance)
print_data('error_out', error)

ax.scatter(x_true, y_true,s=30, color='g', label='point')
ax.errorbar(x_true, y_true, xerr=0, yerr=error, linestyle="None", color='black',label = 'Error')
ax.plot(x_true, true_function, color='b', label='true function')
ax.plot(x_true, mean_vector, color='r', label='avg g(x)')

ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)
plt.show()
