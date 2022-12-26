from PolynomialRegression import PolynomialRegression as Pregression
from generate_points import *

true_theta = [0.2, -1, 0.9, 0.7, 0, -0.2]
N = 20
min_x = 0
max_x = 2
mu = 0
sigma = 0.1

x_pts, y_pts,true_model = generatePolyPoints(min_x, max_x, N, true_theta, mu, sigma, plot=1)

PR = Pregression(x_pts, y_pts)
# apply the Least Squares method to estimate the parameter vector
theta_zero = PR.fit(method='normal_equation', order=5)
y_pred_small_training_set=PR.plot_predictedPolyLine(x_pts, y_pts,true_model, plot=1)
mse_small_training_set = np.mean(np.subtract(y_pred_small_training_set,y_pts)**2)
pts = 2 *  np.random.random_sample((1000, 2))
x_ptss = pts[:, 0]
y_ptss = pts[:, 1]
PR = Pregression(x_ptss, y_ptss)
theta_1000 = PR.fit(method='normal_equation', order=5)
y_pred_large_training_set=PR.plot_predictedPolyLine(x_ptss, y_ptss)
# Calculate the Mean Square Error of y over the training set
mse_test_set = np.mean((y_pred_large_training_set - y_ptss) ** 2)
