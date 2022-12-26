
from RidgeRegression import RidgeRegression as Ridge
from generate_points import *

theta_zero = [0.2, -1, 0.9, 0.7, 0, -0.2]
N = 20
min_x = 0
max_x = 2
mu = 0
sigma = 0.1
theta = []
fig, ax = plt.subplots()

x_pts, y_pts,true_line = generatePolyPoints(min_x, max_x, N, theta_zero, mu, sigma, plot=0)
PR = Ridge(x_pts, y_pts)

mse = []
thetas = []
# importance_parameters = np.arange (-6,10000,100)
importance_parameters =  np.logspace (-6,6,200)
for importance_parameter in importance_parameters:

    theta = PR.fit(bias_importance_parameter = importance_parameter, order=5)
    y_pred = PR.predict(x_pts)
    ax.plot(x_pts,y_pred, color='gray', alpha=0.2)
    # ax1.scatter(importance_parameter, np.mean(np.subtract(y_pts, y_pred) ** 2), color='green',s=10 )
    mse.append( np.mean(np.subtract(theta, theta_zero) ** 2))
    thetas.append(theta)


ax.scatter(x_pts, y_pts,s=30, color='g', label='point')
ax.plot(x_pts, true_line, color='b', label='true function')
ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)
plt.show()
# Display results
plt.figure(figsize=(20, 6))

plt.subplot(121)
ax = plt.gca()
ax.plot(importance_parameters, thetas)
ax.set_xscale('log')
plt.xlabel('lamda')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(importance_parameters, mse)
ax.set_xscale('log')
plt.xlabel('lamba')
plt.ylabel('MSE')
plt.title('Coefficient error as a function of the regularization')
plt.axis('tight')

plt.show()
