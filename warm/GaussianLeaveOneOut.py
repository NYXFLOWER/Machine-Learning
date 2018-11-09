import numpy as np
import pods
import matplotlib.pyplot as plt
from warm.GaussianHoldOut import polynomial


class GaussianLeaveOneOut:
    def __init__(self, x, y, basis=polynomial):
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.degree = 0
        self.basis = basis
        self.x_train = 0.
        self.y_train = 0.
        self.x_valid = 0
        self.y_valid = 0.
        self.w_cov = 0.
        self.w_mean = 0.

    def init_leave_one_out(self, ith):
        self.x_train = np.delete(self.x, ith, axis=0)
        self.y_train = np.delete(self.y, ith, axis=0)
        self.x_valid = self.x[ith]
        self.y_valid = self.y[ith]

    def fit(self, sigma_square=0.01, loc=1950, scale=100., alpha=2):
        Phi = self.basis(self.x_train, self.degree, loc, scale)
        k = self.degree + 1
        self.w_cov = np.linalg.inv((1/sigma_square) * np.dot(Phi.T, Phi) + (1/alpha**0.5) * np.eye(k))
        self.w_mean = (1 / sigma_square) * np.dot(np.dot(self.w_cov, Phi.T), self.y_train)

    def prediction(self, x_pred, loc=1950, scale=100.):
        Phi_pred = self.basis(x_pred, self.degree, loc, scale)
        f_pred_mean = np.dot(Phi_pred, self.w_mean)
        return f_pred_mean

    def objective(self, degree):
        self.degree = degree
        sum_error_valid = 0.
        for i in range(self.n):
            self.init_leave_one_out(i)
            self.fit()
            f_valid_mean = self.prediction(x_pred=self.x_valid)
            sum_error_valid += (self.y_valid - f_valid_mean)**2
        return sum_error_valid/self.n

    def variance_std(self, x, loc=1950, scale=100):
        Phi = self.basis(x, self.degree, loc, scale)
        f_var = np.dot(np.dot((Phi, self.w_mean), Phi.T)).diagonal()[:, None]
        return np.sqrt(f_var)


data = pods.datasets.olympic_marathon_men()
x = data['X']
y = data['Y']
num_data = x.shape[0]
max_degree = 8
num_pred_data = 100 # how many points to use for plotting predictions
x_pred = np.linspace(1890, 2016, num_pred_data)[:, None] # input locations for predictions
f_pred_mean = {}
degree_plot = np.array(range(9), dtype=int)[:, None]
error_valid = np.zeros((max_degree+1, 1))

plt.figure(20, figsize=(9, 6))
plt.title('Leave-one-out Error and Hold-out Error on Validation Set for Model with Degree between 0 and 8')
gau_l = GaussianLeaveOneOut(x, y)
for i in range(9):
    error_valid[i, 0] = gau_l.objective(i)
plt.stem(degree_plot, error_valid, 'ro', label='Leave one out')
# plt.stem(degree_plot, sum_squares, label='Hold out')
plt.xlabel('model degree')
plt.ylabel('error on validation set')
plt.yscale('log')
plt.legend()
plt.show()

print("The model with degree ", error_valid.argmin(), " has the smallest error on the validation set")

