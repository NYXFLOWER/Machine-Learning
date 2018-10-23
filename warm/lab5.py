import numpy as np
import pods
import pandas as pd
import matplotlib.pyplot as plt


def linear(x, data_limits):
    """
    :param x: np.array((n, 1))
    :param data_limits: array [min, max] of whole data
    :return: np.array [x, 1]
    """
    n_row = x.shape[0]
    span = data_limits[1] - data_limits[0]
    center = (data_limits[1] + data_limits[0]) / 2.
    x_normalized = 2 * (x - center) / span
    Phi = np.column_stack((x_normalized, np.ones((n_row, 1))))

    return Phi


def quadratic(x, data_limits):
    """
    :param x: np.array((n, 1))
    :param data_limits: array [min, max] of whole data
    :return: np.array [1, x, x^2]
    """
    n_row = x.shape[0]
    span = data_limits[1] - data_limits[0]
    center = (data_limits[1] + data_limits[0]) / 2.
    x_normalized = 2 * (x - center) / span
    Phi = np.column_stack((np.ones((n_row, 1)), x_normalized))
    Phi = np.column_stack((Phi, np.square(x_normalized)))

    return Phi


def prediction(w, x, data_limits, linear):
    """
    :param w: np.array (2,1)
    :param x: np.array (n,1)
    :param linear: linear basis function
    :return f: np.array (n,1)
    """
    Phi = linear(x, data_limits)    # (n, 2)
    f = np.dot(Phi, w)

    return f


def objective(w, x, y, data_limits, linear):
    """
    :param w: np.array (2,1)
    :param x: np.array (n,1)
    :param y: np.array (n,1)
    :param linear: linear basis function
    :return: scalar
    """
    diff = np.subtract(y, prediction(w, x, data_limits, linear))
    e = np.dot(diff.T, diff)

    return e


def fit(x, y, data_limits, linear):
    Phi = linear(x, data_limits)
    w = np.linalg.solve(np.dot(Phi.T, Phi), np.dot(Phi.T, y))

    return w


# init data
data = pd.read_csv("F:\mycode\Machine-Learning\warm\olympicMarathonTimes.csv")
n_row = data.shape[0]
x = np.array(data.iloc[:, 0]).reshape((n_row, 1))
y = np.array(data.iloc[:, 1]).reshape((n_row, 1))
data_limits = [x[0], x[-1]]

# fit and predict with linear model
w = fit(x, y, data_limits, linear)
y_linear = prediction(w, x, data_limits, linear)
e_linear = objective(w, x, y, data_limits, linear)
print("The objective is ", np.asscalar(e_linear))

# plot linear model
plt.plot(x, y, 'r*')
plt.plot(x, y_linear, 'b-')
plt.title("The error of the fit on the training data")
plt.show()

# ---------------------------Assignment 2----------------------------- #
# fit and predict with quadratic model
w = fit(x, y, data_limits, quadratic)
y_quadratic = prediction(w, x, data_limits, quadratic)
e_quadratic = objective(w, x, y, data_limits, quadratic)
print("The objective is ", np.asscalar(e_quadratic))

# plot quadratic model
plt.plot(x, y, 'r*')
plt.plot(x, y_quadratic, 'b-')
plt.title("The error of the fit on the training data")
plt.show()

# ---------------------------Assignment 3----------------------------- #
# select indices of data to 'hold out'
indices_hold_out = np.flatnonzero(x>1980)

# Create a training set
x_train = np.delete(x, indices_hold_out, axis=0)
y_train = np.delete(y, indices_hold_out, axis=0)

# Create a hold out set
x_valid = np.take(x, indices_hold_out, axis=0)
y_valid = np.take(y, indices_hold_out, axis=0)

# fit and predict with linear model
w_linear = fit(x_train, y_train, data_limits, linear)
y_linear = prediction(w_linear, x, data_limits, linear)
e_linear = objective(w_linear, x_valid, y_valid, data_limits, linear)
print("The error of the fit on the held out data for linear model is ", np.asscalar(e_linear))

# fit and predict with quadratic model
w_quadratic = fit(x_train, y_train, data_limits, quadratic)
y_quadratic = prediction(w_quadratic, x, data_limits, quadratic)
e_quadratic = objective(w_quadratic, x_valid, y_valid, data_limits, quadratic)
print("The error of the fit on the held out data for quadratic model is ", np.asscalar(e_quadratic))

# plot linear model and quadratic model
plt.figure(1)
plt.plot(x_valid, y_valid, 'r*', label='the held out data')
plt.plot(x_train, y_train, 'b*', label='the train data')
plt.plot(x, y_linear, 'c-', label='linear model')
plt.plot(x, y_quadratic, 'm-', label='quadratic model')
plt.title("The error of the fit on the held out data for both linear and quadratic models")
plt.legend()
plt.show()


# ---------------------------Assignment 4----------------------------- #
def polynomial(x, degree, loc=1956, scale=120):
    """
    :param x:
    :param degree: the actual degree of polynomial
    :param loc:
    :param scale:
    :return: np.array (x.shape[0],degree+1)
    """
    degrees = np.arange(degree+1)
    return ((x-loc)/scale)**degrees


def fit(x, y, degree):
    Phi = polynomial(x, degree)
    w = np.linalg.solve(np.dot(Phi.T, Phi), np.dot(Phi.T, y))

    return w


def prediction(w, x, degree):
    Phi = polynomial(x, degree)
    f = np.dot(Phi, w)

    return f


def objective(w, x, y, degree):
    diff = np.subtract(y, prediction(w, x, degree))
    e = np.dot(diff.T, diff)

    return e


max_degree = 17
error_training = np.zeros(shape=(max_degree+1, 1))
error_validation = np.zeros(shape=(max_degree+1, 1))

for i in range(max_degree+1):
    w = fit(x_train, y_train, i)
    y_predication = prediction(w, x, i)
    error_training[i, 0] = objective(w, x_train, y_train, i)[0, 0]
    error_validation[i, 0] = objective(w, x_valid, y_valid, i)[0, 0]

degrees = np.linspace(0, 17, 18)
plt.figure(figsize=(10, 16))
plt.stem(degrees, np.log(error_training), 'ro--', label='training error')
plt.stem(degrees, np.log(error_validation), label='validation error')
plt.show()


