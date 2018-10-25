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
plt.xlabel('Year')
plt.ylabel('Time(s)')
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
plt.xlabel('Year')
plt.ylabel('Time(s)')
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
plt.xlabel('Year')
plt.ylabel('Time(s)')
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
    # plot for each polynomial degree
    # plt.figure(i+10)
    # plt.plot(x_valid, y_valid, 'r*', label='the held out data')
    # plt.plot(x_train, y_train, 'b*', label='the train data')
    # plt.plot(x, y_predication, 'm-', label='predication model')

# minimum training error & validation error
argmin_training = np.argmin(error_training)
min_training = np.min(error_training)
argmin_validation = np.argmin(error_validation)
min_validation = np.min(error_validation)
print('The polynomial with degree ', argmin_training,
      ' has the minimum training error ', min_training)
print('The polynomial with degree ', argmin_validation,
      ' has the minimum validation error ', min_validation)

degrees = np.linspace(0, 17, 18)
x_ticks = np.linspace(0, 17, 18)
plt.figure(2)
plt.xticks(x_ticks)
plt.xlabel('Polynomial Order')
plt.yscale('log')
plt.ylabel('Log Validation Loss')
plt.stem(degrees, error_training, 'ro--', label='training error')
plt.stem(degrees, error_validation, label='validation error')
plt.title("Log error of both training and validation error")
plt.legend()
plt.show()


# ---------------------------Assignment 5----------------------------- #
# select indices of data - leave one out
def model_estimation_leave_one_out(x, y, degree):
    n_row = x.shape[0]
    validation = 0.
    for i in range(n_row):
        x_train = np.delete(x, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        w = fit(x_train, y_train, degree)
        validation += objective(w, x[i], y[i], degree)
    return validation/n_row


max_degree = 17
average_validation = np.zeros(shape=(max_degree+1, 1))
degrees = np.linspace(0, 17, 18)

# compute leave-one-out cross validation error
print('the leave-one-out cross validation error for basis functions')
for i in range(max_degree+1):
    average_validation[i, 0] = model_estimation_leave_one_out(x, y, i)
    print('basis degree ', i, ': ', average_validation[i, 0])

# plot
plt.figure(3)
plt.xticks(degrees)
plt.yscale('log')
plt.xlabel('Polynomial Order')
plt.ylabel('Log Validation Loss')
plt.stem(degrees, average_validation)
plt.show()


# ---------------------------Assignment 6----------------------------- #
def model_estimation_k_fold_cross(x, y, degree, k):
    n_row = x.shape[0]
    i_x_y = np.random.permutation(np.linspace(0, n_row-1, n_row, dtype=np.int))
    n_extra = n_row % k
    n_each_fold = n_row // k
    validation = 0.
    temp = 0
    for i in range(k):
        if n_extra > 0:
            index = i_x_y[temp: temp + n_each_fold + 1]
            n_extra -= 1
            temp += n_each_fold + 1
        else:
            index = i_x_y[temp: temp + n_each_fold]
            temp += n_each_fold

        x_train = np.delete(x, index, axis=0)
        y_train = np.delete(y, index, axis=0)
        x_valid = np.take(x, index, axis=0)
        y_valid = np.take(y, index, axis=0)
        w = fit(x_train, y_train, degree)
        validation += objective(w, x_valid, y_valid, degree)
    return validation/k


# The model selected by hold out validation
print('-------------------- The model selected by hold out validation --------------------')
argmin_validation = np.argmin(error_validation)
min_validation = np.min(error_validation)
print('The polynomial with degree ', argmin_validation,
      ' has the minimum validation error ', min_validation)
print()

# The model selected by leave-one-out validation
print('-------------------- The model selected by leave-one-out validation --------------------')
argmin_validation = np.argmin(average_validation)
min_validation = np.min(average_validation)
print('The polynomial with degree ', argmin_validation,
          ' has the minimum validation error ', min_validation)
print()


# compute k-fold cross validation error
print('-------------------- The model selected by five-fold cross validation --------------------')
max_degree = 17
k = 5
average_validation_k = np.zeros(shape=(max_degree+1, 1))
degrees = np.linspace(0, 17, 18, dtype=np.int)

# compute k-fold cross validation error
for j in range(10):
    print('the ', j+1, 'th k-fold cross validation error for basis functions')
    for i in range(max_degree + 1):
        average_validation_k[i, 0] = model_estimation_k_fold_cross(x, y, i, k)
    argmin_validation = np.argmin(average_validation_k)
    min_validation = np.min(average_validation_k)
    print('The polynomial with degree ', argmin_validation,
          ' has the minimum validation error ', min_validation)
    print()

print('---------------------------------- Answer ----------------------------------')
print('Both the hold out validation and the leave-one-out validation select the polynomial model with degree 2.')
print('Different five-fold cross validations select different models.')

#     plt.figure(j + 20)
#     plt.xticks(degrees)
#     plt.yscale('log')
#     plt.xlabel('Polynomial Order')
#     plt.ylabel('Log Validation Loss')
#     plt.stem(degrees, average_validation_k)
#
# plt.show()
