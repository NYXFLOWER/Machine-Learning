import pods
import numpy as np
import matplotlib.pyplot as plt

# # # set prior variance on w
# # alpha = 4.
# # # set the order of the polynomial basis set
# # order = 5
# # # set the noise variance
# # sigma2 = 0.01
#
# # # define polynomial baisi
# # loc = 1950.
# # scale = 1.
# # degree = 5.
# # Phi_pred = polynomial(x_pred, degree=degree, loc=loc, scale=scale)
# # Phi = polynomial(x, degree=degree, loc=loc, scale=scale)    # shape(27, 6)
#
# # scale and shift standard normal to Gaussian normal with mu and sigma^2
# # mu = 4 # mean of the distribution
# # alpha = 2 # variance of the distribution
# # w_vec = np.random.normal(size=200)*np.sqrt(alpha) + mu
#
# # create prior sample of w ~ N(0, alpha)
# # K = int(degree) + 1     # dimension of parameter
# # z_vec = np.random.normal(size=K)
# # w_sample = z_vec*np.sqrt(alpha)
# # print(w_sample)
#
# # create function f = \Phi w
# # scale = 100.
# # Phi_pred = polynomial(x_pred, degree=degree, loc=loc, scale=scale)
# # Phi = polynomial(x, degree=degree, loc=loc, scale=scale)
# # f_sample = np.dot(Phi_pred,w_sample)
# # plt.plot(x_pred.flatten(), f_sample.flatten(), 'r-')    # flatten(): change (3, 4) to (12,)
#
#
def polynomial(x, degree, loc, scale):
    degrees = np.arange(degree+1)

    return ((x-loc)/scale)**degrees

#
# import data and set predication set of X
data = pods.datasets.olympic_marathon_men()
x = data['X']
y = data['Y']
# num_data = x.shape[0]
# num_pred_data = 100 # how many points to use for plotting predictions
# x_pred = np.linspace(1890, 2016, num_pred_data)[:, None] # input locations for predictions

# alpha = 2.
# degree = 5.
loc = 1950.
scale = 100
# K = int(degree) + 1     # dimension of parameter
# Phi = polynomial(x, degree=degree, loc=loc, scale=scale)    # shape(27, 6)
#
# # ----------------Question 1----------------- #
# # posterior p(w|y,x) Gaussian
# sigma2 = 0.01
# w_cov = np.linalg.inv((1/sigma2)*np.dot(Phi.T, Phi) + (1/sigma2**0.5)*np.eye(6))    # shapp(6,6)
# w_mean = (1/sigma2) * np.dot(np.dot(w_cov, Phi.T), y)   #shape(6,1)
#
# # ------------------------------------------- #
# # create prior sample of w ~ N(0, alpha)
# z_vec = np.random.normal(size=K)
# w_sample = z_vec*np.sqrt(alpha)
# Phi_pred = polynomial(x_pred, degree=degree, loc=loc, scale=scale) #shape(100,6)
#
# # ----------------Question 2----------------- #
# # marginal likelihood p(y) Gaussian
# # compute mean under posterior density
# f_pred_mean = np.dot(Phi_pred, w_mean)
#
# # plot the predictions
# plt.plot(x_pred.flatten(), f_pred_mean.flatten())
# plt.plot(x, y, 'rx')
#
# # compute mean at the training data and sum of squares error
# f_mean = np.dot(Phi, w_mean)
# sum_squares = np.power(y-f_mean, 2).sum()
# print('The error is: ', sum_squares)
#
#
# # ----------------Question 3----------------- #
# # Write code for you answer to this question in this box
# # Do not delete these comments, otherwise you will get zero for this answer.
# # Make sure your code has run and the answer is correct *before* submitting your notebook for marking.
#
# # Compute variance at function values
# f_pred_var = np.dot(np.dot(Phi, w_cov), Phi.T).diagonal()
# f_pred_std = np.sqrt(f_pred_var)
#
# # plot the mean and error bars at 2 standard deviations above and below the mean
# plt.figure(figsize=(9, 9))
# plt.errorbar(x.tolist(), f_mean.flatten(), yerr=(f_pred_std, f_pred_std),
#              fmt='.', capsize=3, ecolor='g')
# plt.plot(x_pred.flatten(), f_pred_mean.flatten())
# plt.plot(x, y, 'rx')
# plt.title(('the mean function and the error bars for the basis').upper())
# plt.ylabel('Time (s)')
# plt.xlabel('Year')
# plt.show()
#
# # ----------------Question 3----------------- #
# # Write code for you answer to this question in this box
# # Do not delete these comments, otherwise you will get zero for this answer.
# # Make sure your code has run and the answer is correct *before* submitting your notebook for marking.
#
# # select indices of data to 'hold out'
# indices_hold_out = np.flatnonzero(x > 1980)
#
# # Create a training set
# x_train = np.delete(x, indices_hold_out, axis=0)
# y_train = np.delete(y, indices_hold_out, axis=0)
#
# # Create a hold out set
# x_valid = np.take(x, indices_hold_out, axis=0)
# y_valid = np.take(y, indices_hold_out, axis=0)
#
# # Create x for prediction plot
# num_pred_data = 150
# x_pred = np.linspace(1890, 2016, num_pred_data)[:, None]
# sum_squares_plot = np.array(np.zeros(9), dtype=float)
# degree_plot = np.array(range(9), dtype=int)
#
# plt.figure(1)
# for i in range(9):
#     k = i + 1  # the dimension of parameter
#     Phi = polynomial(x_train, degree=i, loc=loc, scale=scale)
#     Phi_valid = polynomial(x_valid, degree=i, loc=loc, scale=scale)
#     Phi_pred = polynomial(x_pred, degree=i, loc=loc, scale=scale)
#
#     # posterior p(w|y,x) Gaussian
#     w_cov = np.linalg.inv((1 / sigma2) * np.dot(Phi.T, Phi) + (1 / sigma2 ** 0.5) * np.eye(k))  # shapp(6,6)
#     w_mean = (1 / sigma2) * np.dot(np.dot(w_cov, Phi.T), y_train)  # shape(6,1)
#
#     # marginal likelihood p(y) Gaussian
#     # compute mean under posterior density
#     f_pred_mean = np.dot(Phi_pred, w_mean)
#
#     # plot the predictions
#     plt.plot(x_pred.flatten(), f_pred_mean.flatten())
#
#     # compute mean at the training data and sum of squares error
#     f_mean = np.dot(Phi_valid, w_mean)
#     sum_squares = np.power(y_valid - f_mean, 2).sum()
#     sum_squares_plot[i] = sum_squares
#
#     print(i, ": ", sum_squares)
#
# plt.plot(x_train, y_train, 'rx')
# plt.plot(x_valid, y_valid, 'bX')
# plt.show()
#
# plt.figure(2)
# plt.stem(degree_plot, sum_squares_plot)
# plt.show()


class Gaussian:

    def __init__(self, x, y, max_degree=8, loc=1950, scale=100, num_pred_data=150):
        self.x = x
        self.y = y
        self.max_degree = max_degree
        self.loc = loc
        self.scale = scale
        self.x_pred = np.linspace(x[0], x[-1], num_pred_data)

    def poly(self, x):
        return polynomial(x, self.degree, self.loc, self.scale).reshape((x.shape[0], self.degree+1))

    def fit(self, sigma2=0.01):
        Phi = self.poly(self.x_train)
        self.w_cov = np.linalg.inv((1 / sigma2) * np.dot(Phi.T, Phi) + (1 / sigma2 ** 0.5) * np.eye(self.degree+1))
        self.w_mean = (1 / sigma2) * np.dot(np.dot(self.w_cov, Phi.T), self.y_train)
        Phi_valid = self.poly(self.x_valid)
        Phi_pred = self.poly(self.x_pred)
        self.f_pred = np.dot(Phi_pred, self.w_mean)
        # Compute variance at function values
        self.f_pred_var = np.dot(np.dot(Phi, self.w_cov), Phi.T).diagonal()
        self.f_pred_std = np.sqrt(self.f_pred_var)
        self.f_pred_mean = np.dot(self.f_pred, self.w_mean)
        self.f_mean = np.dot(Phi_valid, self.w_mean)
        sum_squares = np.power(self.y_valid - self.f_mean, 2).sum()
        return sum_squares

    def hold_out(self, by_year):
        # select indices of data to 'hold out'
        indices_hold_out = np.flatnonzero(self.x > by_year)
        # Create a training set
        self.x_train = np.delete(self.x, indices_hold_out, axis=0)
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 1))
        self.y_train = np.delete(self.y, indices_hold_out, axis=0).reshape((self.y_train.shape[0], 1))
        # Create a hold out set
        self.x_valid = np.take(self.x, indices_hold_out, axis=0).reshape((self.x_valid.shape[0], 1))
        self.y_valid = np.take(self.y, indices_hold_out, axis=0).reshape((self.y_valid.shape[0], 1))

        k = self.max_degree+1
        sum_squares_array = np.array(np.zeros(k), dtype=float)
        for i in range(k):
            self.degree = i
            sum_squares_array[i] = self.fit()
        return sum_squares_array

    def leave_one_out_with_degree(self):
        n_row = self.x.shape[0]
        sum_squares = 0.
        for j in range(n_row):
            self.x_train = np.delete(self.x, j, axis=0).reshape((self.x_train.shape[0], 1))
            self.y_train = np.delete(self.y, j, axis=0).reshape((self.y_train.shape[0], 1))
            self.x_valid = self.x[j].reshape((self.x_valid.shape[0], 1))
            self.y_valid = self.y[j].reshape((self.y_valid.shape[0], 1))
            sum_squares += self.fit()
        return sum_squares/n_row

    def leave_one_out(self):
        dimension = self.max_degree + 1

        sum_squares_array = np.array(np.zeros(k), dtype=float)
        for k in range(dimension):
            self.degree = k
            sum_squares_array[k] = self.leave_one_out_with_degree()
        return sum_squares_array


degree = 8
# degree_array = np.array(range(degree+1), dtype=int)
# model = Gaussian(x, y, degree)
# sum_squares_hold_out = model.hold_out(1980)

n = x.shape[0]
k = degree + 1
sigma2 = 0.01
degree_array = np.array(range(degree + 1), dtype=int)
sum_squares_array = np.array(np.zeros(k), dtype=float)
num_pred_data = 150
x_pred = np.linspace(1890, 2016, num_pred_data)[:, None]

for i in range(9):
    for j in range(n):
        x_train = np.delete(x, j, axis=0)
        y_train = np.delete(y, j, axis=0)
        x_valid = x[j]
        y_valid = y[j]

        k = i + 1  # the dimension of parameter
        Phi = polynomial(x_train, degree=i, loc=loc, scale=scale)
        Phi_valid = polynomial(x_valid, degree=i, loc=loc, scale=scale)
        Phi_pred = polynomial(x_pred, degree=i, loc=loc, scale=scale)

        # posterior p(w|y,x) Gaussian
        w_cov = np.linalg.inv((1 / sigma2) * np.dot(Phi.T, Phi) + (1 / sigma2 ** 0.5) * np.eye(k))  # shapp(6,6)
        w_mean = (1 / sigma2) * np.dot(np.dot(w_cov, Phi.T), y_train)  # shape(6,1)

        f_pred_mean = np.dot(Phi_pred, w_mean)

        # plot the predictions
        plt.figure(i + 10)
        plt.plot(x_pred.flatten(), f_pred_mean.flatten())
        plt.plot(x_train, y_train, 'rx')
        plt.plot(x_valid, y_valid, 'bX')

        # compute mean at the training data and sum of squares error
        f_mean = np.dot(Phi_valid, w_mean)
        sum_squares = np.power(y_valid - f_mean, 2)
        sum_squares_array[i] += sum_squares /n

        f_pred_mean = np.dot(Phi, w_mean)
        f_pred_var = np.dot(np.dot(Phi, w_cov), Phi.T).diagonal()
        f_pred_std = np.sqrt(f_pred_var)
        plt.errorbar(x_train.tolist(), f_pred_mean.flatten(), yerr=f_pred_std, fmt='.', capsize=3, ecolor='g')

    print(i, ": ", sum_squares_array[i])

plt.figure(0)
plt.stem(degree_array, sum_squares_array)
plt.show()
