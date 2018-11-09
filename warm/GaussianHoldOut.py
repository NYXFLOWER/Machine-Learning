import numpy as np
import pods
import matplotlib.pyplot as plt





# plt
# sigma2 = 0.01
   # shape(6,1)

# f_pred_mean = np.dot(Phi_pred, w_mean)

# plt.figure(0)
# plt.plot(x_pred, f_pred_mean, label='The Mean Function')
# plt.plot(x, y, 'rx', label='Observation')
# plt.title('The Predictions Function')
# plt.xlabel('year')
# plt.ylabel('time (h)')

# f_mean = np.dot(Phi, w_mean)
# sum_squares = np.sum((y - f_mean)**2)

# compute variance at function values
# f_pred_var = np.dot(np.dot(Phi_pred, w_cov), Phi_pred.T).diagonal()[:, None]
# f_pred_std = np.sqrt(f_pred_var)

# plt.figure(1)
# plt.title('The Mean Function and The Error Bars for The Basis')
# plt.plot(x_pred, f_pred_mean, label='The Mean Function')
# plt.plot(x, y, 'rx', label='Observation')
# plt.errorbar(x_pred, f_pred_mean, yerr=f_pred_std, fmt='.', capsize=3, ecolor='g', label='The Error Bar')
# plt.xlabel('year')
# plt.ylabel('time (h)')
# plt.legend()

def polynomial(x, degree, loc, scale):
    degrees = np.arange(degree+1)
    return ((x-loc)/scale)**degrees


class GaussianHoldOut:
    def __init__(self, x, y, year=1980, basis=polynomial):
        self.x = x
        self.y = y
        self.year = year
        self.basis = basis
        self.x_train, self.y_train, self.x_valid, self.y_valid = self.init_hold_out()
        self.w_mean = 0.
        self.w_cov = 0.
        self.degree = 0

    def init_hold_out(self):
        # select indices of data to 'hold out'
        indices = np.flatnonzero(x > self.year)
        return np.delete(x,indices,axis=0), np.delete(y,indices,axis=0), np.take(x,indices,axis=0), np.take(y,indices,axis=0)

    def fit(self, degree, loc=1950., scale=100., sigma2=0.01, alpha=2):
        Phi = self.basis(self.x_train, degree, loc, scale)
        self.degree = degree
        k = degree + 1
        self.w_cov = np.linalg.inv((1 / sigma2) * np.dot(Phi.T, Phi) + (1 / alpha ** 0.5) * np.eye(k))
        self.w_mean = (1 / sigma2) * np.dot(np.dot(self.w_cov, Phi.T), self.y_train)

    def prediction(self, x_pred, loc=1950., scale=100.):
        Phi_pred = self.basis(x_pred, self.degree, loc, scale)
        f_pred_mean = np.dot(Phi_pred, self.w_mean)
        return f_pred_mean

    def objective(self):
        f_valid_mean = self.prediction(x_pred=self.x_valid)
        sum_squares = np.sum((self.y_valid - f_valid_mean)**2)
        return sum_squares

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
sum_squares = np.zeros((max_degree+1, 1))

plt.figure(10, figsize=(9, 6))
plt.title("The Prediction for Different Model Orders between 0 and 8")
gau = GaussianHoldOut(x, y)
for i in range(9):
    gau.fit(i)
    f_pred_mean[i] = gau.prediction(x_pred=x_pred)
    sum_squares[i, 0] = gau.objective()
    plt.plot(x_pred, f_pred_mean[i], label=('Model with Degree ' + str(i)))

plt.plot(gau.x_train, gau.y_train, 'bx', label='Training Set')
plt.plot(gau.x_valid, gau.y_valid, 'r.', label='Hold-out Validation Set')
plt.xlabel('year')
plt.ylabel('time / h')
plt.legend()
plt.show()

plt.figure(11, figsize=(9, 6))
plt.title('The Sum Square on the Validation Set for Different Model Orders between 0 and 8')
plt.stem(degree_plot, sum_squares)
plt.yscale('log')
plt.xlabel('degree')
plt.ylabel('error')
plt.show()

print("The selected model which has the maximum likelihood from last week by three validations in last week is the polynomial model with degree 2. \n")
print("According to the figure, it is clear that the model with degree 2 is the best fitting model this time. Its degree is the same as that of the model most likely to be selected in last week.")
