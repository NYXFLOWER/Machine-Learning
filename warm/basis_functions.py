import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import pods
from ipywidgets import *


# Run in ipython or Jupyter Notebook
def display_prediction(basis, num_basis=4, wlim=(-1., 1.), fig=None, ax=None, xlim=None, ylim=None,
                       num_points=1000, offset=0.0, **kwargs):
    """Interactive widget for displaying a prediction function based on summing separate basis functions.
    :param basis: a function handle that calls the basis functions.
    :type basis: function handle.
    :param xlim: limits of the x axis to use.
    :param ylim: limits of the y axis to use.
    :param wlim: limits for the basis function weights."""
    if fig is not None:
        if ax is None:
            ax = fig.gca()
    if xlim is None:
        if ax is not None:
            xlim = ax.get_xlim()
        else:
            xlim = (-2., 2.)
    if ylim is None:
        if ax is not None:
            ylim = ax.get_ylim()
        else:
            ylim = (-1., 1.)

    # initialise X and set up W arguments.
    x = np.zeros((num_points, 1))
    x[:, 0] = np.linspace(xlim[0], xlim[1], num_points)
    param_args = {}
    for i in range(num_basis):
        lim = list(wlim)
        if i == 0:
            lim[0] += offset
            lim[1] += offset
        param_args['w_' + str(i)] = tuple(lim)

    # helper function for making basis prediction.
    def predict_basis(w, basis, x, num_basis, **kwargs):
        Phi = basis(x, num_basis, **kwargs)
        f = np.dot(Phi, w)
        return f, Phi

    if type(basis) is dict:
        use_basis = basis[list(basis.keys())[0]]
    else:
        use_basis = basis
    f, Phi = predict_basis(np.zeros((num_basis, 1)),
                           use_basis, x, num_basis,
                           **kwargs)
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

    predline = ax.plot(x, f, linewidth=2)[0]
    basislines = []
    for i in range(num_basis):
        basislines.append(ax.plot(x, Phi[:, i], 'r')[0])

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    def generate_function(basis, num_basis, predline, basislines, basis_args, display_basis, offset,
                          **kwargs):
        w = np.zeros((num_basis, 1))
        for i in range(num_basis):
            w[i] = kwargs['w_' + str(i)]
        f, Phi = predict_basis(w, basis, x, num_basis, **basis_args)
        predline.set_xdata(x[:, 0])
        predline.set_ydata(f)
        for i in range(num_basis):
            basislines[i].set_xdata(x[:, 0])
            basislines[i].set_ydata(Phi[:, i])

        if display_basis:
            for i in range(num_basis):
                basislines[i].set_alpha(1)  # make visible
        else:
            for i in range(num_basis):
                basislines[i].set_alpha(0)
        display(fig)

    if type(basis) is not dict:
        basis = fixed(basis)

    plt.close(fig)
    interact(generate_function,
             basis=basis,
             num_basis=fixed(num_basis),
             predline=fixed(predline),
             basislines=fixed(basislines),
             basis_args=fixed(kwargs),
             offset=fixed(offset),
             display_basis=False,
             **param_args)


def polynomial(x, num_basis=4, data_limits=[-1., 1.]):
    "Polynomial basis"
    centre = data_limits[0]/2. + data_limits[1]/2.
    span = data_limits[1] - data_limits[0]
    z = x - centre
    z = 2*z/span
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i+1] = z**i
    return Phi


def fourier(x, num_basis=4, data_limits=[-2., 2.]):
    "Fourier basis"
    tau = 2*np.pi
    span = float(data_limits[1]-data_limits[0])
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        count = float((i+1)//2)
        frequency = count/span
        if i % 2:
            Phi[:, i:i+1] = np.sin(tau*frequency*x)
        else:
            Phi[:, i:i+1] = np.cos(tau*frequency*x)
    return Phi


def relu(x, num_basis=4, data_limits=[-1., 1.], gain=None):
    "Rectified linear units basis"
    if num_basis>2:
        centres=np.linspace(data_limits[0], data_limits[1], num_basis)
    else:
        centres = np.asarray([data_limits[0]/2. + data_limits[1]/2.])
    if gain is None:
        gain = np.ones(num_basis-1)
    Phi = np.zeros((x.shape[0], num_basis))
    # Create the bias
    Phi[:, 0] = 1.0
    for i in range(1, num_basis):
        Phi[:, i:i+1] = (gain[i-1]*x>centres[i-1])*(x-centres[i-1])
    return Phi


def radial(x, num_basis=4, data_limits=[-1., 1.]):
    "Radial basis constructed using exponentiated quadratic form."
    if num_basis > 1:
        centres = np.linspace(data_limits[0], data_limits[1], num_basis)
        width = (centres[1] - centres[0]) / 2.
    else:
        centres = np.asarray([data_limits[0] / 2. + data_limits[1] / 2.])
        width = (data_limits[1] - data_limits[0]) / 2.

    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i + 1] = np.exp(-0.5 * ((x - centres[i]) / width) ** 2)
    return Phi


def compute_square_error(y, y_approx):
    square = np.square(y - y_approx)
    return square.sum()


# initial data
data = pods.datasets.olympic_marathon_men()
y = data['Y']
x = data['X']

# normalize y
y -= y.mean()
y /= y.std()

# design matrix on x with 4 dimensions radial basis
phi = radial(x, num_basis=4, data_limits=[x[0], x[-1]])  # z shape (n, 4)

# compute the model parameters w
# phii = np.dot(phi.T, phi)
# wwww = np.dot(np.linalg.inv(phii), phi.T)
# w = np.dot(wwww, y)
Q, R = np.linalg.qr(phi)
w = sp.linalg.solve_triangular(R, np.dot(Q.T, y))

# y of approximated function
y_approx = np.dot(phi, w)
error_app = compute_square_error(y, y_approx)
print("The square error of radial basis function (computed): ", error_app)

# y of observed function
w_rbf = np.array([[1.9], [0], [0], [-1]])
y_rbf = np.dot(phi, w_rbf)
error_rbf = compute_square_error(y, y_rbf)
print("The square error of radial basis function (observed): ", error_rbf)

# plot both to compare
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(x, y, 'rx')
ax.plot(x, y_approx, 'b-', label='fit by observing')
ax.plot(x, y_rbf, 'g--', label='fit by computing')
ax.legend()

# plot and compare
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(x, y, 'rx')
display_prediction(basis=dict(radial=radial, polynomial=polynomial, fourier=fourier, relu=relu),
                                 ylim=[-2.0, 4.],
                                 data_limits=(1888, 2020),
                                 fig=fig1, ax=ax1,
                                 offset=0.,
                                 wlim=(-4, 4),
                                 num_basis=4)
