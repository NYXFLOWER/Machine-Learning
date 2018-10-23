import numpy as np
import matplotlib.pyplot as plt
import pods
import pandas as pd
import warm.objective_squares_error as se

data = pd.read_csv("F:\mycode\Machine-Learning\warm\olympicMarathonTimes.csv")
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Coordinate Descent
m = -0.4
c = 80
epoch = 100000
pre_objective = se.sum_squares_error(Y, X, m, c)

for i in range(1, epoch):
    c = se.offset_update(Y, X, m)
    m = se.gradient_update(Y, X, c)
    if i % 10 == 0:
        objective = se.sum_squares_error(Y, X, m, c)
        print("iteration ", i, " objective: ", objective)
        if pre_objective - objective < 1e-4:
            break
        pre_objective = objective
    if i % 2000 == 0:
        x_test = np.linspace(1890, 2020, 130)[:, None]
        f_test = m * x_test + c
        plt.plot(x_test, f_test, 'b-')
        plt.plot(X, Y, 'rx')
