import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 100, 10000)
y1 = 8*np.cos(2*x) + 9*np.cos(3*x) + 7*np.cos(8*x)
y2 = 8*np.cos(2*x) + 9*np.cos(3*x)

plt.figure()
plt.plot(x, y1)
plt.show()
plt.figure()
plt.plot(x, y2)
plt.show()