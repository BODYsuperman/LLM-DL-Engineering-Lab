


import numpy as np
import matplotlib.pyplot as plt

from common.gradient import numerical_diff



def f(x):
    return 0.01*x**2 + 0.1*x


def tangent_line(f, x):
    y = f(x)
    #slope a
    a = numerical_diff(f, x)
    print("slope of the tangent line is:", a)
    #Calculate the intercept based on the tangent passing through (x, y)

    b = y - a*x

    return lambda x: a * x + b

x = np.arange(0, 25, 0.1)
y = f(x)


f_line = tangent_line(f, x= 6)
y_line = f_line(x)
plt.plot(x, y)
plt.plot(x, y_line)
plt.show()