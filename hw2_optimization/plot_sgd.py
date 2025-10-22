import numpy as np
import matplotlib.pyplot as plt
import time
import random
import SGDtest
import sgd

# Starting point:
x_0 = -5

# Create plots:
xi, x_next = sgd.sgd(1, 1000, -5)
yvals = SGDtest.fsum(np.asarray(xi))

# plt.figure()
plt.plot(yvals)
# plt.xlim(0, 1000)
plt.xlabel('i')
plt.ylabel('fsum(x(i))')
plt.title('fsum(x(i)) vs i')
plt.grid()
plt.show()