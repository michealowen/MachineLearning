import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r = np.linspace(1,10,100)
x = r
t = np.linspace(1,10,100)
y = r
z = (1 - 2*y - 3*x)/4
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z, label='parametric curve')
ax.legend()
 
plt.show()