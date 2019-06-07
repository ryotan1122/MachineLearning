import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

wine = pd.read_csv("winequality-white.csv", delimiter=";")

y = wine["quality"]

xname = "alcohol"
yname = "sulphates"
zname = "total sulfur dioxide"

plt.style.use('ggplot')
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(xname)
ax.set_ylabel(yname)
ax.set_zlabel(zname)
ax.scatter3D(
    wine[xname],
    wine[yname],
    wine[zname],
    c=y, s=y**2, cmap="cool")
plt.show()
