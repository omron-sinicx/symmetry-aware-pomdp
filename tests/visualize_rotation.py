from math import cos, sin

import numpy as np
from matplotlib import pyplot

theta = np.deg2rad(120)

rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

v = np.array([1, -1])
w = np.array([2, -2])

v2 = np.dot(rot, v)
w2 = np.dot(rot, w)

theta = np.deg2rad(240)

rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

v3 = np.dot(rot, v)
w3 = np.dot(rot, w)

axes = pyplot.gca()

axes.set_xlim([-6, 6])
axes.set_ylim([-6, 6])

c1 = pyplot.Circle((0, 0), 1, fill=False)
c2 = pyplot.Circle((0, 0), 5, fill=False)

axes.set_aspect(1)
axes.add_artist(c1)
axes.add_artist(c2)

x = np.array([v[0], w[0], v2[0], w2[0], v3[0], w3[0]])
y = np.array([v[1], w[1], v2[1], w2[1], v3[1], w3[1]])

pyplot.scatter(x, y)
pyplot.show()