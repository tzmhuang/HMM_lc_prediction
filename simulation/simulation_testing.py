import simulation
import numpy as np
import matplotlib.pyplot as plt
import math

pt1 = [0, 0]
pt2 = [1.5, 0]
pt3 = [1.5, 0]
pt4 = [3, 0]

pts = [pt1, pt2, pt3, pt4]
bcurve = simulation.Bezier(pt1, pt2, pt3, pt4)


v = 0.01
dt = 0.03
a, b, c = bcurve.get_coeff(order=1)
s = [0]
for i in range(10000):
    l = s[-1]**2*a + s[-1]*b + c
    L = np.sqrt(l[0]**2 + l[1]**2)
    s += [s[-1] + v*dt/L]

s = np.array(s)
x, y = bcurve.eval(s, 0)
dx, dy = bcurve.eval(s, 1)

plt.figure()
plt.scatter([x[0] for x in pts], [x[1] for x in pts])
plt.plot(x, y, 'x')

plt.figure()
plt.plot(x, dx)
plt.plot(x, dy)

plt.figure()
plt.plot(x, np.arctan2(dy, dx)*180/math.pi)

plt.figure()
plt.plot(np.sqrt(dx**2 + dy**2))

plt.figure()
plt.plot(np.sqrt(np.diff(x)**2+np.diff(y)**2))
plt.show()
