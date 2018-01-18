#### Original code from
#### http://www.rorydriscoll.com/2009/01/07/better-sampling/

# Vector3 Sample::CosineSampleHemisphere(float u1, float u2)
# {
#     const float r = Sqrt(u1);
#     const float theta = 2 * kPi * u2;
#
#     const float x = r * Cos(theta);
#     const float y = r * Sin(theta);
#
#     return Vector3(x, y, Sqrt(Max(0.0f, 1 - u1)));
# }

#### Python port
import math

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cosineSampleHemisphere(u1, u2):
    r = math.sqrt(u1)
    theta = 2 * math.pi * u2

    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = math.sqrt(max(0.0, 1.0 - u1))

    return x, y, z


resolution = 100

xs = np.zeros((resolution, resolution), dtype=np.float32)
ys = np.zeros((resolution, resolution), dtype=np.float32)
zs = np.zeros((resolution, resolution), dtype=np.float32)

interval = np.linspace(0, 1, resolution)

x3 = []
y3 = []
z3 = []

for x_i in interval:
    for y_i in interval:
        if y_i == 1 or x_i == 1:
            continue
        x, y, z = cosineSampleHemisphere(x_i, y_i)
        xs[int(y_i * resolution), int(x_i * resolution)] = x
        ys[int(y_i * resolution), int(x_i * resolution)] = y
        zs[int(y_i * resolution), int(x_i * resolution)] = z

        x3.append(x)
        y3.append(y)
        z3.append(z)

fig = plt.figure(1)
ax = fig.add_subplot(221)
ax.imshow(xs, label="x values")

ax = fig.add_subplot(222)
ax.imshow(ys, label="y values")

ax = fig.add_subplot(223)
ax.imshow(zs, label="z values")
fig.legend()

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

cmap = matplotlib.cm.viridis
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

x3 = np.array(x3)
y3 = np.array(y3)
z3 = np.array(z3)
ax.scatter(x3, y3, z3, c=cmap(norm(x3+y3)))
ax.set_zlim(-1,1)

plt.show()
