from utils.ICP import icp
from utils.UtilsPointcloud import random_sampling
from tf import Tf2D

import matplotlib.pyplot as plt
import numpy as np

data1 = np.load("data1.npy")
data2 = np.load("data2.npy")

def polar2xy(data):
    return np.array([ [r*np.cos(np.radians(theta)), r*np.sin(np.radians(theta))] for theta, r in data ])

def sample(data, npoints):
    n = data.shape[0]
    assert npoints <= n
    return np.take(data, np.linspace(0, n, points, endpoint=False, dtype=int), axis=0)

coord1 = polar2xy(data1)
coord2 = polar2xy(data2)

#points = min(coord1.shape[0], coord2.shape[0])
points = 300
print(f"Points: {points}")

A = sample(coord1, points)
B = sample(coord2, points)

initial = Tf2D((-100, 0), np.radians(7))

T, e, i = icp(initial(coord1), coord2, max_iterations=10000, tolerance=0.00001)

print(np.mean(e), i)

tf = Tf2D.from_matrix(T)
#tf = initial

#coord1_new = np.array([ [x,y,1] for x, y in coord1 ]).T
coord1_new = tf(coord1)

print(tf)

plt.scatter(coord1_new[:, 0], coord1_new[:, 1], color="red", marker=".")
plt.scatter(coord2[:, 0], coord2[:, 1], color="blue", marker=".")

plt.show()