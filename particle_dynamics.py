#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


x = np.load('Langevin.npy')
Nt = len(x)
dt = 0.25

fig = plt.figure()
ax = plt.gca()
ax.set_xlim((-30, 30))
ax.set_ylim((-10, 10))
ax.set_aspect('equal')

particles = ax.scatter(x[0], np.zeros_like(x[0]))


def update(t):
    x_t = np.column_stack((x[t], np.zeros_like(x[t])))

    particles.set_offsets(x_t)
    print(f"{t*dt:.2f}, {np.mean(x[t]**2)}")

    return particles


ani = animation.FuncAnimation(fig, update, Nt, interval=100, repeat=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=2000)
ani.save("langevin.mp4", writer=writer)

#plt.show()

