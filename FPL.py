#!/usr/bin/env python3
import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# Fokker-Planck and Langevin

x = np.load('Langevin.npy')
Nt = len(x)
dt = 0.25

fig = plt.figure()

ax1 = plt.subplot(211)
ax1.set_xlim((-30, 30))
ax1.set_xticks([])

ax2 = plt.subplot(212)
ax2.set_xlim((-30, 30))
ax2.set_ylim((-1, 1))
ax2.set_aspect('equal', adjustable='box')
ax2.set_yticks([])

plt.subplots_adjust(hspace=0.01)



bins = np.linspace(-30, 30, 60)
_, _, bar_container = ax1.hist(x[0], bins=bins, density=True)

X = np.arange(-30, 30, 0.1)
Y = np.exp(-X*X*0.5)/np.sqrt(2*np.pi)
FP = ax1.plot(X, Y)

particles = ax2.scatter(x[0], np.zeros_like(x[0]))


def update(t, bar_container):
    n, _ = np.histogram(x[t], bins=bins, density=True)

    for count, rect in zip(n, bar_container.patches):
        rect.set_height(count)

    x_t = np.column_stack((x[t], np.zeros_like(x[t])))

    particles.set_offsets(x_t)
    print(f"{t*dt:.2f}, {np.mean(x[t]**2)}")

    return bar_container.patches, particles


anim = functools.partial(update, bar_container=bar_container)
ani = animation.FuncAnimation(fig, anim, Nt, interval=100, repeat=False)

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=2000)
#ani.save("Density.mp4", writer=writer)

plt.show()

