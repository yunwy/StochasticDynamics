#!/usr/bin/env python3
import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# Fokker-Planck and Langevin

x = np.load('Langevin.npy')
Nt = len(x)

fig = plt.figure()
ax = plt.gca()
ax.set_xlim((-30, 30))

X = np.arange(-30, 30, 0.1)
Y = np.exp(-X*X*0.5)/np.sqrt(2*np.pi)
FP = ax.plot(X, Y)

bins = np.linspace(-30, 30, 60)
_, _, bar_container = ax.hist(x[0], bins=bins, density=True)


def update(t, bar_container):
    n, _ = np.histogram(x[t], bins=bins, density=True)

    for count, rect in zip(n, bar_container.patches):
        rect.set_height(count)

    return bar_container.patches


anim = functools.partial(update, bar_container=bar_container)
ani = animation.FuncAnimation(fig, anim, Nt, interval=100, repeat=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=2000)
ani.save("Density.mp4", writer=writer)

#plt.show()

