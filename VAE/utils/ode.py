"""
Collection of ODE systems.

This module provides a collection of ODE systems that can be integrated with
:func:`scipy.integrate.solveivp`.

"""

import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from scipy.integrate import solve_ivp


@jit
def roessler(t, X, a=0.15, b=0.1, c=8.5):
    """Roessler model."""
    x, y, z = X
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return [dx, dy, dz]


@jit
def van_der_pol(t, X):
    """van-der-Pol model."""
    x, y = X
    dx = y
    dy = 5 * (1 - x**2) * y - x
    return [dx, dy]


@jit
def lorenz63(t, X):
    """Lorenz'63 model."""
    x, y, z = X
    dx = 10 * (y - x)
    dy = x * (28 - z) - y
    dz = x * y - 8 / 3 * z
    return [dx, dy, dz]


@jit
def lorenz96a(t, X, F):
    """Lorenz'96 model with single scale."""
    Xm1 = np.roll(X, 1)
    Xm2 = np.roll(X, 2)
    Xp1 = np.roll(X, -1)

    return Xm1 * (Xp1 - Xm2) - X + F


@jit
def lorenz96b(t, Xin, K, J, F, h, b, c):
    """Lorenz'96 model with two scales."""
    X = Xin[:K]
    Y = Xin[K:].reshape((K, J))

    # large scale
    X_m1 = np.roll(X, 1)
    X_m2 = np.roll(X, 2)
    X_p1 = np.roll(X, -1)
    dX = -X_m1 * (X_m2 - X_p1) - X + F

    # small scale
    Y_m1 = np.roll(Y, 1)
    Y_p1 = np.roll(Y, -1)
    Y_p2 = np.roll(Y, -2)
    dY = -c * b * Y_p1 * (Y_p2 - Y_m1) - c * Y

    # coupling between scales
    dX -= h * c / b * np.sum(Y, axis=1)
    dY += h * c / b * X.reshape((K, 1))

    dY = dY.flatten()

    return np.concatenate((dX, dY))


def sample_lorenz96a():
    """Sample run of Lorenz'96 model with single scale."""
    F = 5.25
    K = 21

    N = 2**12
    dt = 0.1
    t_start = -50
    t_eval = np.arange(0, N) * dt

    x0 = F + 0.1 * np.random.randn(K)

    sol = solve_ivp(lorenz96a, [t_start, t_eval[-1]], x0, t_eval=t_eval, method='RK45', args=(F, ))

    x = sol.y
    t = sol.t

    return x, t


def sample_lorenz96b():
    """Sample run of Lorenz'96 model with two scales.

    Parameters from:
        Christensen, H. M., I. M. Moroz, and T. N. Palmer, 2015: Simulating weather regimes: impact of
        stochastic and perturbed parameter schemes in a simple atmospheric model. Climate Dynamics, 44,
        2195–2214, https://doi.org/10.1007/s00382-014-2239-9.
    """
    K = 8
    J = 32
    F = 20
    h = 1
    b = 10
    c = 10

    N = 2**10
    dt = 0.05
    t_start = -5
    t_eval = np.arange(0, N) * dt

    x0 = F + 0.1 * np.random.randn(K)
    y0 = 0.01 * np.random.randn(J * K)

    sol = solve_ivp(lorenz96b, [t_start, t_eval[-1]],
                    np.concatenate((x0, y0)),
                    t_eval=t_eval,
                    method='RK45',
                    args=(K, J, F, h, b, c))

    x = sol.y[:K]
    y = sol.y[K:]
    t = sol.t

    return x, y, t


# =================================================================
# Sample script
# =================================================================
if __name__ == '__main__':
    x, y, t = sample_lorenz96b()

    fig = plt.figure(0)
    fig.clf()
    fig, axs = plt.subplots(4, 1, sharex=True, num=0)

    imx = axs[0].pcolormesh(t, np.arange(len(x)) + 1, x, cmap=plt.cm.seismic, shading='nearest')
    # fig.colorbar(imx, ax=axs[0], fraction=0.05)

    imy = axs[1].pcolormesh(t, np.arange(len(y)) + 1, y, cmap=plt.cm.seismic, shading='nearest')
    # fig.colorbar(imy, ax=axs[1], fraction=0.05)

    axs[2].plot(t, 0.5 * np.sum(x**2, axis=0))
    axs[2].grid('both')

    axs[3].plot(t, x[0, :])
    axs[3].plot(t, y[0, :])
    axs[3].grid('both')
