"""Lorenz39/35 recordings used in the paper."""

import numpy as np
from numba import njit

DT = 0.02


@njit(cache=True)
def stationary_field(z, q, g):
    d = np.empty(9)
    b = 1 / q
    d[0] = q * 10 * (z[1] - z[0])
    d[1] = q * (39 * z[0] - z[1] - z[0] * z[2])
    d[2] = q * (z[0] * z[1] - 8 / 3 * z[2])
    for j in [3, 6]:
        d[j] = b * 10 * (z[j + 1] - z[j]) + g * (z[0] - z[j])
        d[j + 1] = b * (35 * z[j] - z[j + 1] - z[j] * z[j + 2])
        d[j + 2] = b * (z[j] * z[j + 1] - 8 / 3 * z[j + 2])
    return d


@njit(cache=True)
def integrate_stationary(z, pars, h=0.002):
    nsub = int(round(0.02 / h))
    h = 0.02 / nsub
    out = np.empty((len(pars) - 1, 9))
    for i in range(len(out)):
        for j in range(nsub):
            p = pars[i] + (pars[i + 1] - pars[i]) * j / nsub
            pm = pars[i] + (pars[i + 1] - pars[i]) * (j + 0.5) / nsub
            pe = pars[i] + (pars[i + 1] - pars[i]) * (j + 1) / nsub
            a = stationary_field(z, p[0], p[1])
            b = stationary_field(z + h * a / 2, pm[0], pm[1])
            c = stationary_field(z + h * b / 2, pm[0], pm[1])
            d = stationary_field(z + h * c, pe[0], pe[1])
            z = z + h * (a + 2 * b + 2 * c + d) / 6
        out[i] = z
    return out


@njit(cache=True)
def switch_field(z, q, gf, gr):
    d = np.empty(6)
    b = 1 / q
    d[0] = q * 10 * (z[1] - z[0]) + gr * (z[3] - z[0])
    d[1] = q * (39 * z[0] - z[1] - z[0] * z[2])
    d[2] = q * (z[0] * z[1] - 8 / 3 * z[2])
    d[3] = b * 10 * (z[4] - z[3]) + gf * (z[0] - z[3])
    d[4] = b * (35 * z[3] - z[4] - z[3] * z[5])
    d[5] = b * (z[3] * z[4] - 8 / 3 * z[5])
    return d


@njit(cache=True)
def integrate_switch(initial, q, gf, gr, count, step=0.001):
    z = initial.copy()
    out = np.empty((count, 6))
    for i in range(count):
        for _ in range(round(DT / step)):
            a = switch_field(z, q, gf, gr)
            b = switch_field(z + step * a / 2, q, gf, gr)
            cc = switch_field(z + step * b / 2, q, gf, gr)
            dd = switch_field(z + step * cc, q, gf, gr)
            z = z + step * (a + 2 * b + 2 * cc + dd) / 6
        out[i] = z
    return out
