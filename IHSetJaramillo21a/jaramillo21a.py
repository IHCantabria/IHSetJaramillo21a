
# @jit
# def jaramillo21a(P, theta, dt, a, b, Lcw, Lccw, alpha0):
#     """
#     Jaramillo et al. 2021a model
#     """
#     alpha_eq = (theta - b) / a
#     alpha = np.zeros_like(P)
#     alpha[0] = alpha0
#     for i in range(0, len(P)-1):
#         if alpha[i] < alpha_eq[i+1]:
#             alpha[i+1] = ((alpha[i]-alpha_eq[i+1])*np.exp(-1 *Lcw * P[i+1] * dt[i]))+alpha_eq[i+1]
#         else:
#             alpha[i+1] = ((alpha[i]-alpha_eq[i+1])*np.exp(-1 *Lccw * P[i+1] * dt[i]))+alpha_eq[i+1]

#     return alpha, alpha_eq

import numpy as np
from numba import njit
import math

@njit(nopython=True, fastmath=True, cache=True)
def jaramillo21a(P, theta, dt, a, b, Lcw, Lccw, alpha0):
    n = P.shape[0]
    alpha_eq = np.empty(n)
    alpha    = np.empty(n)
    inv_a = 1.0 / a
    lcw   = Lcw
    lccw  = Lccw

    for i in range(n):
        alpha_eq[i] = (theta[i] - b) * inv_a

    alpha[0] = alpha0

    for i in range(n - 1):
        prev  = alpha[i]
        eq    = alpha_eq[i + 1]
        delta = prev - eq
        # exponenciales precomputadas
        exp_cw  = math.exp(-lcw  * P[i+1] * dt[i])
        exp_ccw = math.exp(-lccw * P[i+1] * dt[i])
        # cond=1.0 si prev<eq (uso Lcw), 0.0 si no (uso Lccw)
        cond = 1.0 if prev < eq else 0.0
        # branchless select
        factor = cond * exp_cw + (1.0 - cond) * exp_ccw
        # actualización
        alpha[i + 1] = delta * factor + eq

    return alpha, alpha_eq


def jaramillo21a_njit(P, theta, dt, a, b, Lcw, Lccw, alpha0):
    """
    Jaramillo et al. 2021a model
    """
    alpha_eq = (theta - b) / a
    alpha = np.zeros_like(P)
    alpha[0] = alpha0
    for i in range(0, len(P)-1):
        if alpha[i] < alpha_eq[i+1]:
            alpha[i+1] = ((alpha[i]-alpha_eq[i+1])*np.exp(-1 *Lcw * P[i+1] * dt[i]))+alpha_eq[i+1]
        else:
            alpha[i+1] = ((alpha[i]-alpha_eq[i+1])*np.exp(-1 *Lccw * P[i+1] * dt[i]))+alpha_eq[i+1]

    return alpha, alpha_eq