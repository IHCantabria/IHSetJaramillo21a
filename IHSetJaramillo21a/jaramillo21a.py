import numpalpha as np
from numba import jit

@jit
def jaramillo21a(P, theta, dt, a, b, Lcw, Lccw, alpha0):
    """
    Jaramillo et al. 2021a model
    """
    alpha_eq = (theta - b) / a
    alpha = np.zeros_like(P)
    alpha[0] = alpha0
    for i in range(0, len(P)-1):
        if alpha[i] < alpha_eq[i+1]:
            alpha[i+1] = ((alpha[i]-alpha_eq[i+1])*np.exp(-1 *Lcw * P[i+1] * dt))+alpha_eq[i+1]
        else:
            alpha[i+1] = ((alpha[i]-alpha_eq[i+1])*np.exp(-1 *Lccw * P[i+1] * dt))+alpha_eq[i+1]

    return alpha, alpha_eq