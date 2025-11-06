import numpy as np
from .jaramillo21a import jaramillo21a
from IHSetUtils.CoastlineModel import CoastlineModel
from typing import Any

class assimilate_Jaramillo21a(CoastlineModel):
    """
    Jaramillo et al. (2021a) shoreline ROTATION model — parameter EnKF assimilation.

    • Yini is fixed to the first observation (no switch_Yini in assimilation).
    • Parameters:
        a   > 0  (use exp transform)
        b   (linear)
        Lcw > 0  (exp)
        Lccw> 0  (exp)
    • model_step/model_step_batch return the LAST state within the current
      assimilation window and carry it forward via a small 'context' dict.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Jaramillo et al. (2021a)',
            mode='assimilation',
            model_type='RT',
            model_key='Jaramillo21a'
        )
        self.setup_forcing()

    # -------------------------
    # Forcing & initial state
    # -------------------------
    def setup_forcing(self):
        # power/forcing terms
        self.P   = (self.hs ** 2) * self.tp
        self.P_s = (self.hs_s ** 2) * self.tp_s

        # FIXED initial shoreline for assimilation (first observation)
        self.Yini = float(self.Obs_splited[0])

    # -------------------------
    # Ensemble initialization
    # -------------------------
    def init_par(self, population_size: int):
        """
        Sample params in TRANSFORM space:
          theta = [ log(a), b, log(Lcw), log(Lccw) ]
        Bounds come from cfg['lb'], cfg['ub'] with same semantics as calibration
        (lb/ub for PHYSICAL space; we log-transform the positive ones here).
        """
        lowers = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3])])
        uppers = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3])])

        pop = np.empty((population_size, lowers.size), dtype=float)
        for i in range(lowers.size):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers

    # -------------------------
    # One EnKF step (single member)
    # -------------------------
    def model_step(self, par: np.ndarray, t_idx: int, context: Any | None = None):
        """
        Map transform-space params -> physical, run model on [i0:i1) of sub-sampled forcing,
        return last shoreline and updated context with that value.
        """
        a    =  np.exp(par[0])
        b    =  par[1]
        Lcw  =  np.exp(par[2])
        Lccw =  np.exp(par[3])

        i0, i1 = self.idx_obs_splited[t_idx-1], self.idx_obs_splited[t_idx]

        if context is None or ('y_old' not in context):
            y0 = float(self.Yini)   # first step starts from initial shoreline
        else:
            y0 = float(context['y_old'])

        if i0 >= i1:
            return np.array([y0], dtype=float), {'y_old': y0}
        else:

            Ymd, _ = jaramillo21a(self.P_s[i0:i1],
                              self.dir_s[i0:i1],
                              self.dt_s[i0:i1],
                              a, b, Lcw, Lccw, y0)

        y_last = float(Ymd[-1])
        return y_last, {'y_old': y_last}

    # -------------------------
    # Batched EnKF step (faster)
    # -------------------------
    def model_step_batch(self,
                         pop: np.ndarray,             # (N, D) in transform space
                         t_idx: int,
                         contexts: list[dict] | None  # len N
                         ):
        N = pop.shape[0]
        y_out   = np.empty((N,), dtype=float)
        new_ctx = [None] * N

        i0, i1 = self.idx_obs_splited[t_idx-1], self.idx_obs_splited[t_idx]
        P_seg   = self.P_s[i0:i1]
        dir_seg = self.dir_s[i0:i1]
        dt_seg  = self.dt_s[i0:i1]

        if i0 >= i1:
            for j in range(N):
                y0 = float(self.Yini) if (contexts is None or contexts[j] is None
                                        or 'y_old' not in contexts[j]) else float(contexts[j]['y_old'])

                y_out[j]   = y0
                new_ctx[j] = {'y_old': y0}
            return y_out, new_ctx
        else:

            for j in range(N):
                pj    = pop[j]
                a     = np.exp(pj[0])
                b     = pj[1]
                Lcw   = np.exp(pj[2])
                Lccw  = np.exp(pj[3])

                y0 = self.Yini if (contexts is None or contexts[j] is None or 'y_old' not in contexts[j]) \
                    else float(contexts[j]['y_old'])

                Ymd, _ = jaramillo21a(P_seg, dir_seg, dt_seg, a, b, Lcw, Lccw, y0)
                y_last = float(Ymd[-1])

                y_out[j]   = y_last
                new_ctx[j] = {'y_old': y_last}

            # EnKF expects (N,) or (N,1) for scalar obs
            return y_out, new_ctx

    # -------------------------
    # Full forward model (for plotting after assimilation)
    # -------------------------
    def run_model(self, par: np.ndarray) -> np.ndarray:
        """
        Run over full forcing using PHYSICAL params (already de-transformed).
        Useful to draw the final red curve from start to end.
        """
        a, b, Lcw, Lccw = par[:4]
        Ymd, _ = jaramillo21a(self.P, self.dir, self.dt, a, b, Lcw, Lccw, float(self.Yini))
        return Ymd

    # -------------------------
    # Names & pretty-printing
    # -------------------------
    def _set_parameter_names(self):
        self.par_names = [r'a', r'b', r'L_cw', r'L_ccw']
        # Convert stored par_values (transform-space) to physical for display
        for idx in [0, 2, 3]:
            self.par_values[idx] = np.exp(self.par_values[idx])
