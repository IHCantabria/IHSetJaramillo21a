import numpy as np
from .jaramillo21a import jaramillo21a
from IHSetUtils.CoastlineModel import CoastlineModel

class cal_Jaramillo21a_2(CoastlineModel):
    """
    cal_Jaramillo21a_2
    
    Configuration to calibrate and run the Jaramillo et al. (2021a) Shoreline Rotation Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Jaramillo et al. (2021a)',
            mode='calibration',
            model_type='RT',
            model_key='Jaramillo21a'
        )
        self.setup_forcing()

    def setup_forcing(self):
        self.switch_Yini = self.cfg['switch_Yini']
        self.P = self.hs ** 2 * self.tp
        self.P_s = self.hs_s ** 2 * self.tp_s

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]

    def init_par(self, population_size: int):
        if self.switch_Yini == 0:
            lowers = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3])])
            uppers = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3])])
        else:
            lowers = np.array([
                np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]),
                np.log(self.lb[3]), 0.5 * np.min(self.Obs_splited)
            ])
            uppers = np.array([
                np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]),
                np.log(self.ub[3]), 1.5 * np.max(self.Obs_splited)
            ])
        pop = np.zeros((population_size, len(lowers)))
        for i in range(len(lowers)):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers
    
    def model_sim(self, par: np.ndarray) -> np.ndarray:
        a = -np.exp(par[0]); b = par[1]
        Lcw = np.exp(par[2]); Lccw = np.exp(par[3])
        if self.switch_Yini== 0:
            Yini = self.Yini
        else:
            Yini = par[4]
        Ymd, _ = jaramillo21a(self.P_s,
                            self.dir_s,
                            self.dt_s,
                            a,
                            b,
                            Lcw,
                            Lccw,
                            Yini)
        return Ymd[self.idx_obs_splited]
    
    def run_model(self, par: np.ndarray) -> np.ndarray:
        a = par[0]; b = par[1]
        Lcw = par[2]; Lccw = par[3]
        if self.switch_Yini == 0:
            Yini = self.Yini
        else:
            Yini = par[4]
        
        Ymd, _ = jaramillo21a(self.P,
                            self.dir,
                            self.dt,
                            a,
                            b,
                            Lcw,
                            Lccw,
                            Yini)
        return Ymd
    
    def _set_parameter_names(self):
        if self.switch_Yini == 0:
            self.par_names = [r'a', r'b', r'L_cw', r'L_ccw']
        elif self.switch_Yini == 1:
            self.par_names = [r'a', r'b', r'L_cw', r'L_ccw', r'Y_i']
        for idx in [0, 2, 3]:
            self.par_values[idx] = np.exp(self.par_values[idx])
