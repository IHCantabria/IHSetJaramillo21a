import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from .jaramillo21a import jaramillo21a
import json
from scipy.stats import circmean

class cal_Jaramillo21a_2(object):
    """
    cal_Jaramillo21a_2
    
    Configuration to calibrate and run the Jaramillo et al. (2021a) Shoreline Rotation Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        self.name = 'Jaramillo et al. (2021a)'
        self.mode = 'calibration'
        self.type = 'RT'

     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['Jaramillo21a'])
        self.cfg = cfg

        self.cal_alg = cfg['cal_alg']
        self.metrics = cfg['metrics']
        self.switch_Yini = cfg['switch_Yini']
        self.lb = cfg['lb']
        self.ub = cfg['ub']

        self.calibr_cfg = fo.config_cal(cfg)

        self.hs = np.mean(data.hs.values, axis=1)
        self.time = pd.to_datetime(data.time.values)
        self.tp = np.mean(data.tp.values, axis=1)
        self.dir = circmean(data.dir.values, axis=1, high=360, low=0)
        self.P = self.hs ** 2 * self.tp
        self.Obs = data.rot.values
        self.Obs = self.Obs[~data.mask_nan_rot]
        self.time_obs = pd.to_datetime(data.time_obs.values)
        self.time_obs = self.time_obs[~data.mask_nan_rot]
        
        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])
        
        data.close()

        self.split_data()

        if self.switch_Yini == 0:
            self.Yini = self.observations[0]


        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))

        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))
        
        mkDTsplited = np.vectorize(lambda i: (self.time_splited[i+1] - self.time_splited[i]).total_seconds()/3600)
        self.dt_splited = mkDTsplited(np.arange(0, len(self.time_splited)-1))

        if self.switch_Yini== 0:
            # @jit
            def model_simulation(par):
                a = np.exp(par[0])
                b = par[1]
                Lcw = np.exp(par[2])
                Lccw = np.exp(par[3])
                Ymd, _ = jaramillo21a(self.P_splited,
                                    self.dir_splited,
                                    self.dt_splited,
                                    a,
                                    b,
                                    Lcw,
                                    Lccw,
                                    self.Yini)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            def run_model(par):
                a = np.exp(par[0])
                b = par[1]
                Lcw = np.exp(par[2])
                Lccw = np.exp(par[3])
                Ymd, _ = jaramillo21a(self.P,
                                    self.dir,
                                    self.dt,
                                    a,
                                    b,
                                    Lcw,
                                    Lccw,
                                    self.Yini)
                return Ymd

            self.run_model = run_model

            # @jit
            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3])])
                log_upper_bounds = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3])])
                population = np.zeros((population_size, 4))
                for i in range(4):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 1:
            def model_simulation(par):
                a = np.exp(par[0])
                b = par[1]
                Lcw = np.exp(par[2])
                Lccw = np.exp(par[3])
                Yini = np.exp(par[4])
                Ymd, _ = jaramillo21a(self.P_splited,
                                    self.dir_splited,
                                    self.dt_splited,
                                    a,
                                    b,
                                    Lcw,
                                    Lccw,
                                    Yini)

                return Ymd[self.idx_obs_splited]
            
            self.model_sim = model_simulation

            def run_model(par):
                a = np.exp(par[0])
                b = par[1]
                Lcw = np.exp(par[2])
                Lccw = np.exp(par[3])
                Yini = np.exp(par[4])
                Ymd, _ = jaramillo21a(self.P,
                                    self.dir,
                                    self.dt,
                                    a,
                                    b,
                                    Lcw,
                                    Lccw,
                                    Yini)
                return Ymd
            
            self.run_model = run_model

            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3]), np.log(0.75*np.min(self.Obs))])
                log_upper_bounds = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3]), np.log(1.25*np.max(self.Obs))])
                population = np.zeros((population_size, 5))
                for i in range(5):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        ii = np.where(self.time>=self.start_date)[0][0]
        self.P = self.P[ii:]
        self.dir = self.dir[ii:]
        self.time = self.time[ii:]

        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))[0]
        self.idx_validation = idx
        
        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.idx_calibration = idx
        self.P_splited = self.P[idx]
        self.dir_splited = self.dir[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
        self.Obs_splited = self.Obs[idx]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.Obs_splited

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))
        self.idx_validation_obs = idx[0]
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []

    def calibrate(self):
        """
        Calibrate the model.
        """
        self.solution, self.objectives, self.hist = self.calibr_cfg.calibrate(self)

        self.full_run = self.run_model(self.solution)

        if self.switch_Yini == 0:
            self.par_names = [r'a', r'b', r'L_{cw}', r'L_{ccw}']
            self.par_values = self.solution.copy()
            self.par_values[0] = np.exp(self.par_values[0])
            self.par_values[2] = np.exp(self.par_values[2])
            self.par_values[3] = np.exp(self.par_values[3])
        elif self.switch_Yini == 1:
            self.par_names = [r'a', r'b', r'L_{cw}', r'L_{ccw}', r'Y_{i}']
            self.par_values = self.solution.copy()
            self.par_values[0] = np.exp(self.par_values[0])
            self.par_values[2] = np.exp(self.par_values[2])
            self.par_values[3] = np.exp(self.par_values[3])
            self.par_values[4] = np.exp(self.par_values[4])            