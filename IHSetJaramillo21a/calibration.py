import numpy as np
import xarray as xr
from datetime import datetime
from spotpy.parameter import Uniform
from .jaramillo21a import jaramillo21a
from IHSetCalibration import objective_functions

class cal_Jaramillo21a(object):
    """
    cal_jaramillo21a
    
    Configuration to calibrate and run the Jaramillo et al. (2021a) Shoreline Rotation Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        
        
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))

        cfg = xr.open_dataset(path+'config.nc')
        wav = xr.open_dataset(path+'wav.nc')
        ens = xr.open_dataset(path+'ens.nc')

        self.cal_alg = cfg['cal_alg'].values
        self.metrics = cfg['metrics'].values
        self.dt = cfg['dt'].values
        self.switch_alpha_ini = cfg['switch_alpha_ini'].values

        if self.cal_alg == 'NSGAII':
            self.n_pop = cfg['n_pop'].values
            self.generations = cfg['generations'].values
            self.n_obj = cfg['n_obj'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, n_pop=self.n_pop, generations=self.generations, n_obj=self.n_obj)
        else:
            self.repetitions = cfg['repetitions'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, repetitions=self.repetitions)

        self.Hs = wav['Hs'].values
        self.Tp = wav['Tp'].values
        self.Dir = wav['Dir'].values
        self.time = mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values)
        self.P = self.Hs ** 2 * self.Tp

        self.alpha_obs = ens['Obs'].values
        self.time_obs = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)

        self.start_date = datetime(int(cfg['Ysi'].values), int(cfg['Msi'].values), int(cfg['Dsi'].values))
        self.end_date = datetime(int(cfg['Ysf'].values), int(cfg['Msf'].values), int(cfg['Dsf'].values))

        self.split_data()

        if self.switch_alpha_ini == 0:
            self.alpha_ini = self.alpha_obs_splited[0]

        cfg.close()
        wav.close()
        ens.close()
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        if self.switch_alpha_ini == 0:
            def model_simulation(par):
                a = par['a']
                b = par['b']
                Lcw = par['Lcw']
                Lccw = par['Lccw']

                Ymd, _ = jaramillo21a(self.P_splited,
                                      self.Dir,
                                      self.dt,
                                      a,
                                      b,
                                      Lcw,
                                      Lccw,
                                      self.alpha_ini
                                    )
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                Uniform('a', -1e+2, 1e+2),
                Uniform('b', -1e+3, 1e+3),
                Uniform('Lcw', 1e-9, 1e-3),
                Uniform('Lccw', 1e-9, 1e-3)
            ]
            self.model_sim = model_simulation

        elif self.switch_alpha_ini == 1:
            def model_simulation(par):
                a = par['a']
                b = par['b']
                Lcw = par['Lcw']
                Lccw = par['Lccw']
                alpha_ini = par['alpha_ini']

                Ymd, _ = jaramillo21a(self.P_splited, 
                                      self.Dir,
                                      self.dt,
                                      a,
                                      b,
                                      Lcw,
                                      Lccw,
                                      alpha_ini
                                    )
            
                return Ymd[self.idx_obs_splited]
            self.params = [
                Uniform('a', -1e+2, 1e+2),
                Uniform('b', -1e+3, 1e+3),
                Uniform('Lcw', 1e-9, 1e-3),
                Uniform('Lccw', 1e-9, 1e-3),
                Uniform('alpha_ini', np.min(self.alpha_obs), np.max(self.alpha_obs))
            ]
            self.model_sim = model_simulation

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        idx = np.where((self.time < self.start_date) & (self.time > self.end_date))
        self.idx_validation = idx
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
        self.idx_validation_obs = mkIdx(self.time_obs)

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
        self.idx_calibration = idx
        self.P_splited = self.P[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))
        self.alpha_obs_splited = self.alpha_obs[idx]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.alpha_obs_splited


        


