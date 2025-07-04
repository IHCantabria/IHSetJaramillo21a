import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from .jaramillo21a import jaramillo21a_njit
import json
from scipy.stats import circmean

class Jaramillo21a_run(object):
    """
    Jaramillo21a_run
    
    Configuration to calibrate and run the Jaramillo et al. (2021a) Shoreline Rotation Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
        self.name = 'Jaramillo et al. (2021a)'
        self.mode = 'standalone'
        self.type = 'RT'
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['run_Jaramillo21a'])
        self.cfg = cfg

        self.switch_Yini = cfg['switch_Yini']

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

        if self.switch_Yini == 1:
            self.Yini = self.Obs[0]


        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))

        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))

        if self.switch_Yini== 0:
            def run_model(par):
                a = par[0]
                b = par[1]
                Lcw = par[2]
                Lccw = par[3]
                Yini = par[4]
                Ymd, _ = jaramillo21a_njit(self.P,
                                    self.dir,
                                    self.dt,
                                    a,
                                    b,
                                    Lcw,
                                    Lccw,
                                    Yini)
                return Ymd

            self.run_model = run_model

        elif self.switch_Yini == 1:

            def run_model(par):
                a = par[0]
                b = par[1]
                Lcw = par[2]
                Lccw = par[3]
                Ymd, _ = jaramillo21a_njit(self.P,
                                    self.dir,
                                    self.dt,
                                    a,
                                    b,
                                    Lcw,
                                    Lccw,
                                    self.Yini)
                return Ymd
            
            self.run_model = run_model


    def run(self, par):
        self.full_run = self.run_model(par)
        if self.switch_Yini == 1:
            self.par_names = [r'a', r'b', r'L_{cw}', r'L_{ccw}']
            self.par_values = par
        elif self.switch_Yini == 0:
            self.par_names = [r'a', r'b', r'L_{cw}', r'L_{ccw}', r'Y_{i}']
            self.par_values = par
        # self.calculate_metrics()

    def calculate_metrics(self):
        self.metrics_names = fo.backtot()[0]
        self.indexes = fo.multi_obj_indexes(self.metrics_names)
        self.metrics = fo.multi_obj_func(self.Obs, self.full_run[self.idx_obs], self.indexes)

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        ii = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.P = self.P[ii]
        self.dir = self.dir[ii]
        self.time = self.time[ii]

        ii = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
        self.Obs = self.Obs[ii]
        self.time_obs = self.time_obs[ii]

        
