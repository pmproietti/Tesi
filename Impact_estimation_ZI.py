# this code is used for the estimation of market impact on the ZI model
# (solo temporary per ora)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../main/")
import scipy.stats
from scipy.optimize import minimize
import warnings

import ZI
import MTY
import MTY_vol
from impact_estimation import *

    
data = pd.read_csv('.../data.csv') # file contenente i dati relativi al 26 e 27 Gennaio, opportunamente modificato e 
# aggregato tramite la funzione prep_for_ratio() presente in impact_estimation

data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S.%f')

k_data = glas_temp_imp(data,1000)


rate_l, rate_m, rate_c = ZI.find_parameters(data)
message, order = ZI.sim_LOB(rate_l, rate_m, rate_c, m0 = 20_000,
                          k = 1500 , iterations = 150_000, burn = 10_000, avg_vol = 1)

tot_zi= pd.concat([message, order], axis = 1)

k_ZI = glas_temp_imp(tot_zi,1000)

