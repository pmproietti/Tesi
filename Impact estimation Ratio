# impact estimation on the Ratio model

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
from impact_estimation_colab import *

SPREAD = 10         # Spread
N_DISTR = 5         # Numero di distribuzioni del LO placement da passare al simulatore
HURST = 0.70        # Esponente di hurst da passare in simulazione
LOOK_BACK = 10      # Look back window del regressore logistico

data = pd.read_csv('.../data.csv')# file contenente i dati relativi al 26 e 27 Gennaio, opportunamente modificato e 
# aggregato tramite la funzione prep_for_ratio() presente in impact_estimation

data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S.%f')
k_data = glas_temp_imp(data,1000)

# Ratio calibration

l,m,c = MTY.ratio_orders(data, "Spread")
pp = MTY.compute_ratio(data, "Spread", (-2.0,2.0,-1.0,-1.12,1.09,-0.25))
ff_m = MTY.Ratio(*pp.x)
ff_c = MTY.Ratio(pp.x[3], pp.x[4], pp.x[5],pp.x[0], pp.x[1], pp.x[2])
ff_l = MTY.Ratio_l(*pp.x)
idx = data[data["Datetime"].dt.day >= 27].index.to_list()[0]
df_train = data.iloc[:idx]
df_test = data.iloc[idx:]
df_test.reset_index(inplace=True, drop = True)
lst_distr = []
for j in range(N_DISTR):
  if j != N_DISTR-1:
      ask_d, bid_d = MTY.distance_spread(df_test,j*SPREAD, (j+1) * SPREAD)
      label = f"Spread = [{j*SPREAD}, {(j+1)*SPREAD})"
  else:
      ask_d, bid_d = MTY.distance_spread(df_test,j*SPREAD, 10e10)
      label = f"Spread = [{j*SPREAD}, " + r"$\infty$)"
  distance = np.concatenate((bid_d, ask_d))
  xx = np.arange(-100,100,1)
  hist = np.histogram(distance, xx)
  dist = scipy.stats.rv_histogram(hist)
  for k in range(SPREAD):
      lst_distr.append(dist)
f_distr = MTY.FamilyDistribution(lst_distr)
p_index = MTY.compute_volume_index(data)
pars_index = minimize(MTY.likelihood_tpl, (-2, 1), bounds = ((-np.inf, 0), (0, None)),
                              args = (p_index))
alpha, scale = pars_index.x
err_a, err_s = np.sqrt(pars_index.hess_inv.todense()).diagonal()
idx_buy  = data[(data.Type == "Market") & (data.Sign == 1)].index.to_numpy()
idx_sell = data[(data.Type == "Market") & (data.Sign == -1)].index.to_numpy()
best_buy  = data['Ask Size1'].iloc[idx_buy -1].to_numpy()
best_sell = data['Bid Size1'].iloc[idx_sell -1].to_numpy()
ratio_buy  = data.Volume.iloc[idx_buy] / best_buy
ratio_sell = data.Volume.iloc[idx_sell] / best_sell
ratio = np.concatenate((ratio_buy, ratio_sell))
xx = np.linspace(0,6,101)
hist = np.histogram(ratio[ratio != 1.], xx)
mo_vol = scipy.stats.rv_histogram(hist)
delta = ratio[ratio == 1.].shape[0] / ratio.shape[0]

# simulation

mess, ordw = MTY_vol.sim_LOB(ff_l, ff_m, ff_c, alpha, scale, f_distr, lo_volume,
                       mo_vol, delta, m0 = 20_000, k = 10_000,
                       iterations = 100_000, n_tot = 50,
                       burn  = 10_000, energy =  False, hurst = HURST)
tot_ratio = pd.concat([mess, ordw], axis = 1)

k_Ratio = glas_temp_imp(tot_Ratio,1000)
