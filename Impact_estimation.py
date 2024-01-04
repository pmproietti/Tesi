from fbm import fgn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from scipy import stats
from scipy.stats import t
from IPython import display
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import quad

import ZI
import MTY_vol

path = '/content/drive/My Drive/'

pathOB_TSLA = path+'TSLA_data_2015_01/orderbook/TSLA_2015-01-02_34200000_57600000_orderbook_10.csv'
pathMSG_TSLA = path+'TSLA_data_2015_01/message/TSLA_2015-01-02_34200000_57600000_message_10.csv'

#############################################################################################################
#############################################################################################################


def load_data(day=2, month=1, path=path, ticker='TSLA'):
    
    """
    if orderbook file has not column 'Time', set 
    miss_time to True
    
    Returns
    -------
    df_OB, df_MSG
    """
    
    def to_ns(df):
        """
        changes timestamp to nanoseconds
        
        Parameters
        ----------
        df : dataframe with column 'Time'

        Returns
        -------
        df
        """
        if 'Time' in df:
            ns = 1e13
            for i, time in enumerate(df['Time']):
                while time < ns:
                    time*=10
                df.at[i, 'Time'] = time
                
                while time >= ns*10:
                    time/=10
                df.at[i, 'Time'] = time    
        else:
            print('Column "Time" not in df')
        
        return df

    
    day = str(day).zfill(2)
    month = str(month).zfill(2)
    pathOB = path + str(ticker)+'_data_2015_'+month+'/orderbook/TSLA_2015-'+month+ \
        '-'+day+'_34200000_57600000_orderbook_10.csv'
    pathMSG = path + str(ticker)+'_data_2015_'+month+'/message/TSLA_2015-'+month+ \
        '-'+day+'_34200000_57600000_message_10.csv'
    
    df_OB = pd.read_csv(pathOB, delimiter=';', header=None) 
    df_MSG = pd.read_csv(pathMSG, delimiter=';') # columns must be already named
    
    time_col = df_MSG['Time']
    df_OB.insert(0, 'Time', time_col)

    counter = 1
    column_mapping = {}
    
    for i, col in enumerate(df_OB.columns[1:], start=1):
        if i % 4 == 1:
            column_mapping[col] = 'Ask Price'+str(counter)
        elif i % 4 == 2:
            column_mapping[col] = 'Ask Size'+str(counter)
        elif i % 4 == 3:
            column_mapping[col] = 'Bid Price'+str(counter)
        elif i % 4 == 0:
            column_mapping[col] = 'Bid Size'+str(counter)
            counter += 1
    
    df_OB = df_OB.rename(columns=column_mapping)
    
    df_OB = to_ns(df_OB)
    df_MSG = to_ns(df_MSG)
    
    df_OB.sort_values('Time', inplace=True)
    df_MSG.sort_values('Time', inplace=True)
    
    date = '2015-'+month+'-'+day
    df_OB['Datetime'] = pd.to_datetime(date) + pd.to_timedelta(df_OB['Time'], unit='ns')
    df_MSG['Datetime'] = pd.to_datetime(date) + pd.to_timedelta(df_MSG['Time'], unit='ns')
    
    #df_MSG = df_MSG[(df_MSG['Time']>=3.78e13) & (df_MSG['Time']<=5.4e13)]
    #df_OB = df_OB[(df_OB['Time']>=3.78e13) & (df_OB['Time']<=5.4e13)]
    
    df_OB.reset_index(inplace=True, drop=True)
    df_MSG.reset_index(inplace=True, drop=True)
    
    return df_OB, df_MSG



def spread_computer(df_OB):
    
    spread = df_OB['Ask Price1'] - df_OB['Bid Price1']
    
    return spread


def TotVolume(df):
        
    size_columns = [col for col in df.columns if 'Size' in col]
    df['TotVolume'] = df[size_columns].sum(axis=1)

    return df
    

def Quote_column(df):
    df['Quote'] = 0
    
    columns = [col for col in df.columns if ' Size' in col or ' Price' in col]
    differ = df[columns].diff().fillna(0)
    
    non_zero_indices, non_zero_columns = np.where(differ.to_numpy() != 0)
    df_indices = df.index.values
    
    prev_idx = None
    
    for i, col_idx in zip(non_zero_indices, non_zero_columns):
        if prev_idx != i:
            df.loc[df_indices[i], 'Quote'] = col_idx // 4 + 1
            prev_idx = i
    
    return df
    


def side_volume(df):
    
    buy_col = [col for col in df.columns if 'Bid Size' in col]
    ask_col = [col for col in df.columns if 'Ask Size' in col]
    df['BuyVolume'] = np.zeros(len(df))
    df['SellVolume'] = np.zeros(len(df))
    
    for i, row in df[buy_col].iterrows():
        df['BuyVolume'].iloc[i] = sum(row)
    for i, row in df[ask_col].iterrows():
        df['SellVolume'].iloc[i] = sum(row)
    
    return df


def prep_for_ratio(days=[26,27,28,29,30]):
    
    type_mapping = {1:'Limit', 2:'Cancel', 3:'Cancel', 4:'Market', 5:'DELETE', 7:'DELETE'}
    
    df = pd.DataFrame()
    time_delta = []
    
    for day in days:
        df_OB, df_MSG = load_data(day)
        df_OB = to_cents(df_OB)
        df_MSG = to_cents(df_MSG)
        spread = spread_computer(df_OB)
        df_OB['Spread'] = spread
        df_OB['Sign'] = df_MSG['Direction']
        df_OB['Type'] = df_MSG['Type']
        df_OB['Volume'] = df_MSG['Size']
        df_OB['Type'] = df_OB['Type'].map(type_mapping)
        df_OB['Price'] = df_MSG['Price']
        df_OB = df_OB[df_OB['Type'] != 'DELETE']
        df_OB.reset_index(drop=True, inplace=True)
        df_OB = Quote_column(df_OB)
        df_OB = TotVolume(df_OB)
        df_OB = side_volume(df_OB)
        df_OB = df_OB[df_OB['Quote'] > 0]
        df_OB.reset_index(drop=True, inplace=True)
        time_d = df_OB['Time'].diff().to_list()
        del(time_d[0])
        time_delta.extend(time_d)
        print('time delta: ', time_delta[:20])
                
        df = pd.concat([df, df_OB])
        
    print('time delta: ', time_delta[:20])
    print(sum(time_delta))
    print(len(time_delta))
    df.reset_index(drop=True, inplace=True)
    mean_time = sum(time_delta)/len(time_delta)
    
    return df, mean_time

def to_cents(df):
    
    price_columns = [col for col in df.columns if 'Price' in col]
    for i in price_columns:
        df[i] = df[i]/100
        
    return df


def net_order_flow(df_MSG):
    """
    computes net order flow of 
    a pandas df with the 'message'
    structure (columns must be named)
    
    Parameters
    ----------
    df_MSG : dataframe of LOBSTER 'message' file

    Returns
    -------
    net_order_flow (int)
    """
    net_order_flow = 0
    
    mkt_orders = df_MSG[df_MSG['Type']=='Market']
    
    for i in range(0, len(mkt_orders)):
        net_order_flow = net_order_flow - mkt_orders['Sign'].iloc[i]*mkt_orders['Volume'].iloc[i]
    
    return net_order_flow



def cj_nof(df_MSG, interval=3e11):
    """
    Net order flow for permanent impact 
    as in Cartea Jaimungal 2016 (intervals of 5min)

    Parameters
    ----------
    df_MSG : dataframe of LOBSTER 'message' file
    interval : int, interval (in nanoseconds) in which n.o.f. 
    is computed (C.J. set to 5min)

    Returns
    -------
    Net order flow (list).

    """
    
    n_o_f = []
    
    start_time = df_MSG['Time'].iloc[0]
    interval_time = start_time + interval
    end_time = df_MSG['Time'].iloc[-1]
    
    while interval_time <= end_time:
        df = df_MSG[(df_MSG['Time'] >= start_time) & (df_MSG['Time'] <= end_time)]
        n_o_f.append(net_order_flow(df))
        
        start_time = interval_time
        interval_time += interval  
        
    return n_o_f


def midprice(df_OB, time):
    
    midprice = (df_OB.loc[df_OB['Time'] >= time, 'Ask Price1'].values[0]+
                df_OB.loc[df_OB['Time'] >= time, 'Bid Price1'].values[0])/2
    
    return midprice


def k_regr(x,y):
    try:
        k = stats.linregress(x, y).slope    
    except:
        k = 0
    return k


def cj_perm_imp(df, interval=6e11):
    
    """
    Permanent impact as in Cartea Jaimungal 2016 (5min)
    
    """
    dS_n = []
    mu_n = []
    
    start = df['Time'].iloc[0]
    end = df['Time'].iloc[-1]
    
    times = [start]
    unit = start
            
    while unit < end-interval:
        unit += interval
        times.append(unit)
    
    
    S_n = []
    
    for i in times:
        midp_df = df[(df['Time'] >= i)]
        time = midp_df['Time'].iloc[0]
        midp = midprice(midp_df, time)
                
        S_n.append(midp)
                
    daily_dS_n = []
    
    for i in range(0, len(S_n)-1):    
        dmidp = S_n[i+1]-S_n[i]
        daily_dS_n.append(dmidp)
    
    daily_mu_n = cj_nof(df, interval)
            
            
    dS_n += daily_dS_n
    mu_n += daily_mu_n
    
    perm_df = pd.DataFrame({'dS_n': dS_n, 'mu_n': mu_n})
    perm_df = perm_df[(np.abs(stats.zscore(perm_df)) < 3).all(axis=1)]  # outliers deletion
    
    perm_imp = sm.OLS(perm_df['dS_n'], perm_df['mu_n']).fit()
    b = perm_imp.params
    r = perm_imp.rsquared
    std = perm_imp.bse
    '''
    plt.scatter(perm_df['mu_n'], perm_df['dS_n'], label='Data Points')
    plt.plot(perm_df['mu_n'], perm_imp.predict(perm_df['mu_n']), color='red', label='Regression Line')
    #plt.ylim(0, 8000)
    plt.xlabel('Net order flow')
    plt.ylabel('Shift in midprice')
    plt.title('Plot temp. imp. as function of time')
    plt.legend()
    #plt.grid(True)
    plt.show()
    '''
    
    print('Permanent Impact as in Cartea Jaimungal 2016')
    display.display('Coefficient: ', b)
    display.display('Standard deviation: ', std)
    display.display('R squared: ', r)

    return b, std



def temp_imp(df_OB, interval, max_vol):
    """
    Auxiliary function for the estimation of
    temporary impact, following the approach of
    Cartea, Jaimungal 2016

    Parameters
    ----------
    df_OB : pandas dataframe with orderbook data.
    interval : int
        interval for the estimation (nanoseconds)
    max_vol : int
        max volume for the market orders to be 
        simulated (range is adjusted accordingly)
        
        
    Returns
    -------
    k_i : list
        list of the estimated parameter (for each time)
    times : list
        list of the times on wich the regression is 
        performed

    """
    
    start = df_OB['Time'].iloc[0]
    end = df_OB['Time'].iloc[-1]
    
    times = []
    unit = start
    
    while unit < end:
        times.append(unit)
        unit+=interval

    k_i = []
    
    V_mkt_orders = []

    bid_cols = [col for col in df_OB.columns if 'Bid Size' in col]
    max_vol_series = df_OB.groupby('Time')[bid_cols].sum()

    for i in times:
        ex_cost_i = []
        
        
        if not max_vol:
            max_vol = int(max_vol_series.loc[i].sum())
        
        
        for j in range(int(max_vol*0.1), max_vol+1, int(max_vol*0.05)):
            V_mkt_orders.append(j)

        for v in V_mkt_orders:
            
            counter=1
            
            Vl = df_OB.loc[df_OB['Time'] >= i, 'Bid Size1'].values[0]
            Vmkt = v
            depletion = Vmkt - Vl
                
            sum_p = 0
                
            while depletion > 0 and counter < 10:
                counter += 1
                Vl = df_OB.loc[df_OB['Time'] >= i, 'Bid Size'+str(counter)].values[0]
                Vmkt = depletion
                depletion = Vmkt - Vl
                
            sum_p = Vmkt * df_OB.loc[df_OB['Time'] >= i, 'Bid Price'+str(counter)].values[0]
                    
            for l in range(counter-1):
                sum_p += df_OB.loc[df_OB['Time'] >= i, 'Bid Price'+str(counter-1-l)].values[0] * \
                df_OB.loc[df_OB['Time'] >= i, 'Bid Size'+str(counter-1-l)].values[0]
                
            Best_price_i = df_OB.loc[df_OB['Time'] >= i, 'Bid Price1'].values[0]
            ex_cost_v = -sum_p/v + Best_price_i
            ex_cost_i.append(ex_cost_v)
        ex_cost_i = np.array(ex_cost_i)
        X = np.array(V_mkt_orders)
        
        k = k_regr(X, ex_cost_i)
        
        k_i.append(k)
    
    return k_i, times



def cj_temp_imp(df_OB, max_vol, interval=3e11):
    
    """
    Temporary impact estimated as
    in Cartea, Jaimungal 2016
    
    "Here we only use the buy side of the LOB because including both sides 
    does not affect the results, the order of magnitude of the estimated k
    is statistically the same for both sides of the book"
    -Cartea, Jaimungal 2016
    
    """
    
    k_i, times  = temp_imp(df_OB, interval, max_vol) 
    ##################### from here Cesa ############################
    x = np.array(times).reshape(-1, 1)
    y = np.array(k_i).reshape(-1, 1)
    regr_df = pd.DataFrame({'y':y.flatten(), 'x':x.flatten()})
    regr_df = regr_df.dropna()
    regr_df = regr_df[regr_df.y > 0]  # delete negative spread
    regr_df = regr_df[(np.abs(stats.zscore(regr_df)) < 3).all(axis=1)]  # outliers deletion
            
    x = np.array(regr_df.x)
    y = np.array(regr_df.y)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    inds = x.ravel().argsort()  # Sort x values and get index
    x = x.ravel()[inds].reshape(-1, 1)
    y = y[inds]  # Sort y according to x sorted index
    
            #print(x,y)
        
    minAic = 999999
    order = 0
    for i in range(1, 20):
        polynomial_features = PolynomialFeatures(degree=i)
        xp = polynomial_features.fit_transform(x)
        #print(xp)
        model = sm.OLS(y, xp).fit()
        if model.aic < minAic:
            minAic = model.aic
            order = i
    polynomial_features = PolynomialFeatures(degree=order)
    xp = polynomial_features.fit_transform(x)
    model = sm.OLS(y, xp).fit()
    ypred = model.predict(xp)

    fr = 1  # 1 for seconds, 60 for minutes, 3600 for hours --> interval for the integral
    kappas = []
        
    def fun(x):
        y = 0
        for i in range(len(model.params)):
            y = y + model.params[i] * (x ** i)
        return y
        
    for i in np.arange(10 * 60 * 60, 16 * 60 * 60, fr):
        res, err = quad(fun, i, i + fr)[0:2]
        kappas.append(res)
        
        
    #plt.plot(x, y)
    plt.scatter(times, k_i, label='K_i')
    plt.plot(x, ypred, color='red', label='Fitting')
    #plt.ylim(0, 8000)
    plt.xlabel('Time (ns. from midnight)')
    plt.ylabel('Temporary impact')
    plt.title('Plot temp. imp. as function of time')
    plt.legend()
    #plt.grid(True)
    plt.show()
               
    return np.mean(kappas)
        
        


def time_weighting(df, column_name, interval):
    
    P_i = 0
    
    for i in range(2, len(df)):
        P_i += df[column_name].iloc[i-1]*(df['Time'].iloc[i]-df['Time'].iloc[i-1])
        
    P_i = P_i/interval
    
    return P_i


def new_LOB_f(df_OB, levels, interval, times):
    
    """
    Time weighted LOB as in Glas et al. 2020
    """
    
    new_LOB = pd.DataFrame()
    
    for i in times:
    
        couples_i = []
    
        for v in levels:
            column_name_p = f'Bid Price{v}'
            column_name_v = f'Bid Size{v}'
            
            df_p = df_OB[['Time', 'Bid Price'+str(v)]][(df_OB['Time'] > i) & (df_OB['Time'] <= i+interval)]              
            Pbar_i = time_weighting(df_p, column_name_p, interval)
        
            df_v = df_OB[['Time', 'Bid Size'+str(v)]][(df_OB['Time'] > i) & (df_OB['Time'] <= i+interval)]              
            Vbar_i = time_weighting(df_v, column_name_v, interval)
        
            level_couple = {Pbar_i: Vbar_i}
            couples_i.append(level_couple)
        
        sorted_couples = sorted(couples_i, key=lambda d: list(d.keys())[0], reverse=True)
    
        data = {'Bid Price'+str(i+1): [list(d.keys())[0]] for i, d in enumerate(sorted_couples)}
        data.update({'Bid Size'+str(i+1): [list(d.values())[0]] for i, d in enumerate(sorted_couples)})

        temp_df = pd.DataFrame(data)
        new_LOB = pd.concat([new_LOB, temp_df], axis=0)

    
    new_LOB.insert(0, 'Time', times)
    
    return new_LOB



def glas_temp_imp(df_OB, max_vol, interval=3e11):
    
    # too slow
    """
    Temporary impact (execution cost) estimation 
    as in Glas et al. 2020 

    """
    kappas = []
    
    start = df_OB['Time'].iloc[0]
    end = df_OB['Time'].iloc[-1]

    times = []
    unit = start

    while unit < end:
        times.append(unit)
        unit+=interval
    
    levels = list(range(1, 11))

    new_LOB = new_LOB_f(df_OB, levels, interval, times)

    k_i, times = temp_imp(new_LOB, 3e11, max_vol)

    x = np.array(times).reshape(-1, 1)
    y = np.array(k_i).reshape(-1, 1)
    df = pd.DataFrame({'y':y.flatten(), 'x':x.flatten()})
    df = df.dropna()
    df = df[df.y > 0]  # delete negative spread
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]  # outliers deletion
    
    x = np.array(df.x)
    y = np.array(df.y)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    inds = x.ravel().argsort()  # Sort x values and get index
    x = x.ravel()[inds].reshape(-1, 1)
    y = y[inds]  # Sort y according to x sorted index

    #print(x,y)
    
    minAic = 999999
    order = 0
    for i in range(1, 20):
        polynomial_features = PolynomialFeatures(degree=i)
        xp = polynomial_features.fit_transform(x)
        #print(xp)
        model = sm.OLS(y, xp).fit()
        if model.aic < minAic:
            minAic = model.aic
            order = i
    #print('Optimal degree of the polynomial: ', order)
    polynomial_features = PolynomialFeatures(degree=order)
    xp = polynomial_features.fit_transform(x)
    model = sm.OLS(y, xp).fit()
    ypred = model.predict(xp)

    fr = 1  # 1 for seconds, 60 for minutes, 3600 for hours --> interval for the integral
    
    
    def fun(x):
        y = 0
        for i in range(len(model.params)):
            y = y + model.params[i] * (x ** i)
        return y
    
    for i in np.arange(10 * 60 * 60, 16 * 60 * 60, fr):
        res, err = quad(fun, i, i + fr)[0:2]
        kappas.append(res)
    
    '''
    #plt.plot(x, y)
    plt.scatter(times, k_i, label='K_i')
    plt.plot(x, ypred, color='red', label='Fitting')
    #plt.ylim(0, 8000)
    plt.xlabel('Time (ns. from midnight)')
    plt.ylabel('Temporary impact (')
    plt.title('Temp. imp. as function of time')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    return np.mean(kappas)


def optimal_strategy(t, chi, Q_CJ, Q_AC, ell, mu, b, k):
    
    nu_CJ = chi[t] * Q_CJ[t] - 0.5 * b / k * ell[t] * mu[t]    
    nu_AC = chi[t] * Q_AC[t]
    
    return nu_CJ, nu_AC
    


def Sim_OptEx_Ratio(f_lo, f_mo, f_c, alpha, sigma, lo_placement, lo_volume, mo_volume, delta,
             m0 = 0, K = 100, iterations = 10_000, burn = 5_000, n_tot = 100, energy = False,
             hurst = None, T = 1, lam = 12000, kappa = 300, eta_mean = 23, inv0 = 0.5*6000,
             vol = 0.46, k = 0.06e-2, b = 0.02e-2, Ndt=60, sign = -1, CJ = True, naive = False):
    """
    CJ : Boolean
        Chooses between the CJ or AC strategy.
        The default is True = Cj.
    """
    print('START')
    if CJ:
        alg = 0
    else:
        alg = 1

    # setting the parameters for the optimal strategy
    dt = 1/Ndt
    t = np.arange(0, T+0.00000001, dt)
    phi = 10*k
    gamma = (phi/k)**0.5
    tau = T-t
    chi = gamma*np.cosh(gamma*tau)/np.sinh(gamma*tau)

    def exp_p(a, tau):
        return (1 - np.exp(-a*tau)) / a

    v1 = np.exp(gamma * tau)
    tot = kappa + gamma
    dif = kappa - gamma
    ell = 0.5*( v1 * exp_p(tot, tau) - v1**-1 * exp_p(dif, tau))/(np.sinh(gamma*tau))

    mu = np.full(Ndt+1, np.nan)  # Order Flow matrix
    # Initializing variables for simulation base on computed strategy
    X_CJ = np.full(Ndt+1, np.nan)  # Gain/loss vector of the trade
    Q_CJ = np.full(Ndt+1, np.nan)  # Inventory matrix
    S_CJ = np.full(Ndt+1, np.nan)  # Execution Price matrix
    nu_CJ = np.full(Ndt+1, np.nan)  # Rate of Trading matrix
    C_CJ = np.full(Ndt+1, np.nan)   #  Cost vector of the strategy
    Q_AC = np.full(Ndt+1, np.nan)  # Inventory matrix, required by optimal_strategy()

    Q_CJ[0] = Q_AC[0] = inv0
    mu[0] = 0
    X_CJ[0] = 0
    C_CJ[0] = 0

    def ex_time(lam):
        """
        in seconds
        """
        events = np.random.poisson(lam)
        event_time = (60*60) / events
        return event_time


    # Inizilizzo lob, vol_lob e i dizioniari dei messagi e dello stato del book
    lob = np.zeros(K, dtype = np.int16)
    lob[int(K//2 - n_tot//2):K//2] = 1
    lob[K//2:int(K//2 + n_tot //2)] = -1

    vol_lob = np.zeros((20,K), dtype = np.int16)
    vol_lob[0, int(K//2 - n_tot//2):K//2] = 1
    vol_lob[0, K//2:int(K//2 + n_tot //2)] = -1

    order, message = ZI.initialize_order_message(iterations)
    # Simulo il segno dei MO utilizzando un il rumore di un processo gaussiano
    # frazionario avente esponente di hurst fissato in input.
    # Se in input non viene dato un esponente il segno dei MO ÃƒÂ¨ scelto casualmente.
    if hurst is None:
        mo_s = None
        mo_n = None
    else:
        mo_s = np.sign(fgn(n=iterations//10, hurst=hurst, length=1, method='daviesharte'))
        mo_n = 0

    # Simulo il LOB scartando le prime iterazioni
    for i in range(burn):
        MTY_vol.do_order(lob, vol_lob, f_lo, f_mo, f_c, lo_placement, lo_volume, mo_volume,
                    delta, K, alpha, sigma, energy)

    clock_time = 0

    # Simulo il LOB
    while clock_time <= 60*60:

        i = 0
        event_time = ex_time(lam)
        clock_time += event_time

        # Simulo un ordine e aggiorno i dizionari dei messaggi e dello stato del LOB
        message["Price"][i], message["Sign"][i], message["Type"][i], \
        message["Volume"][i], mo_n = MTY_vol.do_order(lob, vol_lob, f_lo, f_mo, f_c, lo_placement,
                    lo_volume, mo_volume, delta, K, alpha, sigma, energy, mo_s, mo_n)

        message['Price'][i] + m0
        message["Spread"][i] = ZI.find_spread(lob)
        message["MidPrice"][i] = ZI.find_mid_price(lob) + m0

        MTY_vol.update_order(lob, order, i, m0)

        i += 1
    S_CJ[0] = message['Price'][-1]
    clock_time = 0
    next_volume = 0
    print('Inizio opt. ex.')
    for t in range(len(tau)-1):
        event_time = ex_time(lam)
        clock_time += event_time
        
        while clock_time < 60*60/Ndt:

            event_time = ex_time(lam)
            clock_time += event_time

            # Simulo un ordine e aggiorno i dizionari dei messaggi e dello stato del LOB
            message["Price"][i], message["Sign"][i], message["Type"][i], \
            message["Volume"][i], mo_n = MTY_vol.do_order(lob, vol_lob, f_lo, f_mo, f_c, lo_placement,
                        lo_volume, mo_volume, delta, K, alpha, sigma, energy, mo_s, mo_n)

            message["Spread"][i] = ZI.find_spread(lob)
            message["MidPrice"][i] = ZI.find_mid_price(lob) + m0

            MTY_vol.update_order(lob, order, i, m0)

            i += 1

        clock_time = 0
        
        if naive:
            volume = inv0/Ndt
            nu = volume/dt
            volume += next_volume
        else:
            nu = optimal_strategy(t, chi, Q_CJ, Q_AC, ell, mu, b, k)[alg]
            volume = round(nu*dt + next_volume)

        volume = round(nu*dt + next_volume)

        nu_CJ[t] = nu
        direction = 1
        if volume < 0:
            direction = -1
        price = MTY_vol.do_market_order(lob, direction*sign)
        volume = np.abs(volume)
            
        S_CJ[t+1] = price + m0
        message['Price'][i] = price + m0
        message['Sign'][i] = direction*sign
        message['Type'][i] = 1
        message['Volume'][i] = volume

        message["Spread"][i] = ZI.find_spread(lob)
        message["MidPrice"][i] = ZI.find_mid_price(lob) + m0

        dQ_CJ = Q_CJ[t]
        dQ_AC = Q_AC[t]
        dC_CJ = 0
        dX_CJ = 0
    
        ideal_ex = volume * direction*(price + m0)
        
        # FinchÃƒÂ© il volume del MO ÃƒÂ¨ maggiore del volume al best price
        # consuma tutto il best price e passa alla quota successiva.
        #print('price: ', price+m0)
        while np.abs(lob[price]) < volume:
                print('inside while loop, volume at best: ', lob[price])
                dX_CJ += lob[price]*(price+m0)
                print('dX_CJ: ', dX_CJ)
                dQ_CJ -= lob[price]
                dQ_AC -= lob[price]

                to_remove = np.abs(lob[price])
                volume -= to_remove
                lob[price] = 0
                MTY_vol.do_mo_queue(vol_lob, price, to_remove, sign*direction)
                price = MTY_vol.do_market_order(lob, sign*direction)
                print('new price: ', price+m0)
                print('remaining volume: ', volume)
                next_volume = 0

        dQ_CJ -= direction*volume
        dQ_AC -= direction*volume
        dX_CJ += volume * direction*(price+m0)
        dC_CJ = ideal_ex + direction*sign*np.abs(dX_CJ)
        
        Q_CJ[t+1] = dQ_CJ
        Q_AC[t+1] = dQ_AC
        X_CJ[t+1] = dX_CJ
        C_CJ[t+1] = dC_CJ

        dn = np.random.poisson(lam)
        # decide if it adds to the buy/sell pressure
        buysell = (np.random.rand(1) < 0.5)
        # generate the size of the order-flow impact
        eta = np.random.exponential(eta_mean)
        mu[t+1] = mu[t]*np.exp(-kappa * dt) + (eta * dn * (2 * buysell - 1))

        lob[price] += volume * sign
        MTY_vol.do_mo_queue(vol_lob, price, volume, sign)

        i += 1

    return X_CJ, S_CJ, nu_CJ, Q_CJ, Q_AC, C_CJ


def Sim_OptEx_ZI(l_rate, m_rate, c_rate, avg_vol = 200, m0 = 0, K = 100, iterations = 10_000, burn = 5000,
                 T = 1, lam = 12000, kappa = 300, eta_mean = 23, inv0 = 0.5*6000, 
                 vol = 0.46, k = 0.06e-2, b = 0.02e-2, Ndt=60, sign = -1, CJ = True, naive = False):
    """
    CJ : Boolean
        Chooses between the CJ or AC strategy.
        The default is True = Cj.
    """
    print('START')
    if CJ:
        alg = 0
    else:
        alg = 1
    
    # setting the parameters for the optimal strategy
    dt = 1/Ndt
    t = np.arange(0, T+0.00000001, dt)
    phi = 10*k
    gamma = (phi/k)**0.5
    tau = T-t
    chi = gamma*np.cosh(gamma*tau)/np.sinh(gamma*tau)
    
    def exp_p(a, tau):
        return (1 - np.exp(-a*tau)) / a
    
    v1 = np.exp(gamma * tau)
    tot = kappa + gamma
    dif = kappa - gamma
    ell = 0.5*( v1 * exp_p(tot, tau) - v1**-1 * exp_p(dif, tau))/(np.sinh(gamma*tau))
    
    mu = np.full(Ndt+1, np.nan)  # Order Flow matrix
    # Initializing variables for simulation base on computed strategy
    X_CJ = np.full(Ndt+1, np.nan)  # Gain/loss vector of the trade
    Q_CJ = np.full(Ndt+1, np.nan)  # Inventory matrix
    S_CJ = np.full(Ndt+1, np.nan)  # Execution Price matrix
    nu_CJ = np.full(Ndt+1, np.nan)  # Rate of Trading matrix
    C_CJ = np.full(Ndt+1, np.nan)   #  Cost vector of the strategy
    Q_AC = np.full(Ndt+1, np.nan)  # Inventory matrix, required by optimal_strategy()
    
    Q_CJ[0] = inv0
    Q_AC[0] = inv0
    mu[0] = 0
    X_CJ[0] = 0
    C_CJ[0] = 0
    
    def ex_time(lam):
        """
        in seconds
        """
        events = np.random.poisson(lam)
        event_time = (60*60) / events
        return event_time
    
    
    # Initialize LOB
    lob = np.ones(K, dtype = np.int16)
    lob[K//2:] = -1
    lob *= avg_vol

    # Initialize order and message dictionaries
    order, message = ZI.initialize_order_message(iterations)
    
    # Simulo il LOB scartando le prime iterazioni
    for i in range(burn):
        ZI.do_order(lob, l_rate, m_rate, c_rate, K, avg_vol)
        
    clock_time = 0
    
    # Simulo il LOB
    while clock_time <= 60*60:
        
        i = 0
        event_time = ex_time(lam)
        clock_time += event_time
        
        message["Price"][i], message["Sign"][i], message["Type"][i], \
            message["Volume"][i] = ZI.do_order(lob, l_rate, m_rate, c_rate, K, avg_vol)
        message["Spread"][i] = ZI.find_spread(lob)
        message["MidPrice"][i] = ZI.find_mid_price(lob)

        ZI.update_order(lob, order, i, avg_vol)
        
        i += 1
        
    S_CJ[0] = message['Price'][-1]
    clock_time = 0
    next_volume = 0
    
    print('Inizio opt. ex.')
    
    for t in range(len(tau)-1):
        event_time = ex_time(lam)
        clock_time += event_time
        
        while clock_time < 60*60/Ndt:
            
            event_time = ex_time(lam)
            clock_time += event_time
        
            # Simulo un ordine e aggiorno i dizionari dei messaggi e dello stato del LOB
            message["Price"][i], message["Sign"][i], message["Type"][i], \
                message["Volume"][i] = ZI.do_order(lob, l_rate, m_rate, c_rate, K, avg_vol)

            message["Spread"][i] = ZI.find_spread(lob)
            message["MidPrice"][i] = ZI.find_mid_price(lob) + m0

            ZI.update_order(lob, order, i, avg_vol)
            
            i += 1
            
        clock_time = 0
    
        if naive:
            volume = inv0/Ndt
            nu = volume/dt
            volume += next_volume
        else:
            nu = optimal_strategy(t, chi, Q_CJ, Q_AC, ell, mu, b, k)[alg]
            volume = round(nu*dt + next_volume)
        
        nu_CJ[t] = nu
        
        direction = 1
        if volume < 0:
            direction = -1
        price = ZI.do_market_order(lob, direction*sign)
        volume = np.abs(volume)
        
        S_CJ[t+1] = price + m0
        message['Price'][i] = price + m0
        message['Sign'][i] = sign
        message['Type'][i] = 1
        message['Volume'][i] = volume
        
        message["Spread"][i] = ZI.find_spread(lob)
        message["MidPrice"][i] = ZI.find_mid_price(lob) + m0
        
        dQ_CJ = Q_CJ[t]
        dQ_AC = Q_AC[t]
        dX_CJ = 0
        
        ideal_ex = volume * direction*(price+m0)
        
        # FinchÃ© il volume del MO Ã¨ maggiore del volume al best price
        # consuma tutto il best price e passa alla quota successiva.
        
        while np.abs(lob[price]) < volume:
                dX_CJ += lob[price]*(price+m0)
                dQ_CJ -= lob[price]
                dQ_AC -= lob[price]
                
                to_remove = np.abs(lob[price])
                volume -= to_remove
                lob[price] = 0
                price = ZI.do_market_order(lob, sign)
            
        dQ_CJ -= volume
        dQ_AC -= volume
        dX_CJ += volume * (price+m0)
        dC_CJ = (-sign)*ideal_ex + sign*dX_CJ
            
        new_mp = ZI.find_mid_price(lob)
        shift = int(new_mp + 0.5 - K//2)

        if shift > 0:
            lob[:-shift] = lob[shift:]
            lob[-shift:] = np.zeros(len(lob[-shift:]))
        elif shift < 0:
            lob[-shift:] = lob[:shift]
            lob[:-shift] = np.zeros(len(lob[:-shift]))

        Q_CJ[t+1] = dQ_CJ
        Q_AC[t+1] = dQ_AC
        X_CJ[t+1] = dX_CJ
        C_CJ[t+1] = dC_CJ
        
        dn = np.random.poisson(lam)
        # decide if it adds to the buy/sell pressure
        buysell = (np.random.rand(1) < 0.5)
        # generate the size of the order-flow impact
        eta = np.random.exponential(eta_mean)
        mu[t+1] = mu[t]*np.exp(-kappa * dt) + (eta * dn * (2 * buysell - 1))
        
        lob[price] += volume * sign
        
        i += 1
    
    return X_CJ, S_CJ, nu_CJ, Q_CJ, Q_AC, C_CJ


def Sim_NaiveEx_Ratio(f_lo, f_mo, f_c, alpha, sigma, lo_placement, lo_volume, mo_volume, 
            delta, m0 = 0, K = 100, iterations = 10_000, burn = 5_000, n_tot = 100,
            energy = False, hurst = None, lam=12000, T = 1, inv0 = 0.5*6000, Ndt=60, sign = -1):
    
    # setting the parameters for the optimal strategy
    dt = 1/Ndt
    t = np.arange(0, T+0.00000001, dt)
    tau = T-t
    
    mu = np.full(Ndt+1, np.nan)  # Order Flow matrix
    # Initializing variables for simulation base on computed strategy
    X_n = np.full(Ndt+1, np.nan)  # Cost matrix of Strategy
    Q_n= np.full(Ndt+1, np.nan)  # Inventory matrix
    S_n = np.full(Ndt+1, np.nan)  # Execution Price matrix
    nu_n = np.full(Ndt+1, np.nan)  # Rate of Trading matrix      
    Q_AC = np.full(Ndt+1, np.nan)  # Inventory matrix, required by optimal_strategy()
    C_n = np.full(Ndt+1, np.nan)   #  Cost vector of the strategy
    
    Q_n[0] = Q_AC[0] = inv0
    mu[0] = 0
    X_n[0] = 0 
    C_n[0] = 0
    
    def ex_time(lam):
        """
        in seconds
        """
        events = np.random.poisson(lam)
        event_time = (60*60) / events
        return event_time
    
    
    # Inizilizzo lob, vol_lob e i dizioniari dei messagi e dello stato del book
    lob = np.zeros(K, dtype = np.int16)
    lob[int(K//2 - n_tot//2):K//2] = 1
    lob[K//2:int(K//2 + n_tot //2)] = -1

    vol_lob = np.zeros((20,K), dtype = np.int16)
    vol_lob[0, int(K//2 - n_tot//2):K//2] = 1
    vol_lob[0, K//2:int(K//2 + n_tot //2)] = -1

    order, message = ZI.initialize_order_message(iterations)
    # Simulo il segno dei MO utilizzando un il rumore di un processo gaussiano
    # frazionario avente esponente di hurst fissato in input.
    # Se in input non viene dato un esponente il segno dei MO ÃƒÂ¨ scelto casualmente.
    if hurst is None:
        mo_s = None
        mo_n = None
    else:
        mo_s = np.sign(fgn(n=iterations//10, hurst=hurst, length=1, method='daviesharte'))
        mo_n = 0
    # Simulo il LOB scartando le prime iterazioni
    for i in range(burn):
        MTY_vol.do_order(lob, vol_lob, f_lo, f_mo, f_c, lo_placement, lo_volume, mo_volume,
                    delta, K, alpha, sigma, energy)
        
    clock_time = 0
    
    # Simulo il LOB
    while clock_time <= 60*60:
        
        i = 0
        event_time = ex_time(lam)
        clock_time += event_time
        
        # Simulo un ordine e aggiorno i dizionari dei messaggi e dello stato del LOB
        message["Price"][i], message["Sign"][i], message["Type"][i], \
        message["Volume"][i], mo_n = MTY_vol.do_order(lob, vol_lob, f_lo, f_mo, f_c, lo_placement,
                    lo_volume, mo_volume, delta, K, alpha, sigma, energy, mo_s, mo_n)
        
        message['Price'][i] + m0
        message["Spread"][i] = ZI.find_spread(lob)
        message["MidPrice"][i] = ZI.find_mid_price(lob) + m0

        MTY_vol.update_order(lob, order, i, m0)
        
        i += 1
    S_n[0] = message['Price'][-1]
    clock_time = 0
    next_volume = 0
    
    print('Inizio opt. ex.')
    for t in range(len(tau)-1):
        event_time = ex_time(lam)
        clock_time += event_time
        while clock_time < 60*60/Ndt:
            
            event_time = ex_time(lam)
            clock_time += event_time
        
            # Simulo un ordine e aggiorno i dizionari dei messaggi e dello stato del LOB
            message["Price"][i], message["Sign"][i], message["Type"][i], \
            message["Volume"][i], mo_n = MTY_vol.do_order(lob, vol_lob, f_lo, f_mo, f_c, lo_placement,
                        lo_volume, mo_volume, delta, K, alpha, sigma, energy, mo_s, mo_n)

            message["Spread"][i] = ZI.find_spread(lob)
            message["MidPrice"][i] = ZI.find_mid_price(lob) + m0

            MTY_vol.update_order(lob, order, i, m0)
            
            i += 1
            
        clock_time = 0
    
        volume = inv0/Ndt
        nu = volume/dt
        print('nu: ', nu)
        volume = inv0/Ndt + next_volume
        nu_n[t] = nu
        #print('array dei nu: ', nu_CJ)
        print('volume: ', volume)
        price = MTY_vol.do_market_order(lob, sign) 
        
        S_n[t+1] = price + m0
        message['Price'][i] = price + m0
        message['Sign'][i] = sign
        message['Type'][i] = 1
        message['Volume'][i] = volume
        
        message["Spread"][i] = ZI.find_spread(lob)
        message["MidPrice"][i] = ZI.find_mid_price(lob) + m0
        
        dQ_n = Q_n[t]
        dQ_AC = Q_AC[t]
        dX_n = 0
        dC_n = 0
        
        ideal_ex = volume * (price+m0)
        
        # FinchÃƒÂ© il volume del MO ÃƒÂ¨ maggiore del volume al best price
        # consuma tutto il best price e passa alla quota successiva.
        
        print("price: ", price+m0)
        
        while np.abs(lob[price]) < volume:
            try:
                print('inside while loop, volume at best: ', lob[price])
                dX_n += lob[price]*(price+m0)
                print('dX_n: ', dX_n)
                dQ_n -= lob[price]
                dQ_AC -= lob[price]
                
                to_remove = np.abs(lob[price])
                volume -= to_remove
                print("volume: ", volume)
                lob[price] = 0
                MTY_vol.do_mo_queue(vol_lob, price, to_remove, sign)
                price = MTY_vol.do_market_order(lob, sign)
                print('new price: ', price+m0)
                print('remaining volume: ', volume)
                next_volume = 0
                
            except:
                next_volume = volume
                
        
        dQ_n -= volume
        dQ_AC -= volume
        dX_n += volume * (price+m0)
        dC_n = (-sign)*ideal_ex + sign*dX_n
        
        print('dX_n: ', dX_n)
        Q_n[t+1] = dQ_n
        Q_AC[t+1] = dQ_AC
        X_n[t+1] = dX_n
        C_n[t+1] = dC_n
        
        lob[price] += volume * sign
        MTY_vol.do_mo_queue(vol_lob, price, volume, sign)
        
        i += 1
    
    return X_n, S_n, nu_n, Q_n, C_n
