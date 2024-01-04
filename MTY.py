from scipy.optimize import minimize
import numpy as np
import pandas as pd
import ZI
import os
import scipy

def distance_from_opposite(df):

    idx_buy = df[(df["Sign"] == 1) & (df["Type"] == "Limit")].index.to_numpy()
    idx_sell = df[(df["Sign"] == -1) & (df["Type"] == "Limit")].index.to_numpy()

    price_bid = df["Price"].iloc[idx_buy[1:]].to_numpy()
    price_ask = df["Price"].iloc[idx_sell[1:]].to_numpy()

    best_ask = df["AskPrice_0"].iloc[idx_buy[1:] - 1].to_numpy()
    best_bid = df["BidPrice_0"].iloc[idx_sell[1:] - 1].to_numpy()

    distance_buy = best_ask - price_bid
    distance_sell = price_ask - best_bid

    return distance_buy, distance_sell

def load_data(filepath):
    df = pd.read_csv(filepath, index_col = 0)
    # transform the column Datetime from string to datetime
    df["Datetime"]= pd.to_datetime(df["Datetime"])

    #create a new column that represent second to midnight
    seconds = np.zeros(len(df))
    for i, date in enumerate(df["Datetime"]):
        seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                                        date.microsecond * 1e-6
    df["Time"] = seconds
    df = df.loc[df["Datetime"].dt.day != 27]
    df = df.loc[df["Datetime"].dt.hour > 6]
    df = df.loc[df["Datetime"].dt.hour < 16]
    df = df.loc[df["Spread"] > 0]
    df.reset_index(inplace = True, drop = True)
    df["Spread"] = df["Spread"].astype(int)
    df["BuyVolume"] = df.iloc[:,2:40:4].sum(axis=1)
    df["SellVolume"] = df.iloc[:,4:44:4].sum(axis=1)
    df["TotVolume"] = df["BuyVolume"] + df["SellVolume"]

    return df

def truncated_power_law(x, a, s):
    """
    PDF della power law troncata in [0,1] presentata nell'articolo di Muni Toke e
    Yoshida "Modelling intensities of order flows in a limit order book"
    """
    ret = s * (a +1) / ((1+ s)**(a +1) - 1)
    ret = ret * (1 + s * x)**a
    return ret

def likelihood_tpl(params, p_index):
    """
    Log likelihood power law troncata
    """
    a = params[0]
    s = params[1]

    N = len(p_index)
    ret = N * np.log(s * (a +1) / ((1+ s)**(a +1) - 1))
    ret += a * (np.log(1 + s * p_index)).sum()
    return -ret

def inverse_cdf_tpr(x,a,s):
    """
    Funzione di densità cumulativa inversa della power law troncata. Utile per
    fare samplng della distribuzione.
    """
    ret = ((1 + s)**(a + 1) - 1) * x + 1
    ret = (ret**(1 / (a + 1)) - 1) / s
    return ret

def compute_volume_index(df, n_quotes = 10):
    """
    Funzione che calcola il priority index di ogni ordine che è stato cancellato
    nel book.

    Input:
        1. df: pd.DataFrame
            DataFrame contenente i dati del LOB.
        2. n_quotes int (default = 10)
            Numero di quote da considerare per calcolare il priority index.
    Output:
        1. p_index np.array
            Array contenente il priority index di ogni ordine che è stato cancellato.
    """
    # Crea due serie (una per il lato ask e una per il lato bid) contenenti le quote
    # di ogni ordine cancellato.
    idx_buy = df.loc[(df["Type"] == "Cancel") & (df["Sign"] == 1)].index.to_numpy()
    idx_sell = df.loc[(df["Type"] == "Cancel") & (df["Sign"] == -1)].index.to_numpy()
    idx_buy = idx_buy[idx_buy > 0]
    idx_sell = idx_sell[idx_sell > 0]
    quote_buy = df["Quote"].iloc[idx_buy]
    quote_sell = df["Quote"].iloc[idx_sell]
    
    # create header list
    h_buy  = [f"Bid Size{int(i)+1}" for i in range(n_quotes)]
    h_sell = [f"Ask Size{int(i)+1}" for i in range(n_quotes)]

    # Crea un DataFrame contente il lato buy del book nell'istante precedente
    # all'arrivo di una cancellazione
    volume_buy = df.loc[idx_buy - 1, h_buy]
    index_buy = np.zeros(volume_buy.shape[0])
    
    # Trova il volume index per ogni cancellazione nel lato buy:
    for k, i in enumerate(volume_buy.index.to_list()):
        header = [f"Bid Size{int(j)+1}" for j in range(int(quote_buy.at[i+1]))]
        # Trova il volume totale degli ordini ad un price level superiore.
        # A questo valore viene poi sommata metà del volume della quota a cui è
        # avvenuta la cancellazione, perché non posso sapere la posizione esatta
        # dell'ordine cancellato all'interno della coda temporale relativa agli
        # ordini piazzati allo stesso prezzo (leggere articolo MTY per maggiori informazioni).
        if len(header) > 1:
            index_buy[k] = volume_buy.loc[i,header[-1]] / 2 + volume_buy.loc[i,header[:-1]].sum()
        else:
            index_buy[k] = volume_buy.loc[i,header[-1]] / 2

    # Ripeto lo stesso procedimento per il lato sell
    volume_sell = df.loc[idx_sell - 1, h_sell]
    index_sell = np.zeros(volume_sell.shape[0])
    for k, i in enumerate(volume_sell.index.to_list()):
        header = [f"Ask Size{int(j)+1}" for j in range(int(quote_sell.at[i+1]))]
        if len(header) > 1:
            index_sell[k] = volume_sell.loc[i,header[-1]] / 2 + volume_sell.loc[i,header[:-1]].sum()
        else:
            index_sell[k] = volume_sell.loc[i,header[-1]] / 2

    # Per trovare il priority index divido il priority volume per il volume totale
    # nello stesso lato della cancellazione
    p_buy = index_buy / df.loc[idx_buy - 1]["BuyVolume"].to_numpy()
    p_sell = index_sell / df.loc[idx_sell - 1]["SellVolume"].to_numpy()
    p_index = np.concatenate((p_buy, p_sell))

    return p_index

def compute_priority_index(df, guess, **kwargs):
    p_index = compute_volume_index(df)
    pars = minimize(likelihood_tpl, guess, method='SLSQP', **kwargs)
    print(pars)
    return pars.x[0], pars.x[1]

def eexp(x, b0,b1,b2):
    "Funzione logistica implementata nell'articolo MTY"
    return np.e**(b0 + b1 * np.log(x + 1) + b2 * (np.log(x + 1))**2)

class FamilyDistribution():
    def __init__(self, lst):
        self.distr = lst
        pass

    def find_distribution(self, x):
        if x < len(self.distr):
            sp_distr = self.distr[x - 1]
        else:
            sp_distr = self.distr[-1]
        return sp_distr

class Ratio():
    """
    Questa classe
    """
    def __init__(self, a0, a1, a2, b0, b1, b2):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        pass

    def find_rate(self, x, y):
        return 1 / (1 +  eexp(x, -self.a0, -self.a1 , -self.a2) + \
                eexp(x, self.b0 - self.a0, self.b1 - self.a1 , self.b2 - self.a2))

class Ratio_l():
    """
    Stessa classe di sopra, ma relativa ai LO
    """
    def __init__(self, a0, a1, a2, b0, b1, b2):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        pass

    def find_rate(self, x, y):
        return 1 / (1 +  eexp(x, self.a0, self.a1 , self.a2) + \
                 eexp(x, self.b0, self.b1, self.b2))

def ratio_orders(df,column):
    """
    Questa funzione ritorna l'intensità relativa empirica per LO, MO e cancellazioni,
    in funzione di una quantità da definire in input.

    Input:
        1. df: pd.DataFrame
            DataFrame contente dati del LOB
        2. column: ["Spread, "TotVolume] (può essere anche diversa)
            Quantità in funzione del quale calcolare le intensità relative

    Output:
        1. ratio_l: pd.Series
            intensità relativa LO in funzione dello Spread o del volume totale nel book.
        2. ratio_m: pd.Series
            intensità relativa MO in funzione dello Spread o del volume totale nel book.
        3. ratio_c: pd.Series
            intensità relativa cancellazioni in funzione dello Spread o del
            volume totale nel book.

    """
    idx_market = df[df["Type"] == "Market"].index.to_numpy()[1:]
    idx_limit = df[df["Type"] == "Limit"].index.to_numpy()[1:]
    idx_cancel = df[df["Type"] == "Cancel"].index.to_numpy()[1:]


    N_m = df.iloc[idx_market-1].groupby(column)["Price"].count()
    N_l = df.iloc[idx_limit-1].groupby(column)["Price"].count()
    N_c = df.iloc[idx_cancel-1].groupby(column)["Price"].count()

    tot = (N_m + N_l + N_c).fillna(0).replace(np.inf, 0)

    ratio_l = (N_l / tot).fillna(0).replace(np.inf, 0)
    ratio_m = (N_m / tot).fillna(0).replace(np.inf, 0)
    ratio_c = (N_c / tot).fillna(0).replace(np.inf, 0)

    return ratio_l, ratio_m, ratio_c

def logl_ratio_energy(params, x1, x2 ,x3, N1,N2,N3):
    """
    Log likelihood della funzione per le intensità relative di LO, MO e cancellazioni,
    descritta nel capitolo 6 della mia tesi.
    """
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    b0 = params[3]
    b1 = params[4]
    b2 = params[5]

    ret = - (np.log(1 + eexp(x1, a0, a1 , a2) + eexp(x1, b0, b1 , b2))).sum()

    ret -= (np.log(1 + eexp(x2, -a0, -a1 , -a2) + eexp(x2, b0 - a0, b1 - a1 , b2 - a2))).sum()

    ret -= (np.log(1 + eexp(x3, -b0, -b1 , -b2) + eexp(x3, a0 - b0, a1 - b1 , a2 - b2))).sum()
    return -ret

def compute_ratio(df, column, guess):
    """
    Questa funzine stima tramite minimizzazione della log likelihood il valore
    dei parametri della funzione delle intensità relative di LO, MO e cancellazioni,
    descritta nel capitolo 6 della mia tesi.
    """

    idx_market = df[df["Type"] == "Market"].index.to_numpy() - 1
    idx_cancel = df[df["Type"] == "Cancel"].index.to_numpy() - 1
    idx_limit  = df[df["Type"] == "Limit"].index.to_numpy() - 1
    # Segna lo spread e il volime totale nel momento precedente all'arrivo di un
    # MO, LO o cancellazione.
    xx1 = df[column].iloc[idx_market[1:]]
    xx0 = df[column].iloc[idx_limit[1:]]
    xx2 = df[column].iloc[idx_cancel[1:]]
    N1 = df["TotVolume"].iloc[idx_market[1:]]
    N0 = df["TotVolume"].iloc[idx_limit[1:]]
    N2 = df["TotVolume"].iloc[idx_cancel[1:]]
    # stima paramtri tramite minimizzazione della log likelihood
    pp = minimize(logl_ratio_energy, x0 = guess, args=(xx0,xx1,xx2,N0,N1,N2))
    print(pp)
    return pp

def compute_weight(new_df, idx, distance):
    ss = pd.unique(new_df["Spread"].iloc[idx])
    weight = np.zeros(distance.shape[0])

    for j, val in enumerate(distance):
        if ss[ss > val].shape[0] != 0:
            weight[j] = ss.shape[0] / ss[ss > val].shape[0]

    return weight

def resample_data(distance, weight, N):
    data = pd.DataFrame(distance, columns= ["Dist"])
    data["Weight"] = weight
    resampled = data["Dist"].sample(N, weights = data["Weight"],
                                    replace = True).to_numpy()
    return resampled

def distance_spread(new_df, min_val, max_val, n_resample = 500_000):
    idx_sp = new_df[(new_df["Spread"] >= min_val) & \
                (new_df["Spread"] < max_val)].index.to_numpy() + 1

    idx_bid = new_df.iloc[idx_sp[:-1]][(new_df["Sign"].iloc[idx_sp[:-1]] == 1) & \
                (new_df["Type"].iloc[idx_sp[:-1]] == "Limit")].index.to_numpy()

    idx_ask = new_df.iloc[idx_sp[:-1]][(new_df["Sign"].iloc[idx_sp[:-1]] == -1) & \
                (new_df["Type"].iloc[idx_sp[:-1]] == "Limit")].index.to_numpy()

    #Compute distance from the same side best price
    best_bid  = new_df["Bid Price1"].iloc[idx_bid - 1].to_numpy()
    bid_price = new_df["Price"].iloc[idx_bid].to_numpy()
    distance_bid =  bid_price - best_bid

    #Compute weighted distance and resample data according to the weight
    weight_bid = compute_weight(new_df, idx_bid - 1, distance_bid)
    resampled_bid = resample_data(distance_bid, weight_bid, n_resample)

    #Compute distance from the same side best price
    best_ask = new_df["Ask Price1"].iloc[idx_ask - 1].to_numpy()
    ask_price =  new_df["Price"].iloc[idx_ask].to_numpy()
    distance_ask = best_ask - ask_price

    #Compute weighted distance and resample data according to the weight

    weight_ask = compute_weight(new_df, idx_ask - 1, distance_ask)
    resampled_ask = resample_data(distance_ask, weight_ask, n_resample)

    return resampled_bid, resampled_ask
