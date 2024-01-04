import numpy as np
import pandas as pd
import ZI
import MTY
from fbm import fgn

def do_cancel_order(arr, sign, alpha = -1, scale = 1):

    n_orders_bid = arr[arr > 0].sum()
    n_orders_ask = -(arr[arr < 0].sum())

    # Estraggo un priority index da una truncated power law
    x = np.random.rand()
    priority_index = MTY.inverse_cdf_tpr(x, alpha, scale)

    # Trovo il primo ordine che ha un priority index <= di quello estratto
    # e ne ritorno il corrispettivo livello di prezzo
    if sign == 1:
        arr_indexes = np.arange(0, n_orders_bid) / n_orders_bid
        pos = n_orders_bid - np.where(arr_indexes <= priority_index)[0][-1] - 1
    else:
        arr_indexes = np.arange(0, n_orders_ask) / n_orders_ask
        pos = np.where(arr_indexes <= priority_index)[0][-1] + n_orders_bid

    pos_orders = np.abs(arr).cumsum()
    price =  np.where(pos_orders > pos)[0][0]

    return price

def do_limit_order(arr, dist, lenght, sign):

    if sign == 1:
        best_price = np.where(arr > 0)[0][-1]
        opposite = np.where(arr < 0)[0][0]
        pos = 10e10
        while  pos >= opposite or pos < 0:
            pos = best_price + int(dist.rvs())


    else:
        best_price = np.where(arr < 0)[0][0]
        opposite = np.where(arr > 0)[0][-1]
        pos = 0
        while pos <= opposite or pos >= lenght:
            pos = best_price - int(dist.rvs())

    return pos

def do_market_order(arr, sign):
    if sign == 1:
        pos  = np.where(arr < 0)[0][0]
    else:
        pos = np.where(arr > 0)[0][-1]

    return pos

def find_next_order(lob, vol_lob, f_lo, f_mo, f_c, k, energy, mo_s, mo_n):
    # Calcola il numero totale di ordini nel bid e ask side del book
    bid_size = vol_lob[vol_lob > 0].shape[0]
    ask_size = vol_lob[vol_lob < 0].shape[0]
    # Calcola i rate di arrivo di LO, MO e cancellazioni utilizzando lo spread
    # per TSLA e utilizzando spread e volume totale per dati energetici.
    sp = ZI.find_spread(lob)
    if energy is False:
        l_rate = f_lo.find_rate(sp, 1)
        m_rate = f_mo.find_rate(sp, 1)
        c_rate = f_c.find_rate(sp, 1)

    if energy is True:
        n_orders = vol_lob[vol_lob > 0].sum() - vol_lob[vol_lob < 0].sum()
        l_rate = f_lo.find_rate(sp, n_orders)
        m_rate = f_mo.find_rate(sp, n_orders)
        c_rate = f_c.find_rate(sp, n_orders)

    tot = l_rate + m_rate + c_rate
    probs = [l_rate / tot, m_rate / tot, c_rate  / tot]
    FLAG = False
    # Scegli un ordine con probabilità uguale al valore dei rate di arrivo
    # e con segno:
    # 1) Casuale per LO.
    # 2) Casuale per MO se non viene dato in input un esponente di Hurst altrimenti
    #    il segno viene estratto dal segno di un rumore gaussiano frazionario.
    # 3) Il segno per cancellazioni viene scelto con probabilità uguale al rapporto
    #    fra il numero di ordini in ogni lato del book.
    while FLAG is False:
        next = np.random.choice([0, 1, 2], p = probs)
        if mo_n is None:
            if next != 2:
                sign = ZI.rand_sign()
            else:
                tt = bid_size + ask_size
                sign = np.random.choice([1, -1], p = [bid_size / tt, ask_size / tt])
        else:
            if next == 0:
                sign = ZI.rand_sign()
            elif next == 1:
                sign = int(mo_s[mo_n])
                mo_n += 1
            else:
                tt = bid_size + ask_size
                sign = np.random.choice([1, -1], p = [bid_size / tt, ask_size / tt])
        # Questa parte serve per assicurare che non vengano cancellati ordini o
        # non vengano eseguiti MO che possono portare il book ad avere 0 quote
        # nell'ask o nel bid
        if bid_size > 2 and ask_size > 2:
            FLAG = True

        elif bid_size < 3 and sign == 1 and next == 2:
            FLAG= False

        elif bid_size < 3 and sign == -1 and next == 1:
            FLAG= False

        elif ask_size < 3 and  sign == -1 and next == 2:
            FLAG= False

        elif ask_size < 3 and sign == 1 and next == 1:
            FLAG= False

        else:
            FLAG = True

    return next, sign, mo_n

def add_order_to_queue(vol_lob, pos, volume):
    # Aggiungo un ordine nella matrice vol_lob
    idx = np.where(vol_lob[:,pos] == 0)[0][0]
    vol_lob[idx, pos] = volume

def remove_order_from_queue(vol_lob, pos, idx):
    # Rimuovo un ordine dalla matrice vol_lob
    removed = vol_lob[idx,pos]
    vol_lob[idx:-1, pos] = vol_lob[idx+1:, pos]
    vol_lob[-1, pos] = 0
    return removed


def do_mo_queue(vol_lob, pos, volume, sign):
    if volume < np.abs(vol_lob[0, pos]):
        vol_lob[0,pos] += volume*sign

    elif volume == 0:
        pass

    elif volume == np.abs(vol_lob[0,pos]):
        _ = remove_order_from_queue(vol_lob, pos, 0)

    elif volume > np.abs(vol_lob[0, pos]):
        volume -= np.abs(vol_lob[0,pos])
        _ = remove_order_from_queue(vol_lob, pos, 0)
        do_mo_queue(vol_lob,pos, volume, sign)

def do_co_queue(vol_lob, pos):
    lenght = vol_lob[:,pos][vol_lob[:,pos] != 0].shape[0]
    idx = np.random.choice(np.arange(lenght))
    removed = remove_order_from_queue(vol_lob, pos, idx)
    return np.abs(removed)


def do_order(lob, vol_lob, f_lo, f_mo, f_c, lo_placement,lo_volume, mo_volume,
             delta, k, a, s, energy, mo_s = None, mo_n = None):
    # Trova il segno e il tipo del prossimo ordine
    if mo_n is None:
        o_type, sign, _ = find_next_order(lob, vol_lob, f_lo, f_mo, f_c, k, energy, mo_s, mo_n)
    else:
        o_type, sign, mo_n = find_next_order(lob, vol_lob, f_lo, f_mo, f_c, k, energy, mo_s, mo_n)
    # Calcola spread e mid price
    mp = ZI.find_mid_price(lob)
    sp = ZI.find_spread(lob)

    # Se il prossimo ordine è un LO, il prezzo e il volume vengono estratti da
    # opportune distribuzioni passate in input.
    if o_type == 0:
        distribution = lo_placement.find_distribution(sp)
        price = do_limit_order(lob, distribution, k, sign)
        volume = int(lo_volume.rvs())
        lob[price] += volume * sign
        add_order_to_queue(vol_lob, price, volume*sign)

    # Se invece l'ordine è un MO:
    elif o_type == 1:
        # Calcola il volume totale nell'ask e nel bid e fissa il prezzo come
        # il prezzo del miglior ask se il segno è +1, o il prezzo del miglior bid
        # altrimenti
        bid_size = vol_lob[vol_lob > 0].sum()
        ask_size = -vol_lob[vol_lob < 0].sum()
        price = do_market_order(lob, sign)

        # Per scegliere il volume, con una certa probabilità delta passata in
        # input fissa il volume del MO uguale al volume del best price,
        # altrimenti fa sampling da una distribuzione passata in input.
        volume = 10e10
        if sign == 1:
            best_vol = -lob[price]
            while volume >= ask_size - 1:
                rdn = np.random.random()
                if rdn < delta:
                    volume = best_vol
                else:
                    volume = np.ceil(mo_volume.rvs() * best_vol)
        else:
            best_vol = lob[price]
            while volume >= bid_size - 1:
                rdn = np.random.random()
                if rdn < delta:
                    volume = best_vol
                else:
                    volume = np.ceil(mo_volume.rvs() * best_vol)

        # Finché il volume del MO è maggiore del volume al best price
        # consuma tutto il best price e passa alla quota successiva.
        while np.abs(lob[price]) < volume:
            to_remove = np.abs(lob[price])
            volume -= to_remove
            lob[price] = 0
            do_mo_queue(vol_lob, price, to_remove, sign)
            price = do_market_order(lob, sign)
        lob[price] += volume * sign
        do_mo_queue(vol_lob, price, volume, sign)

    # Infine se l'ordine è una cancellazione, l'ordine da cancellare viene scelto
    # facendo sampling dalla distribuzine del priority index, mentre il volume
    # viene scelto uguale al volume dell'ordine da cancellare
    else:
        price = do_cancel_order(lob, sign, a, s)
        volume = do_co_queue(vol_lob, price)
        lob[price] -= sign * volume

    return price, sign, o_type, volume, mo_n

def update_order(arr, order, i, p0):
    # Check the number of non empty levels
    n_quote_bid = arr[arr > 0].shape[0]
    n_quote_ask = arr[arr < 0].shape[0]
    # Update bid price and bid volume for the first 10 level
    for n in range(min(10, n_quote_bid)):
        order[f"Bid Price{n+1}"][i] = np.where(arr > 0)[0][-n-1] + p0
        order[f"Bid Size{n+1}"][i] = arr[arr > 0][-n-1]
    # Update ask price and ask volume for the first 10 level
    for n in range(min(10, n_quote_ask)):
        order[f"Ask Price{n+1}"][i] = np.where(arr < 0)[0][n] + p0
        order[f"Ask Size{n+1}"][i] = -arr[arr < 0][n]


def sim_LOB(f_lo, f_mo, f_c, alpha, sigma, lo_placement, lo_volume, mo_volume, delta,
             m0 = 0, k = 100, iterations = 10_000, burn = 5_000, n_tot = 100, energy = False,
             hurst = None, lam = 12000):
    """
    Codice per le simulazioni del LOB utilizzando il modello Ratio.

    Input:
        1. f_lo: Ratio_l()
            Funzione logistica utilizzata per calcolare il rate di arrivo dei
            LO a partire dallo Spread e dal volume totale nel book.
        2. f_mo: Ratio()
            Funzione logistica utilizzata per calcolare il rate di arrivo dei
            MO a partire dallo Spread e dal volume totale nel book.
        3. f_c: Ratio()
            Funzione logistica utilizzata per calcolare il rate di arrivo delle
            cancellazioni a partire dallo Spread e dal volume totale nel book.
        4. alpha: float
            Esponente della power law utilizzata per modellare la distribuzione del
            priority index.
        5. sigma: float
            Parametro di scaling utilizzato per modellare la distribuzione del
            priority index.
        6. lo_placement: FamilyDistribution()
            Distribuzioni utilizzate per calcolare il piazzamento dei LO.
            L'oggetto in input è una custom class chiamata FamilyDistribution
            che permette di utilizzare distribuzioni differenti in base allo spread.
        7. lo_volume: scipy.stats.rv_histogram o scipy.stats.rv_discrete
            Distribuzione dei volumi dei LO.
        8. mo_volume: scipy.stats.rv_histogram, .rv_discrete, o .rv_continuous
            Distribuzione dei volumi dei MO condizionati al volume del best price.
        9. delta: float
            Frazione dei MO che hanno volume pari al volume del best price.
        10. m0: int (default = 0)
            Mid Price iniziale
        11. k: int (default  = 100)
            Numero totale di price levels nel LOB da simulare, meglio avere un
            numero il più alto possibile.
        12. iterations: int (default = 10_000)
            Numero totale delle iterazioni della simulazione
        13. burn: int (default = 5_000)
            Numero delle iterazioni iniziali da scartare.
        14. n_tot: int (default = 100)
            Numero iniziale di ordini del book.
        15. energy: {True, False} (default = True)
            Variabile booleana che indica se i rate di arrivo degli ordini
            dipende dal volume totale nel book.
        16. hurst [min = 0.5, max = 1] (default, None)
            Valore dell'esponente di Hurst da usare per simulare il segno dei MO.
            Più il valore è vicino ad 1 e più gli ordini sono correlati.
            Se invece non viene inserito nessun valore, gli il segno degli ordini
            viene considerato totalmente scorrelato.
    Output:
        1. df_m: pd.DataFrame
            Dataframe contenente lo spread, il mid price, il prezzo e il tipo di
            ogni ordine fatto nelle simulazioni.
        2. df_o: pd.DataFrame
            DataFrame contenente il prezzo e il volume delle 10 migliori quote
            dell'ask e del bid, per ogni iterazione della simulazione.
    """

    # La variabile lob è un array che mi indica il volume totale degli ordini ad
    # ogni livello di prezzo, mentre la variabile vol_lob è un array 2D in cui
    # le colonne sono diversi livelli di prezzo e le righe rappresentano la
    # priorità temporale, ovvero gli ordini nella prima riga sono gli ordini
    # che sono da più tempo nel book, gli ordini nella seconda riga sono i
    # penultimi etc.
    #
    # Esempio: mettiamo conto di avere 4 ordini nel book, tre sono al primo livello
    # di prezzo mentre uno è al quarto, in questo caso vol_lob ha forma:
    # 1 0 0 -2
    # 2 0 0 0
    # 1 0 0 0
    #
    # e lob ha forma:
    # [4 0 0 ,-2]
    #
    # Nel caso in cui viene fatto un MO di volume 1 al primo livello di prezzo,
    # vol_lob diventa:
    # 2 0 0 -2
    # 1 0 0 0
    # 0 0 0 0
    
    def ex_time(lam):
        """
        in seconds
        """
        events = np.random.poisson(lam)
        event_time = (60*60) / events
        return event_time


    # Inizilizzo lob, vol_lob e i dizioniari dei messagi e dello stato del book
    lob = np.zeros(k, dtype = np.int16)
    lob[int(k//2 - n_tot//2):k//2] = 1
    lob[k//2:int(k//2 + n_tot //2)] = -1

    vol_lob = np.zeros((20,k), dtype = np.int16)
    vol_lob[0, int(k//2 - n_tot//2):k//2] = 1
    vol_lob[0, k//2:int(k//2 + n_tot //2)] = -1

    order, message = ZI.initialize_order_message(iterations)
    message['Time'] = np.full((iterations, ), 0)

    # Simulo il segno dei MO utilizzando un il rumore di un processo gaussiano
    # frazionario avente esponente di hurst fissato in input.
    # Se in input non viene dato un esponente il segno dei MO è scelto casualmente.
    if hurst is None:
        mo_s = None
        mo_n = None
    else:
        mo_s = np.sign(fgn(n=iterations//10, hurst=hurst, length=1, method='daviesharte'))
        mo_n = 0

    # Simulo il LOB scartando le prime iterazioni
    for i in range(burn):
        do_order(lob, vol_lob, f_lo, f_mo, f_c, lo_placement, lo_volume, mo_volume,
                    delta, k, alpha, sigma, energy)
    
    # Simulo il LOB
    for i in range(iterations):
        # Printo la percentuale di completamentp
        percentage = i / iterations * 100
        print(f"{percentage:.2f}", end = "\r")
        # Simulo un ordine e aggiorno i dizionari dei messaggi e dello stato del LOB
        message["Price"][i], message["Sign"][i], message["Type"][i], \
        message["Volume"][i], mo_n = do_order(lob, vol_lob, f_lo, f_mo, f_c, lo_placement,
                    lo_volume, mo_volume, delta, k, alpha, sigma, energy, mo_s, mo_n)

        message["Spread"][i] = ZI.find_spread(lob)
        message["MidPrice"][i] = ZI.find_mid_price(lob) + m0
        if i > 0:
            message['Time'][i] = message['Time'][i-1] + ex_time(lam)*1e9

        update_order(lob, order, i, m0)

    # Converto i dizionari dei messaggi e stato del book in DatFrame
    df_m = pd.DataFrame(message)
    df_o = pd.DataFrame(order)

    # Piccole correzioni
    df_m["Price"] += m0
    df_m["Type"].replace([0,1,2], ["Limit", "Market", "Cancel"], inplace = True)
    ZI.fix_zero_volume(df_o)

    return df_m, df_o
