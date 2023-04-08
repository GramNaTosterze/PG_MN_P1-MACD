#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import math
#%% definicje funkcji
def EMA(current_sample, span, samples):
    alpha = 2/(span+1)
    g = 0
    d = 0
    for i in range(span):
        alphaI = (1 - alpha) ** i
        if current_sample - i >= 0:
            g += alphaI * samples[current_sample - i]
        else:
            g += alphaI * samples[0]
        d += alphaI
    return g/d

def MACD(current_sample):
    return EMA(current_sample, 12, samples) - EMA(current_sample, 26, samples)

def SIGNAL(current_sample):
    return EMA(current_sample, 9, macd)

def plot_samples(sample_plot):
    sample_plot.plot(date, samples)
    sample_plot.set_title("Notowania Nintendo Co. Ltd ADR")
    sample_plot.set_ylabel("Notowania firmy($)")
    plt.grid()
    sample_plot.xaxis.set_tick_params(labelsize=5, rotation=45)
    return sample_plot
    
def plot_macd(macd_plot):
    macd_plot.set_ylabel("wartości MACD")
    macd_plot.set_xlabel("Kolejne dni")
    macd_plot.set_title("Wskaźnik MACD")
    
    macd_plot.plot(date, macd, label = 'MACD')
    macd_plot.plot(date, sig, label = 'SIGNAL')
    macd_plot.legend(loc = 'lower right')
    plt.grid()
    macd_plot.xaxis.set_tick_params(labelsize=5, rotation=45)

def plot_trends(sample_plot):
    buy_price = []
    sell_price = []
    for i in range(n):
        if( intersections[i] == 'Buy'):
            buy_price.append(samples[i])
            sell_price.append(np.nan)
        elif ( intersections[i] == 'Sell'):
            buy_price.append(np.nan)
            sell_price.append(samples[i])
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)

    sample_plot.plot(date, buy_price, marker = '^', color = 'green', label = 'sygnał kupna', linewidth = 0)
    sample_plot.plot(date, sell_price, marker = 'v', color = 'r', label = 'sygnał sprzedarzy', linewidth = 0)
    sample_plot.legend()

def plot_gains():
    sell1 = simulation(None,None, '1')
    sell50p = simulation(1/2,1/2, '50p')
    sellAll = simulation(1,1, 'All')

    plt.clf()
    plt.grid()
    s1 = plt.subplot2grid((8,1), (0,0), rowspan = 2, colspan = 1)
    plt.title("Zestawienie zysków dla 3 metod kupna i sprzedarzy")
    s1.bar(date, sell1)
    plt.grid()
    s1.xaxis.set_tick_params(labelsize=5, rotation=45)
    s50p = plt.subplot2grid((8,1), (2,0), rowspan = 3, colspan = 1)
    plt.ylabel("Zysk ($)")
    plt.grid()
    s50p.xaxis.set_tick_params(labelsize=5, rotation=45)
    s50p.bar(date, sell50p, color = 'green')
    sAll = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)
    plt.grid()
    sAll.bar(date, sellAll, color = 'purple')
    sAll.xaxis.set_tick_params(labelsize=5, rotation=45)

def signals():
    intersections = ['Ignore']
    for i in range(1,n):
        if(macd[i-1]  < sig[i-1] and macd[i] > sig[i] and macd[i] < 0):
            intersections.append('Buy')
        elif(macd[i-1]  > sig[i-1] and macd[i] < sig[i] and macd[i] > 0):
            intersections.append('Sell')
        else:
            intersections.append('Ignore')
    return intersections

def simulation(buy_mull, sell_mull, fileName):
    
    df_capital = []
    df_stocks = []
    df_earnings = []
    capital = 1000
    stocks: int = 0
    days_to_check = [100, 200, 300, 400, 500, 700, 900, 999]
    for i in range(n):
        if( intersections[i] == 'Buy' and capital >= samples[i]):
            #kupno
            buy_stocks = int(sell_mull * capital / samples[i]) if buy_mull is not None else 1
            capital -= samples[i] * buy_stocks
            stocks += buy_stocks
        elif ( intersections[i] == 'Sell' and stocks >= 1):
            #sprzedarz
            sell_stocks = int(stocks * sell_mull) if sell_mull is not None else 1
            capital += samples[i] * sell_stocks
            stocks -= sell_stocks   
        if (i in days_to_check):
            df_capital.append(capital)
            df_stocks.append(stocks)
        df_earnings.append(round(capital + samples[i]*stocks - 1000, 2))
    print("kapitał ostateczny: ", round(capital + samples[n-1]*stocks, 2))
    print("zysk :", round(capital + samples[n-1]*stocks - 1000, 2), '\n')
    
    tradeHistory = pd.DataFrame({
        'Dzień': days_to_check, 
        'Kapitał': df_capital,
        'Akcje': df_stocks, 
        'Zarobek': np.array(df_earnings)[np.array(days_to_check)]
    })
    tradeHistory.to_latex('PG_MN_Proj1_Data/TradeHistory'+fileName+'.tex', index=False)
    return df_earnings



# %% wczytanie danych
HistoricalData = pd.read_csv('PG_MN_Proj1_Data/NTDOY.csv')

# obliczanie wskaźnika MACD i SIGNAL
n = 1000
samples = np.array( [HistoricalData.loc[i]['Close'] for i in range(n)] )
date = HistoricalData.loc[0:n-1]['Date']
date = [datetime.strptime(d, '%Y-%m-%d') for d in date]
macd = np.array([MACD(i) for i in range(n)])
sig = np.array([SIGNAL(i) for i in range(n)])

# Wykres wziętych próbek

#   Wykres cen
plot_samples(plt.subplot())
plt.savefig('Images/Samples.png')
plt.clf()
plot_macd(plt.subplot())
plt.savefig('Images/MACD.png')


#  %% Symulacja - {handel po 1, sprzedarz 50% posiadanych i kupno 50% możliwych, sprzedarz i kupno wszystkiego}
intersections = signals()
plot_gains()
plt.savefig('Images/MethodComparision.png')

#%matplotlib auto
plt.clf()
subplot = plot_samples(plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1))
subplot.set_title("Zestawienie Notowań ze wskaźnikiem MACD")
plot_macd(plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1))
plt.title("") # usunięcie tytułu dolnego wykresu aby nie nachodził na górny
plot_trends(subplot)

plt.savefig('Images/TradeSignals.png')
plt.show()

# %%
