# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:31:07 2020

@author: khoih
"""


import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm


def graph_candles(df):
    df.reset_index(inplace=True)
    print (df.head())
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.show()
def close_price(df):
    for c in df.columns.values:
        plt.plot(df[c], label = c)
        #plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close price $USD', fontsize = 18)
    plt.legend(df.columns.values, loc = 'upper left')
    plt.show()
def normalized_price(df):
    (df/df.iloc[0]*100).plot(figsize = (8,5))
    plt.show()
    
def hist_returns(returns):
    returns.hist(bins = 50, figsize = (9,6))
    
def qqplot(df):
    sm.qqplot(df, line = 's')
    plt.grid(True)
    plt.xlabel('Theoretical quantiles')

def monte_carlo_plot(pvol, prets):
    plt.figure(figsize = (8,4))
    plt.scatter(pvol,prets, c = prets/pvol, marker = 'o')
    plt.grid = True
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label = 'Sharpe ratio')
