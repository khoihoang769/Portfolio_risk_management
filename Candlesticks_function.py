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



def graph_candles(df):
    df.reset_index(inplace=True)
    print (df.head())
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.show()
