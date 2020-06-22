# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:24:10 2020

@author: khoih
"""

import pandas as pd
import numpy as np
from OOP_Financial_Info_2 import FinancialInfo
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
from datetime import datetime
# from Candlesticks_function import add_day_num
from Candlesticks_function import graph_candles


FinancialInfo.change_start_end('1d','2008-1-01','2009-12-31')

citibank = FinancialInfo('C')
morgan = FinancialInfo('MS')
goldman = FinancialInfo('GS')
jpm = FinancialInfo('JPM')

citibank.get_data()
morgan.get_data()
goldman.get_data()
jpm.get_data()


# graph_candles(apple.tickerdf)


citibank.get_financial_info()
morgan.get_financial_info()
goldman.get_financial_info()
jpm.get_financial_info()


# FinancialInfo.change_risk_free(0.002)

FinancialInfo.port_rets([0.25, 0.25, 0.25, 0.25])

asset_returns = FinancialInfo.Stock_data_frames

portfolio_returns = FinancialInfo.Portfolio_returns

covariance = asset_returns.cov() * 252

print (covariance)

portfolio_variance = np.transpose(FinancialInfo.portfolio_weights)@covariance@FinancialInfo.portfolio_weights
portfolio_volatility = np.sqrt(portfolio_variance)
print (portfolio_volatility)

returns_windowed = portfolio_returns.rolling(30)

volatility_series = returns_windowed.std()*np.sqrt(252)

volatility_series.plot().set_ylabel('Annualized volatility, 30-day window')
# FinancialInfo.port_weight([0.2,0.2,0.3,0.1,0.1,0.1])

# FinancialInfo.port_weight([0.12,0.12,0.16,0.15,0.15,0.3])
# FinancialInfo.get_capm()

# FinancialInfo.simulation(25)

# FinancialInfo.get_var()






