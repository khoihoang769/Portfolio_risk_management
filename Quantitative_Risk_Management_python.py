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
import Plotting_functions
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.cla import CLA
# from OOP_Financial_Info_2 import PortfolioOptm
FinancialInfo.change_start_end('1d','2010-12-31','2019-12-31')

citibank = FinancialInfo('C')
morgan = FinancialInfo('MS')
goldman = FinancialInfo('GS')
jpm = FinancialInfo('JPM')

citibank.get_data()
morgan.get_data()
goldman.get_data()
jpm.get_data()


# graph_candles(apple.tickerdf)


# FinancialInfo.change_risk_free(0.002)

FinancialInfo.port_rets([0.25, 0.25, 0.25, 0.25])

asset_returns = FinancialInfo.Stock_data_frames

prices = FinancialInfo.Stock_close_prices

log_returns = FinancialInfo.log_returns

mean_returns = FinancialInfo.mean_returns_avg 

jpm.rolling_ma(20)


# FinancialInfo.monte_carlo(2500)
# FinancialInfo.get_efficient_frontier(FinancialInfo.portfolio_weights)
# FinancialInfo.optimization(100000)

# Plotting_functions.monte_carlo_plot(FinancialInfo.pvol, FinancialInfo.prets)


# Plotting_functions.hist_returns(log_returns)

# Plotting_functions.qqplot(log_returns['C_Close'])
# Plotting_functions.normalized_price(prices)

# plt.plot(mean_returns, linestyle = 'None', marker = 'o')

# e_cov = FinancialInfo.e_cov

# sample_cov = FinancialInfo.cov_mat_annual

# FinancialInfo.get_efficient_frontier()

# FinancialInfo.get_var()

