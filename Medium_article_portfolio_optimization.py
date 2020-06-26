# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 08:47:47 2020

@author: khoih
"""

from OOP_Financial_Info_2 import FinancialInfo
import Plotting_functions
import Trade_strategies
FinancialInfo.change_start_end('1d','2013-1-01','2020-3-18')


google = FinancialInfo('GOOGL')
apple = FinancialInfo('AAPL')
netflix = FinancialInfo('NFLX')
amazon = FinancialInfo('AMZN')
facebook = FinancialInfo('FB')

google.get_data()
apple.get_data()
netflix.get_data()
amazon.get_data()
facebook.get_data()

FinancialInfo.port_rets([0.2,0.2,0.2,0.2,0.2])

df = FinancialInfo.Stock_close_prices

# Plotting_functions.normalized_price(df)
# Plotting_functions.close_price(df)

returns = FinancialInfo.Stock_data_frames

cov_matrix_annual = FinancialInfo.cov_mat_annual

port_volatility = FinancialInfo.portfolio_volatility

portfolioSimpleAnnualReturn = FinancialInfo.simple_returns_annual

FinancialInfo.optimization(100000)
FinancialInfo.monte_carlo(2500)
FinancialInfo.get_efficient_frontier(FinancialInfo.portfolio_weights)


# Trade_strategies.strat_rwb(netflix.tickerdf)

# netflix.rolling_ma(50)

# netflix.rolling_ma(100)

# netflix_df = netflix.tickerdf

