# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 06:25:24 2020

@author: khoih
"""

from OOP_Financial_Info_2 import FinancialInfo
import Plotting_functions
import Trade_strategies
from BSM_fct import bsm_call_value

FinancialInfo.change_risk_free(0.05)
FinancialInfo.change_start_end('1d','2020-1-01','2020-7-23')

amazon = FinancialInfo('AMZN')
abbv = FinancialInfo('ABBV')
pfe = FinancialInfo('PFE')
azn = FinancialInfo('AZN')
microsoft = FinancialInfo('MSFT')
tesla = FinancialInfo('TSLA')

amazon.get_data()
abbv.get_data()
pfe.get_data()
azn.get_data()
microsoft.get_data()
tesla.get_data()
FinancialInfo.port_rets([0.4416, 0.1559, 0.1417, 0.0855, 0.0587,0.04])

portfolio_volatility = FinancialInfo.portfolio_volatility

print (portfolio_volatility)

FinancialInfo.optimization(900000)
FinancialInfo.monte_carlo(2500)
FinancialInfo.get_efficient_frontier(FinancialInfo.portfolio_weights)

S = tesla.tickerdf['Close'][-1]

print (S)

amzn3800c = amazon.call_value(3800,0.25)
amzn3000p = amazon.put_value(3000,0.25)
print (round(amzn3800c,3))
print (round(amzn3000p,3))
print (round(tesla.call_value(2000,0.25),3))

abbv.rolling_ma(20)
abbv.get_stock_sharpe()
