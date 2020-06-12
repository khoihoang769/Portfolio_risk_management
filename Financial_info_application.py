# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:36:19 2020

@author: khoih
"""
import pandas as pd
import numpy as np
from OOP_Financial_info import FinancialInfo
import matplotlib.pyplot as plt
# Stock_data_frames = pd.DataFrame()

google = FinancialInfo('GOOGL','1d','2020-1-31','2020-5-31')
microsoft = FinancialInfo('MSFT','1d','2020-1-31','2020-5-31')
apple = FinancialInfo('AAPL','1d','2020-1-31','2020-5-31')
jpm = FinancialInfo('JPM','1d','2020-1-31','2020-5-31')
amazon = FinancialInfo('AMZN','1d','2020-1-31','2020-5-31')
facebook = FinancialInfo('FB','1d','2020-1-31','2020-5-31')


google.get_data()
microsoft.get_data()
apple.get_data()
jpm.get_data()
amazon.get_data()
facebook.get_data()

google.get_financial_info()
microsoft.get_financial_info()
apple.get_financial_info()
jpm.get_financial_info()
amazon.get_financial_info()
facebook.get_financial_info()


FinancialInfo.change_risk_free(0.05)
# FinancialInfo.change_weight([0.1,0.1,0.1,0.1,0.1,0.5])

FinancialInfo.port_weight([0.1,0.1,0.1,0.1,0.1,0.5])

# FinancialInfo.change_weight([0.2,0.2,0.3,0.1,0.1,0.1])

FinancialInfo.port_weight([0.2,0.2,0.3,0.1,0.1,0.1])

FinancialInfo.port_weight([0.12,0.12,0.16,0.15,0.15,0.3])








