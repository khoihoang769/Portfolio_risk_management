# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:48:49 2020

@author: khoih
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import datetime
from scipy.stats import norm 

class FinancialInfo:
    num_stocks = 0
    frames = []
    weight = []
    symbol_list = []
    risk_free = 0
    wList = []
    returns_list = []
    vol_list = []
    sharpe_list = []
    start = ''
    end = ''
    period = ''

    @classmethod
    def change_start_end(cls,period,start,end):
        cls.period = period
        cls.start = start
        cls.end = end
        
    def __init__(self, symbol):
        self.symbol = symbol
        FinancialInfo.num_stocks += 1
        FinancialInfo.symbol_list.append(self.symbol)
    
    def get_data(self):
        self.ticker = yf.Ticker(self.symbol)
        self.tickerdf = self.ticker.history(period = FinancialInfo.period, start = FinancialInfo.start, end = FinancialInfo.end)
        
    def get_financial_info(self):
        self.tickerdf[self.symbol] = self.tickerdf['Close'].pct_change()
        self.percent_return = self.tickerdf[self.symbol]*100
        self.mean_return_daily = np.mean(self.tickerdf[self.symbol])
        self.mean_return_annualized = ((1 + self.mean_return_daily)**252)-1
        self.sigma_daily = np.std(self.tickerdf[self.symbol])
        self.variance_daily = self.sigma_daily**2
        self.sigma_annualized = self.sigma_daily*np.sqrt(252)
        self.variance_annualized = self.sigma_annualized**2
        
        self.clean_returns = self.tickerdf[self.symbol].dropna()
        FinancialInfo.frames.append(self.tickerdf[self.symbol])
        
    def get_stock_sharpe(self):
        self.tickerdf['Sharpe'] = (self.tickerdf[self.symbol] - FinancialInfo.risk_free)/ self.sigma_annualized
        print (self.tickerdf['Sharpe'].describe()[['min' , 'max']])
        
    def get_plots(self):
        return self.tickerdf[self.symbol].plot()

    
    @classmethod
    def port_rets(cls, list):
        global weightedreturns
        global Stock_data_frames
        global Portfolio
        global Portfolio_returns
        cls.weight = list
        #Create panel for stocks returns
        cls.Stock_data_frames = pd.concat(FinancialInfo.frames, axis = 1)
        cls.Stock_data_frames.dropna(inplace = True)
    
        #Create panel for stocks return multipled by their weights in portfolio
        cls.weightedreturns = FinancialInfo.Stock_data_frames.mul(FinancialInfo.weight, axis = 1)
        
        cls.cummulativereturns = ((1 + FinancialInfo.weightedreturns.sum(axis = 1)).cumprod() - 1)
        # cummulativereturns = weightedreturns.sum(axis = 1)
        portfolio_weights_ew = np.repeat(1/FinancialInfo.num_stocks, FinancialInfo.num_stocks)
        cummulativereturns_ew = FinancialInfo.Stock_data_frames.iloc[:,0:FinancialInfo.num_stocks].mul(portfolio_weights_ew, axis = 1).sum(axis =1)
        
        FinancialInfo.cummulativereturns.plot(color = 'Red')
        # cummulativereturns_ew.plot(color = 'Blue')
        
        #Calculate portfolio volatility
        cls.cov_mat = FinancialInfo.Stock_data_frames.cov()
        cls.cov_mat_annual = cls.cov_mat*252
        cls.portfolio_weights = np.array(FinancialInfo.weight)
        cls.portfolio_volatility = np.sqrt(np.dot(cls.portfolio_weights.T, np.dot(cls.cov_mat_annual, cls.portfolio_weights)))
        # print (f'The volatility of this portfoliois:  {portfolio_volatility}')
        FinancialInfo.vol_list.append(cls.portfolio_volatility)
        cls.pfvol = pd.DataFrame(FinancialInfo.vol_list)
        

        #Create the sum of returns of portfolio's stocks
        cls.Portfolio_returns = FinancialInfo.weightedreturns.sum(axis = 1, skipna = True)
        cls.Portfolio = pd.DataFrame(data = cls.Portfolio_returns, columns = ['Portfolio'])
        cls.Portfolio.drop(cls.Portfolio.index[0], inplace = True)
        
        # Annualized portfolio volatility 30-day windows
        cls.returns_windowed = cls.Portfolio_returns.rolling(30)
        cls.volatility_series = cls.returns_windowed.std() * np.sqrt(252)

        #Create a total accmulative returns 
        cls.total_returns = cls.Portfolio_returns.sum()
        
        #Calculate sharpe ratio
        cls.Portfolio_sharpe = (cls.total_returns - FinancialInfo.risk_free)/cls.portfolio_volatility
        # print (f'The total returns for this portfolio is: {total_returns}')
        FinancialInfo.returns_list.append(cls.total_returns)
        cls.pfreturns = pd.DataFrame(FinancialInfo.returns_list)
        FinancialInfo.sharpe_list.append(cls.Portfolio_sharpe)
        cls.pfsharpe = pd.DataFrame(FinancialInfo.sharpe_list)
    
        #Create a dataframe with total returns, sharpe ratio, and volatility
        FinancialInfo.wList.append(cls.weight)
        cls.df = pd.DataFrame(np.array(FinancialInfo.wList),columns = FinancialInfo.symbol_list)
        cls.df.insert(FinancialInfo.num_stocks,'Returns', cls.pfreturns)
        cls.df.insert(FinancialInfo.num_stocks + 1,'Volatility', cls.pfvol)
        cls.df.insert(FinancialInfo.num_stocks + 2, 'Sharpe', cls.pfsharpe)

        #Sorting the dataframe by Sharpe ratio
        cls.df_sorted = cls.df.sort_values(by = ['Sharpe'], ascending = False)
        cls.MSR_weights = cls.df_sorted.iloc[0,0:FinancialInfo.num_stocks]        
        cls.MSR_weights_array = np.array(cls.MSR_weights)
        cls.MSRreturns = FinancialInfo.Stock_data_frames.iloc[:,0:FinancialInfo.num_stocks].mul(cls.MSR_weights_array, axis = 1).sum(axis = 1)
        # MSRreturns.plot(color = 'Orange')
        
        #Sorting the dataframe by volatility
        cls.df_vol_sorted = cls.df.sort_values(by = ['Volatility'], ascending = True)
        cls.GMV_weights = cls.df_vol_sorted.iloc[0,0:FinancialInfo.num_stocks]
        cls.GMV_weights_array = np.array(cls.GMV_weights)
        cls.GMVreturns = FinancialInfo.Stock_data_frames.iloc[:,0:FinancialInfo.num_stocks].mul(cls.GMV_weights_array, axis = 1).sum(axis = 1)
        # GMVreturns.plot (color = 'Green')
        
    @classmethod
    def simulation(cls, num):
        for i in range (1,num):
            value = np.random.dirichlet(np.ones(len(FinancialInfo.symbol_list)),size=1)
            lst = value.tolist()
            FinancialInfo.port_weight(lst[0])
    
    @classmethod
    def get_var(cls):
        
        var_95 = np.percentile(Portfolio,5)
        
        #Estimate the average daily return and volatility
        mu = np.mean(Portfolio['Portfolio'])
        vol = np.std(Portfolio['Portfolio'])
        confidence_level = 0.05
        #Calculate the parametric VaR
        para_var_95 = norm.ppf(confidence_level,mu,vol)
        print('Mean: ', str(mu), '\nVolatility: ', str(vol), '\nVaR(95): ', str(var_95), '\nParaVaR(95): ', str(para_var_95))
        
        
    @classmethod
    def get_capm(cls):
        #Building Portfolio Dataframe for CAPM
        Portfolio.insert(1,'RF', FinancialInfo.risk_free)
        Portfolio['Portfolio_excess'] = Portfolio['Portfolio'] - Portfolio['RF']
        # CumulativeReturns = ((1 + Portfolio[['Portfolio','Portfolio_excess']]).cumprod()-1)
        # CumulativeReturns.plot()
        
        #Getting Market Returns
        start = datetime.datetime(2020,1,31)
        end = datetime.datetime(2020,5,31)
        SP500 = web.DataReader(['sp500'], 'fred', start,end)
        SP500['daily_return'] = (SP500['sp500']/SP500['sp500'].shift(1))-1
        SP500.dropna(inplace = True)
        
        
        #Inserting Market Excess into Portfolio Dataframe
        Portfolio.insert(3, 'Market_returns', SP500['daily_return'])
        Portfolio['Market_excess'] = Portfolio['Market_returns'] - Portfolio['RF']
        
        #Getting CAPM benchmarks
        
        covariance_matrix = Portfolio[['Portfolio_excess', 'Market_excess']].cov()
        covariance_coefficient = covariance_matrix.iloc[0,1]
        
        benchmark_variance = Portfolio['Market_excess'].var()
        portfolio_beta = covariance_coefficient / benchmark_variance
        
        print (f'The benchmark variance is {benchmark_variance}')
    @classmethod
    def change_risk_free(cls, amount):
        cls.risk_free = amount
    

        
        