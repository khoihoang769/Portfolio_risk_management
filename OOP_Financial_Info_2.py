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
import scipy.optimize as sco

from scipy.stats import norm 
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns
from pypfopt.expected_returns import mean_historical_return
from pypfopt import risk_models
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.cla import CLA
from pypfopt.discrete_allocation import DiscreteAllocation,get_latest_prices

class FinancialInfo:
    num_stocks = 0
    frames = []
    prices = []
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
        
        self.tickerdf[self.symbol] = self.tickerdf['Close'].pct_change()
        self.tickerdf[self.symbol + '_Close'] = self.tickerdf['Close']
        self.percent_return = self.tickerdf[self.symbol]*100
        self.mean_return_daily = np.mean(self.tickerdf[self.symbol])
        self.mean_return_annualized = ((1 + self.mean_return_daily)**252)-1
        self.sigma_daily = np.std(self.tickerdf[self.symbol])
        self.variance_daily = self.sigma_daily**2
        self.sigma_annualized = self.sigma_daily*np.sqrt(252)
        self.variance_annualized = self.sigma_annualized**2
        
        self.clean_returns = self.tickerdf[self.symbol].dropna()
        FinancialInfo.frames.append(self.tickerdf[self.symbol])
        FinancialInfo.prices.append(self.tickerdf[self.symbol + '_Close'])
        
    def get_stock_sharpe(self):
        self.tickerdf['Sharpe'] = (self.tickerdf[self.symbol] - FinancialInfo.risk_free)/ self.sigma_annualized
        print (self.tickerdf['Sharpe'].describe()[['min' , 'max']])
        
    def rolling_ma(self, num):
        ma = num
        smaString = 'Sma_' + str(ma)
        numH = 0
        numC = 0
        self.tickerdf[smaString] = self.tickerdf.iloc[:,3].rolling(window = ma).mean()
        self.tickerdf = self.tickerdf.iloc[ma:]
        
        fig, ax = plt.subplots(figsize = (8,4))
        ax.plot(self.tickerdf.index, self.tickerdf['Close'], label = self.symbol)
        ax.plot(self.tickerdf[smaString].index, self.tickerdf[smaString], label = f'{num} days SMA')
        ax.legend(loc = 'best')
        for i in self.tickerdf.index:
            if (self.tickerdf['Close'][i]>self.tickerdf[smaString][i]):
                numH +=1
            else:
                numC +=1
        print (numH)
        print (numC)
    @classmethod
    def change_risk_free(cls, amount):
        cls.risk_free = amount
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
        
        #Create panel for stocks' close price
        cls.Stock_close_prices = pd.concat(FinancialInfo.prices, axis = 1)
        cls.Stock_close_prices.dropna(inplace = True)
        
        # Compute the annualized average historical return
        cls.mean_returns_avg = mean_historical_return(cls.Stock_close_prices, frequency = 252)
        
        # Calculate the portfolio annual simple return
        cls.simple_returns_annual = np.sum(cls.Stock_data_frames.mean() * cls.weight)*252
        # Calculate the log returns
        
        cls.log_returns = np.log(cls.Stock_close_prices/cls.Stock_close_prices.shift(1))
        cls.log_returns.dropna(inplace = True)
        
        # Calculate the annualized log returns and covariance
        cls.log_returns_annualized = cls.log_returns.mean() * 252
        cls.log_returns_covariance = cls.log_returns.cov() * 252
    
        #Create panel for stocks return multipled by their weights in portfolio
        cls.weightedreturns = FinancialInfo.Stock_data_frames.mul(FinancialInfo.weight, axis = 1)
        
        cls.cummulativereturns = ((1 + FinancialInfo.weightedreturns.sum(axis = 1)).cumprod() - 1)
        # cummulativereturns = weightedreturns.sum(axis = 1)
        portfolio_weights_ew = np.repeat(1/FinancialInfo.num_stocks, FinancialInfo.num_stocks)
        # cummulativereturns_ew = FinancialInfo.Stock_data_frames.iloc[:,0:FinancialInfo.num_stocks].mul(portfolio_weights_ew, axis = 1).sum(axis =1)
        
        # FinancialInfo.cummulativereturns.plot(color = 'Red')
        # cummulativereturns_ew.plot(color = 'Blue')
        
        #Calculate portfolio volatility
        cls.cov_mat = FinancialInfo.Stock_data_frames.cov()
        cls.cov_mat_annual = cls.cov_mat*252
        cls.portfolio_weights = np.array(FinancialInfo.weight)
        cls.portfolio_volatility = np.sqrt(np.dot(cls.portfolio_weights.T, np.dot(cls.cov_mat_annual, cls.portfolio_weights)))
        # print (f'The volatility of this portfoliois:  {portfolio_volatility}')
        FinancialInfo.vol_list.append(cls.portfolio_volatility)
        cls.pfvol = pd.DataFrame(FinancialInfo.vol_list)
        
        # Create the covariance shrinkage instance variable
        cls.cov_shrinkage = CovarianceShrinkage(cls.Stock_close_prices)
        cls.e_cov = cls.cov_shrinkage.ledoit_wolf()
        

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
        # cls.MSRreturns.plot(color = 'Orange')
        
        #Sorting the dataframe by volatility
        cls.df_vol_sorted = cls.df.sort_values(by = ['Volatility'], ascending = True)
        cls.GMV_weights = cls.df_vol_sorted.iloc[0,0:FinancialInfo.num_stocks]
        cls.GMV_weights_array = np.array(cls.GMV_weights)
        cls.GMVreturns = FinancialInfo.Stock_data_frames.iloc[:,0:FinancialInfo.num_stocks].mul(cls.GMV_weights_array, axis = 1).sum(axis = 1)
        # cls.GMVreturns.plot (color = 'Green')
        
        

        
    @classmethod
    def simulation(cls, num):
        for i in range (1,num):
            value = np.random.dirichlet(np.ones(len(FinancialInfo.symbol_list)),size=1)
            lst = value.tolist()
            FinancialInfo.port_weight(lst[0])
    
    @classmethod
    def monte_carlo(cls,num):
        cls.prets = []
        cls.pvol = []
        for p in range (num):
            random_weights = np.random.random(FinancialInfo.num_stocks)
            random_weights /= np.sum(random_weights)
            cls.prets.append(np.sum(cls.Stock_data_frames.mean()*random_weights)*252)
            cls.pvol.append(np.sqrt(np.dot(random_weights.T,
                                           np.dot(cls.e_cov, random_weights))))
        cls.prets = np.array(cls.prets)
        cls.pvol = np.array(cls.pvol)
        
    @classmethod
    def get_efficient_frontier(cls,weights):
        cls.efficient_portfolio = CLA(cls.mean_returns_avg, cls.e_cov)
        (ret,vol,weight) = cls.efficient_portfolio.efficient_frontier()
        plt.show()       
        def statistics(weights):
            pret = np.sum(cls.Stock_data_frames.mean() * weights)*252
            pvol = np.sqrt(np.dot(weights.T, np.dot(cls.e_cov,weights)))
            return np.array([pret, pvol, pret/pvol])
        
        def min_func_sharpe(weights):
            return -statistics(weights)[2]
        
        def min_func_variance(weights):
            return statistics(weights)[1]**2
        
        cons = ({'type':'eq','fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0,1) for x in range (cls.num_stocks))
        ew = FinancialInfo.num_stocks * [1/ cls.num_stocks]
        
        cls.opts = sco.minimize(min_func_sharpe, ew, method = 'SlSQP', bounds = bnds, constraints = cons)
        cls.optv = sco.minimize(min_func_variance, ew, method = 'SlSQP', bounds = bnds, constraints = cons)
        plt.figure(figsize = (10,4))
        plt.scatter(cls.pvol,cls.prets, c = cls.prets/cls.pvol, marker = 'o',cmap='RdYlBu')
        plt.scatter(vol, ret, s = 4, c = 'g', marker = 'x')
        plt.plot(statistics(cls.opts['x'])[1], statistics(cls.opts['x'])[0],
                 'r*', markersize=15)
        plt.plot(statistics(cls.optv['x'])[1], statistics(cls.optv['x'])[0],
                 'y*', markersize=15)
        plt.grid = True
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label = 'Sharpe ratio')
        
    @classmethod
    def optimization(cls, fund):
        mu = mean_historical_return(cls.Stock_close_prices)
        S = risk_models.sample_cov(cls.Stock_close_prices)
        ef = EfficientFrontier(mu,S)
        weights = ef.max_sharpe() #Maximize the Sharpe ratio, and get the raw weights
        cleaned_weights = ef.clean_weights()
        print (cleaned_weights)
        ef.portfolio_performance (verbose = True)
        latest_price = get_latest_prices(cls.Stock_close_prices)
        weights = cleaned_weights
        da = DiscreteAllocation(weights, latest_price, total_portfolio_value = fund)
        allocation, leftover = da.lp_portfolio()
        print("Discrete allocation:", allocation)
        print ("Funds remaining: ${:.2f}".format(leftover))
        
        # Don't know what exactly VaR can contribute
    @classmethod
    def get_var(cls):
        
        var_95 = np.percentile(cls.Portfolio,5)
        
        #Estimate the average daily return and volatility
        mu = np.mean(cls.Portfolio['Portfolio'])
        vol = np.std(cls.Portfolio['Portfolio'])
        confidence_level = 0.05
        #Calculate the parametric VaR
        para_var_95 = norm.ppf(confidence_level,mu,vol)
        print('Mean: ', str(mu), '\nVolatility: ', str(vol), '\nVaR(95): ', str(var_95), '\nParaVaR(95): ', str(para_var_95))
        
        #Need more changes
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

    

        
        