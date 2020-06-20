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

    
    def __init__(self, symbol, period, start, end):
        self.symbol = symbol
        self.period = period
        self.start = start
        self.end = end
        FinancialInfo.num_stocks += 1
        FinancialInfo.symbol_list.append(self.symbol)
    
    def get_data(self):
        self.ticker = yf.Ticker(self.symbol)
        self.tickerdf = self.ticker.history(period = self.period, start = self.start, end = self.end)
        
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
    def port_weight(cls, list):
        global weightedreturns
        global Stock_data_frames
        global Portfolio
        global Portfolio_returns
        cls.weight = list
        #Create panel for stocks returns
        Stock_data_frames = pd.concat(FinancialInfo.frames, axis = 1)
        Stock_data_frames.dropna(inplace = True)
        
        #Create panel for stocks return multipled by their weights in portfolio
        weightedreturns = Stock_data_frames.mul(FinancialInfo.weight, axis = 1)
        cummulativereturns = ((1 + weightedreturns.sum(axis = 1)).cumprod() - 1)
        # cummulativereturns = weightedreturns.sum(axis = 1)
        portfolio_weights_ew = np.repeat(1/FinancialInfo.num_stocks, FinancialInfo.num_stocks)
        cummulativereturns_ew = Stock_data_frames.iloc[:,0:FinancialInfo.num_stocks].mul(portfolio_weights_ew, axis = 1).sum(axis =1)
        
        # cummulativereturns.plot(color = 'Red')
        # cummulativereturns_ew.plot(color = 'Blue')
        
        
        #Calculate portfolio volatility
        cov_mat = Stock_data_frames.cov()
        cov_mat_annual = cov_mat*252
        portfolio_weights = np.array(FinancialInfo.weight)
        portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
        print (f'The volatility of this portfoliois:  {portfolio_volatility}')
        FinancialInfo.vol_list.append(portfolio_volatility)
        pfvol = pd.DataFrame(FinancialInfo.vol_list)

        #Create the sum of returns of portfolio's stocks
        Portfolio_returns = weightedreturns.sum(axis = 1, skipna = True)
        Portfolio = pd.DataFrame(data = Portfolio_returns, columns = ['Portfolio'])
        Portfolio.drop(Portfolio.index[0], inplace = True)

        #Create a total accmulative returns 
        total_returns = Portfolio_returns.sum()
        
        #Calculate sharpe ratio
        Portfolio_sharpe = (total_returns - FinancialInfo.risk_free)/portfolio_volatility
        print (f'The total returns for this portfolio is: {total_returns}')
        FinancialInfo.returns_list.append(total_returns)
        pfreturns = pd.DataFrame(FinancialInfo.returns_list)
        FinancialInfo.sharpe_list.append(Portfolio_sharpe)
        pfsharpe = pd.DataFrame(FinancialInfo.sharpe_list)
    
        #Create a dataframe with total returns, sharpe ratio, and volatility
        FinancialInfo.wList.append(cls.weight)
        df = pd.DataFrame(np.array(FinancialInfo.wList),columns = FinancialInfo.symbol_list)
        df.insert(6,'Returns', pfreturns)
        df.insert(7,'Volatility', pfvol)
        df.insert(8, 'Sharpe', pfsharpe)

        #Sorting the dataframe by Sharpe ratio
        df_sorted = df.sort_values(by = ['Sharpe'], ascending = False)
        MSR_weights = df_sorted.iloc[0,0:FinancialInfo.num_stocks]        
        MSR_weights_array = np.array(MSR_weights)
        MSRreturns = Stock_data_frames.iloc[:,0:FinancialInfo.num_stocks].mul(MSR_weights_array, axis = 1).sum(axis = 1)
        # MSRreturns.plot(color = 'Orange')
        
        #Sorting the dataframe by volatility
        df_vol_sorted = df.sort_values(by = ['Volatility'], ascending = True)
        GMV_weights = df_vol_sorted.iloc[0,0:FinancialInfo.num_stocks]
        GMV_weights_array = np.array(GMV_weights)
        GMVreturns = Stock_data_frames.iloc[:,0:FinancialInfo.num_stocks].mul(GMV_weights_array, axis = 1).sum(axis = 1)
        # GMVreturns.plot (color = 'Green')
        
    @classmethod
    def simulation(cls, num):
        for i in range (1,num):
            value = np.random.dirichlet(np.ones(len(FinancialInfo.symbol_list)),size=1)
            lst = value.tolist()
            FinancialInfo.port_weight(lst[0])
    
    @classmethod
    def get_var(cls):
        Portfolio_perc = Portfolio*100
        var_95 = np.percentile(Portfolio_perc,5)
        #Sort the returns for ploting
        sorted_rets = sorted(Portfolio_perc['Portfolio'])
        #Plot the probability of each sorted return quantile
        plt.hist(sorted_rets, density = True)
        
        #Denote the VaR 95 quantile
        plt.axvline(x = var_95, color = 'r', linestyle = '-', label = 'VaR 95: {0:.2f}%'. format(var_95))
        plt.show()
        
        #Estimate the average daily return and volatility
        mu = np.mean(Portfolio['Portfolio'])
        vol = np.std(Portfolio['Portfolio'])
        confidence_level = 0.05
        #Calculate the parametric VaR
        para_var_95 = norm.ppf(confidence_level,mu,vol)
        print('Mean: ', str(mu), '\nVolatility: ', str(vol), '\nParaVaR(95): ', str(para_var_95))
        
        
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
        
        print (benchmark_variance)
    @classmethod
    def change_risk_free(cls, amount):
        cls.risk_free = amount
    

        
        