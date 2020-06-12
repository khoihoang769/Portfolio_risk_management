import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # @classmethod
    # def change_weight(cls, list):
    #     cls.weight = list
    
    @classmethod
    def port_weight(cls, list):
        cls.weight = list
        Stock_data_frames = pd.concat(FinancialInfo.frames, axis = 1)
        Stock_data_frames.dropna()
        # print(Stock_data_frames)
        weightedreturns = Stock_data_frames.mul(FinancialInfo.weight, axis = 1)
        # print (Stock_data_frames)
        cummulativereturns = ((1 + weightedreturns.sum(axis = 1)).cumprod() - 1)
        # print (cummulativereturns)
        portfolio_weights_ew = np.repeat(1/FinancialInfo.num_stocks, FinancialInfo.num_stocks)
        cummulativereturns_ew = Stock_data_frames.iloc[:,0:FinancialInfo.num_stocks].mul(portfolio_weights_ew, axis = 1).sum(axis =1)
        
        cummulativereturns.plot(color = 'Red')
        cummulativereturns_ew.plot(color = 'Blue')
        # plt.show()
    
    # @classmethod
    # def get_cov(cls):
    #     global portfolio_volatility
        # correlation_matrix = Stock_data_frames.corr()
        # sns.heatmap(correlation_matrix,
        #             annot = True,
        #             cmap = 'YlGnBu',        
        #             linewidths=0.3,
        #             annot_kws={"size": 8})
        # plt.xticks(rotation = 90)
        # plt.yticks(rotation = 0)
        # plt.show()
        cov_mat = Stock_data_frames.cov()
        cov_mat_annual = cov_mat*252
        portfolio_weights = np.array(FinancialInfo.weight)
        portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
        print (f'The volatility of this portfoliois:  {portfolio_volatility}')
        FinancialInfo.vol_list.append(portfolio_volatility)
        pfvol = pd.DataFrame(FinancialInfo.vol_list)

        Portfolio_returns = weightedreturns.sum(axis = 1)
        total_returns = Portfolio_returns.sum()
        Portfolio_sharpe = (total_returns - FinancialInfo.risk_free)/portfolio_volatility
        print (f'The total returns for this portfolio is: {total_returns}')
        FinancialInfo.returns_list.append(total_returns)
        pfreturns = pd.DataFrame(FinancialInfo.returns_list)
        FinancialInfo.sharpe_list.append(Portfolio_sharpe)
        pfsharpe = pd.DataFrame(FinancialInfo.sharpe_list)
    
        
        FinancialInfo.wList.append(cls.weight)
        df = pd.DataFrame(np.array(FinancialInfo.wList),columns = FinancialInfo.symbol_list)
        df.insert(6,'Returns', pfreturns)
        df.insert(7,'Volatility', pfvol)
        df.insert(8, 'Sharpe', pfsharpe)

        # df['Volatility'] = portfolio_volatility
        print (df)
        
    @classmethod
    def change_risk_free(cls, amount):
        cls.risk_free = amount
    

        
        