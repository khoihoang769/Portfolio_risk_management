#Valuation of European call options in Black-Scholes-Merton model
#incl. Vega function and implied volatility estimation
#bsm_functions.py

#Analytical Black-Scholes-Merton (BSM) formula
from math import log,sqrt,exp
from scipy import stats

def bsm_call_value(S0,K,T,r,sigma):

    
    S0= float(S0)
    d1= (log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2= (log(S0/K)+(r-0.5*sigma**2)*T)/(sigma*sqrt(T))
    value = (S0*stats.norm.cdf(d1,0.0,1.0))-K*exp(-r*T)*stats.norm.cdf(d2,0.0,1.0)
    #stat.norm.cdf => cummulative distribution function for normal distribution
    
    return value

def bsm_put_value(S,K,T,r,sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    
    put = (K * exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0) - S * stats.norm.cdf(-d1, 0.0, 1.0))
    
    return put
def bsm_vega(S0,K,T,r,sigma):
    S0= float(S0)
    d1= (log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    vega = S0*stats.norm.cdf(d1,0.0,1.0)*sqrt(T)
    return vega

def bsm_call_imp_vol(S0,K,T,r,C0,sigma_est,it=100):
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0,K,T,r,sigma_est)-C0)/bsm_vega(S0,K,T,r,sigma_est))
    return sigma_est

# S0=100
# K=120
# T=4
# r=0.05
# sigma=0.2

# bsm_call_value(S0,K,T,r,sigma)
# value = bsm_call_value(S0,K,T,r,sigma)
# put = bsm_put_value(S0,K,T,r,sigma)
# vega = bsm_vega(S0,K,T,r,sigma)

