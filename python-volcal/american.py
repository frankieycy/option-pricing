import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import bisect, newton

amPrx_Stree = dict()
amPrx_Otree = dict()

def PriceAmericanOption(spotPrice, forwardPrice, strike, maturity, riskFreeRate, impliedVol, optionType, timeSteps=2000):
    # Price American option via Cox binomial tree (d = 1/u)
    # Assume continuous dividend, reflected in forward price
    # TO-DO: proportional/discrete dividend (difficult!)
    global amPrx_Stree, amPrx_Otree
    S,F,K,T,r,sig,n = spotPrice,forwardPrice,strike,maturity,riskFreeRate,impliedVol,timeSteps
    if optionType == 'call':
        payoff = lambda St: np.maximum(St-K,0)
    else:
        payoff = lambda St: np.maximum(K-St,0)
    dt = T/n
    D = np.exp(-r*dt)
    u = np.exp(sig*np.sqrt(dt))
    d = 1/u
    R = (F/S)**(1/n)
    p = (R-d)/(u-d)
    q = 1-p
    Stag = 'S=%.4f,F=%.4f,T=%.4f,r=%.4f,sig=%.4f,n=%d' % (S,F,T,r,sig,n)

    # Tree caching
    if Stag in amPrx_Stree:
        Stree = amPrx_Stree[Stag]
    else:
        Stree = np.zeros((n+1,n+1))
        for i in range(n+1):
            Stree[i,:i+1] = S*u**np.arange(-i,i+1,2)
        amPrx_Stree[Stag] = Stree
    if n in amPrx_Otree:
        Otree = amPrx_Otree[n]
    else:
        Otree = np.zeros((n+1,n+1))
        amPrx_Otree[n] = Otree

    Otree[-1] = payoff(Stree[-1])
    for i in range(n-1,-1,-1):
        Otree[i,:i+1] = np.maximum(D*(p*Otree[i+1,1:i+2]+q*Otree[i+1,0:i+1]), payoff(Stree[i,:i+1]))

    return Otree[0,0]

def AmericanOptionImpliedVol(spotPrice, forwardPrice, strike, maturity, riskFreeRate, priceMkt, optionType, timeSteps=1000, method="Bisection"):
    # Implied flat volatility under Cox binomial tree
    def objective(impVol):
        return PriceAmericanOption(spotPrice, forwardPrice, strike, maturity, riskFreeRate, impVol, optionType, timeSteps) - priceMkt
    impVol = 0
    if method == "Bisection":
        impVol = bisect(objective, 0.01, 1)
    elif method == "Newton":
        impVol = newton(objective, 0.4)
    return impVol

def AmericanOptionImpliedForwardAndRate(spotPrice, strike, maturity, priceMktPut, priceMktCall, timeSteps):
    # Implied forward & riskfree rate from ATM put/call prices
    # Iterate on fwd price & rate until put/call implied vols converge ATM
    pass

def DeAmericanizedOptionsChainDataset(df, spotPrice, stepSize):
    # De-Americanization of listed option prices into European pseudo-prices
    # Return standardized options chain dataset with columns: "Contract Name","Expiry","Texp","Put/Call","Strike","Bid","Ask"
    # Routine: (1) implied fwd prices (2) back out implied vols (3) cast to European prices (4) standardize dataset
    pass

PriceAmericanOption_vec = np.vectorize(PriceAmericanOption)
AmericanOptionImpliedVol_vec = np.vectorize(AmericanOptionImpliedVol)
