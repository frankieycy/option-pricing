import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def PriceAmericanOption(strike, maturity, spotPrice, forwardPrice, impliedVol, timeStep=0.0002):
    # Price American option via Cox binomial tree (d = 1/u)
    # Assume continuous dividend, reflected in forward price
    # TO-DO: proportional/discrete dividend (difficult!)
    pass

def AmericanOptionImpliedVol(priceMkt, strike, maturity, spotPrice, forwardPrice, timeStep=0.0002):
    # Implied flat volatility under Cox binomial tree
    pass

def AmericanOptionImpliedForward(priceMktPut, priceMktCall, strike, maturity, spotPrice, timeStep=0.0002):
    # Implied forward from ATM put/call prices
    # Iterate on fwd price until put/call implied vols converge ATM
    pass

def DeAmericanizedOptionsChainDataset(df, spotPrice, timeStep=0.0002):
    # De-Americanization of listed option prices into European pseudo-prices
    # Return standardized options chain dataset with columns: "Contract Name","Expiry","Texp","Put/Call","Strike","Bid","Ask"
    # Routine: (1) implied fwd prices (2) back out implied vols (3) cast to European prices (4) standardize dataset
    pass
