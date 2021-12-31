import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
plt.switch_backend("Agg")

def BlackScholesFormulaCall(currentPrice, strike, maturity, riskFreeRate, impliedVol):
    logMoneyness = np.log(currentPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    riskFreeRateFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol
    price = currentPrice * norm.cdf(d1) - riskFreeRateFactor * strike * norm.cdf(d2)
    return price

def BlackScholesFormulaPut(currentPrice, strike, maturity, riskFreeRate, impliedVol):
    logMoneyness = np.log(currentPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    riskFreeRateFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol
    price = riskFreeRateFactor * strike * norm.cdf(-d2) - currentPrice * norm.cdf(-d1)
    return price

def BlackScholesImpliedVolCall(currentPrice, strike, maturity, riskFreeRate, price):
    nStrikes = len(strike) if isinstance(strike, np.ndarray) else 1
    impVol0 = np.repeat(1e-10, nStrikes)
    impVol1 = np.repeat(10., nStrikes)
    price0 = BlackScholesFormulaCall(currentPrice, strike, maturity, riskFreeRate, impVol0)
    price1 = BlackScholesFormulaCall(currentPrice, strike, maturity, riskFreeRate, impVol1)
    while np.mean(impVol1-impVol0) > 1e-10:
        impVol2 = (impVol0+impVol1)/2
        price2 = BlackScholesFormulaCall(currentPrice, strike, maturity, riskFreeRate, impVol2)
        price0 += (price2<price)*(price2-price0)
        impVol0 += (price2<price)*(impVol2-impVol0)
        price1 += (price2>=price)*(price2-price1)
        impVol1 += (price2>=price)*(impVol2-impVol1)
    return impVol2

def BlackScholesImpliedVolPut(currentPrice, strike, maturity, riskFreeRate, price):
    nStrikes = len(strike) if isinstance(strike, np.ndarray) or isinstance(strike, list) else 1
    impVol0 = np.repeat(1e-10, nStrikes)
    impVol1 = np.repeat(10., nStrikes)
    price0 = BlackScholesFormulaPut(currentPrice, strike, maturity, riskFreeRate, impVol0)
    price1 = BlackScholesFormulaPut(currentPrice, strike, maturity, riskFreeRate, impVol1)
    while np.mean(impVol1-impVol0) > 1e-10:
        impVol2 = (impVol0+impVol1)/2
        price2 = BlackScholesFormulaPut(currentPrice, strike, maturity, riskFreeRate, impVol2)
        price0 += (price2<price)*(price2-price0)
        impVol0 += (price2<price)*(impVol2-impVol0)
        price1 += (price2>=price)*(price2-price1)
        impVol1 += (price2>=price)*(impVol2-impVol1)
    return impVol2

def BlackScholesImpliedVolOTM(currentPrice, strike, maturity, riskFreeRate, price):
    forwardPrice = currentPrice*np.exp(riskFreeRate*maturity)
    impVol = BlackScholesImpliedVolCall(currentPrice, strike, maturity, riskFreeRate, price) if strike > forwardPrice else \
        BlackScholesImpliedVolPut(currentPrice, strike, maturity, riskFreeRate, price)
    return impVol

def LewisFormulaOTM(charFunc, logStrike, maturity):
    integrand = lambda u: (np.exp(-1j*u*logStrike) * charFunc(u-1j/2, maturity) / (u**2+.25)).real
    logStrikeMinus = (logStrike<0)*logStrike
    price = np.exp(logStrikeMinus) - np.exp(logStrike/2)/np.pi * quad(integrand, 0, np.inf)[0]
    return price

def CharFuncImpliedVol(charFunc):
    def impVolFunc(logStrike, maturity):
        return BlackScholesImpliedVolOTM(1, np.exp(logStrike), maturity, 0, LewisFormulaOTM(charFunc, logStrike, maturity))
    return impVolFunc

def HestonCharFunc(meanRevRate, correlation, volOfVol, meanVol, currentVol):
    def charFunc(u, maturity):
        alpha = -u**2/2-1j*u/2
        beta = meanRevRate-correlation*volOfVol*1j*u
        gamma = volOfVol**2/2
        d = np.sqrt(beta**2-4*alpha*gamma)
        rp = (beta+d)/(2*gamma)
        rm = (beta-d)/(2*gamma)
        g = rm/rp
        D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
        C = meanRevRate*(rm*maturity-2/volOfVol**2*np.log((1-g*np.exp(-d*maturity))/(1-g)))
        return np.exp(C*meanVol+D*currentVol)
    return charFunc
