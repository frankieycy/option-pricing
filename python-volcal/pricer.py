import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from math import isclose
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.optimize import fsolve, minimize
from scipy.integrate import quad
from scipy.interpolate import splrep, splev, pchip, interp1d
plt.switch_backend("Agg")

#### Black-Scholes #############################################################

def BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black Scholes formula for call/put
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    riskFreeRateFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol
    if isinstance(optionType, str): # scalar calculations
        return spotPrice * norm.cdf(d1) - riskFreeRateFactor * strike * norm.cdf(d2) if optionType == "call" else \
            riskFreeRateFactor * strike * norm.cdf(-d2) - spotPrice * norm.cdf(-d1)
    else: # vector calculations
        call = (optionType == "call")
        price = np.zeros(len(strike))
        price[call] = spotPrice * norm.cdf(d1[call]) - riskFreeRateFactor * strike[call] * norm.cdf(d2[call]) # call
        price[~call] = riskFreeRateFactor * strike[~call] * norm.cdf(-d2[~call]) - spotPrice * norm.cdf(-d1[~call]) # put
        return price
    # price = np.where(optionType == "call",
    #     spotPrice * norm.cdf(d1) - riskFreeRateFactor * strike * norm.cdf(d2),
    #     riskFreeRateFactor * strike * norm.cdf(-d2) - spotPrice * norm.cdf(-d1))
    # return price

def BlackScholesVega(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black Scholes vega for call/put (first deriv wrt sigma)
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    return spotPrice * np.sqrt(maturity) * norm.pdf(d1)

def WithinNoArbBound(spotPrice, strike, maturity, riskFreeRate, priceMkt, optionType):
    # Whether priceMkt lies within no-arb bounds
    riskFreeRateFactor = np.exp(-riskFreeRate*maturity)
    noArb = np.where(optionType == "call",
        (priceMkt > np.maximum(spotPrice-strike*riskFreeRateFactor,0)) & (priceMkt < spotPrice),
        (priceMkt > np.maximum(strike*riskFreeRateFactor-spotPrice,0)) & (priceMkt < strike*riskFreeRateFactor))
    return noArb

def BlackScholesImpliedVol(spotPrice, strike, maturity, riskFreeRate, priceMkt, optionType="OTM", method="Bisection"):
    # Black Scholes implied volatility for call/put/OTM
    # Generally, strike & priceMkt are vectors (called from CharFuncImpliedVol)
    # Make this very efficient!
    forwardPrice = spotPrice*np.exp(riskFreeRate*maturity)
    nStrikes = len(strike) if isinstance(strike, np.ndarray) else 1
    impVol = np.repeat(0., nStrikes)
    if isinstance(optionType, str): # cast optionType as vector
        if optionType == "OTM": optionType = np.where(strike > forwardPrice, "call", "put")
        else: optionType = np.repeat(optionType, nStrikes)
    if method == "Bisection": # Bisection search
        impVol0 = np.repeat(1e-10, nStrikes)
        impVol1 = np.repeat(10., nStrikes)
        price0 = BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impVol0, optionType)
        price1 = BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impVol1, optionType)
        while np.mean(impVol1-impVol0) > 1e-10:
            impVol = (impVol0+impVol1)/2
            price = BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impVol, optionType)
            price0 += (price<priceMkt)*(price-price0)
            impVol0 += (price<priceMkt)*(impVol-impVol0)
            price1 += (price>=priceMkt)*(price-price1)
            impVol1 += (price>=priceMkt)*(impVol-impVol1)
        return impVol
    elif method == "Newton": # Newton-Raphson method (NTM options)
        k = np.log(strike/forwardPrice)
        noArb = WithinNoArbBound(spotPrice, strike, maturity, riskFreeRate, priceMkt, optionType)
        ntm = (k>-1)&(k<1)&(noArb) # near-the-money & arb-free
        strikeNtm, optionTypeNtm, priceMktNtm = strike[ntm], optionType[ntm], priceMkt[ntm]
        def objective(impVol):
            return BlackScholesFormula(spotPrice, strikeNtm, maturity, riskFreeRate, impVol, optionTypeNtm) - priceMktNtm
        def objectiveDeriv(impVol):
            return BlackScholesVega(spotPrice, strikeNtm, maturity, riskFreeRate, impVol, optionTypeNtm)
        impVol0 = np.repeat(0.4, np.sum(ntm))
        impVol1 = np.repeat(0., np.sum(ntm))
        for i in range(40): # iterate for NTM options
            step = objective(impVol0) / objectiveDeriv(impVol0)
            step[np.abs(step)>1] = 0 # abnormal step due to small derivs
            impVol1 = impVol0 - step
            if np.mean(np.abs(impVol1-impVol0)) < 1e-10: break
            impVol0 = impVol1.copy()
        impVol[ntm] = impVol1
        if np.sum(~ntm):
            # delegate far-OTM options to Bisection
            # small derivs make Newton unstable
            impVol[~ntm] = BlackScholesImpliedVol(spotPrice, strike[~ntm], maturity, riskFreeRate, priceMkt[~ntm], optionType[~ntm], method="Bisection")
        return impVol
    elif method == "Chebychev": # Chebychev IV-interpolation
        # Ref: Glau, The Chebyshev Method for the Implied Volatility
        # TO-DO
        return impVol

#### Pricing Formula ###########################################################
# Return prices at S0=1 given logStrike k=log(K/F) (scalar/vector) and maturity T (scalar)
# NOTE: **kwargs for deployment in Implied Vol functions
# Forward measure (riskFreeRate=0 in charFunc) is assumed, so price is undiscounted
# TO-DO: non-zero riskFreeRate case? just multiply discount factor to formula?

cmFFT_init = False
cmFFT_du = None
cmFFT_u = None
cmFFT_dk = None
cmFFT_k = None
cmFFT_b = None
cmFFT_w = None
cmFFT_ntm = None
cmFFT_Imult = None
cmFFT_cpImult = None
cmFFT_otmImult = None
cmFFT_cpCFarg = None
cmFFT_cpCFmult = None
cmFFT_charFunc = None
cmFFT_charFuncLog = None

def LewisFormula(charFunc, logStrike, maturity, optionType="OTM", **kwargs):
    # Lewis formula for call/put/OTM
    # Works for scalar logStrike only
    if optionType == "call": k0 = 0
    elif optionType == "put": k0 = logStrike
    elif optionType == "OTM": k0 = (logStrike<0)*logStrike
    integrand = lambda u: np.real(np.exp(-1j*u*logStrike) * charFunc(u-1j/2, maturity) / (u**2+.25))
    price = np.exp(k0) - np.exp(logStrike/2)/np.pi * quad(integrand, 0, np.inf)[0]
    return price

def LewisFormulaFFT(charFunc, logStrike, maturity, optionType="OTM", interp="cubic", N=2**12, B=1000, **kwargs):
    # Lewis FFT formula for call/put/OTM
    # Works for vector logStrike
    # Unstable for short maturity
    du = B/N
    u = np.arange(N)*du
    w = np.arange(N)
    w = 3+(-1)**(w+1)
    w[0] = 1; w[N-1] = 1
    dk = 2*np.pi/B
    b = N*dk/2
    k = -b+np.arange(N)*dk
    I = w * np.exp(1j*b*u) * charFunc(u-1j/2, maturity) / (u**2+0.25) * du/3
    Ifft = np.real(fft(I))
    spline = interp1d(k, Ifft, kind=interp)
    if optionType == "call": k0 = 0
    elif optionType == "put": k0 = logStrike
    elif optionType == "OTM": k0 = (logStrike<0)*logStrike
    price = np.exp(k0) - np.exp(logStrike/2)/np.pi * spline(logStrike)
    return price

def CarrMadanFormula(charFunc, logStrike, maturity, optionType="OTM", alpha=2, **kwargs):
    # Carr-Madan formula for call/put/OTM
    # Works for scalar logStrike only
    if optionType in ["call", "put"]:
        def modCharFunc(u, maturity):
            return charFunc(u-(alpha+1)*1j, maturity) / (alpha**2+alpha-u**2+1j*(2*alpha+1)*u)
        integrand = lambda u: np.real(np.exp(-1j*u*logStrike) * modCharFunc(u, maturity))
        price = np.exp(-alpha*logStrike)/np.pi * quad(integrand, 0, np.inf)[0]
        if optionType == "call": return price
        elif optionType == "put": return price-1+np.exp(logStrike)
    elif optionType == "OTM":
        if np.abs(logStrike) < 1e-10:
            price0 = CarrMadanFormula(charFunc, -1e-4, maturity, "put", alpha, **kwargs)
            price1 = CarrMadanFormula(charFunc, +1e-4, maturity, "call", alpha, **kwargs)
            return (price0+price1)/2
        def modCharFunc(u, maturity):
            return 1 / (1+1j*u) - 1 / (1j*u) - charFunc(u-1j, maturity) / (u**2-1j*u)
        def gamCharFunc(u, maturity):
            return (modCharFunc(u-1j*alpha, maturity) - modCharFunc(u+1j*alpha, maturity)) / 2
        integrand = lambda u: np.real(np.exp(-1j*u*logStrike) * gamCharFunc(u, maturity))
        price = 1/(np.pi*np.sinh(alpha*logStrike)) * quad(integrand, 0, np.inf)[0]
        return price

def CarrMadanFormulaFFT(charFunc, logStrike, maturity, optionType="OTM", interp="cubic",
    alpha=2, N=2**16, B=4000, useGlobal=False, curryCharFunc=False, **kwargs):
    # Carr-Madan FFT formula for call/put/OTM
    # Works for vector logStrike; useGlobal=True in order for curryCharFunc=True
    # Make this very efficient! (0.0087s per maturity * 28 maturities = 0.244s)
    if useGlobal:
        global cmFFT_init, cmFFT_du, cmFFT_u, cmFFT_dk, cmFFT_k, cmFFT_b, cmFFT_w, cmFFT_ntm, \
            cmFFT_Imult, cmFFT_cpImult, cmFFT_otmImult, cmFFT_cpCFarg, cmFFT_cpCFmult, cmFFT_charFunc, cmFFT_charFuncLog
        if not cmFFT_init:
            # initialize global params (only ONCE!)
            #### FFT params ####
            cmFFT_du = B/N
            cmFFT_u = np.arange(N)*cmFFT_du
            cmFFT_w = np.arange(N)
            cmFFT_w = 3+(-1)**(cmFFT_w+1)
            cmFFT_w[0] = 1; cmFFT_w[N-1] = 1
            cmFFT_dk = 2*np.pi/B
            cmFFT_b = N*cmFFT_dk/2
            cmFFT_k = -cmFFT_b+np.arange(N)*cmFFT_dk
            cmFFT_ntm = (cmFFT_k>-5)&(cmFFT_k<5)
            #### pre-calculations ####
            cmFFT_Imult = cmFFT_w * np.exp(1j*cmFFT_b*cmFFT_u) * cmFFT_du/3
            cmFFT_cpImult = np.exp(-alpha*cmFFT_k)/np.pi
            with np.errstate(divide='ignore'):
                cmFFT_otmImult = 1/(np.pi*np.sinh(alpha*cmFFT_k))
            cmFFT_cpCFarg = cmFFT_u-(alpha+1)*1j
            cmFFT_cpCFmult = 1/(alpha**2+alpha-cmFFT_u**2+1j*(2*alpha+1)*cmFFT_u)
            cmFFT_init = True
        if charFunc != cmFFT_charFuncLog:
            # update charFunc (for every NEW charFunc)
            cmFFT_charFuncLog = charFunc
            if curryCharFunc: cmFFT_charFunc = charFunc(cmFFT_cpCFarg)
            else: cmFFT_charFunc = charFunc
        du = cmFFT_du
        u = cmFFT_u
        dk = cmFFT_dk
        k = cmFFT_k
        b = cmFFT_b
        w = cmFFT_w
        ntm = cmFFT_ntm
        Imult = cmFFT_Imult
        cpImult = cmFFT_cpImult
        otmImult = cmFFT_otmImult
        cpCFarg = cmFFT_cpCFarg
        cpCFmult = cmFFT_cpCFmult
        charFunc = cmFFT_charFunc
    else:
        du = B/N
        u = np.arange(N)*du
        w = np.arange(N)
        w = 3+(-1)**(w+1)
        w[0] = 1; w[N-1] = 1
        dk = 2*np.pi/B
        b = N*dk/2
        k = -b+np.arange(N)*dk
        ntm = (k>-5)&(k<5)
        Imult = w * np.exp(1j*b*u) * du/3
        cpImult = np.exp(-alpha*k)/np.pi
        with np.errstate(divide='ignore'):
            otmImult = 1/(np.pi*np.sinh(alpha*k))
    if optionType in ["call", "put"]:
        if useGlobal:
            def modCharFunc(u, maturity): # pre-calculated cpCFarg/cpCFmult
                return charFunc(cpCFarg, maturity) * cpCFmult
        else:
            def modCharFunc(u, maturity):
                return charFunc(u-(alpha+1)*1j, maturity) / (alpha**2+alpha-u**2+1j*(2*alpha+1)*u)
        # I = w * np.exp(1j*b*u) * modCharFunc(u, maturity) * du/3 # 0.010s (make this fast!)
        I = Imult * modCharFunc(u, maturity) # 0.0068s
        Ifft = cpImult * np.real(fft(I)) # 0.0010s
        spline = interp1d(k[ntm], Ifft[ntm], kind=interp) # 0.0009s
        price = spline(logStrike)
        if optionType == "call": return price
        elif optionType == "put": return price-1+np.exp(logStrike)
    elif optionType == "OTM":
        def modCharFunc(u, maturity):
            return 1 / (1+1j*u) - 1 / (1j*u) - charFunc(u-1j, maturity) / (u**2-1j*u)
        def gamCharFunc(u, maturity):
            return (modCharFunc(u-1j*alpha, maturity) - modCharFunc(u+1j*alpha, maturity)) / 2
        # I = w * np.exp(1j*b*u) * gamCharFunc(u, maturity) * du/3
        I = Imult * gamCharFunc(u, maturity)
        Ifft = otmImult * np.real(fft(I))
        Ifft[N//2] = CarrMadanFormula(charFunc, 0, maturity, "OTM", alpha, **kwargs) # k = 0
        spline = interp1d(k[ntm], Ifft[ntm], kind=interp)
        price = spline(logStrike)
        return price

#### Implied Vol ###############################################################
# Given charFunc, return impVolFunc with arguments (logStrike, maturity)

def CharFuncImpliedVol(charFunc, optionType="OTM", riskFreeRate=0, FFT=False, formulaType="CarrMadan", inversionMethod="Bisection", **kwargs):
    # Implied volatility for call/put/OTM priced with charFunc
    # CAUTION: formula assumes forward measure (forward option price)
    # so riskFreeRate has to be 0 (for now, just a dummy)
    if formulaType == "Lewis":
        formula = LewisFormulaFFT if FFT else LewisFormula
    elif formulaType == "CarrMadan":
        formula = CarrMadanFormulaFFT if FFT else CarrMadanFormula
    def impVolFunc(logStrike, maturity):
        return BlackScholesImpliedVol(1, np.exp(logStrike), maturity, riskFreeRate, formula(charFunc, logStrike, maturity, optionType, **kwargs), optionType, inversionMethod)
    return impVolFunc

def LewisCharFuncImpliedVol(charFunc, optionType="OTM", riskFreeRate=0, **kwargs):
    # Implied volatility for call/put/OTM priced with charFunc, based on Lewis formula
    def impVolFunc(logStrike, maturity):
        def objective(vol):
            integrand = lambda u:  np.real(np.exp(-1j*u*logStrike) * (charFunc(u-1j/2, maturity) - BlackScholesCharFunc(vol, riskFreeRate)(u-1j/2, maturity)) / (u**2+.25))
            return quad(integrand, 0, np.inf)[0]
        impVol = fsolve(objective, 0.4)[0]
        return impVol
    return impVolFunc

#### Characteristic Function ###################################################
# Characteristic Function: E(exp(i*u*XT)), XT = log(ST/S0)
# Return charFunc with arguments (u, maturity)

def BlackScholesCharFunc(vol, riskFreeRate=0, curry=False):
    # Characteristic function for Black-Scholes model
    def charFunc(u, maturity):
        return np.exp(1j*u*riskFreeRate*maturity-vol**2*maturity/2*u*(u+1j))
    return charFunc

def HestonCharFunc(meanRevRate, correlation, volOfVol, meanVar, currentVar, riskFreeRate=0, curry=False):
    # Characteristic function for Heston model
    if curry:
        def charFunc(u):
            iur = 1j*u*riskFreeRate
            alpha = -u**2/2-1j*u/2
            beta = meanRevRate-correlation*volOfVol*1j*u
            gamma = volOfVol**2/2
            d = np.sqrt(beta**2-4*alpha*gamma)
            rp = (beta+d)/(2*gamma)
            rm = (beta-d)/(2*gamma)
            g = rm/rp
            def charFuncFixedU(u, maturity): # u is dummy
                D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
                C = meanRevRate*(rm*maturity-2/volOfVol**2*np.log((1-g*np.exp(-d*maturity))/(1-g)))
                return np.exp(iur*maturity+C*meanVar+D*currentVar)
            return charFuncFixedU
    else:
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
            return np.exp(1j*u*riskFreeRate*maturity+C*meanVar+D*currentVar)
    return charFunc

def MertonJumpCharFunc(vol, jumpInt, jumpMean, jumpSd, riskFreeRate=0, curry=False):
    # Characteristic function for Merton-Jump model
    if curry:
        def charFunc(u):
            chExp = 1j*u*riskFreeRate-vol**2/2*u*(u+1j)-1j*u*jumpInt*(np.exp(jumpMean+jumpSd**2/2)-1)+jumpInt*(np.exp(1j*u*jumpMean-u**2*jumpSd**2/2)-1)
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(chExp*maturity)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            return np.exp(1j*u*riskFreeRate*maturity-vol**2*maturity/2*u*(u+1j)-1j*u*jumpInt*maturity*(np.exp(jumpMean+jumpSd**2/2)-1)+jumpInt*maturity*(np.exp(1j*u*jumpMean-u**2*jumpSd**2/2)-1))
    return charFunc

def VarianceGammaCharFunc(vol, drift, timeChgVar, riskFreeRate=0, curry=False):
    # Characteristic function for Variance-Gamma model
    # Xt = drift*gamma(t;1,timeChgVar) + vol*W(gamma(t;1,timeChgVar))
    # Ref: Madan, The Variance Gamma Process and Option Pricing
    if curry:
        def charFunc(u):
            chExp = 1j*u*(riskFreeRate+1/timeChgVar*np.log(1-(drift+vol**2/2)*timeChgVar))-np.log(1-1j*u*drift*timeChgVar+u**2*vol**2*timeChgVar/2)/timeChgVar
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(chExp*maturity)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            return np.exp(1j*u*(riskFreeRate+1/timeChgVar*np.log(1-(drift+vol**2/2)*timeChgVar))*maturity)*(1-1j*u*drift*timeChgVar+u**2*vol**2*timeChgVar/2)**(-maturity/timeChgVar)
    return charFunc

def SVJCharFunc(meanRevRate, correlation, volOfVol, meanVar, currentVar, jumpInt, jumpMean, jumpSd, riskFreeRate=0, curry=False):
    # Characteristic function for SVJ model (Heston-MertonJump)
    if curry:
        def charFunc(u):
            alpha = -u**2/2-1j*u/2
            beta = meanRevRate-correlation*volOfVol*1j*u
            gamma = volOfVol**2/2
            d = np.sqrt(beta**2-4*alpha*gamma)
            rp = (beta+d)/(2*gamma)
            rm = (beta-d)/(2*gamma)
            g = rm/rp
            chExp = 1j*u*riskFreeRate-1j*u*jumpInt*(np.exp(jumpMean+jumpSd**2/2)-1)+jumpInt*(np.exp(1j*u*jumpMean-u**2*jumpSd**2/2)-1)
            def charFuncFixedU(u, maturity): # u is dummy
                D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
                C = meanRevRate*(rm*maturity-2/volOfVol**2*np.log((1-g*np.exp(-d*maturity))/(1-g)))
                return np.exp(chExp*maturity+C*meanVar+D*currentVar)
            return charFuncFixedU
    else:
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
            return np.exp(1j*u*riskFreeRate*maturity+C*meanVar+D*currentVar-1j*u*jumpInt*maturity*(np.exp(jumpMean+jumpSd**2/2)-1)+jumpInt*maturity*(np.exp(1j*u*jumpMean-u**2*jumpSd**2/2)-1))
    return charFunc

def SVJJCharFunc(meanRevRate, correlation, volOfVol, meanVar, currentVar, jumpInt, jumpMean, jumpSd, riskFreeRate=0, curry=False):
    # Characteristic function for SVJJ model (HestonJump-MertonJump)
    # TO-DO
    pass

def rHestonPoorMansCharFunc(hurstExp, correlation, volOfVol, currentVar, riskFreeRate=0, curry=False):
    # Characteristic function for rHeston model (poor man's Heston approx)
    # Heston approx: meanRevRate=0 hence no meanVar-dependence, currentVar is var swap price
    # Ref: Gatheral, Roughening Heston
    volOfVolFactor = np.sqrt(3/(2*hurstExp+2))*volOfVol/sp.special.gamma(hurstExp+1.5)
    if curry:
        def charFunc(u):
            iur = 1j*u*riskFreeRate
            alpha = -u**2/2-1j*u/2
            def charFuncFixedU(u, maturity): # u is dummy
                volOfVolMod = volOfVolFactor/maturity**(0.5-hurstExp)
                beta = -correlation*volOfVolMod*1j*u
                gamma = volOfVolMod**2/2
                d = np.sqrt(beta**2-4*alpha*gamma)
                rp = (beta+d)/(2*gamma)
                rm = (beta-d)/(2*gamma)
                g = rm/rp
                D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
                return np.exp(iur*maturity+D*currentVar)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            volOfVolMod = volOfVolFactor/maturity**(0.5-hurstExp)
            alpha = -u**2/2-1j*u/2
            beta = -correlation*volOfVolMod*1j*u
            gamma = volOfVolMod**2/2
            d = np.sqrt(beta**2-4*alpha*gamma)
            rp = (beta+d)/(2*gamma)
            rm = (beta-d)/(2*gamma)
            g = rm/rp
            D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
            return np.exp(1j*u*riskFreeRate*maturity+D*currentVar)
    return charFunc

def rHestonPadeCharFunc(hurstExp, correlation, meanVar, currentVar, riskFreeRate=0, curry=False):
    # Characteristic function for rHeston model (poor man's crude approx)
    # Ref: Gatheral, Rational Approximation of the Rough Heston Solution
    # TO-DO
    pass

#### Calibration ###############################################################

def CalibrateModelToOptionPrice(logStrike, maturity, optionPrice, model, params0, paramsLabel,
    bounds=None, w=None, optionType="call", formulaType="CarrMadan", **kwargs):
    # Calibrate model params to option prices (pricing measure)
    # NOT the standard practice!
    if w is None: w = 1
    maturity = np.array(maturity)
    if formulaType == "Lewis":
        formula = LewisFormulaFFT
    elif formulaType == "CarrMadan":
        formula = CarrMadanFormulaFFT
    def objective(params):
        params = {paramsLabel[i]: params[i] for i in range(len(params))}
        charFunc = model(**params)
        # price = LewisFormulaFFT(charFunc, logStrike, maturity, optionType, **kwargs) # single fixed maturity
        price = np.concatenate([formula(charFunc, logStrike[maturity==T], T, optionType, **kwargs) for T in np.unique(maturity)], axis=None)
        loss = np.sum(w*(price-optionPrice)**2)
        print(f"params: {params}")
        print(f"loss: {loss}")
        return loss
    opt = minimize(objective, x0=params0, bounds=bounds, method="SLSQP")
    print("Optimization output:", opt, sep="\n")
    return opt.x

def CalibrateModelToImpliedVol(logStrike, maturity, optionImpVol, model, params0, paramsLabel,
    bounds=None, w=None, optionType="call", formulaType="CarrMadan", curryCharFunc=False, **kwargs):
    # Calibrate model params to implied vols (pricing measure)
    # NOTE: include useGlobal=True for curryCharFunc=True
    if w is None: w = 1
    maturity = np.array(maturity)
    bidVol = optionImpVol["Bid"].to_numpy()
    askVol = optionImpVol["Ask"].to_numpy()
    def objective(params):
        params = {paramsLabel[i]: params[i] for i in range(len(params))}
        # charFunc = model(**params)
        # impVolFunc = CharFuncImpliedVol(charFunc, optionType=optionType, FFT=True, formulaType=formulaType, **kwargs)
        charFunc = model(**params, curry=curryCharFunc)
        impVolFunc = CharFuncImpliedVol(charFunc, optionType=optionType, FFT=True, formulaType=formulaType, curryCharFunc=curryCharFunc, **kwargs)
        impVol = np.concatenate([impVolFunc(logStrike[maturity==T], T) for T in np.unique(maturity)], axis=None) # most costly
        loss = np.sum(w*((impVol-bidVol)**2+(askVol-impVol)**2))
        print(f"params: {params}")
        print(f"loss: {loss}")
        return loss
    opt = minimize(objective, x0=params0, bounds=bounds)
    print("Optimization output:", opt, sep="\n")
    return opt.x

#### Plotting Function #########################################################

def PlotImpliedVol(df, figname=None, ncol=6):
    # Plot bid-ask implied volatilities based on df
    # Columns: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    if not figname:
        figname = "impliedvol.png"
    Texp = df["Texp"].unique()
    Nexp = len(Texp)
    nrow = int(np.ceil(Nexp/ncol))
    ncol = min(len(Texp),6)
    fig, ax = plt.subplots(nrow,ncol,figsize=(15,10))
    for i in range(nrow*ncol):
        ix,iy = i//ncol,i%ncol
        idx = (ix,iy) if nrow*ncol>6 else iy
        ax_idx = ax[idx] if ncol>1 else ax
        if i < Nexp:
            T = Texp[i]
            dfT = df[df["Texp"]==T]
            k = np.log(dfT["Strike"]/dfT["Fwd"])
            bid = dfT["Bid"]
            ask = dfT["Ask"]
            ax_idx.scatter(k,bid,c='r',s=5)
            ax_idx.scatter(k,ask,c='b',s=5)
            if "Fit" in dfT:
                fit = dfT["Fit"]
                ax_idx.scatter(k,fit,c='k',s=5)
            ax_idx.set_title(rf"$T={np.round(T,3)}$")
            ax_idx.set_xlabel("log-strike")
            ax_idx.set_ylabel("implied vol")
        else:
            ax_idx.axis("off")
    fig.tight_layout()
    plt.savefig(figname)
    plt.close()

#### Fwd Var Curve #############################################################

def VarianceSwapFormula(logStrike, maturity, impliedVol, showPlot=False):
    # Fukasawa robust variance swap formula
    logStrike,impliedVol = np.array(logStrike),np.array(impliedVol)
    totalImpVol = impliedVol*np.sqrt(maturity)
    d2 = -logStrike/totalImpVol-totalImpVol/2
    y = norm.cdf(d2)
    ord = np.argsort(y)
    ymin,ymax = np.min(y),np.max(y)
    y,d2,logStrike,totalImpVol = y[ord],d2[ord],logStrike[ord],totalImpVol[ord]

    if showPlot:
        yint = np.linspace(ymin,ymax,200)
        fig = plt.figure(figsize=(6,4))
        plt.scatter(y, totalImpVol, c='k', s=5)
        plt.title(rf"$T={np.round(maturity,3)}$")
        plt.xlabel("$y=N(d_2)$")
        plt.ylabel("total implied vol")
        fig.tight_layout()
        plt.savefig("test_totalImpVolVsY.png")
        plt.close()

    # tck = splrep(y, totalImpVol**2, s=0)
    # intTotalImpVar = lambda x: splev(x, tck, der=0)
    pch = pchip(y, totalImpVol**2)
    intTotalImpVar = lambda x: pch(x)

    areaMid = quad(intTotalImpVar, ymin, ymax, limit=1000)[0]
    areaMin = totalImpVol[0]**2*norm.cdf(d2[0])
    areaMax = totalImpVol[-1]**2*norm.cdf(-d2[-1])
    price = areaMin + areaMid + areaMax
    return price

def GammaSwapFormula(logStrike, maturity, impliedVol):
    # Fukasawa robust gamma swap formula
    logStrike,impliedVol = np.array(logStrike),np.array(impliedVol)
    totalImpVol = impliedVol*np.sqrt(maturity)
    d1 = -logStrike/totalImpVol+totalImpVol/2
    y = norm.cdf(d1)
    ord = np.argsort(y)
    ymin,ymax = np.min(y),np.max(y)
    y,d1,logStrike,totalImpVol = y[ord],d1[ord],logStrike[ord],totalImpVol[ord]

    # tck = splrep(y, totalImpVol**2, s=0)
    # intTotalImpVar = lambda x: splev(x, tck, der=0)
    pch = pchip(y, totalImpVol**2)
    intTotalImpVar = lambda x: pch(x)

    areaMid = quad(intTotalImpVar, ymin, ymax, limit=1000)[0]
    areaMin = totalImpVol[0]**2*norm.cdf(d1[0])
    areaMax = totalImpVol[-1]**2*norm.cdf(-d1[-1])
    price = areaMin + areaMid + areaMax
    return price

def LeverageSwapFormula(logStrike, maturity, impliedVol):
    # Fukasawa robust leverage swap formula
    return GammaSwapFormula(logStrike, maturity, impliedVol) - VarianceSwapFormula(logStrike, maturity, impliedVol)

def CalcSwapCurve(df, swapFormula):
    # Calculate swap curves based on implied volatilities in df
    Texp = df["Texp"].unique()
    Nexp = len(Texp)
    curve = {c: list() for c in ["bid","mid","ask"]}
    for T in Texp:
        dfT = df[df["Texp"]==T]
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        bid = dfT["Bid"]
        ask = dfT["Ask"]
        mid = (bid+ask)/2
        curve["bid"].append(swapFormula(k,T,bid)/T)
        curve["mid"].append(swapFormula(k,T,mid)/T)
        curve["ask"].append(swapFormula(k,T,ask)/T)
    curve = pd.DataFrame(curve)
    curve["Texp"] = Texp
    curve = curve[["Texp","bid","mid","ask"]]
    return curve

def CalcFwdVarCurve(curveVS):
    # Calculate forward variance curve based on VS curve
    Texp = curveVS["Texp"]
    diffTexp = curveVS["Texp"].diff()
    price = curveVS[["bid","mid","ask"]].multiply(Texp,axis=0)
    curve = price.diff()
    curve = curve.div(diffTexp,axis=0)
    curve.iloc[0] = price.iloc[0]/Texp.iloc[0]
    curve["Texp"] = Texp
    curve = curve[["Texp","bid","mid","ask"]]
    return curve

def FwdVarCurveFunc(maturity, fwdVar, fitType="const"):
    # Smooth out forward variance curve
    Texp = maturity
    Nexp = len(Texp)
    curveFunc = None
    if fitType == "const":
        curveFunc = lambda t: fwdVar[min(sum(Texp<t),Nexp-1)]
    return curveFunc
