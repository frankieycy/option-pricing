import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from math import isclose
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.optimize import fsolve, minimize, dual_annealing, \
    shgo, differential_evolution, basinhopping
from scipy.integrate import quad, quad_vec
from scipy.interpolate import splrep, splev, pchip, interp1d, interp2d, \
    InterpolatedUnivariateSpline, RectBivariateSpline
plt.switch_backend("Agg")

#### Global Variables ##########################################################

bsIv_interpInit = False
bsIv_interpFunc = None

bsIv_rationalInit = False
bsIv_rationalLBnd = None
bsIv_rationalUBnd = None
bsIv_rationalIv = None

cmFFT_init = False
cmFFT_du = None
cmFFT_u = None
cmFFT_dk = None
cmFFT_k = None
cmFFT_b = None
cmFFT_w = None
cmFFT_ntm = None
cmFFT_kntm = None
cmFFT_Imult = None
cmFFT_cpImult = None
cmFFT_otmImult = None
cmFFT_cpCFarg = None
cmFFT_cpCFmult = None
cmFFT_charFunc = None
cmFFT_charFuncLog = None

cosFmla_init = False
cosFmla_n = None
cosFmla_expArg = None
cosFmla_cfArg = None
cosFmla_cosInt = None
cosFmla_dict = dict()
cosFmla_charFunc = None
cosFmla_charFuncLog = None
cosFmla_adptParams = {
    #### Adaptive params #################
    # 0.005:  {'a': -5, 'b': 5, 'N': 6000},
    # 0.5:    {'a': -3, 'b': 3, 'N': 2000},
    # 99:     {'a': -5, 'b': 5, 'N': 1000},
    #### Default #########################
    99:     {'a': -5, 'b': 5, 'N': 4000},
}

rhPadeCF_w = None
rhPadeCF_dict = dict()

#### Black-Scholes #############################################################

def BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black Scholes formula for call/put
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    discountFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol

    if isinstance(optionType, str): # Uniform optionType
        return spotPrice * norm.cdf(d1) - discountFactor * strike * norm.cdf(d2) if optionType == "call" else \
            discountFactor * strike * norm.cdf(-d2) - spotPrice * norm.cdf(-d1)
    else: # Vector optionType
        # strike & optionType must be vectors
        call = (optionType == "call")
        price = np.zeros(len(strike))
        price[call] = (spotPrice[call] if isinstance(spotPrice, np.ndarray) else spotPrice) * norm.cdf(d1[call]) - (discountFactor[call] if isinstance(discountFactor, np.ndarray) else discountFactor) * strike[call] * norm.cdf(d2[call]) # call
        price[~call] = (discountFactor[~call] if isinstance(discountFactor, np.ndarray) else discountFactor) * strike[~call] * norm.cdf(-d2[~call]) - (spotPrice[~call] if isinstance(spotPrice, np.ndarray) else spotPrice) * norm.cdf(-d1[~call]) # put
        return price

    # price = np.where(optionType == "call",
    #     spotPrice * norm.cdf(d1) - discountFactor * strike * norm.cdf(d2),
    #     discountFactor * strike * norm.cdf(-d2) - spotPrice * norm.cdf(-d1))
    # return price

def BlackScholesVega(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black Scholes vega for call/put (first deriv wrt sigma)
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    return spotPrice * np.sqrt(maturity) * norm.pdf(d1)

def WithinNoArbBound(spotPrice, strike, maturity, riskFreeRate, priceMkt, optionType):
    # Whether priceMkt lies within no-arb bounds
    discountFactor = np.exp(-riskFreeRate*maturity)
    noArb = np.where(optionType == "call",
        (priceMkt > np.maximum(spotPrice-strike*discountFactor,0)) & (priceMkt < spotPrice),
        (priceMkt > np.maximum(strike*discountFactor-spotPrice,0)) & (priceMkt < strike*discountFactor))
    return noArb

def BlackScholesImpliedVol(spotPrice, strike, maturity, riskFreeRate, priceMkt, optionType="OTM", method="Bisection"):
    # Black Scholes implied volatility for call/put/OTM
    # Generally, strike & priceMkt are input vectors (called from e.g. CharFuncImpliedVol)
    # Within function, optionType & maturity are cast to vectors
    # Make this very efficient!
    forwardPrice = spotPrice*np.exp(riskFreeRate*maturity)
    nStrikes = len(strike) if isinstance(strike, np.ndarray) else 1
    impVol = np.repeat(0., nStrikes)

    if isinstance(optionType, str): # Cast optionType as vector
        if optionType == "OTM": optionType = np.where(strike > forwardPrice, "call", "put")
        else: optionType = np.repeat(optionType, nStrikes)
    if np.isscalar(maturity): # Cast maturity as vector
        maturity = np.repeat(maturity, nStrikes)

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
        ntm = (k>-1)&(k<1)&(noArb) # Near-the-money & arb-free
        strikeNtm, maturityNtm, optionTypeNtm, priceMktNtm = strike[ntm], maturity[ntm], optionType[ntm], priceMkt[ntm]
        def objective(impVol):
            return BlackScholesFormula(spotPrice, strikeNtm, maturityNtm, riskFreeRate, impVol, optionTypeNtm) - priceMktNtm
        def objectiveDeriv(impVol):
            return BlackScholesVega(spotPrice, strikeNtm, maturityNtm, riskFreeRate, impVol, optionTypeNtm)
        impVol0 = np.repeat(0.4, np.sum(ntm))
        impVol1 = np.repeat(0., np.sum(ntm))
        for i in range(40): # Iterate for NTM options
            step = objective(impVol0) / objectiveDeriv(impVol0)
            step[np.abs(step)>1] = 0 # Abnormal step due to small derivs
            impVol1 = impVol0 - step
            if np.mean(np.abs(impVol1-impVol0)) < 1e-10: break
            impVol0 = impVol1.copy()
        impVol[ntm] = impVol1
        if np.sum(~ntm):
            # Delegate far-OTM options to Bisection
            # Small derivs make Newton unstable
            impVol[~ntm] = BlackScholesImpliedVol(spotPrice, strike[~ntm], maturity[~ntm], riskFreeRate, priceMkt[~ntm], optionType[~ntm], method="Bisection")
        return impVol

    elif method == "Interp": # Cubic interpolation (vectors input ONLY)
        # Accuracy is compromised for speed!
        # In practice, batch Bisection (all T) is faster!
        # Params: Kgrid ~ 4e-3, Vgrid ~ 2e-4
        global bsIv_interpInit, bsIv_interpFunc
        if not bsIv_interpInit:
            def call(k,v):
                return np.exp(-k/2)*norm.cdf(-k/v+v/2) - np.exp(k/2)*norm.cdf(-k/v-v/2)
            K = np.arange(-5,5,4e-3) # Log-moneyness: log(K/F)
            V = np.arange(1e-3,1,2e-4) # Total implied vol: sig*sqrt(T)
            C = call(*np.meshgrid(K,V))
            bsIv_interpFunc = RectBivariateSpline(K,V,C.T)
            bsIv_interpInit = True
        k = np.log(strike/forwardPrice)
        put = (optionType == "put")
        priceMkt /= np.sqrt(spotPrice*strike*np.exp(-riskFreeRate*maturity))
        priceMkt[put] += -np.exp(k[put]/2)+np.exp(-k[put]/2) # Cast as call prices
        def callInterp(impVol):
            return bsIv_interpFunc(k, impVol, grid=False)
        impVol0 = np.repeat(1e-10, nStrikes)
        impVol1 = np.repeat(1., nStrikes)
        price0 = callInterp(impVol0)
        price1 = callInterp(impVol1)
        while np.mean(impVol1-impVol0) > 1e-10:
            impVol = (impVol0+impVol1)/2
            price = callInterp(impVol)
            price0 += (price<priceMkt)*(price-price0)
            impVol0 += (price<priceMkt)*(impVol-impVol0)
            price1 += (price>=priceMkt)*(price-price1)
            impVol1 += (price>=priceMkt)*(impVol-impVol1)
        impVol /= np.sqrt(maturity)
        return impVol

    elif method == "Rational": # Rational approx
        # Ref: Li, Approximate Inversion of the Black–Scholes Formula using Rational Functions
        # TO-DO: Bisection-polishing
        global bsIv_rationalInit, bsIv_rationalLBnd, bsIv_rationalUBnd, bsIv_rationalIv
        if not bsIv_rationalInit:
            bsIv_rationalLBnd = lambda x: (-0.00424532412773*x+0.00099075112125*x**2)/(1+0.26674393279214*x+0.03360553011959*x**2)
            bsIv_rationalUBnd = lambda x: (0.38292495908775+0.31382372544666*x+0.07116503261172*x**2)/(1+0.01380361926221*x+0.11791124749938*x**2)
            bsIv_rationalIv = lambda x,c: (-0.969271876255*x+0.097428338274*c**0.5+1.750081126685*c)+(-0.068098378725*c**0.5-0.263473754689*c+4.714393825758*c**1.5+14.749084301452*c**2.0+0.440639436211*x-5.792537721792*x*c**0.5+3.529944137559*x*c-32.570660102526*x*c**1.5-5.267481008429*x**2-23.636495876611*x**2*c**0.5+76.398155779133*x**2*c-9.020361771283*x**3+41.855161781749*x**3*c**0.5-12.150611865704*x**4)/(1+6.268456292246*c**0.5+30.068281276567*c-11.473184324152*c**1.5-13.954993561151*c**2.0-6.284840445036*x-11.780036995036*x*c**0.5-230.101682610568*x*c+261.950288864225*x*c**1.5-2.310966989723*x**2+86.127219899668*x**2*c**0.5+20.090690444187*x**2*c+3.730181294225*x**3-50.117067019539*x**3*c**0.5+13.723711519422*x**4)
            bsIv_rationalCall = lambda x,v: norm.cdf(x/v+v/2)-np.exp(-x)*norm.cdf(x/v-v/2)
            bsIv_rationalVega = lambda x,v: norm.pdf(x/v+v/2)
        x = np.log(forwardPrice/strike)
        c = priceMkt/spotPrice
        itm = (x > 0)
        put = (optionType == "put")
        c[put] += 1-np.exp(-x[put]) # Cast as call prices
        c[itm] = np.exp(x[itm])*(c[itm]-1)+1; x[itm] *= -1 # Cast as OTM
        domain = (x>=-0.5)&(x<=0)&(c>bsIv_rationalLBnd(x))&(c<bsIv_rationalUBnd(x))
        xD,cD,mD = x[domain],c[domain],maturity[domain]
        vD = bsIv_rationalIv(xD,cD)
        for i in range(2): # NR-polishing
            vD -= (bsIv_rationalCall(xD,vD)-cD)/bsIv_rationalVega(xD,vD)
        impVol[domain] = vD/np.sqrt(mD)
        if np.sum(~domain):
            # Delegate options not in domain to Bisection
            impVol[~domain] = BlackScholesImpliedVol(spotPrice, strike[~domain], maturity[~domain], riskFreeRate, priceMkt[~domain], optionType[~domain], method="Bisection")
        return impVol

    elif method == "Chebychev": # Chebychev IV-interpolation
        # Ref: Glau, The Chebyshev Method for the Implied Volatility
        # TO-DO
        return impVol

#### Characteristic Function ###################################################
# Characteristic Function: E(exp(i*u*XT)), XT = log(ST/S0)
# Return charFunc with arguments (u, maturity)

def BlackScholesCharFunc(vol, riskFreeRate=0, curry=False):
    # Characteristic function for Black-Scholes model
    if curry:
        def charFunc(u):
            chExp = 1j*u*riskFreeRate-vol**2/2*u*(u+1j)
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(chExp*maturity)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            return np.exp((1j*u*riskFreeRate-vol**2/2*u*(u+1j))*maturity)
    return charFunc

#### Diffusion

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

#### Jump

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
            return np.exp((1j*u*riskFreeRate-vol**2/2*u*(u+1j)-1j*u*jumpInt*(np.exp(jumpMean+jumpSd**2/2)-1)+jumpInt*(np.exp(1j*u*jumpMean-u**2*jumpSd**2/2)-1))*maturity)
    return charFunc

#### Jump-diffusion

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

def SVJJCharFunc(meanRevRate, correlation, volOfVol, meanVar, currentVar, varJump, jumpInt, jumpMean, jumpSd, riskFreeRate=0, curry=False):
    # Characteristic function for SVJJ model (HestonJump-MertonJump)
    # Ref: Gatheral, Volatility Workshop VW2.pdf; Andrew Matytsin (1999)
    if curry:
        def charFunc(u):
            global svjjCF_I, svjjCF_matPrev
            svjjCF_I = 0; svjjCF_matPrev = 0
            alpha = -u**2/2-1j*u/2
            beta = meanRevRate-correlation*volOfVol*1j*u
            gamma = volOfVol**2/2
            d = np.sqrt(beta**2-4*alpha*gamma)
            rp = (beta+d)/(2*gamma)
            rm = (beta-d)/(2*gamma)
            g = rm/rp
            J = np.exp(1j*u*jumpMean-u**2*jumpSd**2/2)
            chExp0 = 1j*u*riskFreeRate-1j*u*jumpInt*(np.exp(jumpMean+jumpSd**2/2)-1)
            Dfunc = lambda t: rm*(1-np.exp(-d*t))/(1-g*np.exp(-d*t))
            Ifunc = lambda t: np.exp(varJump*Dfunc(t))
            def charFuncFixedU(u, maturity): # u is dummy
                global svjjCF_I, svjjCF_matPrev
                # I = quad_vec(lambda t: np.exp(varJump*Dfunc(t)),0,maturity)[0]/maturity
                svjjCF_I += quad_vec(Ifunc,svjjCF_matPrev,maturity)[0]
                svjjCF_matPrev = maturity; I = svjjCF_I/maturity
                D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
                C = meanRevRate*(rm*maturity-2/volOfVol**2*np.log((1-g*np.exp(-d*maturity))/(1-g)))
                chExp1 = jumpInt*(J*I-1)
                return np.exp((chExp0+chExp1)*maturity+C*meanVar+D*currentVar)
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
            Dfunc = lambda t: rm*(1-np.exp(-d*t))/(1-g*np.exp(-d*t))
            I = quad_vec(lambda t: np.exp(varJump*Dfunc(t)),0,maturity)[0]/maturity
            D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
            C = meanRevRate*(rm*maturity-2/volOfVol**2*np.log((1-g*np.exp(-d*maturity))/(1-g)))
            J = jumpInt*maturity*(np.exp(1j*u*jumpMean-u**2*jumpSd**2/2)*I-1-1j*u*(np.exp(jumpMean+jumpSd**2/2)-1))
            return np.exp(1j*u*riskFreeRate*maturity+C*meanVar+D*currentVar+J)
    return charFunc

#### Time-changed

def VarianceGammaCharFunc(vol, drift, timeChgVar, riskFreeRate=0, curry=False):
    # Characteristic function for Variance-Gamma model (Brownian parametrization)
    # Xt = drift*gamma(t;1,timeChgVar) + vol*W(gamma(t;1,timeChgVar))
    # Ref: Madan, The Variance Gamma Process and Option Pricing
    if curry:
        def charFunc(u):
            chExp = 1j*u*(riskFreeRate+1/timeChgVar*np.nan_to_num(np.log(1-(drift+vol**2/2)*timeChgVar)))-np.log(1-1j*u*drift*timeChgVar+u**2*vol**2*timeChgVar/2)/timeChgVar
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(chExp*maturity)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            return np.exp(1j*u*(riskFreeRate+1/timeChgVar*np.log(1-(drift+vol**2/2)*timeChgVar))*maturity)*(1-1j*u*drift*timeChgVar+u**2*vol**2*timeChgVar/2)**(-maturity/timeChgVar)
    return charFunc

def VarianceGammaLevyCharFunc(C, G, M, riskFreeRate=0, curry=False):
    # Characteristic function for Variance-Gamma model (Levy measure parametrization)
    # Levy measure k(x) = C*exp(G*x)/|x| for x<0, C*exp(-M*x)/x for x>0
    # Ref: Madan, The Variance Gamma Process and Option Pricing
    if curry:
        def charFunc(u):
            chExp = 1j*u*riskFreeRate+C*np.log(((M-1)*(G+1))/((M-1j*u)*(G+1j*u)))
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(chExp*maturity)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            return np.exp((1j*u*riskFreeRate+C*np.log(((M-1)*(G+1))/((M-1j*u)*(G+1j*u))))*maturity)
    return charFunc

def CGMYCharFunc(C, G, M, Y, riskFreeRate=0, curry=False):
    # Characteristic function for CGMY model
    # Levy measure k(x) = C*exp(G*x)/|x|^(Y+1) for x<0, C*exp(-M*x)/x^Y for x>0
    # Ref: CGMY, The Fine Structure of Asset Returns: An Empirical Investigation
    gammaY = sp.special.gamma(-Y)
    if curry:
        def charFunc(u):
            chExp = 1j*u*riskFreeRate+C*gammaY*((M-1j*u)**Y+(G+1j*u)**Y-(M-1)**Y-(G+1)**Y)
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(chExp*maturity)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            return np.exp((1j*u*riskFreeRate+C*gammaY*((M-1j*u)**Y+(G+1j*u)**Y-(M-1)**Y-(G+1)**Y))*maturity)
    return charFunc

def eCGMYCharFunc(vol, C, G, M, Y, riskFreeRate=0, curry=False):
    # Characteristic function for extended CGMY model
    # Ref: CGMY, The Fine Structure of Asset Returns: An Empirical Investigation
    gammaY = sp.special.gamma(-Y)
    if curry:
        def charFunc(u):
            chExp = 1j*u*riskFreeRate-vol**2/2*u*(u+1j)+C*gammaY*((M-1j*u)**Y+(G+1j*u)**Y-(M-1)**Y-(G+1)**Y)
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(chExp*maturity)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            return np.exp((1j*u*riskFreeRate-vol**2/2*u*(u+1j)+C*gammaY*((M-1j*u)**Y+(G+1j*u)**Y-(M-1)**Y-(G+1)**Y))*maturity)
    return charFunc

def pnCGMYCharFunc(C, CRatio, G, M, Yp, Yn, riskFreeRate=0, curry=False):
    # Characteristic function for pn-CGMY model
    # Levy measure k(x) = Cn*exp(G*x)/|x|^(Yn+1) for x<0, Cp*exp(-M*x)/x^(Yp+1) for x>0
    # Ref: CGMY, Stochastic Volatility for Levy Processes
    gammaYp = sp.special.gamma(-Yp)
    gammaYn = sp.special.gamma(-Yn)
    if curry:
        def charFunc(u):
            chExp = 1j*u*riskFreeRate+C*(gammaYp*((M-1j*u)**Yp-(M-1)**Yp)+CRatio*gammaYn*((G+1j*u)**Yn-(G+1)**Yn))
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(chExp*maturity)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            return np.exp((1j*u*riskFreeRate+C*(gammaYp*((M-1j*u)**Yp-(M-1)**Yp)+CRatio*gammaYn*((G+1j*u)**Yn-(G+1)**Yn)))*maturity)
    return charFunc

def NIGCharFunc(vol, drift, timeChgDrift, riskFreeRate=0, curry=False):
    # Characteristic function for NIG model
    # Ref: CGMY, Stochastic Volatility for Levy Processes
    if curry:
        def charFunc(u):
            chExp = (1j*u*riskFreeRate+np.nan_to_num(np.sqrt(timeChgDrift**2-2*drift-vol**2))-np.sqrt(timeChgDrift**2-2*drift*1j*u+vol**2*u**2))
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(chExp*maturity)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            return np.exp((1j*u*riskFreeRate+np.sqrt(timeChgDrift**2-2*drift-vol**2)-np.sqrt(timeChgDrift**2-2*drift*1j*u+vol**2*u**2))*maturity)
    return charFunc

#### Stochastic-arrival

def CIRCharFunc(initVal, meanRevRate, mean, vol, curry=False):
    # Characteristic function for CIR process (NOT for stock price!)
    # Ref: CGMY, Stochastic Volatility for Levy Processes
    if curry:
        def charFunc(u):
            gamma = np.sqrt(meanRevRate**2-2*vol**2*1j*u)
            A0 = np.exp(meanRevRate**2*mean/vol**2)
            A1 = meanRevRate/gamma
            A2 = 2*meanRevRate*mean/vol**2
            B0 = 2*1j*u
            def charFuncFixedU(u, maturity): # u is dummy
                A = A0**maturity/(np.cosh(gamma*maturity/2)+A1*np.sinh(gamma*maturity/2))**A2
                B = B0/(meanRevRate+gamma/np.tanh(gamma*maturity/2))
                return A*np.exp(B*initVal)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            gamma = np.sqrt(meanRevRate**2-2*vol**2*1j*u)
            A = np.exp(meanRevRate**2*mean*maturity/vol**2)/(np.cosh(gamma*maturity/2)+meanRevRate/gamma*np.sinh(gamma*maturity/2))**(2*meanRevRate*mean/vol**2)
            B = 2*1j*u/(meanRevRate+gamma/np.tanh(gamma*maturity/2))
            return A*np.exp(B*initVal)
    return charFunc

def VGSACharFunc(C, G, M, saMeanRevRate, saMean, saVol, riskFreeRate=0, curry=False):
    # Characteristic function for VG-SA model
    # Ref: CGMY, Stochastic Volatility for Levy Processes
    chExp = lambda u: np.log((M*G)/((M-1j*u)*(G+1j*u))) # VG char exponent
    if curry:
        def charFunc(u):
            iu = 1j*u
            iur = 1j*u*riskFreeRate
            chExp0 = -1j*chExp(u)
            chExp1 = -1j*chExp(-1j)
            cirCF0 = CIRCharFunc(C, saMeanRevRate, saMean, saVol, curry=True)(chExp0)
            cirCF1 = CIRCharFunc(C, saMeanRevRate, saMean, saVol, curry=True)(chExp1)
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(iur*maturity)*np.nan_to_num(cirCF0(chExp0,maturity)/cirCF1(chExp1,maturity)**iu)
            return charFuncFixedU
    else:
        cirCF = CIRCharFunc(C, saMeanRevRate, saMean, saVol)
        def charFunc(u, maturity):
            return np.exp(1j*u*riskFreeRate*maturity)*np.nan_to_num(cirCF(-1j*chExp(u),maturity)/cirCF(-1j*chExp(-1j),maturity)**(1j*u))
    return charFunc

def CGMYSACharFunc(C, CRatio, G, M, Yp, Yn, saMeanRevRate, saMean, saVol, riskFreeRate=0, curry=False):
    # Characteristic function for CGMY-SA model
    # Ref: CGMY, Stochastic Volatility for Levy Processes
    gammaYp = sp.special.gamma(-Yp)
    gammaYn = sp.special.gamma(-Yn)
    chExp = lambda u: C*(gammaYp*((M-1j*u)**Yp-M**Yp)+CRatio*gammaYn*((G+1j*u)**Yn-G**Yn)) # CGMY char exponent
    if curry:
        def charFunc(u):
            iu = 1j*u
            iur = 1j*u*riskFreeRate
            chExp0 = -1j*chExp(u)
            chExp1 = -1j*chExp(-1j)
            cirCF0 = CIRCharFunc(C, saMeanRevRate, saMean, saVol, curry=True)(chExp0)
            cirCF1 = CIRCharFunc(C, saMeanRevRate, saMean, saVol, curry=True)(chExp1)
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(iur*maturity)*np.nan_to_num(cirCF0(chExp0,maturity)/cirCF1(chExp1,maturity)**iu)
            return charFuncFixedU
    else:
        cirCF = CIRCharFunc(C, saMeanRevRate, saMean, saVol)
        def charFunc(u, maturity):
            return np.exp(1j*u*riskFreeRate*maturity)*np.nan_to_num(cirCF(-1j*chExp(u),maturity)/cirCF(-1j*chExp(-1j),maturity)**(1j*u))
    return charFunc

def NIGSACharFunc(vol, drift, timeChgDrift, saMeanRevRate, saMean, saVol, riskFreeRate=0, curry=False):
    # Characteristic function for NIG-SA model
    # Ref: CGMY, Stochastic Volatility for Levy Processes
    chExp = lambda u: timeChgDrift-np.sqrt(timeChgDrift**2-2*drift*1j*u+u**2) # NIG char exponent
    if curry:
        def charFunc(u):
            iu = 1j*u
            iur = 1j*u*riskFreeRate
            chExp0 = -1j*chExp(u)
            chExp1 = -1j*chExp(-1j)
            cirCF0 = CIRCharFunc(vol, saMeanRevRate, saMean, saVol, curry=True)(chExp0)
            cirCF1 = CIRCharFunc(vol, saMeanRevRate, saMean, saVol, curry=True)(chExp1)
            def charFuncFixedU(u, maturity): # u is dummy
                return np.exp(iur*maturity)*np.nan_to_num(cirCF0(chExp0,maturity)/cirCF1(chExp1,maturity)**iu)
            return charFuncFixedU
    else:
        cirCF = CIRCharFunc(vol, saMeanRevRate, saMean, saVol)
        def charFunc(u, maturity):
            return np.exp(1j*u*riskFreeRate*maturity)*np.nan_to_num(cirCF(-1j*chExp(u),maturity)/cirCF(-1j*chExp(-1j),maturity)**(1j*u))
    return charFunc

#### Fractional BM

def rHestonPoorMansCharFunc(hurstExp, correlation, volOfVol, currentVar, riskFreeRate=0, curry=False):
    # Characteristic function for rHeston model (poor man's Heston approx)
    # Heston approx: meanRevRate=0 hence no meanVar-dependence, currentVar is var swap price
    # Ref: Gatheral, Roughening Heston
    volOfVolFactor = np.sqrt(3/(2*hurstExp+2))*volOfVol/sp.special.gamma(hurstExp+1.5)
    if curry:
        def charFunc(u):
            iur = 1j*u*riskFreeRate
            alpha = -u**2/2-1j*u/2
            betaFactor = -correlation*1j*u
            def charFuncFixedU(u, maturity): # u is dummy
                volOfVolMod = volOfVolFactor/maturity**(0.5-hurstExp)
                beta = betaFactor*volOfVolMod
                gamma = volOfVolMod**2/2
                d = np.sqrt(beta**2-4*alpha*gamma)
                rp = (beta+d)/(2*gamma)
                rm = (beta-d)/(2*gamma)
                # g = rm/rp
                g = np.nan_to_num(rm/rp)
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
            # g = rm/rp
            g = np.nan_to_num(rm/rp)
            D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
            return np.exp(1j*u*riskFreeRate*maturity+D*currentVar)
    return charFunc

def rHestonPoorMansModCharFunc(hurstExp, meanRevRate, correlation, volOfVol, meanVar, currentVar, riskFreeRate=0, curry=False):
    # Characteristic function for rHeston model (poor man's Heston approx modified)
    # Ref: Gatheral, Roughening Heston
    volOfVolFactor = np.sqrt(3/(2*hurstExp+2))*volOfVol/sp.special.gamma(hurstExp+1.5)
    if curry:
        def charFunc(u):
            iur = 1j*u*riskFreeRate
            alpha = -u**2/2-1j*u/2
            betaFactor = -correlation*1j*u
            def charFuncFixedU(u, maturity): # u is dummy
                volOfVolMod = volOfVolFactor/maturity**(0.5-hurstExp)
                beta = betaFactor*volOfVolMod
                gamma = volOfVolMod**2/2
                d = np.sqrt(beta**2-4*alpha*gamma)
                rp = (beta+d)/(2*gamma)
                rm = (beta-d)/(2*gamma)
                # g = rm/rp
                g = np.nan_to_num(rm/rp)
                D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
                C = meanRevRate*(rm*maturity-2/volOfVolMod**2*np.log((1-g*np.exp(-d*maturity))/(1-g)))
                return np.exp(iur*maturity+C*meanVar+D*currentVar)
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
            # g = rm/rp
            g = np.nan_to_num(rm/rp)
            D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
            C = meanRevRate*(rm*maturity-2/volOfVolMod**2*np.log((1-g*np.exp(-d*maturity))/(1-g)))
            return np.exp(1j*u*riskFreeRate*maturity+C*meanVar+D*currentVar)
    return charFunc

def dhPade33(hurstExp, correlation, volOfVol, curry=False):
    # D^alpha(h) where h = solution to fractional Riccati equation
    # Return kernel matrix of dimension (a,x), a = cfArg, x = maturity
    # Ref: Gatheral, Rational Approximation of the Rough Heston Solution
    H = hurstExp
    al = hurstExp + .5
    rho = correlation
    nu = volOfVol
    if curry:
        def kernel(a):
            aa = np.sqrt(a * (a + 1j) - rho**2 * a**2)
            rm = -1j * rho * a - aa
            rp = -1j * rho * a + aa

            b1 = -a*(a+1j)/(2*sp.special.gamma(1+al))
            b2 = (1-a*1j)*a**2*rho/(2*sp.special.gamma(1+2*al))
            b3 = sp.special.gamma(1+2*al)/sp.special.gamma(1+3*al)*(a**2*(1j+a)**2/(8*sp.special.gamma(1+al)**2)+(a+1j)*a**3*rho**2/(2*sp.special.gamma(1+2*al)))

            g0 = rm
            g1 = -rm/(aa*sp.special.gamma(1-al))
            g2 = rm/aa**2/sp.special.gamma(1-2*al)*(1+rm/(2*aa)*sp.special.gamma(1-2*al)/sp.special.gamma(1-al)**2)

            den = g0**3+2*b1*g0*g1-b2*g1**2+b1**2*g2+b2*g0*g2

            p1 = b1
            p2 = (b1**2*g0**2 + b2*g0**3 + b1**3*g1 + b1*b2*g0*g1 - b2**2*g1**2 + b1*b3*g1**2 + b2**2*g0*g2 - b1*b3*g0*g2)/den
            q1 = (b1*g0**2 + b1**2*g1 - b2*g0*g1 + b3*g1**2 - b1*b2*g2 - b3*g0*g2)/den
            q2 = (b1**2*g0 + b2*g0**2 - b1*b2*g1 - b3*g0*g1 + b2**2*g2 - b1*b3*g2)/den
            q3 = (b1**3 + 2*b1*b2*g0 + b3*g0**2 - b2**2*g1 + b1*b3*g1)/den
            p3 = g0*q3

            def kernelFixedA(a, x):
                y = x**al
                h = (np.outer(p1,y) + np.outer(p2,y**2) + np.outer(p3,y**3))/(1 + np.outer(q1,y) + np.outer(q2,y**2) + np.outer(q3,y**3))
                return .5*(h.T-rm).T*(h.T-rp).T
                # h = (p1*y + p2*y**2 + p3*y**3)/(1 + q1*y + q2*y**2 + q3*y**3)
                # return .5*(h-rm)*(h-rp)
            return kernelFixedA
    else:
        def kernel(a, x):
            aa = np.sqrt(a * (a + 1j) - rho**2 * a**2)
            rm = -1j * rho * a - aa
            rp = -1j * rho * a + aa

            b1 = -a*(a+1j)/(2*sp.special.gamma(1+al))
            b2 = (1-a*1j)*a**2*rho/(2*sp.special.gamma(1+2*al))
            b3 = sp.special.gamma(1+2*al)/sp.special.gamma(1+3*al)*(a**2*(1j+a)**2/(8*sp.special.gamma(1+al)**2)+(a+1j)*a**3*rho**2/(2*sp.special.gamma(1+2*al)))

            g0 = rm
            g1 = -rm/(aa*sp.special.gamma(1-al))
            g2 = rm/aa**2/sp.special.gamma(1-2*al)*(1+rm/(2*aa)*sp.special.gamma(1-2*al)/sp.special.gamma(1-al)**2)

            den = g0**3+2*b1*g0*g1-b2*g1**2+b1**2*g2+b2*g0*g2

            p1 = b1
            p2 = (b1**2*g0**2 + b2*g0**3 + b1**3*g1 + b1*b2*g0*g1 - b2**2*g1**2 + b1*b3*g1**2 + b2**2*g0*g2 - b1*b3*g0*g2)/den
            q1 = (b1*g0**2 + b1**2*g1 - b2*g0*g1 + b3*g1**2 - b1*b2*g2 - b3*g0*g2)/den
            q2 = (b1**2*g0 + b2*g0**2 - b1*b2*g1 - b3*g0*g1 + b2**2*g2 - b1*b3*g2)/den
            q3 = (b1**3 + 2*b1*b2*g0 + b3*g0**2 - b2**2*g1 + b1*b3*g1)/den
            p3 = g0*q3

            y = x**al
            h = (np.outer(p1,y) + np.outer(p2,y**2) + np.outer(p3,y**3))/(1 + np.outer(q1,y) + np.outer(q2,y**2) + np.outer(q3,y**3))
            return .5*(h.T-rm).T*(h.T-rp).T
            # h = (p1*y + p2*y**2 + p3*y**3)/(1 + q1*y + q2*y**2 + q3*y**3)
            # return .5*(h-rm)*(h-rp)
    return kernel

def dhPade44(hurstExp, correlation, volOfVol, curry=False):
    # D**alpha(h) where h = solution to fractional Riccati equation
    # Return kernel matrix of dimension (a,x), a = cfArg, x = maturity
    # Ref: Gatheral, Rational Approximation of the Rough Heston Solution
    H = hurstExp
    al = hurstExp + .5
    rho = correlation
    nu = volOfVol
    if curry:
        def kernel(a):
            aa = np.sqrt(a * (a + 1j) - rho**2 * a**2)
            rm = -1j * rho * a - aa
            rp = -1j * rho * a + aa

            b1 = -a*(a+1j)/(2*sp.special.gamma(1+al))
            b2 = (1-a*1j)*a**2* ho/(2*sp.special.gamma(1+2*al))
            b3 = sp.special.gamma(1+2*al)/sp.special.gamma(1+3*al)*(a**2*(1j+a)**2/(8*sp.special.gamma(1+al)**2)+(a+1j)*a**3*rho**2/(2*sp.special.gamma(1+2*al)))
            b4 = ((a**2*(1j+a)**2)/(8*sp.special.gamma(1+al)**2)+(1j*rho**2*(1-1j*a)*a**3)/(2*sp.special.gamma(1+2*al)))*sp.special.gamma(1+2*al)/sp.special.gamma(1+3*al)

            g0 = rm
            g1 = -rm/(aa*sp.special.gamma(1-al))
            g2 = rm/aa**2/sp.special.gamma(1-2*al)*(1+rm/(2*aa)*sp.special.gamma(1-2*al)/sp.special.gamma(1-al)**2)
            g3 = (rm*(-1-(rm*sp.special.gamma(1-2*al))/(2.*aa*sp.special.gamma(1-al)**2)-(rm*sp.special.gamma(1-3*al)*(1+(rm*sp.special.gamma(1-2*al))/(2.*aa*sp.special.gamma(1-al)**2)))/(aa*sp.special.gamma(1-2*al)*sp.special.gamma(1- al))))/(aa**3*sp.special.gamma(1-3*al))

            den = (g0**4 + 3*b1*g0**2*g1 + b1**2*g1**2 - 2*b2*g0*g1**2 + b3*g1**3 +
                    2*b1**2*g0*g2 + 2*b2*g0**2*g2 - 2*b1*b2*g1*g2 - 2*b3*g0*g1*g2 +
                    b2**2*g2**2 - b1*b3*g2**2 + b1**3*g3 + 2*b1*b2*g0*g3 + b3*g0**2*g3 -
                    b2**2*g1*g3 + b1*b3*g1*g3)

            p1 = b1
            p2 = (b1**2*g0**3 + b2*g0**4 + 2*b1**3*g0*g1 + 2*b1*b2*g0**2*g1 -
                   b1**2*b2*g1**2 - 2*b2**2*g0*g1**2 + b1*b3*g0*g1**2 + b2*b3*g1**3 -
                   b1*b4*g1**3 + b1**4*g2 + 2*b1**2*b2*g0*g2 + 2*b2**2*g0**2*g2 -
                   b1*b3*g0**2*g2 - b1*b2**2*g1*g2 + b1**2*b3*g1*g2 - 2*b2*b3*g0*g1*g2 +
                   2*b1*b4*g0*g1*g2 + b2**3*g2**2 - 2*b1*b2*b3*g2**2 + b1**2*b4*g2**2 +
                   b1*b2**2*g0*g3 - b1**2*b3*g0*g3 + b2*b3*g0**2*g3 - b1*b4*g0**2*g3 -
                   b2**3*g1*g3 + 2*b1*b2*b3*g1*g3 - b1**2*b4*g1*g3)/den
            p3 = (b1**3*g0**2 + 2*b1*b2*g0**3 + b3*g0**4 + b1**4*g1 + 2*b1**2*b2*g0*g1 - b2**2*g0**2*g1 + 2*b1*b3*g0**2*g1 -
                   2*b1*b2**2*g1**2 + 2*b1**2*b3*g1**2 - b2*b3*g0*g1**2 + b1*b4*g0*g1**2 + b3**2*g1**3 - b2*b4*g1**3 +
                   b1*b2**2*g0*g2 - b1**2*b3*g0*g2 + b2*b3*g0**2*g2 - b1*b4*g0**2*g2 + b2**3*g1*g2 - 2*b1*b2*b3*g1*g2 +
                   b1**2*b4*g1*g2 - 2*b3**2*g0*g1*g2 + 2*b2*b4*g0*g1*g2 - b2**3*g0*g3 + 2*b1*b2*b3*g0*g3 - b1**2*b4*g0*g3 +
                   b3**2*g0**2*g3 - b2*b4*g0**2*g3)/den

            q1 = (b1*g0**3 + 2*b1**2*g0*g1 - b2*g0**2*g1 - 2*b1*b2*g1**2 + b3*g0*g1**2 - b4*g1**3 + b1**3*g2 -
                   b3*g0**2*g2 + b2**2*g1*g2 + b1*b3*g1*g2 + 2*b4*g0*g1*g2 - b2*b3*g2**2 + b1*b4*g2**2 - b1**2*b2*g3 - b2**2*g0*g3 -
                   b1*b3*g0*g3 - b4*g0**2*g3 + b2*b3*g1*g3 - b1*b4*g1*g3)/den
            q2 = (b1**2*g0**2 + b2*g0**3 + b1**3*g1 - b3*g0**2*g1 + b1*b3*g1**2 + b4*g0*g1**2 - b1**2*b2*g2 + b2**2*g0*g2 -
                   3*b1*b3*g0*g2 - b4*g0**2*g2 - b2*b3*g1*g2 + b1*b4*g1*g2 + b3**2*g2**2 - b2*b4*g2**2 + b1*b2**2*g3 - b1**2*b3*g3 +
                   b2*b3*g0*g3 - b1*b4*g0*g3 - b3**2*g1*g3 + b2*b4*g1*g3)/den
            q3 = (b1**3*g0 + 2*b1*b2*g0**2 + b3*g0**3 - b1**2*b2*g1 - 2*b2**2*g0*g1 - b4*g0**2*g1 + b2*b3*g1**2 - b1*b4*g1**2 +
                   b1*b2**2*g2 - b1**2*b3*g2 + b2*b3*g0*g2 - b1*b4*g0*g2 - b3**2*g1*g2 + b2*b4*g1*g2 - b2**3*g3 + 2*b1*b2*b3*g3 -
                   b1**2*b4*g3 + b3**2*g0*g3 - b2*b4*g0*g3)/den
            q4 = (b1**4 + 3*b1**2*b2*g0 + b2**2*g0**2 + 2*b1*b3*g0**2 + b4*g0**3 - 2*b1*b2**2*g1 + 2*b1**2*b3*g1 -
                   2*b2*b3*g0*g1 + 2*b1*b4*g0*g1 + b3**2*g1**2 - b2*b4*g1**2 + b2**3*g2 - 2*b1*b2*b3*g2 + b1**2*b4*g2 - b3**2*g0*g2 +
                   b2*b4*g0*g2)/den

            p4 = g0*q4

            def kernelFixedA(a, x):
                y = x**al
                h = (np.outer(p1,y) + np.outer(p2,y**2) + np.outer(p3,y**3) + np.outer(p4,y**4))/(1 + np.outer(q1,y) + np.outer(q2,y**2) + np.outer(q3,y**3) + np.outer(q4,y**4))
                return .5*(h.T-rm).T*(h.T-rp).T
            return kernelFixedA
    else:
        def kernel(a, x):
            aa = sqrt(a * (a + 1j) - rho**2 * a**2)
            rm = -1j * rho * a - aa
            rp = -1j * rho * a + aa

            b1 = -a*(a+1j)/(2*sp.special.gamma(1+al))
            b2 = (1-a*1j)*a**2* ho/(2*sp.special.gamma(1+2*al))
            b3 = sp.special.gamma(1+2*al)/sp.special.gamma(1+3*al)*(a**2*(1j+a)**2/(8*sp.special.gamma(1+al)**2)+(a+1j)*a**3*rho**2/(2*sp.special.gamma(1+2*al)))
            b4 = ((a**2*(1j+a)**2)/(8*sp.special.gamma(1+al)**2)+(1j*rho**2*(1-1j*a)*a**3)/(2*sp.special.gamma(1+2*al)))*sp.special.gamma(1+2*al)/sp.special.gamma(1+3*al)

            g0 = rm
            g1 = -rm/(aa*sp.special.gamma(1-al))
            g2 = rm/aa**2/sp.special.gamma(1-2*al)*(1+rm/(2*aa)*sp.special.gamma(1-2*al)/sp.special.gamma(1-al)**2)
            g3 = (rm*(-1-(rm*sp.special.gamma(1-2*al))/(2.*aa*sp.special.gamma(1-al)**2)-(rm*sp.special.gamma(1-3*al)*(1+(rm*sp.special.gamma(1-2*al))/(2.*aa*sp.special.gamma(1-al)**2)))/(aa*sp.special.gamma(1-2*al)*sp.special.gamma(1- al))))/(aa**3*sp.special.gamma(1-3*al))

            den = (g0**4 + 3*b1*g0**2*g1 + b1**2*g1**2 - 2*b2*g0*g1**2 + b3*g1**3 +
                    2*b1**2*g0*g2 + 2*b2*g0**2*g2 - 2*b1*b2*g1*g2 - 2*b3*g0*g1*g2 +
                    b2**2*g2**2 - b1*b3*g2**2 + b1**3*g3 + 2*b1*b2*g0*g3 + b3*g0**2*g3 -
                    b2**2*g1*g3 + b1*b3*g1*g3)

            p1 = b1
            p2 = (b1**2*g0**3 + b2*g0**4 + 2*b1**3*g0*g1 + 2*b1*b2*g0**2*g1 -
                   b1**2*b2*g1**2 - 2*b2**2*g0*g1**2 + b1*b3*g0*g1**2 + b2*b3*g1**3 -
                   b1*b4*g1**3 + b1**4*g2 + 2*b1**2*b2*g0*g2 + 2*b2**2*g0**2*g2 -
                   b1*b3*g0**2*g2 - b1*b2**2*g1*g2 + b1**2*b3*g1*g2 - 2*b2*b3*g0*g1*g2 +
                   2*b1*b4*g0*g1*g2 + b2**3*g2**2 - 2*b1*b2*b3*g2**2 + b1**2*b4*g2**2 +
                   b1*b2**2*g0*g3 - b1**2*b3*g0*g3 + b2*b3*g0**2*g3 - b1*b4*g0**2*g3 -
                   b2**3*g1*g3 + 2*b1*b2*b3*g1*g3 - b1**2*b4*g1*g3)/den
            p3 = (b1**3*g0**2 + 2*b1*b2*g0**3 + b3*g0**4 + b1**4*g1 + 2*b1**2*b2*g0*g1 - b2**2*g0**2*g1 + 2*b1*b3*g0**2*g1 -
                   2*b1*b2**2*g1**2 + 2*b1**2*b3*g1**2 - b2*b3*g0*g1**2 + b1*b4*g0*g1**2 + b3**2*g1**3 - b2*b4*g1**3 +
                   b1*b2**2*g0*g2 - b1**2*b3*g0*g2 + b2*b3*g0**2*g2 - b1*b4*g0**2*g2 + b2**3*g1*g2 - 2*b1*b2*b3*g1*g2 +
                   b1**2*b4*g1*g2 - 2*b3**2*g0*g1*g2 + 2*b2*b4*g0*g1*g2 - b2**3*g0*g3 + 2*b1*b2*b3*g0*g3 - b1**2*b4*g0*g3 +
                   b3**2*g0**2*g3 - b2*b4*g0**2*g3)/den

            q1 = (b1*g0**3 + 2*b1**2*g0*g1 - b2*g0**2*g1 - 2*b1*b2*g1**2 + b3*g0*g1**2 - b4*g1**3 + b1**3*g2 -
                   b3*g0**2*g2 + b2**2*g1*g2 + b1*b3*g1*g2 + 2*b4*g0*g1*g2 - b2*b3*g2**2 + b1*b4*g2**2 - b1**2*b2*g3 - b2**2*g0*g3 -
                   b1*b3*g0*g3 - b4*g0**2*g3 + b2*b3*g1*g3 - b1*b4*g1*g3)/den
            q2 = (b1**2*g0**2 + b2*g0**3 + b1**3*g1 - b3*g0**2*g1 + b1*b3*g1**2 + b4*g0*g1**2 - b1**2*b2*g2 + b2**2*g0*g2 -
                   3*b1*b3*g0*g2 - b4*g0**2*g2 - b2*b3*g1*g2 + b1*b4*g1*g2 + b3**2*g2**2 - b2*b4*g2**2 + b1*b2**2*g3 - b1**2*b3*g3 +
                   b2*b3*g0*g3 - b1*b4*g0*g3 - b3**2*g1*g3 + b2*b4*g1*g3)/den
            q3 = (b1**3*g0 + 2*b1*b2*g0**2 + b3*g0**3 - b1**2*b2*g1 - 2*b2**2*g0*g1 - b4*g0**2*g1 + b2*b3*g1**2 - b1*b4*g1**2 +
                   b1*b2**2*g2 - b1**2*b3*g2 + b2*b3*g0*g2 - b1*b4*g0*g2 - b3**2*g1*g2 + b2*b4*g1*g2 - b2**3*g3 + 2*b1*b2*b3*g3 -
                   b1**2*b4*g3 + b3**2*g0*g3 - b2*b4*g0*g3)/den
            q4 = (b1**4 + 3*b1**2*b2*g0 + b2**2*g0**2 + 2*b1*b3*g0**2 + b4*g0**3 - 2*b1*b2**2*g1 + 2*b1**2*b3*g1 -
                   2*b2*b3*g0*g1 + 2*b1*b4*g0*g1 + b3**2*g1**2 - b2*b4*g1**2 + b2**3*g2 - 2*b1*b2*b3*g2 + b1**2*b4*g2 - b3**2*g0*g2 +
                   b2*b4*g0*g2)/den

            p4 = g0*q4

            y = x**al
            h = (np.outer(p1,y) + np.outer(p2,y**2) + np.outer(p3,y**3) + np.outer(p4,y**4))/(1 + np.outer(q1,y) + np.outer(q2,y**2) + np.outer(q3,y**3) + np.outer(q4,y**4))
            return .5*(h.T-rm).T*(h.T-rp).T
    return kernel

def rHestonPadeCharFunc(hurstExp, correlation, volOfVol, fvFunc, dhPade=dhPade33, n=100, riskFreeRate=0, curry=False):
    # Characteristic function for rHeston model (poor man's crude approx)
    # Ref: Gatheral, Rational Approximation of the Rough Heston Solution
    global rhPadeCF_w, rhPadeCF_dict
    H = hurstExp
    al = hurstExp + .5
    rho = correlation
    nu = volOfVol
    if rhPadeCF_w is None:
        w = np.arange(n+1)
        w = 3+(-1)**(w+1)
        w[0] = 1; w[n] = 1
        rhPadeCF_w = w
    else:
        w = rhPadeCF_w
    if curry:
        def charFunc(u):
            kernel = dhPade(H,rho,nu,curry)(u)
            def charFuncFixedU(u, maturity): # u is dummy
                if maturity in rhPadeCF_dict:
                    dt  = rhPadeCF_dict[maturity]["dt"]
                    t   = rhPadeCF_dict[maturity]["t"]
                    # xi  = rhPadeCF_dict[maturity]["xi"]
                    xiw = rhPadeCF_dict[maturity]["xiw"]
                else:
                    dt  = maturity/n
                    t   = np.linspace(0,maturity,n+1)
                    xi  = fvFunc(t)
                    xiw = xi[::-1]*w
                    rhPadeCF_dict[maturity] = {
                        "dt":  dt,
                        "t":   t,
                        "xi":  xi,
                        "xiw": xiw
                    }
                x = nu**(1/al)*t
                dah = np.nan_to_num(kernel(u,x))
                return np.exp(dah.dot(xiw)*dt/3)
                # return np.exp(dah.dot(xi[::-1])*dt)
            return charFuncFixedU
    else:
        def charFunc(u, maturity):
            dt  = maturity/n
            t   = np.linspace(0,maturity,n+1)
            x   = nu**(1/al)*t
            xi  = fvFunc(t)
            dah = np.nan_to_num(dhPade(H,rho,nu)(u,x))
            return np.exp(dah.dot(xi[::-1]*w)*dt/3)
            # return np.exp(dah.dot(xi[::-1])*dt)
    return charFunc

#### Pricing Formula ###########################################################
# Return prices at S0=1 given logStrike k=log(K/F) (scalar/vector) and maturity T (scalar)
# **kwargs for deployment in Implied Vol functions
# Forward measure (riskFreeRate=0 in charFunc) is assumed, so price is undiscounted
# TO-DO: non-zero riskFreeRate case? just multiply discount factor to formula?

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
    # Does NOT support useGlobal, curryCharFunc
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
    # Make this very efficient! (0.0081s per maturity * 28 maturities = 0.227s)
    if useGlobal:
        global cmFFT_init, cmFFT_du, cmFFT_u, cmFFT_dk, cmFFT_k, cmFFT_b, cmFFT_w, cmFFT_ntm, cmFFT_kntm, \
            cmFFT_Imult, cmFFT_cpImult, cmFFT_otmImult, cmFFT_cpCFarg, cmFFT_cpCFmult, cmFFT_charFunc, cmFFT_charFuncLog

        if not cmFFT_init:
            # Initialize global params (only ONCE!)
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
            cmFFT_kntm = cmFFT_k[cmFFT_ntm]
            #### pre-calculations ####
            cmFFT_Imult = cmFFT_w * np.exp(1j*cmFFT_b*cmFFT_u) * cmFFT_du/3
            cmFFT_cpImult = np.exp(-alpha*cmFFT_k)/np.pi
            with np.errstate(divide='ignore'):
                cmFFT_otmImult = 1/(np.pi*np.sinh(alpha*cmFFT_k))
            cmFFT_cpCFarg = cmFFT_u-(alpha+1)*1j
            cmFFT_cpCFmult = 1/(alpha**2+alpha-cmFFT_u**2+1j*(2*alpha+1)*cmFFT_u)
            cmFFT_init = True

        if charFunc != cmFFT_charFuncLog:
            # Update charFunc (for every NEW charFunc)
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
        kntm = cmFFT_kntm
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
        kntm = k[ntm]
        Imult = w * np.exp(1j*b*u) * du/3
        cpImult = np.exp(-alpha*k)/np.pi
        with np.errstate(divide='ignore'):
            otmImult = 1/(np.pi*np.sinh(alpha*k))

    if optionType in ["call", "put"]:
        if useGlobal:
            def modCharFunc(u, maturity): # Pre-calculated cpCFarg/cpCFmult
                return charFunc(cpCFarg, maturity) * cpCFmult
        else:
            def modCharFunc(u, maturity):
                return charFunc(u-(alpha+1)*1j, maturity) / (alpha**2+alpha-u**2+1j*(2*alpha+1)*u)
        # I = w * np.exp(1j*b*u) * modCharFunc(u, maturity) * du/3 # 0.010s (make this fast!)
        I = Imult * modCharFunc(u, maturity) # 0.0067s
        # Ifft = cpImult * np.real(fft(I)) # 0.0009s
        Ifft = cpImult * np.real(np.fft.fft(I)) # 0.0008s
        # spline = interp1d(kntm, Ifft[ntm], kind=interp) # 0.0008s
        spline = InterpolatedUnivariateSpline(kntm, Ifft[ntm], k=(3 if interp=="cubic" else 1)) # 0.0006s
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
        # Ifft = otmImult * np.real(fft(I))
        Ifft = otmImult * np.real(np.fft.fft(I))
        Ifft[N//2] = CarrMadanFormula(charFunc, 0, maturity, "OTM", alpha, **kwargs) # k = 0
        # spline = interp1d(kntm, Ifft[ntm], kind=interp)
        spline = InterpolatedUnivariateSpline(kntm, Ifft[ntm], k=(3 if interp=="cubic" else 1))
        price = spline(logStrike)
        return price

def COSFormula(charFunc, logStrike, maturity, optionType="call", N=4000, a=-5, b=5, useGlobal=False, curryCharFunc=False, **kwargs):
    # COS formula for call/put
    def cosInt0(k,c,d):
        return (np.cos(k*np.pi*(d-a)/(b-a))*np.exp(d)-np.cos(k*np.pi*(c-a)/(b-a))*np.exp(c)+(k*np.pi/(b-a))*(np.sin(k*np.pi*(d-a)/(b-a))*np.exp(d)-np.sin(k*np.pi*(c-a)/(b-a))*np.exp(c)))/(1+(k*np.pi/(b-a))**2)
    def cosInt1(k,c,d):
        with np.errstate(divide='ignore'):
            return np.nan_to_num(((b-a)/(k*np.pi)))*(np.sin(k*np.pi*(d-a)/(b-a))-np.sin(k*np.pi*(c-a)/(b-a)))*(k!=0)+(d-c)*(k==0)
    def cpCosInt(k):
        if optionType == "call":
            return 2/(b-a)*(cosInt0(k,0,b)-cosInt1(k,0,b))
        elif optionType == "put":
            return 2/(b-a)*(-cosInt0(k,a,0)+cosInt1(k,a,0))

    if useGlobal:
        global cosFmla_init, cosFmla_n, cosFmla_expArg, cosFmla_cfArg, cosFmla_cosInt, cosFmla_dict, cosFmla_charFunc, cosFmla_charFuncLog

        if not cosFmla_init:
            cosFmla_n = np.arange(N)
            cosFmla_expArg = 1j*np.pi*cosFmla_n
            cosFmla_cfArg = np.pi*cosFmla_n/(b-a)
            cosFmla_cosInt = cpCosInt(cosFmla_n)
            cosFmla_init = True

        if charFunc != cosFmla_charFuncLog:
            # Update charFunc (for every NEW charFunc)
            cosFmla_charFuncLog = charFunc
            if curryCharFunc: cosFmla_charFunc = charFunc(cosFmla_cfArg)
            else: cosFmla_charFunc = charFunc

        n = cosFmla_n
        expArg = cosFmla_expArg
        cfVec = cosFmla_charFunc(cosFmla_cfArg, maturity)
        cosInt = cosFmla_cosInt

        if maturity in cosFmla_dict:
            # x = cosFmla_dict[maturity]['x']
            # K = cosFmla_dict[maturity]['K']
            # ftMtrx = cosFmla_dict[maturity]['ftMtrx']
            # ftMtrxK = cosFmla_dict[maturity]['ftMtrxK']
            ftMtrxKI = cosFmla_dict[maturity]['ftMtrxKI']
        else:
            x = -logStrike
            K = np.exp(logStrike)
            ftMtrx = np.exp(np.multiply.outer((x-a)/(b-a), expArg))
            ftMtrx[:,0] *= 0.5
            ftMtrxK = (ftMtrx.T*K).T
            ftMtrxKI = ftMtrxK*cosInt
            cosFmla_dict[maturity] = {
                'x': x,
                'K': K,
                'ftMtrx': ftMtrx,
                'ftMtrxK': ftMtrxK,
                'ftMtrxKI': ftMtrxKI,
            }

    else:
        n = np.arange(N)
        expArg = 1j*np.pi*n
        cfVec = charFunc(np.pi*n/(b-a), maturity)
        cosInt = cpCosInt(n)

        x = -logStrike
        K = np.exp(logStrike)
        ftMtrx = np.exp(np.multiply.outer((x-a)/(b-a), expArg))
        ftMtrx[:,0] *= 0.5
        ftMtrxK = (ftMtrx.T*K).T
        ftMtrxKI = ftMtrxK*cosInt

    # price = np.real(ftMtrxK*cfVec).dot(cosInt) # 0.0015s
    price = np.real(np.sum(ftMtrxKI*cfVec,axis=1)) # 0.0007s
    return price

def COSFormulaAdpt(charFunc, logStrike, maturity, optionType="call", curryCharFunc=False, **kwargs):
    # COS formula for call/put with adaptive (a,b,N)
    # Use global params by default, i.e. useGlobal=True
    global cosFmla_dict, cosFmla_charFunc, cosFmla_charFuncLog, cosFmla_adptParams

    if maturity in cosFmla_dict: # Use cache
        ftMtrxKI = cosFmla_dict[maturity]['ftMtrxKI']
        cfArg = cosFmla_dict[maturity]['cfArg']

    else: # Initialize
        for T in cosFmla_adptParams:
            if maturity < T: dictT = cosFmla_adptParams[T]
        a,b,N = dictT['a'],dictT['b'],dictT['N']

        def cosInt0(k,c,d):
            return (np.cos(k*np.pi*(d-a)/(b-a))*np.exp(d)-np.cos(k*np.pi*(c-a)/(b-a))*np.exp(c)+(k*np.pi/(b-a))*(np.sin(k*np.pi*(d-a)/(b-a))*np.exp(d)-np.sin(k*np.pi*(c-a)/(b-a))*np.exp(c)))/(1+(k*np.pi/(b-a))**2)
        def cosInt1(k,c,d):
            with np.errstate(divide='ignore'):
                return np.nan_to_num(((b-a)/(k*np.pi)))*(np.sin(k*np.pi*(d-a)/(b-a))-np.sin(k*np.pi*(c-a)/(b-a)))*(k!=0)+(d-c)*(k==0)
        def cpCosInt(k):
            if optionType == "call":
                return 2/(b-a)*(cosInt0(k,0,b)-cosInt1(k,0,b))
            elif optionType == "put":
                return 2/(b-a)*(-cosInt0(k,a,0)+cosInt1(k,a,0))

        n = np.arange(N)
        expArg = 1j*np.pi*n
        cfArg = np.pi*n/(b-a)
        cosInt = cpCosInt(n)

        x = -logStrike
        K = np.exp(logStrike)
        ftMtrx = np.exp(np.multiply.outer((x-a)/(b-a), expArg))
        ftMtrx[:,0] *= 0.5
        ftMtrxK = (ftMtrx.T*K).T
        ftMtrxKI = ftMtrxK*cosInt
        cosFmla_dict[maturity] = {
            # 'x': x,
            # 'K': K,
            # 'ftMtrx': ftMtrx,
            # 'ftMtrxK': ftMtrxK,
            'ftMtrxKI': ftMtrxKI,
            'cfArg': cfArg,
        }

    if charFunc != cosFmla_charFuncLog:
        # Update charFunc (for every NEW charFunc)
        cosFmla_charFuncLog = charFunc
        if curryCharFunc: cosFmla_charFunc = charFunc(cfArg)
        else: cosFmla_charFunc = charFunc

    cfVec = cosFmla_charFunc(cfArg, maturity)

    price = np.real(np.sum(ftMtrxKI*cfVec,axis=1))
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
    elif formulaType == "COS":
        formula = COSFormula
    elif formulaType == "COSAdpt":
        formula = COSFormulaAdpt
    def impVolFunc(logStrike, maturity):
        return BlackScholesImpliedVol(1, np.exp(logStrike), maturity, riskFreeRate, formula(charFunc, logStrike, maturity, optionType, **kwargs), optionType, inversionMethod)
    return impVolFunc

def CharFuncImpliedVolAtm(charFunc, optionType="OTM", riskFreeRate=0, FFT=False, formulaType="CarrMadan", inversionMethod="Bisection", **kwargs):
    # Implied skew for call/put/OTM priced with charFunc, based on Lewis formula
    impVolFunc = CharFuncImpliedVol(charFunc, optionType, riskFreeRate, FFT, formulaType, inversionMethod, **kwargs)
    def atmVolFunc(maturity):
        # Works for scalar maturity only
        atmVol = impVolFunc(np.array([0]), maturity)
        return atmVol
    return atmVolFunc

def LewisCharFuncImpliedVol(charFunc, optionType="OTM", riskFreeRate=0, **kwargs):
    # Implied volatility for call/put/OTM priced with charFunc, based on Lewis formula
    # Much SLOWER than Bisection
    def impVolFunc(logStrike, maturity):
        def objective(vol):
            integrand = lambda u: np.real(np.exp(-1j*u*logStrike) * (charFunc(u-1j/2, maturity) - BlackScholesCharFunc(vol, riskFreeRate)(u-1j/2, maturity)) / (u**2+.25))
            return quad(integrand, 0, np.inf)[0]
        impVol = fsolve(objective, 0.4)[0]
        return impVol
    return impVolFunc

def LewisCharFuncImpliedSkewAtm(charFunc, optionType="OTM", riskFreeRate=0, FFT=False, formulaType="CarrMadan", inversionMethod="Bisection", **kwargs):
    # Implied skew for call/put/OTM priced with charFunc, based on Lewis formula
    impVolFunc = CharFuncImpliedVol(charFunc, optionType, riskFreeRate, FFT, formulaType, inversionMethod, **kwargs)
    def atmSkewFunc(maturity):
        # Works for scalar maturity only
        atmVol = impVolFunc(np.array([0]), maturity)
        integrand = lambda u: np.imag(u * charFunc(u-1j/2, maturity) / (u**2+.25))
        atmSkew = -np.exp(atmVol**2*maturity/8) * np.sqrt(2/(np.pi*maturity)) * quad(integrand, 0, np.inf)[0]
        return atmSkew
    return atmSkewFunc

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
    elif formulaType == "COS":
        formula = COSFormula
    elif formulaType == "COSAdpt":
        formula = COSFormulaAdpt

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
    # Include useGlobal=True for curryCharFunc=True
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
        impVol = np.concatenate([impVolFunc(logStrike[maturity==T], T) for T in np.unique(maturity)], axis=None) # most costly: BS inversion for each T
        loss = np.sum(w*((impVol-bidVol)**2+(askVol-impVol)**2))
        print(f"params: {params}")
        print(f"loss: {loss}")
        return loss

    opt = minimize(objective, x0=params0, bounds=bounds)
    print("Optimization output:", opt, sep="\n")
    return opt.x

def CalibrateModelToImpliedVolFast(logStrike, maturity, optionImpVol, model, params0, paramsLabel,
    bounds=None, w=None, optionType="call", formulaType="CarrMadan", curryCharFunc=False, optMethod="Gradient", kwargsCF={}, **kwargs):
    # Calibrate model params to implied vols (pricing measure)
    # Include useGlobal=True for curryCharFunc=True
    if w is None: w = 1
    maturity = np.array(maturity)
    matUniq = np.unique(maturity)
    strike = np.exp(logStrike)
    bidVol = optionImpVol["Bid"].to_numpy()
    askVol = optionImpVol["Ask"].to_numpy()
    logStrikeDict = {T: logStrike[maturity==T] for T in matUniq}

    if formulaType == "Lewis":
        formula = LewisFormulaFFT
    elif formulaType == "CarrMadan":
        formula = CarrMadanFormulaFFT
    elif formulaType == "COS":
        formula = COSFormula
    elif formulaType == "COSAdpt":
        formula = COSFormulaAdpt
    riskFreeRate = kwargs["riskFreeRate"] if "riskFreeRate" in kwargs else 0
    inversionMethod = kwargs["inversionMethod"] if "inversionMethod" in kwargs else "Bisection"

    def objective(params):
        params = {paramsLabel[i]: params[i] for i in range(len(params))}
        # charFunc = model(**params)
        # impVolFunc = CharFuncImpliedVol(charFunc, optionType=optionType, FFT=True, formulaType=formulaType, **kwargs)
        charFunc = model(**params, **kwargsCF, curry=curryCharFunc)
        price = np.concatenate([formula(charFunc, logStrikeDict[T], T, optionType, curryCharFunc=curryCharFunc, **kwargs) for T in matUniq], axis=None) # most costly
        impVol = BlackScholesImpliedVol(1, strike, maturity, riskFreeRate, price, optionType, inversionMethod) # BS inversion for all T
        loss = np.sum(w*((impVol-bidVol)**2+(askVol-impVol)**2))
        # loss = np.sum(w*(impVol-(bidVol+askVol)/2)**2)
        print(f"params: {params}")
        print(f"loss: {loss}")
        return loss

    if optMethod == "Gradient":
        opt = minimize(objective, x0=params0, bounds=bounds)

    elif optMethod == "Annealing":
        opt0 = dual_annealing(objective, bounds=bounds, maxfun=1000)
        opt = minimize(objective, x0=opt0.x, bounds=bounds)

    elif optMethod == "SHGO":
        opt0 = shgo(objective, bounds=bounds)
        opt = minimize(objective, x0=opt0.x, bounds=bounds)

    elif optMethod == "Evolution": # Best performance so far!
        opt0 = differential_evolution(objective, bounds=bounds)
        opt = minimize(objective, x0=opt0.x, bounds=bounds)

    elif optMethod == "Basin":
        class basinBnd:
            def __init__(self):
                self.xmin = np.array([b[0] for b in bounds])
                self.xmax = np.array([b[1] for b in bounds])
            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                tmax = bool(np.all(x <= self.xmax))
                tmin = bool(np.all(x >= self.xmin))
                return tmax and tmin
        opt0 = basinhopping(objective, x0=params0, accept_test=basinBnd())
        opt = minimize(objective, x0=opt0.x, bounds=bounds)

    print("Optimization output:", opt, sep="\n")
    return opt.x

#### Plotting/Param Function ###################################################

def PlotImpliedVol(df, figname=None, ncol=6):
    # Plot bid-ask implied volatilities based on df
    # Columns: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    if not figname:
        figname = "impliedvol.png"
    Texp = df["Texp"].unique()
    Nexp = len(Texp)
    nrow = int(np.ceil(Nexp/ncol))
    ncol = min(len(Texp),6)
    fig, ax = plt.subplots(nrow,ncol,figsize=(2.5*ncol,2*nrow))

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
            ax_idx.scatter(k,bid,c='r',s=2,marker="^")
            ax_idx.scatter(k,ask,c='b',s=2,marker="v")
            if "Fit" in dfT:
                fit = dfT["Fit"]
                i = (fit>1e-2)
                # ax_idx.scatter(k[i],fit[i],c='k',s=2)
                ax_idx.plot(k[i],fit[i],'k',linewidth=1)
            ax_idx.set_title(rf"$T={np.round(T,3)}$")
            ax_idx.set_xlabel("log-strike")
            ax_idx.set_ylabel("implied vol")
        else:
            ax_idx.axis("off")

    fig.tight_layout()
    plt.savefig(figname)
    plt.close()

def PlotImpliedVolSurface(df, figname=None, model=None):
    # Plot implied vol surface based on df
    # Columns: "Log-strike","Texp","IV"
    if not figname: figname = "IVS.png"

    df = df.dropna()

    logStrike = df["Log-strike"]
    maturity  = df["Texp"]
    impVol    = df["IV"]*100

    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection="3d")
    surf = ax.plot_trisurf(logStrike,maturity,impVol,cmap='summer')
    ax.set_xlabel("log-strike")
    ax.set_ylabel("maturity")
    ax.set_zlabel("implied vol")
    if model: ax.set_title(model)

    plt.savefig(figname)
    plt.close()

def CalcAtmVolAndSkew(df):
    # Calculate implied vols & skews based on df, with cubic interpolation
    # Columns: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    Texp = df["Texp"].unique()
    atmVol = list()
    atmSkew = list()

    for T in Texp:
        dfT = df[df["Texp"]==T]
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        mid = ((dfT["Bid"]+dfT["Ask"])/2).to_numpy()
        ntm = (k>-0.05)&(k<0.05)
        spline = InterpolatedUnivariateSpline(k[ntm], mid[ntm])
        atmVol.append(spline(0).item())
        atmSkew.append(spline.derivatives(0)[1])

    atmVol = np.array(atmVol)
    atmSkew = np.array(atmSkew)
    return {"Texp": Texp, "atmVol": atmVol, "atmSkew": atmSkew}

def CalcLocalVolSurface(df):
    # Calculate local vols by Dupire formula based on IVS df, with cubic interpolation
    # Columns: "Log-strike","Texp","IV"
    # Ref: Gatheral, The Volatility Surface, a Practitioner's Guide
    # NOT stable, values blow up!
    k = df["Log-strike"].to_numpy()
    T = df["Texp"].to_numpy()
    sigI = df["IV"].to_numpy()
    w = sigI**2*T

    k0 = np.unique(k)
    T0 = np.unique(T)

    fw = interp2d(k,T,w,kind='cubic')
    dk1 = fw(k0,T0,dx=1).reshape(-1)
    dk2 = fw(k0,T0,dx=2).reshape(-1)
    dT1 = fw(k0,T0,dy=1).reshape(-1)

    sigL = np.sqrt(dT1/((1-0.5*k/w*dk1)**2-0.25*(0.25+1/w)*dk1**2+0.5*dk2)) # Dupire PDE
    sigL = pd.DataFrame(np.array([k,T,sigL]).T,columns=["Log-strike","Texp","LV"])
    return sigL

def PlotLocalVolSurface(df, figname=None, model=None):
    # Plot local vol surface based on df
    # Columns: "Log-strike","Texp","LV"
    if not figname: figname = "LVS.png"

    df = df.dropna()

    logStrike = df["Log-strike"]
    maturity  = df["Texp"]
    impVol    = df["LV"]*100

    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection="3d")
    surf = ax.plot_trisurf(logStrike,maturity,impVol,cmap='summer')
    ax.set_xlabel("log-strike")
    ax.set_ylabel("maturity")
    ax.set_zlabel("local vol")
    if model: ax.set_title(model)

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

def CalcFwdVarCurve(curveVS, eps=0):
    # Calculate forward variance curve based on VS curve
    # price = integrated fwd var (vswap price w/o time avg)
    Texp = curveVS["Texp"]
    diffTexp = curveVS["Texp"].diff()
    price = curveVS[["bid","mid","ask"]].multiply(Texp,axis=0)
    if eps > 0: # smooth VS prices
        def objective(Texp, price):
            def loss(err):
                adjPrice = price+2*np.sqrt(price*Texp)*err
                curve = np.concatenate([[adjPrice[0]/Texp[0]],np.diff(adjPrice)/np.diff(Texp)])
                curveDiff = np.diff(curve)/np.diff(Texp)
                return sum((adjPrice-price)**2)+sum(curveDiff**2)
            return loss
        Nexp = len(Texp)
        for vs in ["bid","mid","ask"]:
            opt = minimize(objective(Texp,price[vs]), x0=np.repeat(0,Nexp), bounds=[(-eps,eps)]*Nexp)
            price[vs] += 2*np.sqrt(price[vs]*Texp)*opt.x
            # print(opt.x)
    curve = price.diff()
    curve = curve.div(diffTexp,axis=0)
    curve.iloc[0] = price.iloc[0]/Texp.iloc[0]
    curve["Texp"] = Texp
    curve = curve[["Texp","bid","mid","ask"]]
    return curve

def FwdVarCurveFunc(maturity, fwdVar, fitType="const"):
    # Forward variance curve function
    Texp = maturity
    Nexp = len(Texp)
    curveFunc = None
    if fitType == "const":
        curveFunc = interp1d(Texp,fwdVar,kind="next",fill_value="extrapolate")
    elif fitType == "spline":
        curveFunc = InterpolatedUnivariateSpline(Texp,fwdVar,ext=3)
    return curveFunc

def SmoothFwdVarCurveFunc(maturity, vsPrice, eps=0):
    # Smoothed forward variance curve function
    # Ref: Filipovic, Willems, Exact Smooth Term-Structure Estimation
    Texp = maturity
    Nexp = len(Texp)
    price = vsPrice*Texp
    def phi(tau, x):
        m = np.minimum(x,tau)
        return 1-m**3/6+x*tau*(2+m)/2
    def phiDeriv(tau, x):
        m = np.minimum(x,tau)
        return tau-m**2/2+tau*m
    A = np.vstack([phi(tau,Texp) for tau in Texp])
    Ainv = np.linalg.inv(A)
    def objective(Texp, price):
        def loss(err):
            adjPrice = price+2*np.sqrt(price*Texp)*err
            return sum(adjPrice.T.dot(Ainv)*adjPrice)
        return loss
    opt = minimize(objective(Texp,price), x0=np.repeat(0,Nexp), bounds=[(-eps,eps)]*Nexp)
    price += 2*np.sqrt(price*Texp)*opt.x
    Z = Ainv.dot(price)
    # print(opt.x)
    def curveFunc(x):
        return sum(Z*phiDeriv(Texp,x))
    return np.vectorize(curveFunc)
