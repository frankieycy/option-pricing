import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import ndtr
from scipy.fftpack import fft
from scipy.integrate import quad
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, RectBivariateSpline
from numba import njit, float64, vectorize
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

#### Black-Scholes #############################################################

INVROOT2PI = 0.3989422804014327

@njit(float64(float64), fastmath=True, cache=True)
def _ndtr_jit(x):
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    g = 0.2316419

    k = 1.0 / (1.0 + g * np.abs(x))
    k2 = k * k
    k3 = k2 * k
    k4 = k3 * k
    k5 = k4 * k

    if x >= 0.0:
        c = (a1 * k + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5)
        phi = 1.0 - c * np.exp(-x*x/2.0) * INVROOT2PI
    else:
        phi = 1.0 - _ndtr_jit(-x)

    return phi

@vectorize([float64(float64)], fastmath=True, cache=True)
def ndtr_jit(x):
    return _ndtr_jit(x)

def BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black Scholes formula for call/put
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    discountFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol

    if isinstance(optionType, str): # Uniform optionType
        return spotPrice * ndtr(d1) - discountFactor * strike * ndtr(d2) if optionType == "call" else \
            discountFactor * strike * ndtr(-d2) - spotPrice * ndtr(-d1)
    else: # Vector optionType
        # strike & optionType must be vectors
        call = (optionType == "call")
        price = np.zeros(len(strike))
        price[call] = (spotPrice[call] if isinstance(spotPrice, np.ndarray) else spotPrice) * ndtr(d1[call]) - (discountFactor[call] if isinstance(discountFactor, np.ndarray) else discountFactor) * strike[call] * ndtr(d2[call]) # call
        price[~call] = (discountFactor[~call] if isinstance(discountFactor, np.ndarray) else discountFactor) * strike[~call] * ndtr(-d2[~call]) - (spotPrice[~call] if isinstance(spotPrice, np.ndarray) else spotPrice) * ndtr(-d1[~call]) # put
        return price

    # price = np.where(optionType == "call",
    #     spotPrice * ndtr(d1) - discountFactor * strike * ndtr(d2),
    #     discountFactor * strike * ndtr(-d2) - spotPrice * ndtr(-d1))
    # return price

@njit
def BlackScholesFormula_jit(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black Scholes formula for call/put
    # Fast implementation for use in BlackScholesImpliedVol_jitBisect
    # spotPrice & riskFreeRate are scalars; optionType is 'call' or 'put'; the rest are vectors
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    discountFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol

    if optionType == 'call':
        return spotPrice * ndtr_jit(d1) - discountFactor * strike * ndtr_jit(d2)
    else:
        return discountFactor * strike * ndtr_jit(-d2) - spotPrice * ndtr_jit(-d1)

def BlackScholesDelta(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black Scholes delta for call/put (first deriv wrt spot)
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    return np.where(optionType == "call", ndtr(d1), -ndtr(-d1))

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

@njit
def BlackScholesImpliedVol_jitBisect(spotPrice, strike, maturity, riskFreeRate, priceMkt, optionType):
    nStrikes = len(strike)
    impVol = np.zeros(nStrikes)
    impVol0 = np.repeat(1e-10, nStrikes)
    impVol1 = np.repeat(10., nStrikes)
    price0 = BlackScholesFormula_jit(spotPrice, strike, maturity, riskFreeRate, impVol0, optionType)
    price1 = BlackScholesFormula_jit(spotPrice, strike, maturity, riskFreeRate, impVol1, optionType)
    while np.mean(impVol1-impVol0) > 1e-10:
        impVol = (impVol0+impVol1)/2
        price = BlackScholesFormula_jit(spotPrice, strike, maturity, riskFreeRate, impVol, optionType)
        price0 += (price<priceMkt)*(price-price0)
        impVol0 += (price<priceMkt)*(impVol-impVol0)
        price1 += (price>=priceMkt)*(price-price1)
        impVol1 += (price>=priceMkt)*(impVol-impVol1)
    return impVol

def BlackScholesImpliedVol(spotPrice, strike, maturity, riskFreeRate, priceMkt, optionType="OTM", method="Bisection"):
    # Black Scholes implied volatility for call/put/OTM
    # Generally, strike & priceMkt are input vectors (called from e.g. CharFuncImpliedVol)
    # Within function, optionType & maturity are cast to vectors
    # Make this very efficient!
    forwardPrice = spotPrice*np.exp(riskFreeRate*maturity)
    if not isinstance(strike, np.ndarray):
        strike = np.array([strike])
    nStrikes = len(strike)
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

    elif method == "Bisection_jit":
        if np.all(optionType == 'call'):
            return BlackScholesImpliedVol_jitBisect(spotPrice, strike, maturity, riskFreeRate, priceMkt, 'call')
        elif np.all(optionType == 'put'):
            return BlackScholesImpliedVol_jitBisect(spotPrice, strike, maturity, riskFreeRate, priceMkt, 'put')
        else:
            return BlackScholesImpliedVol(spotPrice, strike, maturity, riskFreeRate, priceMkt, optionType, method="Bisection")

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
                return np.exp(-k/2)*ndtr(-k/v+v/2) - np.exp(k/2)*ndtr(-k/v-v/2)
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
            bsIv_rationalCall = lambda x,v: ndtr(x/v+v/2)-np.exp(-x)*ndtr(x/v-v/2)
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

#### Pricing Formula ###########################################################

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

@njit(fastmath=True, cache=True)
def COSFormula_prxMult(ftMtrxKI, cfVec):
    # return np.real(np.sum(ftMtrxKI*cfVec,axis=1))
    return np.real(ftMtrxKI.dot(cfVec))

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
        cfVec = cosFmla_charFunc(cosFmla_cfArg, maturity) # 0.0005s (depending on form of CF)
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
    # price = np.real(np.sum(ftMtrxKI*cfVec,axis=1)) # 0.0007s
    price = COSFormula_prxMult(ftMtrxKI,cfVec) # 0.0002s
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

#### Characteristic Function ###################################################

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
                # return HestonCharFuncFixedU_jit(u, maturity, meanRevRate, volOfVol, meanVar, currentVar, iur, d, rm, g)
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

#### Event Model

def GaussianEventJumpCharFunc(spotCharFunc, eventTime, jumpUpProb, jumpUpMean, jumpUpStd, jumpDnMean, jumpDnStd):
    # Characteristic function for Gaussian event jump model
    def charFunc(u, maturity):
        return spotCharFunc(u, maturity) * ((maturity < eventTime) + (maturity >= eventTime) * (jumpUpProb*np.exp(1j*u*jumpUpMean-u**2*jumpUpStd**2/2)+(1-jumpUpProb)*np.exp(1j*u*jumpDnMean-u**2*jumpDnStd**2/2))/(jumpUpProb*np.exp(jumpUpMean+jumpUpStd**2/2)+(1-jumpUpProb)*np.exp(jumpDnMean+jumpDnStd**2/2))**(1j*u))
    return charFunc

def PointEventJumpCharFunc(spotCharFunc, eventTime, jumpProb, jump):
    # Characteristic function for point event jump model
    def charFunc(u, maturity):
        return spotCharFunc(u, maturity) * ((maturity < eventTime) + (maturity >= eventTime) * (jumpProb*np.exp(1j*u*jump)+(1-jumpProb)*np.exp(-1j*u*jump))/(jumpProb*np.exp(jump)+(1-jumpProb)*np.exp(-jump))**(1j*u))
    return charFunc

#### Event Bump Expansion ######################################################

def PointEventJumpOptionPrice(spotPrice, strike, maturity, riskFreeRate, impliedVol, eventTime, optionType, jumpProb, jump):
    meanExpJump = jumpProb * np.exp(jump) + (1-jumpProb) * np.exp(-jump)
    return (maturity < eventTime) * BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType) + (maturity >= eventTime) * \
        (jumpProb * BlackScholesFormula(spotPrice*np.exp(jump)/meanExpJump, strike, maturity, riskFreeRate, impliedVol, optionType) + (1-jumpProb) * BlackScholesFormula(spotPrice*np.exp(-jump)/meanExpJump, strike, maturity, riskFreeRate, impliedVol, optionType))

def PointEventJumpOptionPriceOTM(spotPrice, strike, maturity, riskFreeRate, impliedVol, eventTime, jumpProb, jump, returnOptionType=False):
    forwardPrice = spotPrice*np.exp(riskFreeRate*maturity)
    optionType = np.where(strike > forwardPrice, "call", "put")
    if returnOptionType:
        return PointEventJumpOptionPrice(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType, eventTime, jumpProb, jump), optionType
    return PointEventJumpOptionPrice(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType, eventTime, jumpProb, jump)

def PointEventJumpMoment(n, jumpProb, jump, **kwargs):
    meanExpJump = jumpProb * np.exp(jump) + (1-jumpProb) * np.exp(-jump)
    return jumpProb * (1-jumpProb) * ((1-jumpProb)**(n-1)-(-jumpProb)**(n-1)) * (2*np.sinh(jump)/meanExpJump)**n

def EventVarianceBump(logStrike, totalVar, jumpMomentFunc, n, **kwargs):
    bump = 0
    d1 = -logStrike/np.sqrt(totalVar)+np.sqrt(totalVar)/2
    d1n = d1/np.sqrt(totalVar)
    if n >= 0:
        bump += jumpMomentFunc(2,**kwargs)/np.math.factorial(2)
    if n >= 1:
        bump += jumpMomentFunc(3,**kwargs)/np.math.factorial(3)*(-d1n-1)
    if n >= 2:
        bump += jumpMomentFunc(4,**kwargs)/np.math.factorial(4)*(d1n**2+3*d1n+2-1/totalVar)
    if n >= 3:
        bump += jumpMomentFunc(5,**kwargs)/np.math.factorial(5)*(-d1n**3-6*d1n**2-11*d1n+3*d1n/totalVar-6+6/totalVar)
    if n >= 4:
        bump += jumpMomentFunc(6,**kwargs)/np.math.factorial(6)*(d1n**4+10*d1n**3+35*d1n**2-6*d1n**2/totalVar+50*d1n-30*d1n/totalVar+24-35/totalVar+3/totalVar**2)
    return 2 * bump

def EventPriceBump(logStrike, totalVar, jumpMomentFunc, n, **kwargs):
    d1 = -logStrike/np.sqrt(totalVar)+np.sqrt(totalVar)/2
    varVega = norm.pdf(d1)/(2*np.sqrt(totalVar))
    varBump = EventVarianceBump(logStrike, totalVar, jumpMomentFunc, n, **kwargs)
    return varVega * varBump

def EventVarianceBump2ndOrder(logStrike, totalVar, jumpMomentFunc, n, **kwargs):
    d1 = -logStrike/np.sqrt(totalVar)+np.sqrt(totalVar)/2
    d2 = d1-np.sqrt(totalVar)
    varBump = EventVarianceBump(logStrike, totalVar, jumpMomentFunc, n, **kwargs)
    u = totalVar/(d1*d2-1)
    return 2 * u * (np.sqrt(1+varBump/u)-1)
