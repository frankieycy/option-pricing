import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import bisect, newton, minimize, minimize_scalar
from pricer import BlackScholesFormula

amPrx_Stree = dict()
amPrx_Otree = dict()

def PriceAmericanOption(spotPrice, forwardPrice, strike, maturity, riskFreeRate, impliedVol, optionType, timeSteps=2000, useGlobal=False):
    # Price American option via Cox binomial tree (d = 1/u)
    # Assume continuous dividend, reflected in forward price
    # CAUTION: cached S tree is only approximate as params are rounded!
    # TO-DO: proportional/discrete dividend (difficult!)
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

    if useGlobal: # Tree caching (accuracy is compromised!)
        global amPrx_Stree, amPrx_Otree
        Stag = 'S=%.4f,F=%.4f,T=%.4f,r=%.4f,sig=%.4f,n=%d' % (S,F,T,r,sig,n)
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

    else:
        Stree = np.zeros((n+1,n+1))
        Otree = np.zeros((n+1,n+1))
        for i in range(n+1):
            Stree[i,:i+1] = S*u**np.arange(-i,i+1,2)

    Otree[-1] = payoff(Stree[-1])
    for i in range(n-1,-1,-1):
        Otree[i,:i+1] = np.maximum(D*(p*Otree[i+1,1:i+2]+q*Otree[i+1,0:i+1]), payoff(Stree[i,:i+1]))

    return Otree[0,0]

PriceAmericanOption_vec = np.vectorize(PriceAmericanOption)

def AmericanOptionImpliedVol(spotPrice, forwardPrice, strike, maturity, riskFreeRate, priceMkt, optionType, timeSteps=1000, method="Bisection", **kwargs):
    # Implied flat volatility under Cox binomial tree
    def objective(impVol):
        return PriceAmericanOption(spotPrice, forwardPrice, strike, maturity, riskFreeRate, impVol, optionType, timeSteps, **kwargs) - priceMkt
    impVol = 0
    try:
        if method == "Bisection":
            impVol = bisect(objective, 1e-8, 1)
        elif method == "Newton":
            impVol = newton(objective, 0.4)
    except Exception: pass
    return impVol

AmericanOptionImpliedVol_vec = np.vectorize(AmericanOptionImpliedVol)

def AmericanOptionImpliedForwardAndRate(spotPrice, strike, maturity, priceMktCall, priceMktPut, timeSteps=1000, method="Bisection", sigPenalty=0, iterLog=False, **kwargs):
    # Implied forward & riskfree rate from ATM put/call prices
    # Iterate on fwd & rate until put/call fwd & implied vols converge ATM
    # Cannot converge on sensible solution! output ~ (S,0) unchanged
    def loss(params):
        F, r = params
        D = np.exp(-r*maturity)
        q = r-np.log(F/spotPrice)/maturity
        sigC = AmericanOptionImpliedVol(spotPrice, F, strike, maturity, r, priceMktCall, 'call', timeSteps, method, **kwargs)
        sigP = AmericanOptionImpliedVol(spotPrice, F, strike, maturity, r, priceMktPut, 'put', timeSteps, method, **kwargs)
        Cbs = BlackScholesFormula(F, strike, maturity, 0, sigC, 'call')
        Pbs = BlackScholesFormula(F, strike, maturity, 0, sigP, 'put')
        Fi = (Cbs-Pbs) + strike
        loss = (Fi-F)**2 + sigPenalty*(sigP-sigC)**2
        if iterLog:
            print(f"params: {params} loss: {loss}")
            print(f"  r={r} q={q} F={F} Fi={Fi} sigC={sigC} sigP={sigP}")
        return loss

    params = (spotPrice, 0)
    bounds = ((0.8*spotPrice,1.2*spotPrice), (-0.05,0.05))

    opt = minimize(loss, x0=params, bounds=bounds)
    return opt.x

def AmericanOptionImpliedDividendAndRate(spotPrice, strike, maturity, priceMktCall, priceMktPut, riskFreeRate=None, timeSteps=1000, method="Bisection", sigPenalty=0, iterLog=False, **kwargs):
    # Implied dividend (& riskfree rate) from ATM put/call prices
    # (1) Iterate on div & rate until put/call div & implied vols converge ATM - typically converge on r=q and loss decreases with q!
    # (2) Solve for unique div that gives zero loss fixing r (assume r is known!)
    if riskFreeRate is None:
        def loss(params):
            q, r = params
            D = np.exp(-r*maturity)
            F = spotPrice*np.exp((r-q)*maturity)
            sigC = AmericanOptionImpliedVol(spotPrice, F, strike, maturity, r, priceMktCall, 'call', timeSteps, method, **kwargs)
            sigP = AmericanOptionImpliedVol(spotPrice, F, strike, maturity, r, priceMktPut, 'put', timeSteps, method, **kwargs)
            Cbs = BlackScholesFormula(F, strike, maturity, 0, sigC, 'call')
            Pbs = BlackScholesFormula(F, strike, maturity, 0, sigP, 'put')
            Fi = (Cbs-Pbs) + strike
            qi = r-np.log(Fi/spotPrice)/maturity
            loss = 10000*(q-qi)**2 + sigPenalty*(sigP-sigC)**2
            if iterLog:
                print(f"params: {params} loss: {loss}")
                print(f"  r={r} q={q} qi={qi} F={F} Fi={Fi} sigC={sigC} sigP={sigP}")
            return loss

        params = (0, 0)
        bounds = ((-0.20,0.20), (-0.05,0.05))

        opt = minimize(loss, x0=params, bounds=bounds)
        return opt.x

    else:
        r = riskFreeRate
        def loss(q):
            D = np.exp(-r*maturity)
            F = spotPrice*np.exp((r-q)*maturity)
            sigC = AmericanOptionImpliedVol(spotPrice, F, strike, maturity, r, priceMktCall, 'call', timeSteps, method, **kwargs)
            sigP = AmericanOptionImpliedVol(spotPrice, F, strike, maturity, r, priceMktPut, 'put', timeSteps, method, **kwargs)
            Cbs = BlackScholesFormula(F, strike, maturity, 0, sigC, 'call')
            Pbs = BlackScholesFormula(F, strike, maturity, 0, sigP, 'put')
            Fi = (Cbs-Pbs) + strike
            qi = r-np.log(Fi/spotPrice)/maturity
            # loss = 100*(q-qi)
            loss = 10000*(q-qi)**2
            if iterLog:
                print(f"params: {q} loss: {loss}")
                print(f"  r={r} q={q} qi={qi} F={F} Fi={Fi} sigC={sigC} sigP={sigP}")
            return loss

        # q = bisect(loss, r-0.02, r+0.02)
        # return q

        bounds = (r-0.05, r+0.05)

        opt = minimize_scalar(loss, bounds=bounds, method="Bounded")
        return opt.x

def DeAmericanizedOptionsChainDataset(df, spotPrice, rfRateFunc=None, timeSteps=1000, iterLog=False, **kwargs):
    # De-Americanization of listed option prices into European pseudo-prices
    # Return standardized options chain dataset with columns: "Contract Name","Expiry","Texp","Put/Call","Strike","Bid","Ask"
    # Routine: (1) imply dividend/rate (2) back out implied vols (3) cast to European prices
    S = spotPrice
    deAmDf = list()
    Texp = df["Texp"].unique()

    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        dfTc = dfT[dfT['Put/Call']=='Call']
        dfTp = dfT[dfT['Put/Call']=='Put']
        Kc = dfTc['Strike']
        Kp = dfTp['Strike']
        K0 = Kc[Kc.isin(Kp)] # common strikes

        if len(K0) > 0:
            ntm = (K0-S).abs().argmin()
            K = K0.iloc[ntm] # NTM strike
            if abs(K-S) < 10:
                *_, Cb, Ca = dfTc[Kc==K].iloc[0] # call bid/ask
                *_, Pb, Pa = dfTp[Kp==K].iloc[0] # put bid/ask
                Cm = (Cb+Ca)/2
                Pm = (Pb+Pa)/2
                print(f"T={T} S={S} K={K} Cm={Cm} Pm={Pm}")

                if rfRateFunc is None: # imply div & rate
                    q,r = AmericanOptionImpliedDividendAndRate(S, K, T, Cm, Pm, None, timeSteps, iterLog=iterLog, **kwargs)
                else: # imply div only (more stable!)
                    r = rfRateFunc(T)
                    q = AmericanOptionImpliedDividendAndRate(S, K, T, Cm, Pm, r, timeSteps, iterLog=iterLog, **kwargs)
                # r = q = 0 # naive case!
                print(f"implied: q={q} r={r}")
                F = S*np.exp((r-q)*T)

                # Consider options with time value (s.t. sig is meaningful)
                idxc = (dfTc['Bid']>=1.01*np.maximum(S-Kc,0))
                idxp = (dfTp['Bid']>=1.01*np.maximum(Kp-S,0))
                dfT = pd.concat([dfTc[idxc],dfTp[idxp]])
                pc = dfT['Put/Call'].str.lower()
                K = dfT['Strike']
                bid = dfT['Bid']
                ask = dfT['Ask']
                D = np.exp(-r*T)
                sigB = AmericanOptionImpliedVol_vec(S, F, K, T, r, bid, pc, timeSteps, **kwargs)
                sigA = AmericanOptionImpliedVol_vec(S, F, K, T, r, ask, pc, timeSteps, **kwargs)
                bsB = D*BlackScholesFormula(F, K, T, 0, sigB, pc)
                bsA = D*BlackScholesFormula(F, K, T, 0, sigA, pc)
                dfT['Bid'] = bsB
                dfT['Ask'] = bsA
                dfT['Fwd'] = F
                dfT['PV'] = D
                deAmDf.append(dfT)
                print(dfT.head(10))
                print('-' * 100)

    deAmDf = pd.concat(deAmDf)
    return deAmDf
