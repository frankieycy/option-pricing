import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from numba import njit
from statistics import NormalDist
from scipy.optimize import bisect, minimize, differential_evolution
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from pricer import BlackScholesFormula, BlackScholesImpliedVol
plt.switch_backend("Agg")

#### Carr-Pelts ################################################################
# Parametrize surface with two functions (tau & h) along time/strike dimension

def QuadraticRoots(a, b, c):
    if a == 0:
        return (-c/b, -c/b)
    D = b**2-4*a*c
    if D >= 0:
        r0 = -b/(2*a)
        r1 = np.sqrt(D)/(2*a)
        return (r0-r1, r0+r1)
    else:
        return (np.nan, np.nan)

def tauFunc(sig, Tgrid):
    # Piecewise-constant total implied vol
    # sig, Tgrid are vectors of size m
    w = np.cumsum(sig**2*np.diff(Tgrid,prepend=0))
    w0 = np.concatenate(([0],w))
    T0 = np.concatenate(([0],Tgrid))
    wFunc = interp1d(T0,w0) # linear
    def tau(T):
        return np.sqrt(wFunc(T))
    return np.vectorize(tau)

@njit
def hParams(alpha0, beta0, gamma0, zgrid):
    # Recursively compute alpha/beta vector
    N = len(zgrid)
    n = (N-1)//2
    alpha = np.zeros(N)
    beta  = np.zeros(N)
    gamma = gamma0
    alpha[n] = alpha[n-1] = alpha0
    beta[n] = beta[n-1] = beta0
    for j in range(n,N-1):
        z0 = zgrid[j+1]-zgrid[j]
        alpha[j+1] = alpha[j]+beta[j]*z0+z0**2/(2*gamma[j])
        beta[j+1] = beta[j]+z0/gamma[j]
    for j in range(n-1,-1,-1):
        z0 = zgrid[j]-zgrid[j+1]
        alpha[j-1] = alpha[j]+beta[j]*z0+z0**2/(2*gamma[j])
        beta[j-1] = beta[j]+z0/gamma[j]
    return alpha, beta, gamma

def hFunc(alpha, beta, gamma, zgrid):
    # Piecewise-quadratic pdf exponent (BS case: h = log(2pi)/2+z^2/2)
    # arguments are vectors of size 2n+1
    # TO-DO: bound guard for z (when < min or > max zgrid)
    N = len(zgrid)
    n = (N-1)//2
    def h(z):
        j = np.argmax(zgrid>z)-1
        jj = j*(j>=n)+(j+1)*(j<n) # anchor
        z0 = z-zgrid[jj]
        return alpha[j]+beta[j]*z0+z0**2/(2*gamma[j])
    return np.vectorize(h)

def ohmFunc(alpha, beta, gamma, zgrid):
    # Cdf under piecewise-quadratic h
    # arguments are vectors of size 2n+1
    # TO-DO: bound guard for z (when < min or > max zgrid)
    N = len(zgrid)
    n = (N-1)//2
    ohm0 = np.zeros(N)
    fac1 = np.sqrt(2*np.pi*gamma)
    fac2 = np.exp(gamma*beta**2/2-alpha)
    fac3 = np.sqrt(gamma)
    fac4 = np.sqrt(gamma)*beta
    cdf = NormalDist().cdf
    for j in range(N-1):
        jj = j*(j>=n)+(j+1)*(j<n)
        ohm0[j+1] = ohm0[j] + fac1[j] * fac2[j] * (cdf((zgrid[j+1]-zgrid[jj])/fac3[j]+fac4[j]) - cdf((zgrid[j]-zgrid[jj])/fac3[j]+fac4[j]))
    alpha += np.log(ohm0[-1])
    ohm0inf = ohm0[-1]
    ohm0 /= ohm0inf # normalize
    fac2 /= ohm0inf
    def ohm(z):
        j = np.argmax(zgrid>z)-1
        jj = j*(j>=n)+(j+1)*(j<n) # anchor
        return ohm0[j] + fac1[j] * fac2[j] * (cdf((z-zgrid[jj])/fac3[j]+fac4[j]) - cdf((zgrid[j]-zgrid[jj])/fac3[j]+fac4[j]))
    return np.vectorize(ohm)

def znegCalc(X, tauT, h, zgrid, alpha=None, beta=None, gamma=None, method='Bisection'):
    # Compute zneg from X = h(z+tauT)-h(z)
    # TO-DO: Make this fast!
    zneg = np.nan
    if method == 'Bisection':
        def objective(z):
            return h(z+tauT)-h(z)-X
        z0, z1 = zgrid[0], zgrid[-1]
        try: zneg = bisect(objective,z0,z1) # very slow!
        except Exception: pass
    elif method == 'Loop':
        N = len(zgrid)
        n = (N-1)//2
        jloop = True
        for j in range(1,N):
            z0 = zgrid[j-1] # bounds for z
            z1 = zgrid[j]
            k0 = np.argmax(zgrid>z0+tauT)
            k1 = np.argmax(zgrid>z1+tauT)
            if k1 < k0: k1 = k0
            kloop = True
            for k in range(k0,k1+1):
                zt0 = zgrid[k-1] # bounds for z+tauT
                zt1 = zgrid[k]
                jj = (j-1)*(j-1>=n)+j*(j-1<n) # anchor idx for z
                kk = (k-1)*(k-1>=n)+k*(k-1<n) # anchor idx for z+tauT
                a0,b0,g0,zjj = alpha[j-1],beta[j-1],gamma[j-1],zgrid[jj]
                a1,b1,g1,zkk = alpha[k-1],beta[k-1],gamma[k-1],zgrid[kk]
                roots = QuadraticRoots(1/(2*g1)-1/(2*g0),b1-b0+(tauT-zkk)/g1+zjj/g0,-X+a1-a0+b1*(tauT-zkk)+b0*zjj+(tauT-zkk)**2/(2*g1)-zjj**2/(2*g0))
                # print(tauT,X,'|',j,z0,z1,zjj,'|',k,zt0,zt1,zkk,'|',roots)
                for z in roots:
                    if (z >= z0 and z <= z1) and (z+tauT >= zt0 and z+tauT <= zt1):
                        zneg = z
                        jloop = False
                        kloop = False
                        break
                if not kloop: break
            if not jloop: break
    return zneg

znegCalc = np.vectorize(znegCalc, excluded=(2,3,'alpha','beta','gamma','method'))

def CarrPeltsPrice(K, T, D, F, tau, h, ohm, zgrid, **kwargs):
    # Compute Carr-Pelts price (via their BS-like formula)
    X = np.log(F/K)
    tauT = tau(T) # 0.035s
    zneg = znegCalc(X,tauT,h,zgrid,**kwargs) # 0.135s
    # print(sum(np.isnan(zneg)))
    # print(zneg[:200])
    zpos = zneg+tauT
    Dpos = ohm(zpos) # 0.070s
    Dneg = ohm(zneg)
    P = D*(F*Dpos-K*Dneg)
    P = np.nan_to_num(P) # small gamma makes D blow up
    return P

def CarrPeltsImpliedVol(K, T, D, F, tau, h, ohm, zgrid, methodIv='Bisection', **kwargs):
    P = CarrPeltsPrice(K, T, D, F, tau, h, ohm, zgrid, **kwargs)
    vol = BlackScholesImpliedVol(F, K, T, 0, P/D, 'call', methodIv)
    return vol

def FitCarrPelts(df, zgridCfg=(-100,150,50), gamma0Cfg=(1,1), guessCP=None, fixVol=False):
    # Fit Carr-Pelts parametrization
    # Left-skewed distribution implied by positive beta and decreasing gamma
    # zgrid boundaries correspond to +/- inf; gamma[-1] is dummy (we chose z1~100>z)
    # Ref: Antonov, A New Arbitrage-Free Parametric Volatility Surface
    df = df.dropna()
    df = df[(df['Bid']>0)&(df['Ask']>0)]
    Texp = df["Texp"].unique()
    Nexp = len(Texp)

    k = np.log(df["Strike"]/df["Fwd"])
    k = k.to_numpy()
    bid = df["Bid"].to_numpy()
    ask = df["Ask"].to_numpy()
    mid = (bid+ask)/2
    midVar = (bid**2+ask**2)/2

    w = 1/(ask-bid)

    #### Init params & bounds
    if guessCP is None:
        w0 = np.zeros(Nexp)
        T0 = df["Texp"].to_numpy()

        for j,T in enumerate(Texp):
            i = (T0==T)
            kT = k[i]
            vT = midVar[i]
            ntm = (kT>-0.05)&(kT<0.05)
            spline = InterpolatedUnivariateSpline(kT[ntm], vT[ntm])
            w0[j] = spline(0).item()*T # ATM total variance

        zgrid = np.arange(*zgridCfg)
        N = len(zgrid)

        sig0 = np.sqrt(w0/Texp)
        # alpha0, beta0, gamma0 = np.log(2*np.pi)/2, 0, np.ones(N) # BS-case
        alpha0, beta0, gamma0 = 1, 0, np.linspace(*gamma0Cfg,N)

        if fixVol:
            params0 = np.concatenate(([alpha0],[beta0],gamma0))
        else:
            params0 = np.concatenate(([alpha0],[beta0],gamma0,sig0))

    else:
        params0 = guessCP

    if fixVol:
        bounds0 = [[0,2],[0,2]]+[[0.0001,5]]*N
    else:
        bounds0 = [[0,2],[0,2]]+[[0.0001,5]]*N+[[0.01,0.5]]*Nexp

    #### Loss function
    K = df['Strike'].to_numpy()
    T = df['Texp'].to_numpy()
    D = df['PV'].to_numpy()
    F = df['Fwd'].to_numpy()
    C = df['CallMid'].to_numpy()

    # Cbid = D*BlackScholesFormula(F,K,T,0,bid,'call')
    # Cask = D*BlackScholesFormula(F,K,T,0,ask,'call')
    # w = 1/(Cask-Cbid)

    if fixVol:
        tau = tauFunc(sig0,Texp)

        def loss(params):
            alpha = params[0]
            beta  = params[1]
            gamma = params[2:2+N]

            print(f'params:\n  alpha={alpha}\n  beta={beta}\n  gamma={gamma}')

            alpha, beta, gamma = hParams(alpha,beta,gamma,zgrid)

            h   = hFunc(alpha,beta,gamma,zgrid)
            ohm = ohmFunc(alpha,beta,gamma,zgrid)

            # P = CarrPeltsPrice(K,T,D,F,tau,h,ohm,zgrid, # most costly!
            #     alpha=alpha,beta=beta,gamma=gamma,method='Loop')
            # # P = CarrPeltsPrice(K,T,D,F,tau,h,ohm,zgrid)
            # L = sum(w*(P-C)**2)

            iv = CarrPeltsImpliedVol(K,T,D,F,tau,h,ohm,zgrid, # most costly!
                alpha=alpha,beta=beta,gamma=gamma,method='Loop')
            L = sum(w*(iv-mid)**2)

            print(f'  loss={L}')

            return L

    else:
        def loss(params):
            alpha = params[0]
            beta  = params[1]
            gamma = params[2:2+N]
            sig   = params[2+N:]

            print(f'params:\n  alpha={alpha}\n  beta={beta}\n  gamma={gamma}\n  sig={sig}')

            alpha, beta, gamma = hParams(alpha,beta,gamma,zgrid)

            tau = tauFunc(sig,Texp)
            h   = hFunc(alpha,beta,gamma,zgrid)
            ohm = ohmFunc(alpha,beta,gamma,zgrid)

            # P = CarrPeltsPrice(K,T,D,F,tau,h,ohm,zgrid, # most costly!
            #     alpha=alpha,beta=beta,gamma=gamma,method='Loop')
            # # P = CarrPeltsPrice(K,T,D,F,tau,h,ohm,zgrid)
            # L = sum(w*(P-C)**2)

            iv = CarrPeltsImpliedVol(K,T,D,F,tau,h,ohm,zgrid, # most costly!
                alpha=alpha,beta=beta,gamma=gamma,method='Loop')
            L = sum(w*(iv-mid)**2)

            print(f'  loss={L}')

            return L

    #### Basic test!
    # loss(params0)
    # return params0

    #### Optimization
    opt = minimize(loss, x0=params0, bounds=bounds0)
    # opt = differential_evolution(loss, bounds=bounds0)

    print(opt)

    #### Output
    alpha = opt.x[0]
    beta  = opt.x[1]
    gamma = opt.x[2:2+N]

    alpha, beta, gamma = hParams(alpha,beta,gamma,zgrid)

    if fixVol:
        sig = sig0
    else:
        sig = opt.x[2+N:]

    CP = {
        'zgrid': zgrid,
        'Tgrid': Texp,
        'alpha': alpha,
        'beta':  beta,
        'gamma': gamma,
        'sig':   sig,
    }

    return CP

#### Ensemble Carr-Pelts #######################################################

def EnsembleCarrPeltsPrice(K, T, D, F, tau, h, ohm):
    # Fit ensemble Carr-Pelts parametrization
    pass
