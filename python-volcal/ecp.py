import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from numba import njit
from statistics import NormalDist
from scipy.special import ndtr
from scipy.optimize import bisect, minimize, differential_evolution
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from pricer import BlackScholesFormula, BlackScholesImpliedVol
plt.switch_backend("Agg")

#### Global Variables ##########################################################

zeps = 1e-8

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
    # return np.vectorize(tau)
    return tau

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
    global zeps
    N = len(zgrid)
    n = (N-1)//2
    zmin = min(zgrid)
    zmax = max(zgrid)
    def h(z): # used only in znegCalc(method='Bisection')
        z = np.maximum(zmin,np.minimum(zmax-zeps,z)) # range guard
        # j = np.argmax(zgrid>z)-1
        j = np.searchsorted(zgrid,z,side='right')-1
        jj = j*(j>=n)+(j+1)*(j<n) # anchor
        z0 = z-zgrid[jj]
        return alpha[j]+beta[j]*z0+z0**2/(2*gamma[j])
    # return np.vectorize(h)
    return h

def ohmFunc(alpha, beta, gamma, zgrid):
    # Cdf under piecewise-quadratic h
    # arguments are vectors of size 2n+1
    global zeps
    N = len(zgrid)
    n = (N-1)//2
    zmin = min(zgrid)
    zmax = max(zgrid)
    ohm0 = np.zeros(N)
    fac1 = np.sqrt(2*np.pi*gamma)
    fac2 = np.exp(gamma*beta**2/2-alpha)
    fac3 = np.sqrt(gamma)
    fac4 = np.sqrt(gamma)*beta
    cdf = ndtr # np.vectorize(NormalDist().cdf)
    j = np.arange(N-1)
    jj = j*(j>=n)+(j+1)*(j<n)
    cdf0 = cdf((zgrid[:-1]-zgrid[jj])/fac3[:-1]+fac4[:-1])
    cdf1 = cdf((zgrid[1:]-zgrid[jj])/fac3[:-1]+fac4[:-1])
    for j in range(N-1):
        jj = j*(j>=n)+(j+1)*(j<n)
        ohm0[j+1] = ohm0[j] + fac1[j] * fac2[j] * (cdf1[j] - cdf0[j])
    alpha += np.log(ohm0[-1])
    ohm0inf = ohm0[-1]
    ohm0 /= ohm0inf # normalize
    fac2 /= ohm0inf
    def ohm(z):
        z = np.maximum(zmin,np.minimum(zmax-zeps,z)) # range guard
        z = np.nan_to_num(z)
        # j = np.argmax(zgrid>z)-1
        j = np.searchsorted(zgrid,z,side='right')-1
        jj = j*(j>=n)+(j+1)*(j<n) # anchor
        return ohm0[j] + fac1[j] * fac2[j] * (cdf((z-zgrid[jj])/fac3[j]+fac4[j]) - cdf0[j])
    # return np.vectorize(ohm)
    return ohm

def znegCalc(X, tauT, h, zgrid, alpha=None, beta=None, gamma=None, method='Bisection'):
    # Compute zneg from X = h(z+tauT)-h(z)
    zneg = np.nan
    if method == 'Bisection':
        def objective(z):
            return h(z+tauT)-h(z)-X
        z0, z1 = zgrid[0], zgrid[-1]-tauT
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

znegCalc = np.vectorize(znegCalc, excluded=(2,3,'alpha','beta','gamma','method')) # vectorize X,tauT

@njit(fastmath=True)
def znegCalc_loop(X_vec, tauT_vec, zgrid, alpha, beta, gamma):
    # Compute zneg from X = h(z+tauT)-h(z) (very fast implementation!)
    zneg_vec = np.zeros(len(X_vec))
    N = len(zgrid)
    n = (N-1)//2
    for i,par in enumerate(zip(X_vec,tauT_vec)):
        zneg = np.nan
        X,tauT = par
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
                q2 = 1/(2*g1)-1/(2*g0)
                q1 = b1-b0+(tauT-zkk)/g1+zjj/g0
                q0 = -X+a1-a0+b1*(tauT-zkk)+b0*zjj+(tauT-zkk)**2/(2*g1)-zjj**2/(2*g0)
                if q2 == 0:
                    roots = [-q0/q1]*2
                else:
                    D = q1**2-4*q0*q2
                    if D >= 0:
                        roots = [(-q1+np.sqrt(D))/(2*q2),(-q1-np.sqrt(D))/(2*q2)]
                    else: roots = [np.nan]*2
                # print(tauT,X,'|',j,z0,z1,zjj,'|',k,zt0,zt1,zkk,'|',roots)
                for z in roots:
                    if (z >= z0 and z <= z1) and (z+tauT >= zt0 and z+tauT <= zt1):
                        zneg = z
                        jloop = False
                        kloop = False
                        break
                if not kloop: break
            if not jloop: break
        zneg_vec[i] = zneg
    return zneg_vec

def CarrPeltsPrice(K, T, D, F, tau, h, ohm, zgrid, X=None, tauT=None, **kwargs):
    # Compute Carr-Pelts price (via their BS-like formula)
    if X is None:
        X = np.log(F/K)
    if tauT is None:
        tauT = tau(T)
    # zneg = znegCalc(X,tauT,h,zgrid,**kwargs) # slow!
    if 'method' in kwargs and kwargs['method'] == 'Loop':
        alpha,beta,gamma = kwargs['alpha'],kwargs['beta'],kwargs['gamma']
        zneg = znegCalc_loop(X,tauT,zgrid,alpha,beta,gamma)
    # print(sum(np.isnan(zneg)))
    # print(zneg[:200])
    zpos = zneg+tauT
    Dpos = ohm(zpos)
    Dneg = ohm(zneg)
    P = D*(F*Dpos-K*Dneg)
    P = np.nan_to_num(P) # small gamma makes Dpos/neg blow up
    return P

def CarrPeltsImpliedVol(K, T, D, F, tau, h, ohm, zgrid, X=None, tauT=None, methodIv='Bisection', **kwargs):
    # Compute Carr-Pelts price and invert to implied vol
    P = CarrPeltsPrice(K, T, D, F, tau, h, ohm, zgrid, X, tauT, **kwargs)
    vol = BlackScholesImpliedVol(F, K, T, 0, P/D, 'call', methodIv)
    # print(np.array([T,F,K,P,vol]).T[:200])
    return vol

def FitCarrPelts(df, zgridCfg=(-100,150,50), gamma0Cfg=(1,1), fixVol=False, guessCP=None, w=None, optMethod='Gradient'):
    # Fit Carr-Pelts parametrization - require trial & error and artisanal knowledge!
    # Left-skewed distribution implied by positive beta and decreasing gamma
    # Calibration: (1) calibrate alpha/beta/gamma via evolution (coarse) (2) calibrate sig via gradient (polish)
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

    if w is None:
        w = 1/(ask-bid)

    zgrid = np.arange(*zgridCfg)
    N = len(zgrid)

    #### ATM vol
    w0 = np.zeros(Nexp)
    T0 = df["Texp"].to_numpy()

    for j,T in enumerate(Texp):
        i = (T0==T)
        kT = k[i]
        vT = midVar[i]
        ntm = (kT>-0.05)&(kT<0.05)
        spline = InterpolatedUnivariateSpline(kT[ntm], vT[ntm])
        w0[j] = spline(0).item()*T # ATM total variance

    sig0 = np.sqrt(w0/Texp)

    #### Init params & bounds
    if guessCP is None:
        # alpha0, beta0, gamma0 = np.log(2*np.pi)/2, 0, np.ones(N) # BS-case
        alpha0, beta0, gamma0 = 1, 0, np.linspace(*gamma0Cfg,N)
        params0 = np.concatenate(([alpha0],[beta0],gamma0))
    else: # User-defined (alpha0,beta0,gamma0)
        params0 = np.array(guessCP)

    if fixVol: # Fix sig at ATM vols
        bounds0 = [[0,2],[-2,2]]+[[0.01,5]]*N
    else:
        params0 = np.concatenate((params0,sig0))
        bounds0 = [[0,2],[-2,2]]+[[0.01,5]]*N+list(zip(np.maximum(sig0-0.03,0),sig0+0.03))

    # print(params0, bounds0)

    #### Loss function
    K = df['Strike'].to_numpy()
    T = df['Texp'].to_numpy()
    D = df['PV'].to_numpy()
    F = df['Fwd'].to_numpy()
    C = df['CallMid'].to_numpy()
    X = np.log(F/K) # pre-compute

    w = np.array(w)
    mid = np.array(mid)

    @njit(fastmath=True, cache=True)
    def ivToL2Loss(iv):
        return np.sum(w*(iv-mid)**2)

    if fixVol:
        tau = tauFunc(sig0,Texp)
        tauT = tau(T) # pre-compute

        def loss(params):
            alpha = params[0]
            beta  = params[1]
            gamma = params[2:2+N]

            print(f'params:\n  alpha={alpha}\n  beta={beta}\n  gamma={gamma}')

            alpha, beta, gamma = hParams(alpha,beta,gamma,zgrid)

            h   = hFunc(alpha,beta,gamma,zgrid)
            ohm = ohmFunc(alpha,beta,gamma,zgrid)

            iv = CarrPeltsImpliedVol(K,T,D,F,tau,h,ohm,zgrid,X,tauT, # most costly!
                alpha=alpha,beta=beta,gamma=gamma,method='Loop')
            # L = sum(w*(iv-mid)**2)
            L = ivToL2Loss(iv)

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

            iv = CarrPeltsImpliedVol(K,T,D,F,tau,h,ohm,zgrid,X, # most costly!
                alpha=alpha,beta=beta,gamma=gamma,method='Loop')
            # L = sum(w*(iv-mid)**2)
            L = ivToL2Loss(iv)

            print(f'  loss={L}')

            return L

    #### Basic test!
    # loss(params0)
    # return params0

    #### Optimization
    if optMethod == 'Gradient':
        opt = minimize(loss, x0=params0, bounds=bounds0)
    elif optMethod == 'Evolution':
        opt = differential_evolution(loss, bounds=bounds0)

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
        'opt.x': opt.x,
    }

    return CP

#### Ensemble Carr-Pelts #######################################################

def EnsembleCarrPeltsPrice(K, T, D, F, a, tau_vec, h_vec, ohm_vec, zgrid, X=None, tauT_vec=None, kwargs=None):
    # Compute ensemble Carr-Pelts price (via their BS-like formula)
    # a = ensemble weights (assume sum to 1)
    n = len(a)
    P = 0
    if X is None:
        X = np.log(F/K)
    if tauT_vec is None:
        tauT_vec = [tau(T) for tau in tau_vec]
    if kwargs is None:
        kwargs = ({} for i in range(n))
    for i,par in enumerate(zip(tauT_vec,h_vec,ohm_vec,kwargs)):
        tauT,h,ohm,kw = par
        # zneg = znegCalc(X,tauT,h,zgrid,**kw) # slow!
        if 'method' in kw and kw['method'] == 'Loop':
            alpha,beta,gamma = kw['alpha'],kw['beta'],kw['gamma']
            zneg = znegCalc_loop(X,tauT,zgrid,alpha,beta,gamma)
        zpos = zneg+tauT
        Dpos = ohm(zpos)
        Dneg = ohm(zneg)
        P += a[i]*D*(F*Dpos-K*Dneg)
    P = np.nan_to_num(P) # small gamma makes Dpos/neg blow up
    return P

def EnsembleCarrPeltsImpliedVol(K, T, D, F, a, tau_vec, h_vec, ohm_vec, zgrid, X=None, tauT_vec=None, methodIv='Bisection_jit', kwargs=None):
    # Compute ensemble Carr-Pelts price and invert to implied vol
    P = EnsembleCarrPeltsPrice(K, T, D, F, a, tau_vec, h_vec, ohm_vec, zgrid, X, tauT_vec, kwargs)
    vol = BlackScholesImpliedVol(F, K, T, 0, P/D, 'call', methodIv)
    # print(np.array([T,F,K,P,vol]).T[:200])
    return vol

def FitEnsembleCarrPelts(df, n=2, zgridCfg=(-100,150,50), gamma0Cfg=(1,1), fixVol=False, guessCP=None, guessA=None, w=None, optMethod='Gradient'):
    # Fit ensemble Carr-Pelts parametrization - require trial & error and artisanal knowledge!
    # Left-skewed distribution implied by positive beta and decreasing gamma
    # Calibration: (1) calibrate alpha/beta/gamma via evolution (coarse) (2) calibrate sig via gradient (polish)
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

    if w is None:
        w = 1/(ask-bid)

    zgrid = np.arange(*zgridCfg)
    N = len(zgrid)

    #### ATM vol
    w0 = np.zeros(Nexp)
    T0 = df["Texp"].to_numpy()

    for j,T in enumerate(Texp):
        i = (T0==T)
        kT = k[i]
        vT = midVar[i]
        ntm = (kT>-0.05)&(kT<0.05)
        spline = InterpolatedUnivariateSpline(kT[ntm], vT[ntm])
        w0[j] = spline(0).item()*T # ATM total variance

    sig0 = np.sqrt(w0/Texp)

    #### Init params & bounds
    # Structure of params: (alpha, beta, gamma) * n + sig * n + a
    if guessCP is None:
        alpha0, beta0, gamma0 = 1, 0, np.linspace(*gamma0Cfg,N)
        params0 = np.concatenate([[alpha0],[beta0],gamma0]*n)
    else: # User-defined (alpha0,beta0,gamma0)
        params0 = np.array(guessCP)

    if fixVol: # Fix sig at ATM vols
        bounds0 = ([[0,2],[-4,4]]+[[0.005,5]]*N)*n
    else:
        params0 = np.concatenate([params0,np.tile(sig0,n)])
        bounds0 = ([[0,2],[-4,4]]+[[0.005,5]]*N)*n+list(zip(np.maximum(sig0-0.03,0),sig0+0.03))*n

    if guessA is None:
        params0 = np.concatenate((params0,[1]*n))
    else:
        params0 = np.concatenate((params0,guessA))

    bounds0 += [[0,1]]*n

    # print(params0, bounds0)

    #### Loss function
    K = df['Strike'].to_numpy()
    T = df['Texp'].to_numpy()
    D = df['PV'].to_numpy()
    F = df['Fwd'].to_numpy()
    C = df['CallMid'].to_numpy()
    X = np.log(F/K) # pre-compute

    w = np.array(w)
    mid = np.array(mid)

    @njit(fastmath=True, cache=True)
    def ivToL2Loss(iv):
        return np.sum(w*(iv-mid)**2)

    if fixVol:
        tau = tauFunc(sig0,Texp)
        tauT = tau(T) # pre-compute
        tau_vec = [tau]*n
        tauT_vec = [tauT]*n

        def loss(params):
            print(f'params:')

            h_vec   = list()
            ohm_vec = list()
            kwargs  = list()

            for k in range(n):
                alpha = params[(2+N)*k]
                beta  = params[(2+N)*k+1]
                gamma = params[(2+N)*k+2:(2+N)*k+2+N]

                print(f'  alpha{k}={alpha}\n  beta{k}={beta}\n  gamma{k}={gamma}')

                alpha, beta, gamma = hParams(alpha,beta,gamma,zgrid)

                h   = hFunc(alpha,beta,gamma,zgrid)
                ohm = ohmFunc(alpha,beta,gamma,zgrid)

                h_vec.append(h)
                ohm_vec.append(ohm)
                kwargs.append({'alpha': alpha, 'beta': beta, 'gamma': gamma, 'method': 'Loop'})

            a = params[(2+N)*n:]
            a /= sum(a) # normalize

            print(f'  a={a}')

            iv = EnsembleCarrPeltsImpliedVol(K,T,D,F,a,tau_vec,h_vec,ohm_vec,zgrid,X,tauT_vec,kwargs=kwargs) # most costly!
            # L = sum(w*(iv-mid)**2)
            L = ivToL2Loss(iv)

            print(f'  loss={L}')

            return L

    else:
        def loss(params):
            print(f'params:')

            tau_vec = list()
            h_vec   = list()
            ohm_vec = list()
            kwargs  = list()

            for k in range(n):
                alpha = params[(2+N)*k]
                beta  = params[(2+N)*k+1]
                gamma = params[(2+N)*k+2:(2+N)*k+2+N]
                sig   = params[(2+N)*n+Nexp*k:(2+N)*n+Nexp*k+Nexp]

                print(f'  alpha{k}={alpha}\n  beta{k}={beta}\n  gamma{k}={gamma}\n  sig{k}={sig}')

                alpha, beta, gamma = hParams(alpha,beta,gamma,zgrid)

                tau = tauFunc(sig,Texp)
                h   = hFunc(alpha,beta,gamma,zgrid)
                ohm = ohmFunc(alpha,beta,gamma,zgrid)

                tau_vec.append(tau)
                h_vec.append(h)
                ohm_vec.append(ohm)
                kwargs.append({'alpha': alpha, 'beta': beta, 'gamma': gamma, 'method': 'Loop'})

            a = params[(2+N+Nexp)*n:]
            a /= sum(a)

            print(f'  a={a}')

            iv = EnsembleCarrPeltsImpliedVol(K,T,D,F,a,tau_vec,h_vec,ohm_vec,zgrid,X,kwargs=kwargs) # most costly!
            # L = sum(w*(iv-mid)**2)
            L = ivToL2Loss(iv)

            print(f'  loss={L}')

            return L

    #### Basic test!
    # loss(params0)
    # return params0

    #### Optimization
    if optMethod == 'Gradient':
        opt = minimize(loss, x0=params0, bounds=bounds0)
    elif optMethod == 'Evolution':
        opt = differential_evolution(loss, bounds=bounds0)

    print(opt)

    #### Output
    CP = {
        'zgrid': zgrid,
        'Tgrid': Texp,
        'opt.x': opt.x,
    }

    for k in range(n):
        alpha = opt.x[(2+N)*k]
        beta  = opt.x[(2+N)*k+1]
        gamma = opt.x[(2+N)*k+2:(2+N)*k+2+N]

        alpha, beta, gamma = hParams(alpha,beta,gamma,zgrid)

        if fixVol:
            sig = sig0
        else:
            sig = opt.x[(2+N)*n+Nexp*k:(2+N)*n+Nexp*k+Nexp]

        CP[k] = {
            'alpha': alpha,
            'beta':  beta,
            'gamma': gamma,
            'sig':   sig,
        }

    a = opt.x[(2+N+Nexp*(1-fixVol))*n:]
    a /= sum(a)

    CP['a'] = a

    return CP
