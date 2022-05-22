import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from numba import njit
from scipy.stats import norm
from scipy.optimize import minimize, bisect
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from pricer import BlackScholesFormula
plt.switch_backend("Agg")

#### Carr-Pelts ################################################################
# Parametrize surface with two functions (tau & h) along time/strike dimension

def tauFunc(sig, Tgrid):
    # Piecewise-constant total implied vol
    # sig, Tgrid are vectors of size m
    w = sig**2*np.diff(Tgrid,prepend=0)
    w0 = np.concatenate(([0],w))
    T0 = np.concatenate(([0],Tgrid))
    wFunc = interp1d(T0,w0) # linear
    def tau(T):
        return np.sqrt(wFunc(T))
    return np.vectorize(tau)

@njit
def hParams(alpha0, beta0, gamma0, zgrid):
    N = len(zgrid)
    n = (N-1)//2
    alpha = np.zeros(N)
    beta  = np.zeros(N)
    gamma = gamma0
    alpha[n] = alpha[n-1] = alpha0
    beta[n] = beta[n-1] = beta0
    for j in range(n,N-1):
        alpha[j+1] = alpha[j]+beta[j]*(zgrid[j+1]-zgrid[j])+(zgrid[j+1]-zgrid[j])**2/(2*gamma[j])
        beta[j+1] = beta[j]+(zgrid[j+1]-zgrid[j])/gamma[j]
    for j in range(n-1,-1,-1):
        alpha[j-1] = alpha[j]+beta[j]*(zgrid[j]-zgrid[j+1])+(zgrid[j]-zgrid[j+1])**2/(2*gamma[j])
        beta[j-1] = beta[j]+(zgrid[j]-zgrid[j+1])/gamma[j]
    return alpha, beta, gamma

def hFunc(alpha, beta, gamma, zgrid):
    # Piecewise-quadratic pdf exponent (BS case: h = log(2pi)/2+z^2/2)
    # arguments are vectors of size 2n+1
    N = len(zgrid)
    n = (N-1)//2
    def h(z):
        j = np.argmax(zgrid>z)-1
        jj = j*(j>=n)+(j+1)*(j<n)
        return alpha[j]+beta[j]*(z-zgrid[jj])+(z-zgrid[jj])**2/(2*gamma[j])
    return np.vectorize(h)

def ohmFunc(alpha, beta, gamma, zgrid):
    # Cdf under piecewise-quadratic h
    # arguments are vectors of size 2n+1
    N = len(zgrid)
    n = (N-1)//2
    ohm0 = np.zeros(N)
    for j in range(N-1):
        jj = j*(j>=n)+(j+1)*(j<n)
        ohm0[j+1] = ohm0[j] + np.sqrt(2*np.pi*gamma[j]) * np.exp(gamma[j]*beta[j]**2/2-alpha[j]) * (norm.cdf((zgrid[j+1]-zgrid[jj])/np.sqrt(gamma[j])+np.sqrt(gamma[j])*beta[j]) - norm.cdf((zgrid[j]-zgrid[jj])/np.sqrt(gamma[j])+np.sqrt(gamma[j])*beta[j]))
    alpha += np.log(ohm0[-1])
    ohm0 /= ohm0[-1] # normalize
    def ohm(z):
        j = np.argmax(zgrid>z)-1
        jj = j*(j>=n)+(j+1)*(j<n)
        return ohm0[j] + np.sqrt(2*np.pi*gamma[j]) * np.exp(gamma[j]*beta[j]**2/2-alpha[j]) * (norm.cdf((z-zgrid[jj])/np.sqrt(gamma[j])+np.sqrt(gamma[j])*beta[j]) - norm.cdf((zgrid[j]-zgrid[jj])/np.sqrt(gamma[j])+np.sqrt(gamma[j])*beta[j]))
    return np.vectorize(ohm)

def znegCalc(X, tauT, h, zgrid, method='Bisection'):
    # Compute zneg from X = h(z+tauT)-h(z)
    # Make this fast!
    zneg = np.nan
    if method == 'Bisection':
        def objective(z):
            return h(z+tauT)-h(z)-X
        z0, z1 = zgrid[0], zgrid[-1]
        try: zneg = bisect(objective,z0,z1) # very slow!
        except Exception: pass
    elif method == 'Loop':
        pass
    return zneg

znegCalc = np.vectorize(znegCalc, excluded=(2,3))

def CarrPeltsPrice(K, T, D, F, tau, h, ohm, zgrid):
    # Compute Carr-Pelts price (via their BS-like formula)
    X = np.log(F/K)
    tauT = tau(T)
    zneg = znegCalc(X,tauT,h,zgrid)
    zpos = zneg+tauT
    Dpos = ohm(zpos)
    Dneg = ohm(zneg)
    P = D*(F*Dpos-K*Dneg)
    return P

def FitCarrPelts(df):
    # Fit Carr-Pelts parametrization
    df = df.dropna()
    df = df[(df['Bid']>0)&(df['Ask']>0)]
    Texp = df["Texp"].unique()
    Nexp = len(Texp)

    k = np.log(df["Strike"]/df["Fwd"])
    k = k.to_numpy()
    bid = df["Bid"].to_numpy()
    ask = df["Ask"].to_numpy()
    midVar = (bid**2+ask**2)/2

    w0 = np.zeros(Nexp)
    T0 = df["Texp"].to_numpy()

    for j,T in enumerate(Texp):
        i = (T0==T)
        kT = k[i]
        vT = midVar[i]
        ntm = (kT>-0.05)&(kT<0.05)
        spline = InterpolatedUnivariateSpline(kT[ntm], vT[ntm])
        w0[j] = spline(0).item()*T # ATM total variance

    zgrid = np.arange(-100,110,10)
    N = len(zgrid)

    sig0 = np.sqrt(w0/Texp)
    alpha0, beta0, gamma0 = np.log(2*np.pi)/2, 0, np.ones(len(zgrid))

    K = df['Strike']
    T = df['Texp']
    D = df['PV']
    F = df['Fwd']
    C = df['CallMid']

    Cbid = D*BlackScholesFormula(F,K,T,0,bid,'call')
    Cask = D*BlackScholesFormula(F,K,T,0,ask,'call')
    w = 1/(Cask-Cbid)

    np.set_printoptions(precision=4, suppress=True, linewidth=80)

    def loss(params):
        alpha = params[0]
        beta = params[1]
        gamma = params[2:2+N]
        sig = params[2+N:]

        tau = tauFunc(sig,Texp)
        alpha, beta, gamma = hParams(alpha,beta,gamma,zgrid)
        h = hFunc(alpha,beta,gamma,zgrid)
        ohm = ohmFunc(alpha,beta,gamma,zgrid)

        P = CarrPeltsPrice(K,T,D,F,tau,h,ohm,zgrid) # most costly!
        L = sum(w*(P-C)**2)

        print(f'params:\n  alpha={alpha}\n  beta={beta}\n  gamma={gamma}\n  sig={sig}\n  loss={L}')

        return L

    params0 = np.concatenate(([alpha0],[beta0],gamma0,sig0))
    bounds0 = [[-10,10],[-10,10]]+[[-10,10]]*N+[[0,1]]*Nexp

    opt = minimize(loss, x0=params0, bounds=bounds0)
    print(opt.x)

#### Ensemble Carr-Pelts #######################################################

def EnsembleCarrPeltsPrice(K, T, D, F, tau, h, ohm):
    # Fit ensemble Carr-Pelts parametrization
    pass
