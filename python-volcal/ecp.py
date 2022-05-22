import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
plt.switch_backend("Agg")

#### Carr-Pelts ################################################################
# Parametrize surface with two functions (tau & h) along time/strike dimension

def tauFunc(sig, Tgrid):
    # Piecewise-constant total implied vol
    # sig, Tgrid are vectors of size m
    w = sig**2*np.diff(Tgrid,prepend=0)
    w = np.concatenate(([0],w))
    Tgrid = np.concatenate(([0],Tgrid))
    wFunc = interp1d(Tgrid,w) # linear
    def tau(T):
        return np.sqrt(wFunc(T))
    return tau

def hFunc(alpha, beta, gamma, zgrid):
    # Piecewise-quadratic pdf exponent (BS case: h = log(2pi)/2+z^2/2)
    # alpha, beta are scalars; gamma, zgrid are vectors of size 2n+1
    N = len(zgrid)
    n = (N-1)//2
    alpha0 = np.zeros(N)
    beta0  = np.zeros(N)
    gamma0 = gamma
    alpha0[n] = alpha0[n-1] = alpha
    beta0[n] = beta0[n-1] = beta
    for j in range(n,N):
        alpha0[j+1] = alpha0[j]+beta0[j]*(zgrid[j+1]-zgrid[j])+(zgrid[j+1]-zgrid[j])**2/(2*gamma0[j])
        beta0[j+1] = beta0[j]+(zgrid[j+1]-zgrid[j])/gamma0[j]
    for j in range(n-1,-1,-1):
        alpha0[j-1] = alpha0[j]+beta0[j]*(zgrid[j1]-zgrid[j+1])+(zgrid[j]-zgrid[j+1])**2/(2*gamma0[j])
        beta0[j-1] = beta0[j]+(zgrid[j]-zgrid[j+1])/gamma0[j]
    def h(z):
        i = np.argmax(zgrid>z)-1
        ii = i*(i>=n)+(i+1)*(i<n)
        return alpha0[i]+beta0[i]*(z-zgrid[ii])+(z-zgrid[ii])**2/(2*gamma0[i])
    return h

def ohmFunc(alpha, beta, gamma, zgrid):
    # Cdf under piecewise-quadratic h
    def ohm(z):
        pass
    return ohm

def znegCalc(X, tauT, h):
    # Compute zneg from X = h(z+tauT)-h(z)
    pass

def CarrPeltsPrice(K, T, D, F, tau, h, ohm):
    # Compute Carr-Pelts price (via their BS-like formula)
    X = np.log(F/K)
    tauT = tau(T)
    zneg = znegCalc(X,tauT,h)
    zpos = zneg+tauT
    Dpos = ohm(zpos)
    Dneg = ohm(zneg)
    P = D*(F*Dpos-K*Dneg)
    return P

def FitCarrPelts(df):
    # Fit Carr-Pelts parametrization
    pass

#### Ensemble Carr-Pelts #######################################################

def EnsembleCarrPeltsPrice(K, T, D, F, tau, h, ohm):
    # Fit ensemble Carr-Pelts parametrization
    pass
