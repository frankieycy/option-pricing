import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

#### Vol Surface ###############################################################

SVI_PARAMS = {'v0': 0.09, 'v1': 0.04, 'v2':0.04, 'k1': 5, 'k2': 0.1, 'rho': -0.5, 'eta': 1, 'gam': 0.5}

def HestonTermStructureKernel(v0, v1, v2, k1, k2):
    def w0(T):
        return v0*T+(v1-v0)*(1-np.exp(-k1*T))/k1+(v2-v0)*k1/(k1-k2)*((1-np.exp(-k2*T))/k2-(1-np.exp(-k1*T))/k1)
    return w0

def HestonTermStructureKernelTimeDeriv(v0, v1, v2, k1, k2):
    def dw0dT(T):
        return v0+(v1-v0)*np.exp(-k1*T)+(v2-v0)*k1/(k1-k2)*(np.exp(-k2*T)-np.exp(-k1*T))
    return dw0dT

def PowerLawSkewKernel(eta, gam):
    def sk0(w):
        return eta/w**gam
    return sk0

def PowerLawSkewKernelVarDeriv(eta, gam):
    def dsk0dw(w):
        return -eta*gam/w**(gam+1)
    return dsk0dw

def SviPowerLaw(v0, v1, v2, k1, k2, rho, eta, gam):
    # PowerLaw-SVI parametrization
    w0 = HestonTermStructureKernel(v0, v1, v2, k1, k2)
    sk0 = PowerLawSkewKernel(eta, gam)
    def sviFunc(k, T):
        w = w0(T)
        sk = sk0(w)
        a = np.sqrt((sk*k+rho)**2+1-rho**2)
        return w/2*(1+rho*sk*k+a)
    return sviFunc

def SviPowerLawLVol(v0, v1, v2, k1, k2, rho, eta, gam):
    # Local-vol under PowerLaw-SVI parametrization
    w0 = HestonTermStructureKernel(v0, v1, v2, k1, k2)
    sk0 = PowerLawSkewKernel(eta, gam)
    dw0dT = HestonTermStructureKernelTimeDeriv(v0, v1, v2, k1, k2)
    dsk0dw = PowerLawSkewKernelVarDeriv(eta, gam)
    def sviLVarFunc(k, T):
        w = w0(T)
        sk = sk0(w)
        dskdw = dsk0dw(w)
        a = np.sqrt((sk*k+rho)**2+1-rho**2)
        ww = w/2*(1+rho*sk*k+a)
        dwdT = dw0dT(T)*(0.5*(1+rho*sk*k+a)+w/2*(rho*dskdw*k+(sk*k+rho)*dskdw*k/a))
        dwdk = w/2*(rho*sk+(sk*k+rho)*sk/a)
        d2wdk2 = w/2*sk**2*(1-rho**2)/a**3
        return dwdT/((k/(2*ww)*dwdk-1)**2+0.5*d2wdk2-0.25*(0.25+1/ww)*dwdk**2)
    return sviLVarFunc

def VolSurfaceMatrixToDataFrame(m, k, T, idx_name='Expiry', col_name='Log-strike'):
    df = pd.DataFrame(m, index=T, columns=k)
    df.index.name = idx_name
    df.columns.name = col_name
    return df

def PlotImpliedVol():
    pass

#### American Option ###########################################################

def LatticeAmerican(K, T):
    pass

def DeAmericanize(V, K, T):
    pass
