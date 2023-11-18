import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
plt.switch_backend('Agg')

#### Vol Surface ###############################################################

SVI_PARAMS = {'v0': 0.09, 'v1': 0.04, 'v2':0.04, 'k1': 5, 'k2': 0.1, 'rho': -0.5, 'eta': 1, 'gam': 0.5}

class SviPowerLaw:
    def __init__(self, v0, v1, v2, k1, k2, rho, eta, gam):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.k1 = k1
        self.k2 = k2
        self.rho = rho
        self.eta = eta
        self.gam = gam
        self.IVolFunc = self.ImpliedVolFunc()
        self.LVolFunc = self.LocalVolFunc()

    def HestonTermStructureKernel(self):
        def w0(T):
            return self.v0*T+(self.v1-self.v0)*(1-np.exp(-self.k1*T))/self.k1+(self.v2-self.v0)*self.k1/(self.k1-self.k2)*((1-np.exp(-self.k2*T))/self.k2-(1-np.exp(-self.k1*T))/self.k1)
        return w0

    def HestonTermStructureKernelTimeDeriv(self):
        def dw0dT(T):
            return self.v0+(self.v1-self.v0)*np.exp(-self.k1*T)+(self.v2-self.v0)*self.k1/(self.k1-self.k2)*(np.exp(-self.k2*T)-np.exp(-self.k1*T))
        return dw0dT

    def PowerLawSkewKernel(self):
        def sk0(w):
            return self.eta/w**self.gam
        return sk0

    def PowerLawSkewKernelVarDeriv(self):
        def dsk0dw(w):
            return -self.eta*self.gam/w**(self.gam+1)
        return dsk0dw

    def TotalImpliedVarFunc(self):
        w0 = self.HestonTermStructureKernel()
        sk0 = self.PowerLawSkewKernel()
        def sviIVarFunc(k, T):
            w = w0(T)
            sk = sk0(w)
            a = np.sqrt((sk*k+self.rho)**2+1-self.rho**2)
            return w/2*(1+self.rho*sk*k+a)
        return sviIVarFunc

    def LocalVarFunc(self):
        w0 = self.HestonTermStructureKernel()
        sk0 = self.PowerLawSkewKernel()
        dw0dT = self.HestonTermStructureKernelTimeDeriv()
        dsk0dw = self.PowerLawSkewKernelVarDeriv()
        def sviLVarFunc(k, T):
            w = w0(T)
            sk = sk0(w)
            dskdw = dsk0dw(w)
            a = np.sqrt((sk*k+self.rho)**2+1-self.rho**2)
            ww = w/2*(1+self.rho*sk*k+a)
            dwdT = dw0dT(T)*(0.5*(1+self.rho*sk*k+a)+w/2*(self.rho*dskdw*k+(sk*k+self.rho)*dskdw*k/a))
            dwdk = w/2*(self.rho*sk+(sk*k+self.rho)*sk/a)
            d2wdk2 = w/2*sk**2*(1-self.rho**2)/a**3
            return dwdT/((k/(2*ww)*dwdk-1)**2+0.5*d2wdk2-0.25*(0.25+1/ww)*dwdk**2)
        return sviLVarFunc

    def ImpliedVolFunc(self):
        sviIVarFunc = self.TotalImpliedVarFunc()
        def sviIVolFunc(k, T):
            return np.sqrt(sviIVarFunc(k, T)/T)
        return sviIVolFunc

    def LocalVolFunc(self):
        sviLVarFunc = self.LocalVarFunc()
        def sviIVolFunc(k, T):
            return np.sqrt(sviLVarFunc(k, T))
        return sviIVolFunc

#### American Option ###########################################################

class Option:
    def __init__(self, K, T, pc, ex, px=None, ivEu=None, ivAm=None, exBdry=None):
        self.K = K
        self.T = T
        self.pc = pc
        self.ex = ex
        self.px = px
        self.ivEu = ivEu
        self.ivAm = ivAm
        self.exBdry = exBdry

class Spot:
    def __init__(self, S0, r, q, impVolFunc, locVolFunc, div=None):
        self.S0 = S0
        self.r = r
        self.q = q
        self.IVolFunc = impVolFunc
        self.LVolFunc = locVolFunc
        self.div = div

class LatticeConfig:
    def __init__(self, nX, nT, rangeX, rangeT, center='spot'):
        self.nX = nX
        self.nT = nT
        self.rangeX = rangeX
        self.rangeT = rangeT
        self.center = center
        self.gridX = None
        self.gridT = None

class LatticeAmerican:
    def __init__(self, options, spot, config):
        self.options = options
        self.spot = spot
        self.config = config

    def AmericanPrice(self):
        pass

    def DeAmericanize(self):
        pass

#### Helper Functions ##########################################################

def VolSurfaceMatrixToDataFrame(m, k, T, idx_name='Expiry', col_name='Log-strike'):
    df = pd.DataFrame(m, index=T, columns=k)
    df.index.name = idx_name
    df.columns.name = col_name
    return df

def PlotImpliedVol(df, dfLVol=None, figname=None, ncol=6, scatterFit=False, atmBar=False, xlim=None, ylim=None):
    # Plot implied volatilities based on df
    # df: Columns -- Log-strike, Index -- Expiry
    if not figname:
        figname = 'impliedvol.png'
    Texp = df.index
    k    = df.columns
    Nexp = len(Texp)
    ncol = min(Nexp,ncol)
    nrow = int(np.ceil(Nexp/ncol))

    if Nexp > 1: # multiple plots
        fig, ax = plt.subplots(nrow,ncol,figsize=(2.5*ncol,2*nrow))
    else: # single plot
        fig, ax = plt.subplots(nrow,ncol,figsize=(6,4))

    for i in range(nrow*ncol):
        ix,iy = i//ncol,i%ncol
        idx = (ix,iy) if nrow>1 else iy
        ax_idx = ax[idx] if ncol>1 else ax
        if i < Nexp:
            T = Texp[i]
            vol = 100*df.loc[T]
            ax_idx.set_title(rf'$T={np.round(T,3)}$')
            ax_idx.set_xlabel('log-strike')
            ax_idx.set_ylabel('vol (%)')
            if scatterFit:
                ax_idx.scatter(k,vol,c='k',s=5,label='implied')
                if dfLVol is not None:
                    volL = 100*dfLVol.loc[T]
                    ax_idx.scatter(k,volL,c='r',s=5,label='local')
            else:
                ax_idx.plot(k,vol,c='k',lw=3,label='implied')
                if dfLVol is not None:
                    volL = 100*dfLVol.loc[T]
                    ax_idx.plot(k,volL,c='r',lw=3,label='local')
            if atmBar:
                ax_idx.axvline(x=0,c='grey',ls='--',lw=1)
            if xlim is not None:
                ax_idx.set_ylim(xlim)
            if ylim is not None:
                ax_idx.set_ylim(ylim)
        else:
            ax_idx.axis('off')

    fig.tight_layout()
    plt.savefig(figname)
    plt.close()
