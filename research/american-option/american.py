import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
from copy import copy
from scipy.sparse import diags
from scipy.sparse.linalg import splu, spsolve
from scipy.interpolate import interp1d
plt.switch_backend('Agg')

#### Note ######################################################################
# 1. Attributes in all classes are freely accessible, i.e. no encapsulation (getter/setter)

#### Consts ####################################################################

MIN_TTX  = 1e-4         # minimum time-to-expiry to avoid blow-up in SviPowerLaw
MAX_LVAR = 10           # maximum local-var to avoid blow-up in SolveLattice

#### Vol Surface ###############################################################

class VolSurface:
    def __init__(self):
        self.IVolFunc = self.ImpliedVolFunc() # implied-vol function
        self.LVolFunc = self.LocalVolFunc()   # local-vol function
        self.LVarFunc = self.LocalVarFunc()   # local-var function

    def __repr__(self):
        return f'VolSurface()'

    def ImpliedVolFunc(self):
        return (lambda k,T: np.zeros(k.shape) if isinstance(k,np.ndarray) else 0)

    def LocalVolFunc(self):
        return (lambda k,T: np.zeros(k.shape) if isinstance(k,np.ndarray) else 0)

    def LocalVarFunc(self):
        return (lambda k,T: np.zeros(k.shape) if isinstance(k,np.ndarray) else 0)

class FlatVol(VolSurface):
    def __init__(self, sig):
        VolSurface.__init__(self)
        self.sig = sig

    def __repr__(self):
        return f'FlatVol(sig={self.sig})'

    def ImpliedVolFunc(self):
        return (lambda k,T: np.full(k.shape,self.sig) if isinstance(k,np.ndarray) else self.sig)

    def LocalVolFunc(self):
        return (lambda k,T: np.full(k.shape,self.sig) if isinstance(k,np.ndarray) else self.sig)

    def LocalVarFunc(self):
        return (lambda k,T: np.full(k.shape,self.sig) if isinstance(k,np.ndarray) else self.sig)

class SviPowerLaw(VolSurface):
    def __init__(self, v0, v1, v2, k1, k2, rho, eta, gam):
        VolSurface.__init__(self)
        self.v0  = v0  # base variance
        self.v1  = v1  # short-term variance
        self.v2  = v2  # long-term variance
        self.k1  = k1  # short-term mean-reversion
        self.k2  = k2  # long-term mean-reversion
        self.rho = rho # smile asymmetry
        self.eta = eta # skew magnitude
        self.gam = gam # skew decay

    def __repr__(self):
        return f'SviPowerLaw(v0={self.v0}, v1={self.v1}, v2={self.v2}, k1={self.k1}, k2={self.k2}, rho={self.rho}, eta={self.eta}, gam={self.gam})'

    def HestonTermStructureKernel(self):
        def w0(T):
            return self.v0*T+(self.v1-self.v0)*(1-np.exp(-self.k1*T))/self.k1+(self.v2-self.v0)*self.k1/(self.k1-self.k2)*((1-np.exp(-self.k2*T))/self.k2-(1-np.exp(-self.k1*T))/self.k1)
        return w0

    def HestonTermStructureKernelTimeDeriv(self):
        # Term-structure kernel derivative wrt. T
        def dw0dT(T):
            return self.v0+(self.v1-self.v0)*np.exp(-self.k1*T)+(self.v2-self.v0)*self.k1/(self.k1-self.k2)*(np.exp(-self.k2*T)-np.exp(-self.k1*T))
        return dw0dT

    def PowerLawSkewKernel(self):
        def sk0(w):
            return self.eta/w**self.gam
        return sk0

    def PowerLawSkewKernelVarDeriv(self):
        # Skew kernel derivative wrt. w
        def dsk0dw(w):
            return -self.eta*self.gam/w**(self.gam+1)
        return dsk0dw

    def TotalImpliedVarFunc(self):
        # SviPowerLaw surface parametrization
        w0  = self.HestonTermStructureKernel()
        sk0 = self.PowerLawSkewKernel()
        def sviIVarFunc(k, T):
            w  = w0(T)
            sk = sk0(w)
            a  = np.sqrt((sk*k+self.rho)**2+1-self.rho**2)
            return w/2*(1+self.rho*sk*k+a)
        return sviIVarFunc

    def LocalVarFunc(self):
        # SviPowerLaw analytic local-var
        w0     = self.HestonTermStructureKernel()
        sk0    = self.PowerLawSkewKernel()
        dw0dT  = self.HestonTermStructureKernelTimeDeriv()
        dsk0dw = self.PowerLawSkewKernelVarDeriv()
        def sviLVarFunc(k, T):
            w      = w0(T)
            sk     = sk0(w)
            dskdw  = dsk0dw(w)
            a      = np.sqrt((sk*k+self.rho)**2+1-self.rho**2)
            ww     = w/2*(1+self.rho*sk*k+a)
            dwdT   = dw0dT(T)*(0.5*(1+self.rho*sk*k+a)+w/2*(self.rho*dskdw*k+(sk*k+self.rho)*dskdw*k/a))
            dwdk   = w/2*(self.rho*sk+(sk*k+self.rho)*sk/a)
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
    def __init__(self, K, T, pc, ex, px=None, ivEu=None, ivAm=None, lvEu=None):
        self.K        = K    # strike
        self.T        = T    # expiry
        self.pc       = pc   # P or C
        self.ex       = ex   # E or A
        self.px       = px   # price
        self.pxFunc   = None # price as function of spot
        self.ivEu     = ivEu # European vol (curved local-vol)
        self.ivAm     = ivAm # de-Americanized vol (flat local-vol)
        self.lvEu     = lvEu # local-vol at (K,T)
        self.exBdryEu = None # ex-boundary under local-vol
        self.exBdryAm = None # ex-boundary under flat local-vol
        self.pxGridEu = None # price-grid under local-vol
        self.pxGridAm = None # price-grid under flat local-vol
        self.gridX    = None # space-grid in log-spot
        self.gridS    = None # space-grid in spot
        self.gridT    = None # time-grid in time-to-expiry

    def __repr__(self):
        return f'Option(K={self.K}, T={self.T}, pc={self.pc}, ex={self.ex}, ivEu={self.ivEu}, ivAm={self.ivAm}, lvEu={self.lvEu})'

    def Payoff(self, S):
        return np.maximum((self.K-S) if self.pc=='P' else (S-self.K),0)

class Spot:
    def __init__(self, S0, r, q, vs, div=None):
        self.S0  = S0  # initial spot
        self.r   = r   # risk-free rate
        self.q   = q   # dividend yield
        self.vs  = vs  # vol surface
        self.div = div # TODO: discrete dividend, use pd.Series? relative/absolute?

    def __repr__(self):
        return f'Spot(S0={self.S0}, r={self.r}, q={self.q}, div={self.div}, vs={self.vs})'

    def ForwardFunc(self, T):
        return np.exp((self.r-self.q)*T)

class LatticeConfig:
    def __init__(self, S0, nX, nT, rangeX, rangeT, centerValue, center='strike', scheme='implicit', invMethod='splu', boundary='value', interpBdry=False):
        self.S0          = S0          # initial spot
        self.nX          = nX          # number of space-grids
        self.nT          = nT          # number of time-grids
        self.rangeX      = rangeX      # range of space-grid
        self.rangeT      = rangeT      # range of time-grid
        self.centerValue = centerValue # grid-center in spot (S0 or K)
        self.center      = center      # grid-center (spot or strike)
        self.scheme      = scheme      # PDE solver scheme (explicit, implicit or crank-nicolson)
        self.invMethod   = invMethod   # sparse matrix inversion method
        self.boundary    = boundary    # PDE grid boundary method (value or gamma)
        self.interpBdry  = interpBdry  # TODO: interp early ex-boundary
        self.SetRangeS()
        self.AdjustRangeX()
        self.dX = (rangeX[1]-rangeX[0])/nX # space-grid size
        self.dT = (rangeT[1]-rangeT[0])/nT # time-grid size
        self.gridX = np.linspace(rangeX[0],rangeX[1],nX+1) # space-grid in log-spot
        self.gridT = np.linspace(rangeT[0],rangeT[1],nT+1) # time-grid in time-to-expiry

    def __repr__(self):
        return f'LatticeConfig(S0={self.S0}, nX={self.nX}, nT={self.nT}, rangeX={self.rangeX}, rangeT={self.rangeT}, centerValue={self.centerValue}, center={self.center}, scheme={self.scheme}, invMethod={self.invMethod}, boundary={self.boundary}, interpBdry={self.interpBdry})'

    def XToS(self, X):
        return self.centerValue*np.exp(X)

    def SToX(self, S):
        return np.log(S/self.centerValue)

    def SetRangeS(self):
        self.rangeS = (self.XToS(self.rangeX[0]),self.XToS(self.rangeX[1]))

    def AdjustRangeX(self):
        x0 = self.SToX(self.S0)
        self.rangeX[0] = np.minimum(x0,self.rangeX[0])
        self.rangeX[1] = np.maximum(x0,self.rangeX[1])

class LatticePricer:
    def __init__(self, spot):
        self.spot   = spot # true spot
        self.spotAm = None # auxiliary spot used in de-Americanization

    def __repr__(self):
        return f'LatticePricer(spot={self.spot})'

    def SolveLattice(self, option, config, isImpliedVolCalc=False):
        # Solve PDE lattice for option price
        spot = self.spotAm if isImpliedVolCalc else self.spot
        K  = option.K
        T  = option.T
        r  = spot.r
        q  = spot.q
        S0 = spot.S0
        x  = config.gridX
        t  = config.gridT
        dx = config.dX
        dt = config.dT
        S  = config.XToS(x)
        x0 = config.SToX(S0)
        xx,tt  = np.meshgrid(x,t)
        varL   = spot.vs.LVarFunc(xx,np.maximum(T-tt,MIN_TTX))
        varL   = np.minimum(varL,MAX_LVAR)
        pxGrid = np.zeros((len(t),len(x)))
        pxGrid[0] = option.Payoff(S)
        exBdry = np.concatenate([[K],[config.rangeS[0] if option.pc=='P' else config.rangeS[1]]*(len(t)-1)])
        if config.boundary == 'value':
            if option.pc == 'P':
                pxGrid[:,0]  = K*np.exp(-r*t)-S[0]*np.exp(-q*t)
                pxGrid[:,-1] = 0
            else:
                pxGrid[:,0]  = 0
                pxGrid[:,-1] = S[-1]*np.exp(-q*t)-K*np.exp(-r*t)
            V0 = np.zeros(len(x)-2) # offset in implicit scheme
            for i in range(len(t)-1): # forward in time-to-expiry
                if config.scheme == 'explicit':
                    v = varL[i,1:-1]
                    a = -(r-q-v/2)*dt/(2*dx)+v*dt/(2*dx**2)
                    b = 1-r*dt-v*dt/dx**2
                    c = (r-q-v/2)*dt/(2*dx)+v*dt/(2*dx**2)
                    D = diags([a[1:],b,c[:-1]],[-1,0,1]).tocsc()
                    pxGrid[i+1,1:-1] = D@pxGrid[i,1:-1]
                    pxGrid[i+1,1]   += a[0]*pxGrid[i,0]
                    pxGrid[i+1,-2]  += c[-1]*pxGrid[i,-1]
                elif config.scheme == 'implicit':
                    v = varL[i+1,1:-1]
                    a = (r-q-v/2)*dt/(2*dx)-v*dt/(2*dx**2)
                    b = 1+r*dt+v*dt/dx**2
                    c = -(r-q-v/2)*dt/(2*dx)-v*dt/(2*dx**2)
                    D = diags([a[1:],b,c[:-1]],[-1,0,1]).tocsc()
                    V0[0]  = a[0]*pxGrid[i+1,0]
                    V0[-1] = c[-1]*pxGrid[i+1,-1]
                    if config.invMethod == 'splu':
                        pxGrid[i+1,1:-1] = splu(D).solve(pxGrid[i,1:-1]-V0)
                    else:
                        pxGrid[i+1,1:-1] = spsolve(D,pxGrid[i,1:-1]-V0)
                elif config.scheme == 'crank-nicolson':
                    # TODO: Crank-Nicolson scheme
                    pass
                if option.ex == 'A':
                    intrinsic = option.Payoff(S)
                    if option.pc == 'P':
                        idxEx = np.argmax(intrinsic<pxGrid[i+1])
                        exBdry[i+1] = S[idxEx]
                        pxGrid[i+1][:idxEx] = intrinsic[:idxEx]
                    else:
                        idxEx = np.argmax(intrinsic>pxGrid[i+1])
                        exBdry[i+1] = S[idxEx]
                        pxGrid[i+1][idxEx:] = intrinsic[idxEx:]
        elif config.boundary == 'gamma':
            # TODO: zero-gamma boundary
            pass
        pxFunc = interp1d(x,pxGrid[-1],kind='cubic')
        px     = pxFunc(x0)
        if isImpliedVolCalc:
            option.ivAm     = spot.vs.IVolFunc(x0,T)
            option.exBdryAm = exBdry
            option.pxGridAm = pxGrid
        else:
            pxGrid = pd.DataFrame(pxGrid,index=t,columns=x)
            exBdry = pd.Series(exBdry,index=t)
            option.px       = px
            option.pxFunc   = pxFunc
            option.ivEu     = spot.vs.IVolFunc(x0,T)
            option.lvEu     = spot.vs.LVolFunc(x0,T)
            option.exBdryEu = exBdry
            option.pxGridEu = pxGrid
            option.gridX    = x
            option.gridS    = S
            option.gridT    = t
        return px

    def DeAmericanize(self, option, config):
        # TODO: De-Americanize option by flat local-vol assumption
        K  = option.K
        T  = option.T
        S0 = self.spot.S0
        x0 = config.SToX(S0)
        px = option.px
        sig0 = self.spot.vs.IVolFunc(x0,T)
        self.spotAm = copy(self.spot)
        def loss(sig):
            self.spotAm.vs = FlatVol(sig)
            return self.SolveLattice(option,config,True)-px
        print(loss(sig0))

#### Helper Functions ##########################################################

def VolSurfaceMatrixToDataFrame(m, k, T, idx_name='Expiry', col_name='Log-strike'):
    # Cast vol surface matrix to pd.DataFrame
    df = pd.DataFrame(m,index=T,columns=k)
    df.index.name = idx_name
    df.columns.name = col_name
    return df

def PlotImpliedVol(df, dfLVol=None, figname=None, title=None, ncol=7, scatter=False, atmBar=False, xlim=None, ylim=None):
    # Plot implied volatilities based on df
    # df: Columns -- Log-strike, Index -- Expiry
    # TODO: add legend -- implied/local vol
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
            if scatter:
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
                ax_idx.set_xlim(xlim)
            if ylim is not None:
                ax_idx.set_ylim(ylim)
        else:
            ax_idx.axis('off')

    fig.tight_layout()
    plt.savefig(figname)
    plt.close()

def PlotPxGrid(df, T, figname=None, title=None, ncol=7, scatter=False, atmBar=False, xlim=None, ylim=None):
    # TODO: Plot option prices based on df
    # df: Columns -- Spot, Index -- time-to-expiry
    if not figname:
        figname = 'optionpx.png'
    pass
