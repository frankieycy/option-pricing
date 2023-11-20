import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
from copy import copy
from scipy.special import ndtr
from scipy.sparse import diags
from scipy.sparse.linalg import splu, spsolve
from scipy.optimize import fsolve
from scipy.interpolate import pchip, interp1d
plt.switch_backend('Agg')

#### Note ######################################################################
# 1. Attributes in all classes are freely accessible, i.e. no encapsulation (getter/setter).
# 2. The study requires fast and accurate PDE pricing of American option. Test PDE solver first.

#### Consts ####################################################################

MIN_TTX  = 1e-6         # minimum time-to-expiry to avoid blow-up in SviPowerLaw
MAX_LVAR = 10           # maximum local-var to avoid blow-up in SolveLattice

#### Vol Surface ###############################################################

def BlackScholesPrice(sig,K,T,D,F,pc):
    k = np.log(K/F)
    v = sig*np.sqrt(T)
    d1 = -k/v+v/2
    d2 = d1-v
    if pc == 'P':
        return D*(K*ndtr(-d2)-F*ndtr(-d1))
    else:
        return D*(F*ndtr(d1)-K*ndtr(d2))

def BlackScholesImpliedVol(px,K,T,D,F,pc,sig0):
    def obj(sig):
        return BlackScholesPrice(sig,K,T,D,F,pc)-px
    return fsolve(obj,sig0)

class VolSurface:
    def __init__(self):
        self.IVolFunc = self.ImpliedVolFunc() # implied-vol function
        self.LVolFunc = self.LocalVolFunc()   # local-vol function
        self.LVarFunc = self.LocalVarFunc()   # local-var function

    def __repr__(self):
        return f'VolSurface()'

    def ImpliedVolFunc(self):
        return np.vectorize(lambda k,T: 0)

    def LocalVolFunc(self):
        return np.vectorize(lambda k,T: 0)

    def LocalVarFunc(self):
        return np.vectorize(lambda k,T: 0)

class FlatVol(VolSurface):
    def __init__(self, sig):
        VolSurface.__init__(self)
        self.sig = sig

    def __repr__(self):
        return f'FlatVol(sig={self.sig})'

    def ImpliedVolFunc(self):
        return np.vectorize(lambda k,T: self.sig)

    def LocalVolFunc(self):
        return np.vectorize(lambda k,T: self.sig)

    def LocalVarFunc(self):
        return np.vectorize(lambda k,T: self.sig**2)

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
    def __init__(self, K, T, pc, ex, px=None, ivLV=None, ivFV=None, lvLV=None):
        self.K        = K    # strike
        self.T        = T    # expiry
        self.pc       = pc   # P or C
        self.ex       = ex   # E or A
        self.px       = px   # price
        self.pxFunc   = None # price as function of spot
        self.ivLV     = ivLV # European vol (curved local-vol)
        self.ivFV     = ivFV # de-Americanized vol (flat local-vol)
        self.lvLV     = lvLV # local-vol at (K,T)
        self.exBdryLV = None # ex-boundary under local-vol
        self.exBdryFV = None # ex-boundary under flat local-vol
        self.pxGridLV = None # price-grid under local-vol
        self.pxGridFV = None # price-grid under flat local-vol
        self.gridX    = None # space-grid in log-spot
        self.gridS    = None # space-grid in spot
        self.gridT    = None # time-grid in time-to-expiry

    def __repr__(self):
        return f'Option(K={self.K}, T={self.T}, pc={self.pc}, ex={self.ex}, px={self.px}, ivLV={self.ivLV}, ivFV={self.ivFV}, lvLV={self.lvLV})'

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
    def __init__(self, S0, scheme='implicit', invMethod='splu', boundary='value', interpBdry=False):
        self.S0          = S0          # initial spot
        self.scheme      = scheme      # PDE solver scheme (explicit, implicit or crank-nicolson)
        self.invMethod   = invMethod   # sparse matrix inversion method
        self.boundary    = boundary    # PDE grid boundary method (value or gamma)
        self.interpBdry  = interpBdry  # TODO: interp early ex-boundary
        self.nX          = None        # number of space-grids
        self.nT          = None        # number of time-grids
        self.rangeX      = None        # range of space-grid
        self.rangeT      = None        # range of time-grid
        self.centerValue = None        # grid-center in spot (S0 or K)
        self.center      = None        # grid-center (spot or strike)

    def __repr__(self):
        return f'LatticeConfig(S0={self.S0}, nX={self.nX}, nT={self.nT}, rangeX={self.rangeX}, rangeT={self.rangeT}, centerValue={self.centerValue}, center={self.center}, scheme={self.scheme}, invMethod={self.invMethod}, boundary={self.boundary}, interpBdry={self.interpBdry})'

    def initGrid(self, nX, nT, rangeX, rangeT, centerValue, center='strike'):
        self.nX          = nX
        self.nT          = nT
        self.rangeX      = rangeX
        self.rangeT      = rangeT
        self.centerValue = centerValue
        self.center      = center
        self.AdjustRangeX()
        self.SetRangeS()
        self.SetGridParams()

    def XToS(self, X):
        return self.centerValue*np.exp(X)

    def SToX(self, S):
        return np.log(S/self.centerValue)

    def AdjustRangeX(self):
        xmin = self.SToX(0.95*self.S0)
        xmax = self.SToX(1.05*self.S0)
        self.rangeX[0] = np.minimum(xmin,self.rangeX[0])
        self.rangeX[1] = np.maximum(xmax,self.rangeX[1])

    def SetRangeS(self):
        self.rangeS = (self.XToS(self.rangeX[0]),self.XToS(self.rangeX[1]))

    def SetGridParams(self):
        self.dX = (self.rangeX[1]-self.rangeX[0])/self.nX # space-grid size
        self.dT = (self.rangeT[1]-self.rangeT[0])/self.nT # time-grid size
        self.gridX = np.linspace(self.rangeX[0],self.rangeX[1],self.nX+1) # space-grid in log-spot
        self.gridT = np.linspace(self.rangeT[0],self.rangeT[1],self.nT+1) # time-grid in time-to-expiry

class LatticePricer:
    def __init__(self, spot):
        self.spot   = spot # true spot
        self.spotFV = None # auxiliary spot used in de-Americanization

    def __repr__(self):
        return f'LatticePricer(spot={self.spot})'

    def SolveLattice(self, option, config, isImpliedVolCalc=False):
        # Solve PDE lattice for option price
        # TODO: speed profiling
        #### 1. Grid initialization
        spot = self.spotFV if isImpliedVolCalc else self.spot
        K    = option.K
        T    = option.T
        r    = spot.r
        q    = spot.q
        S0   = spot.S0
        x    = config.gridX
        t    = config.gridT
        dx   = config.dX
        dt   = config.dT
        S    = config.XToS(x)
        k    = config.SToX(K)
        x0   = config.SToX(S0)
        F    = spot.ForwardFunc(T-t)
        xF   = config.SToX(F)
        xF0  = xF[0]
        xx,tt  = np.meshgrid(x,t)
        varL   = spot.vs.LVarFunc(xx-xF[:,None],np.maximum(T-tt,MIN_TTX)) # local-var fetched at log(S/F)=log(S/C)-log(F/C)=xx-xF
        varL   = np.minimum(varL,MAX_LVAR)
        pxGrid = np.zeros((len(t),len(x)))
        intrinsic = option.Payoff(S)
        pxGrid[0] = intrinsic
        exBdry = np.concatenate([[K],[config.rangeS[0] if option.pc=='P' else config.rangeS[1]]*(len(t)-1)])
        #### 2. Forward iteration
        if config.boundary == 'value':
            if option.pc == 'P':
                pxGrid[:,0]  = K*np.exp(-r*t)-S[0]*np.exp(-q*t) if option.ex=='E' else intrinsic[0]
                pxGrid[:,-1] = 0
            else:
                pxGrid[:,0]  = 0
                pxGrid[:,-1] = S[-1]*np.exp(-q*t)-K*np.exp(-r*t) if option.ex=='E' else intrinsic[-1]
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
                    if option.pc == 'P':
                        idxEx = np.argmax(intrinsic<pxGrid[i+1])
                    else:
                        idxEx = np.argmax(intrinsic>pxGrid[i+1])
                        idxEx = -1 if idxEx==0 else idxEx
                    exBdry[i+1] = S[idxEx]
                    pxGrid[i+1] = np.maximum(intrinsic,pxGrid[i+1])
        elif config.boundary == 'gamma':
            # TODO: zero-gamma boundary
            pass
        #### 3. Post-processing
        # pxFunc = interp1d(x,pxGrid[-1],kind='cubic')
        pxFunc = pchip(x,pxGrid[-1])
        px     = pxFunc(x0)
        pxGrid = pd.DataFrame(pxGrid,index=t,columns=x)
        exBdry = pd.Series(exBdry,index=t)
        # implied/local-vol fetched at log(K/F)=log(K/C)-log(F/C)=k-xF0
        if isImpliedVolCalc:
            option.ivFV     = spot.vs.IVolFunc(k-xF0,T)
            option.exBdryFV = exBdry
            option.pxGridFV = pxGrid
        else:
            option.px       = px
            option.pxFunc   = pxFunc
            option.ivLV     = spot.vs.IVolFunc(k-xF0,T)
            option.lvLV     = spot.vs.LVolFunc(k-xF0,T)
            option.exBdryLV = exBdry
            option.pxGridLV = pxGrid
            option.gridX    = x
            option.gridS    = S
            option.gridT    = t
        return px

    def DeAmericanize(self, option, config):
        # De-Americanize option by flat local-vol assumption
        K    = option.K
        T    = option.T
        F    = self.spot.ForwardFunc(T)
        k    = config.SToX(K)
        xF0  = config.SToX(F)
        px   = option.px
        sig0 = self.spot.vs.IVolFunc(k-xF0,T)
        self.spotFV = copy(self.spot)
        def obj(sig):
            self.spotFV.vs = FlatVol(sig)
            return self.SolveLattice(option,config,True)-px
        obj = np.vectorize(obj)
        ivFV = fsolve(obj,sig0)
        return ivFV

class AmericanVolSurface(VolSurface):
    def __init__(self, spot, config, nX=200, nT=200, rangeX=[-2,2]):
        VolSurface.__init__(self)
        self.nX     = nX                   # number of space-grids
        self.nT     = nT                   # number of time-grids
        self.rangeX = rangeX               # range of space-grid
        self.spot   = spot                 # spot for fetching true European vols
        self.config = config               # template config with K,T subject to modification
        self.latt   = LatticePricer(spot)  # American lattice pricer
        self.log    = []                   # log for computed options

    def __repr__(self):
        return f'AmericanVolSurface(spot={self.spot})'

    def clearLog(self):
        self.log = []

    def getConfig(self, K, T):
        config = copy(self.config)
        config.initGrid(self.nX,self.nT,self.rangeX,[0,T],K,'strike')
        return config

    def ImpliedVolFunc(self):
        def amIVolFunc(k, T):
            F  = self.spot.ForwardFunc(T)
            K  = F*np.exp(k)
            pc = 'P' if k<=0 else 'C'
            O  = Option(K,T,pc,'A')
            C  = self.getConfig(K,T)
            L  = self.latt
            L.SolveLattice(O,C)
            L.DeAmericanize(O,C)
            self.log.append(O)
            return O.ivFV
        return np.vectorize(amIVolFunc,otypes=[float])

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
