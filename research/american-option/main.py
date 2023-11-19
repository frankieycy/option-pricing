import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from american import *
plt.switch_backend('Agg')

FLAT_VOL        = {'sig': 0.2}
SVI_PARAMS_FLAT = {'v0': 0.04, 'v1': 0.04, 'v2':0.04, 'k1': 1, 'k2': 2,   'rho': 0,    'eta': 0, 'gam': 0  }
SVI_PARAMS_SPX  = {'v0': 0.09, 'v1': 0.04, 'v2':0.04, 'k1': 5, 'k2': 0.1, 'rho': -0.5, 'eta': 1, 'gam': 0.5}

def test_SviPowerLaw():
    k = np.arange(-1,1.05,0.05)
    T = np.arange(0.1,1.05,0.05)
    kk,TT = np.meshgrid(k,T)
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    sigI = svi.IVolFunc(kk,TT)
    sigL = svi.LVolFunc(kk,TT)
    sigI = VolSurfaceMatrixToDataFrame(sigI,k,T)
    sigL = VolSurfaceMatrixToDataFrame(sigL,k,T)
    print('sigI')
    print(sigI)
    print('sigL')
    print(sigL)
    PlotImpliedVol(sigI,sigL,xlim=(-1,1),ylim=(12,100),figname='out/impliedvol.png')

def test_LatticePricer():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0.05
    nX = 200
    nT = 200
    x0 = -2
    x1 = 2
    O = Option(K,T,'P','A')
    S = Spot(S0,r,0,svi)
    C = LatticeConfig(S0,nX,nT,[x0,x1],[0,T],K,scheme='implicit')
    L = LatticePricer(S)
    L.SolveLattice(O,C)
    print(O)
    print(S)
    print(C)
    print(L)
    print('O.exBdryEu')
    print(O.exBdryEu)
    print('O.pxGridEu')
    print(O.pxGridEu)

def test_DeAmericanize():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0.05
    nX = 200
    nT = 200
    x0 = -2
    x1 = 2
    O = Option(K,T,'P','A')
    S = Spot(S0,r,0,svi)
    C = LatticeConfig(S0,nX,nT,[x0,x1],[0,T],K,scheme='implicit')
    L = LatticePricer(S)
    L.SolveLattice(O,C)
    L.DeAmericanize(O,C)

if __name__ == '__main__':
    # test_SviPowerLaw()
    # test_LatticePricer()
    test_DeAmericanize()
