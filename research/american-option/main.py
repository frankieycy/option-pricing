import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from american import *
plt.switch_backend('Agg')

# SVI_PARAMS = {'v0': 0.04, 'v1': 0.04, 'v2':0.04, 'k1': 1, 'k2': 2, 'rho': 0, 'eta': 0, 'gam': 1}
SVI_PARAMS = {'v0': 0.09, 'v1': 0.04, 'v2':0.04, 'k1': 5, 'k2': 0.1, 'rho': -0.5, 'eta': 1, 'gam': 0.5}

def test_SviPowerLaw():
    k = np.arange(-1,1.05,0.05)
    T = np.arange(0.1,1.05,0.05)
    kk,TT = np.meshgrid(k,T)
    svi = SviPowerLaw(**SVI_PARAMS)
    sigI = svi.IVolFunc(kk,TT)
    sigL = svi.LVolFunc(kk,TT)
    sigI = VolSurfaceMatrixToDataFrame(sigI,k,T)
    sigL = VolSurfaceMatrixToDataFrame(sigL,k,T)
    print(sigI)
    print(sigL)
    PlotImpliedVol(sigI,sigL,figname='out/impliedvol.png')

def test_LatticePricer():
    svi = SviPowerLaw(**SVI_PARAMS)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0.05
    nX = 200
    nT = 200
    x0 = -2
    x1 = 2
    O = Option(K,T,'P','A')
    S = Spot(S0,r,0,svi.IVolFunc,svi.LVolFunc,svi.LVarFunc)
    C = LatticeConfig(S0,nX,nT,[x0,x1],[0,T],K,scheme='implicit')
    L = LatticePricer(S)
    L.SolveLattice(O,C)
    print(O.px)
    print(O.ivEu)
    print(O.exBdryEu)
    print(O.pxGridEu)

if __name__ == '__main__':
    # test_SviPowerLaw()
    test_LatticePricer()
