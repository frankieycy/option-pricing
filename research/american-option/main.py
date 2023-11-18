import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from american import *
plt.switch_backend('Agg')

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
    PlotImpliedVol(sigI, sigL)

def test_LatticeAmerican():
    svi = SviPowerLaw(**SVI_PARAMS)
    O = Option(1,1,'P','A')
    S = Spot(1,0.05,0,svi.IVolFunc,svi.LVolFunc)
    C = LatticeConfig(200,200,(-2,2),(0,1))
    A = LatticeAmerican([O],S,C)

if __name__ == '__main__':
    # test_SviPowerLaw()
    test_LatticeAmerican()
