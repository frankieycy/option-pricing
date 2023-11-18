import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from american import *

def test_SviPowerLaw():
    k = np.arange(-1,1.1,0.1)
    T = np.arange(0.1,1.1,0.1)
    kk,TT = np.meshgrid(k,T)
    svi = SviPowerLaw(**SVI_PARAMS)
    w = svi(kk,TT)
    sig = VolSurfaceMatrixToDataFrame(np.sqrt(w/TT),k,T)
    print('Implied Vol Surface')
    print(sig)

def test_SviPowerLawLVol():
    k = np.arange(-1,1.1,0.1)
    T = np.arange(0.1,1.1,0.1)
    kk,TT = np.meshgrid(k,T)
    sviLVar = SviPowerLawLVol(**SVI_PARAMS)
    v = sviLVar(kk,TT)
    sigL = VolSurfaceMatrixToDataFrame(np.sqrt(v),k,T)
    print('Local Vol Surface')
    print(sigL)

if __name__ == '__main__':
    test_SviPowerLaw()
    test_SviPowerLawLVol()
