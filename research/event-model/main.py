import numpy as np
import matplotlib.pyplot as plt
from event import *

dataFolder = "plt/"

paramsBS = {"vol": 0.2}
paramsBCC = {"meanRevRate": 1.15, "correlation": -0.64, "volOfVol": 0.39, "meanVar": 0.04, "currentVar": 0.04}
paramsPEJ = {"eventTime": 0, "jumpProb": 0.5, "jump": 0.02}

def test_FlatSmileEventVarBumpAccuracy():
    N = 2
    S = 1
    T = 1
    k = np.arange(-1,1,0.01)
    K = np.exp(k)
    impVolFunc = CharFuncImpliedVol(PointEventJumpCharFunc(BlackScholesCharFunc(**paramsBS),**paramsPEJ),FFT=True)
    impVolFunc0 = CharFuncImpliedVol(BlackScholesCharFunc(**paramsBS),FFT=True)
    iv = impVolFunc(k,T)
    iv0 = impVolFunc0(k,T)
    w = iv**2
    w0 = iv0**2
    dw = EventVarianceBump2ndOrder(k,w0,PointEventJumpMoment,N,**paramsPEJ)
    fig = plt.figure(figsize=(6,4))
    plt.plot(k,w-w0,'k',lw=5,label='Exact')
    plt.plot(k,dw,'r--',lw=5,label='Approx')
    plt.title(f"Accuracy of $\Delta w$ expansion at $p={paramsPEJ['jumpProb']},\epsilon={paramsPEJ['jump']}$")
    plt.xlabel("log-strike")
    plt.ylabel("event var")
    plt.legend()
    fig.tight_layout()
    plt.savefig(dataFolder+"test_FlatSmileEventVarBumpAccuracy.png")
    plt.close()

def test_EventVarBumpAccuracy():
    N = 2
    S = 1
    T = 1
    k = np.arange(-1,1,0.01)
    K = np.exp(k)
    impVolFunc = CharFuncImpliedVol(PointEventJumpCharFunc(HestonCharFunc(**paramsBCC),**paramsPEJ),FFT=True)
    impVolFunc0 = CharFuncImpliedVol(HestonCharFunc(**paramsBCC),FFT=True)
    iv = impVolFunc(k,T)
    iv0 = impVolFunc0(k,T)
    w = iv**2*T
    w0 = iv0**2*T
    dw = EventVarianceBump2ndOrder(k,w0,PointEventJumpMoment,N,**paramsPEJ)
    fig = plt.figure(figsize=(6,4))
    plt.plot(k,w-w0,'k',lw=5,label='Exact')
    plt.plot(k,dw,'r--',lw=5,label='Approx')
    plt.title(f"Accuracy of $\Delta w$ expansion at $p={paramsPEJ['jumpProb']},\epsilon={paramsPEJ['jump']}$")
    plt.xlabel("log-strike")
    plt.ylabel("event var")
    plt.legend()
    fig.tight_layout()
    plt.savefig(dataFolder+"test_EventVarBumpAccuracy.png")
    plt.close()

def test_EventVarBumpAccuracyForVariousJumpProbs():
    pass

def test_EventVarBumpAccuracyForVariousJumpSizes():
    pass

if __name__ == '__main__':
    # test_FlatSmileEventVarBumpAccuracy()
    test_EventVarBumpAccuracy()
