import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pricer import *
plt.switch_backend("Agg")

def test_BlackScholesImpVol():
    vol = np.array([0.23,0.20,0.18])
    strike = np.array([0.9,1.0,1.1])
    price = BlackScholesFormulaCall(1,strike,1,0,vol)
    impVol = BlackScholesImpliedVolCall(1,strike,1,0,price)
    print(impVol)

def test_HestonSmile():
    paramsBCC = {"meanRevRate": 1.15, "correlation": -0.64, "volOfVol": 0.39, "meanVol": 0.04, "currentVol": 0.04}
    vol = lambda logStrikes: np.array([CharFuncImpliedVol(HestonCharFunc(**paramsBCC))(k,1) for k in logStrikes]).reshape(-1)
    k = np.arange(-0.4,0.4,0.02)
    iv = vol(k)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(k, 100*iv, c='k', s=5)
    plt.title("Heston 1-Year Smile (BCC Params)")
    plt.xlabel("log-strike")
    plt.ylabel("implied vol (%)")
    fig.tight_layout()
    plt.savefig("test_HestonSmileBCC.png")
    plt.close()

if __name__ == '__main__':
    # test_BlackScholesImpVol()
    test_HestonSmile()
