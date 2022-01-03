import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pricer import *
plt.switch_backend("Agg")

paramsBCC = {"meanRevRate": 1.15, "correlation": -0.64, "volOfVol": 0.39, "meanVol": 0.04, "currentVol": 0.04}

def test_BlackScholesImpVol():
    vol = np.array([0.23,0.20,0.18])
    strike = np.array([0.9,1.0,1.1])
    price = BlackScholesFormulaCall(1,strike,1,0,vol)
    impVol = BlackScholesImpliedVolCall(1,strike,1,0,price)
    print(impVol)

def test_HestonSmile():
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

def test_HestonSmileSensitivity():
    k = np.arange(-0.4,0.4,0.02)
    var = ["rho","eta","lambda"]
    png = ["rho","eta","lam"]
    bcc = ["correlation","volOfVol","meanRevRate"]
    inc = [0.1,0.1,0.5]
    for j in range(3):
        paramsBCCnew = paramsBCC.copy()
        fig = plt.figure(figsize=(6,4))
        plt.title(rf"Heston 1-Year Smile $\{var[j]}$ Sensitivity (BCC Params)")
        plt.xlabel("log-strike")
        plt.ylabel("implied vol (%)")
        for i in range(5):
            vol = lambda logStrikes: np.array([CharFuncImpliedVol(HestonCharFunc(**paramsBCCnew))(k,1) for k in logStrikes]).reshape(-1)
            iv = vol(k)
            c = 'k' if i == 0 else 'k--'
            plt.plot(k, 100*iv, c)
            paramsBCCnew[bcc[j]] += inc[j]
        plt.ylim(10,30)
        fig.tight_layout()
        plt.savefig(f"test_HestonSmileBCC_{png[j]}.png")
        plt.close()

def test_plotImpliedVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    plotImpliedVol(df, "test_impliedvol.png")

def test_VarianceSwapFormula():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    T = Texp[2]
    dfT = df[df["Texp"]==T]
    k = np.log(dfT["Strike"]/dfT["Fwd"])
    bid = dfT["Bid"]
    ask = dfT["Ask"]
    mid = (bid+ask)/2
    price = VarianceSwapFormula(k,T,mid,showPlot=True)/T
    print(price)

def test_calcSwapCurve():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    curveVS = calcSwapCurve(df,VarianceSwapFormula)
    curveGS = calcSwapCurve(df,GammaSwapFormula)
    vsMid = curveVS["mid"]
    gsMid = curveGS["mid"]
    print(curveVS)
    print(curveGS)
    fig = plt.figure(figsize=(6,4))
    # plt.scatter(Texp, vsMid, c='r', s=5, label="variance swap")
    # plt.scatter(Texp, gsMid, c='b', s=5, label="gamma swap")
    plt.plot(Texp, vsMid, c='r', label="variance swap")
    plt.plot(Texp, gsMid, c='b', label="gamma swap")
    plt.title("Swap Curve (SPX 20170424)")
    plt.xlabel("maturity")
    plt.ylabel("swap price")
    plt.legend()
    fig.tight_layout()
    plt.savefig("test_SwapCurve.png")
    plt.close()

def test_LevSwapCurve():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    curveLS = calcSwapCurve(df,LeverageSwapFormula)
    lsMid = curveLS["mid"]
    print(curveLS)
    fig = plt.figure(figsize=(6,4))
    plt.plot(Texp, lsMid, c='k', label="leverage swap")
    plt.title("Leverage Swap Curve (SPX 20170424)")
    plt.xlabel("maturity")
    plt.ylabel("swap price")
    plt.legend()
    fig.tight_layout()
    plt.savefig("test_LevSwapCurve.png")
    plt.close()

def test_calcFwdVarCurve():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    curveVS = calcSwapCurve(df,VarianceSwapFormula)
    curveFV = calcFwdVarCurve(curveVS)
    fvMid = curveFV["mid"]
    fvFunc = FwdVarCurveFunc(Texp,fvMid)
    T = np.linspace(0,5,500)
    print(curveFV)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(Texp, fvMid, c='k', s=5)
    plt.plot(T, [fvFunc(t) for t in T], 'k--', lw=0.5)
    plt.title("Forward Variance Curve (SPX 20170424)")
    plt.xlabel("maturity")
    plt.ylabel("forward variance")
    fig.tight_layout()
    plt.savefig("test_FwdVarCurve.png")
    plt.close()

if __name__ == '__main__':
    # test_BlackScholesImpVol()
    # test_HestonSmile()
    # test_HestonSmileSensitivity()
    # test_plotImpliedVol()
    # test_VarianceSwapFormula()
    # test_calcSwapCurve()
    # test_LevSwapCurve()
    test_calcFwdVarCurve()
