import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pricer import *
plt.switch_backend("Agg")

paramsBCC = {"meanRevRate": 1.15, "correlation": -0.64, "volOfVol": 0.39, "meanVar": 0.04, "currentVar": 0.04}
paramsBCCkey = list(paramsBCC.keys())
paramsBCCval = list(paramsBCC.values())
paramsBCCBnd = ((0,10), (-1,1), (0,10), (0.01,1), (0,1))

paramsMER = {"vol": 0.1, "jumpInt": 0.1, "jumpMean": -0.4, "jumpSd": 0.2}
paramsMERkey = list(paramsMER.keys())
paramsMERval = list(paramsMER.values())
paramsMERBnd = ((0,1), (0,10), (-1,1), (0,10))

dataFolder = "test/"

#### Imp Vol ###################################################################

def test_BlackScholesImpVol():
    vol = np.array([0.23,0.20,0.18])
    strike = np.array([0.9,1.0,1.1])
    price = BlackScholesFormula(1,strike,1,0,vol,"call")
    impVol = BlackScholesImpliedVol(1,strike,1,0,price,"call")
    print(impVol)

def test_PlotImpliedVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    PlotImpliedVol(df, dataFolder+"test_impliedvol.png")

#### Fwd Var Curve #############################################################

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

def test_CalcSwapCurve():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    curveVS = CalcSwapCurve(df,VarianceSwapFormula)
    curveGS = CalcSwapCurve(df,GammaSwapFormula)
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
    plt.savefig(dataFolder+"test_SwapCurve.png")
    plt.close()

def test_LevSwapCurve():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    curveLS = CalcSwapCurve(df,LeverageSwapFormula)
    lsMid = curveLS["mid"]
    print(curveLS)
    fig = plt.figure(figsize=(6,4))
    plt.plot(Texp, lsMid, c='k', label="leverage swap")
    plt.title("Leverage Swap Curve (SPX 20170424)")
    plt.xlabel("maturity")
    plt.ylabel("swap price")
    plt.legend()
    fig.tight_layout()
    plt.savefig(dataFolder+"test_LevSwapCurve.png")
    plt.close()

def test_CalcFwdVarCurve():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    curveVS = CalcSwapCurve(df,VarianceSwapFormula)
    curveFV = CalcFwdVarCurve(curveVS)
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
    plt.savefig(dataFolder+"test_FwdVarCurve.png")
    plt.close()

#### Heston ####################################################################

def test_HestonSmile():
    impVolFunc = CharFuncImpliedVol(HestonCharFunc(**paramsBCC))
    vol = lambda K,T: np.array([impVolFunc(k,T) for k in K]).reshape(-1)
    k = np.arange(-0.4,0.4,0.02)
    iv = vol(k,1)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(k, 100*iv, c='k', s=5)
    plt.title("Heston 1-Year Smile (BCC Params)")
    plt.xlabel("log-strike")
    plt.ylabel("implied vol (%)")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_HestonSmileBCC.png")
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
            impVolFunc = CharFuncImpliedVol(HestonCharFunc(**paramsBCCnew))
            vol = lambda K,T: np.array([impVolFunc(k,T) for k in K]).reshape(-1)
            iv = vol(k,1)
            c = 'k' if i == 0 else 'k--'
            plt.plot(k, 100*iv, c)
            paramsBCCnew[bcc[j]] += inc[j]
        plt.ylim(10,30)
        fig.tight_layout()
        plt.savefig(dataFolder+f"test_HestonSmileBCC_{png[j]}.png")
        plt.close()

def test_HestonSmileFFT():
    impVolFunc = CharFuncImpliedVol(HestonCharFunc(**paramsBCC),FFT=True,N=2**14)
    k = np.arange(-0.4,0.4,0.02)
    iv = impVolFunc(k,1)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(k, 100*iv, c='k', s=5)
    plt.title("Heston 1-Year Smile (BCC Params)")
    plt.xlabel("log-strike")
    plt.ylabel("implied vol (%)")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_HestonSmileBCC_FFT.png")
    plt.close()

def test_ShortDatedHestonSmileFFT():
    T = 1e-2
    for B in range(500,6500,500):
        impVolFunc = CharFuncImpliedVol(HestonCharFunc(**paramsBCC),FFT=True,B=B)
        k = np.arange(-0.4,0.4,0.02)
        iv = impVolFunc(k,T)
        fig = plt.figure(figsize=(6,4))
        plt.scatter(k, 100*iv, c='k', s=5)
        plt.title(f"Heston {np.round(T,3)}-Year Smile (BCC Params)")
        plt.xlabel("log-strike")
        plt.ylabel("implied vol (%)")
        fig.tight_layout()
        plt.savefig(dataFolder+f"test_HestonSmileBCC_FFT_T={np.round(T,3)}_B={B}.png")
        plt.close()

def test_HestonSmileFFTForVariousDates():
    # TO-DO
    pass

def test_HestonSmileLewis():
    impVolFunc = LewisCharFuncImpliedVol(HestonCharFunc(**paramsBCC))
    vol = lambda K,T: np.array([impVolFunc(k,T) for k in K]).reshape(-1)
    k = np.arange(-0.4,0.4,0.02)
    iv = vol(k,1)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(k, 100*iv, c='k', s=5)
    plt.title("Heston 1-Year Smile (BCC Params)")
    plt.xlabel("log-strike")
    plt.ylabel("implied vol (%)")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_HestonSmileBCC_Lewis.png")
    plt.close()

def test_CalibrateHestonModelToCallPrice():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    xT = list()
    for T in Texp:
        dfT = df[df["Texp"]==T]
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        mid = (dfT["CallMid"]/dfT["Fwd"]).to_numpy()
        w = 1/(dfT["Ask"]-dfT["Bid"]).to_numpy()
        x = CalibrateModelToOptionPrice(k,T,mid,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCBnd,w=w,optionType="call")
        xT.append([T]+x.tolist())
        print(f"T={np.round(T,3)}", x)
    xT = pd.DataFrame(xT, columns=["Texp"]+paramsBCCkey)
    xT.to_csv(dataFolder+"test_HestonCalibration.csv", index=False)

def test_CalibrateHestonModelToCallPriceNew():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()
    x = CalibrateModelToOptionPrice(k,T,mid,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCBnd,w=w,optionType="call")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsBCCkey)
    x.to_csv(dataFolder+"test_HestonCalibrationNew.csv", index=False)

def test_ImpVolFromHestonCalibration():
    cal = pd.read_csv(dataFolder+"test_HestonCalibration.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        params = cal[cal["Texp"]==T][paramsBCCkey]
        params = params.iloc[0].to_dict()
        impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),FFT=True)
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_HestonImpliedVol.png")

def test_ImpVolFromHestonCalibrationNew():
    cal = pd.read_csv(dataFolder+"test_HestonCalibrationNew.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsBCCkey].iloc[0].to_dict()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),FFT=True)
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_HestonImpliedVolNew.png")

#### Merton ####################################################################

def test_MertonJumpSmile():
    impVolFunc = CharFuncImpliedVol(MertonJumpCharFunc(**paramsMER),FFT=True,N=2**14)
    k = np.arange(-0.4,0.4,0.02)
    iv = impVolFunc(k,1)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(k, 100*iv, c='k', s=5)
    plt.title("Merton 1-Year Smile (MER Params)")
    plt.xlabel("log-strike")
    plt.ylabel("implied vol (%)")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_MertonSmile_FFT.png")
    plt.close()

def test_MertonJumpSmileSensitivity():
    k = np.arange(-0.4,0.4,0.02)
    var = ["mu_J","sigma_J","lambda"]
    png = ["muJ","sigJ","lam"]
    bcc = ["jumpMean","jumpSd","jumpInt"]
    inc = [0.05,0.05,0.1]
    for j in range(3):
        paramsMERnew = paramsMER.copy()
        fig = plt.figure(figsize=(6,4))
        plt.title(rf"Merton 1-Year Smile $\{var[j]}$ Sensitivity (MER Params)")
        plt.xlabel("log-strike")
        plt.ylabel("implied vol (%)")
        for i in range(5):
            impVolFunc = CharFuncImpliedVol(MertonJumpCharFunc(**paramsMERnew))
            vol = lambda K,T: np.array([impVolFunc(k,T) for k in K]).reshape(-1)
            iv = vol(k,1)
            c = 'k' if i == 0 else 'k--'
            plt.plot(k, 100*iv, c)
            paramsMERnew[bcc[j]] += inc[j]
        plt.ylim(10,40)
        fig.tight_layout()
        plt.savefig(dataFolder+f"test_MertonSmile_{png[j]}.png")
        plt.close()

def test_CalibrateMertonJumpModelToCallPrice():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    xT = list()
    for T in Texp:
        dfT = df[df["Texp"]==T]
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        mid = (dfT["CallMid"]/dfT["Fwd"]).to_numpy()
        w = 1/(dfT["Ask"]-dfT["Bid"]).to_numpy()
        x = CalibrateModelToOptionPrice(k,T,mid,MertonJumpCharFunc,paramsMERval,paramsMERkey,bounds=paramsMERBnd,w=w,optionType="call")
        xT.append([T]+x.tolist())
        print(f"T={np.round(T,3)}", x)
    xT = pd.DataFrame(xT, columns=["Texp"]+paramsMERkey)
    xT.to_csv(dataFolder+"test_MertonCalibration.csv", index=False)

def test_CalibrateMertonJumpModelToCallPriceNew():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()
    x = CalibrateModelToOptionPrice(k,T,mid,MertonJumpCharFunc,paramsMERval,paramsMERkey,bounds=paramsMERBnd,w=w,optionType="call")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsMERkey)
    x.to_csv(dataFolder+"test_MertonCalibrationNew.csv", index=False)

def test_ImpVolFromMertonJumpCalibration():
    cal = pd.read_csv(dataFolder+"test_MertonCalibration.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        params = cal[cal["Texp"]==T][paramsMERkey]
        params = params.iloc[0].to_dict()
        impVolFunc = CharFuncImpliedVol(MertonJumpCharFunc(**params),FFT=True)
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_MertonImpliedVol.png")

def test_ImpVolFromMertonJumpCalibrationNew():
    cal = pd.read_csv(dataFolder+"test_MertonCalibrationNew.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsMERkey].iloc[0].to_dict()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        impVolFunc = CharFuncImpliedVol(MertonJumpCharFunc(**params),FFT=True)
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_MertonImpliedVolNew.png")

def test_FitShortDatedMertonSmile():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    T = Texp[2]
    dfT = df[df["Texp"]==T].copy()
    k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
    mid = (dfT["CallMid"]/dfT["Fwd"]).to_numpy()
    w = (k>-0.2)&(k<0.2)
    x = CalibrateModelToOptionPrice(k,T,mid,MertonJumpCharFunc,paramsMERval,paramsMERkey,bounds=paramsMERBnd,w=w,optionType="call")
    print(f"T={np.round(T,3)}", x)
    params = {paramsMERkey[i]: x[i] for i in range(len(x))}
    impVolFunc = CharFuncImpliedVol(MertonJumpCharFunc(**params),FFT=True)
    iv = impVolFunc(k,T)
    dfT["Fit"] = iv
    PlotImpliedVol(dfT, dataFolder+f"test_MertonImpliedVol_T={np.round(T,3)}.png")

if __name__ == '__main__':
    # test_BlackScholesImpVol()
    # test_PlotImpliedVol()
    # test_VarianceSwapFormula()
    # test_CalcSwapCurve()
    # test_LevSwapCurve()
    # test_CalcFwdVarCurve()
    #### Heston ####
    # test_HestonSmile()
    # test_HestonSmileSensitivity()
    # test_HestonSmileFFT()
    # test_ShortDatedHestonSmileFFT()
    test_HestonSmileFFTForVariousDates()
    # test_HestonSmileLewis()
    # test_CalibrateHestonModelToCallPrice()
    # test_CalibrateHestonModelToCallPriceNew()
    # test_ImpVolFromHestonCalibration()
    # test_ImpVolFromHestonCalibrationNew()
    #### Merton ####
    # test_MertonJumpSmile()
    # test_MertonJumpSmileSensitivity()
    # test_CalibrateMertonJumpModelToCallPrice()
    # test_CalibrateMertonJumpModelToCallPriceNew()
    # test_ImpVolFromMertonJumpCalibration()
    # test_ImpVolFromMertonJumpCalibrationNew()
    # test_FitShortDatedMertonSmile()
