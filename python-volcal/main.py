import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pricer import *
from params import *
plt.switch_backend("Agg")

dataFolder = "test/"

#### Imp Vol ###################################################################

def test_BlackScholesImpVol():
    vol = np.array([0.23,0.20,0.18])
    strike = np.array([0.9,1.0,1.1])
    price = BlackScholesFormula(1,strike,1,0,vol,"call")
    impVol = BlackScholesImpliedVol(1,strike,1,0,price,"call")
    print(impVol)

def test_BlackScholesImpVolInterp():
    vol = np.array([0.23,0.20,0.18])
    strike = np.array([0.9,1.0,1.1])
    price = BlackScholesFormula(1,strike,1,0,vol,"call")
    impVol = BlackScholesImpliedVol(1,strike,1,0,price,"call",method="Interp")
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
    impVolFunc = CharFuncImpliedVol(HestonCharFunc(**paramsBCC),FFT=True)
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
    impVolFunc = CharFuncImpliedVol(HestonCharFunc(**paramsBCC),FFT=True)
    for T in np.arange(1,6,1):
        k = np.arange(-0.4,0.4,0.02)
        iv = impVolFunc(k,T)
        fig = plt.figure(figsize=(6,4))
        plt.scatter(k, 100*iv, c='k', s=5)
        plt.title(f"Heston {np.round(T,2)}-Year Smile (BCC Params)")
        plt.xlabel("log-strike")
        plt.ylabel("implied vol (%)")
        fig.tight_layout()
        plt.savefig(dataFolder+f"test_HestonSmileBCC_FFT_T={np.round(T,2)}.png")
        plt.close()

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
        x = CalibrateModelToOptionPrice(k,T,mid,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call")
        xT.append([T]+x.tolist())
        print(f"T={np.round(T,3)}", x)
    xT = pd.DataFrame(xT, columns=["Texp"]+paramsBCCkey)
    xT.to_csv(dataFolder+"test_HestonCalibration.csv", index=False)

def test_CalibrateHestonModelToCallPricePrx():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    x = CalibrateModelToOptionPrice(k,T,mid,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsBCCkey)
    x.to_csv(dataFolder+"test_HestonCalibrationPrx.csv", index=False)

def test_CalibrateHestonModelToImpVol(): # Benchmark!
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    # x = CalibrateModelToImpliedVol(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call")
    # x = CalibrateModelToImpliedVol(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Newton")
    # x = CalibrateModelToImpliedVol(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    x = CalibrateModelToImpliedVol(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Interp",useGlobal=True,curryCharFunc=True)
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsBCCkey)
    x.to_csv(dataFolder+"test_HestonCalibrationIv.csv", index=False)

def test_ImpVolFromHestonCalibration():
    # Wrong implementation: calibrate model to EVERY maturity
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

def test_ImpVolFromHestonCalibrationPrx():
    cal = pd.read_csv(dataFolder+"test_HestonCalibrationPrx.csv")
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
    PlotImpliedVol(dfnew, dataFolder+"test_HestonImpliedVolPrx.png")

def test_ImpVolFromHestonIvCalibration():
    cal = pd.read_csv(dataFolder+"test_HestonCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsBCCkey].iloc[0].to_dict()
    # impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),FFT=True)
    # impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),FFT=True,inversionMethod="Newton")
    impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),FFT=True,inversionMethod="Interp")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_HestonImpliedVolIv.png")

#### Merton ####################################################################

def test_MertonJumpSmile():
    impVolFunc = CharFuncImpliedVol(MertonJumpCharFunc(**paramsMER),FFT=True)
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

def test_FitShortDatedMertonSmile():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    T = Texp[2]
    dfT = df[df["Texp"]==T].copy()
    k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
    mid = (dfT["CallMid"]/dfT["Fwd"]).to_numpy()
    w = (k>-0.2)&(k<0.2)
    x = CalibrateModelToOptionPrice(k,T,mid,MertonJumpCharFunc,paramsMERval,paramsMERkey,bounds=paramsMERbnd,w=w,optionType="call")
    print(f"T={np.round(T,3)}", x)
    params = {paramsMERkey[i]: x[i] for i in range(len(x))}
    impVolFunc = CharFuncImpliedVol(MertonJumpCharFunc(**params),FFT=True)
    iv = impVolFunc(k,T)
    dfT["Fit"] = iv
    PlotImpliedVol(dfT, dataFolder+f"test_MertonImpliedVol_T={np.round(T,3)}.png")

def test_CalibrateMertonJumpModelToCallPrice():
    # Wrong implementation: calibrate model to EVERY maturity
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    xT = list()
    for T in Texp:
        dfT = df[df["Texp"]==T]
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        mid = (dfT["CallMid"]/dfT["Fwd"]).to_numpy()
        w = 1/(dfT["Ask"]-dfT["Bid"]).to_numpy()
        x = CalibrateModelToOptionPrice(k,T,mid,MertonJumpCharFunc,paramsMERval,paramsMERkey,bounds=paramsMERbnd,w=w,optionType="call")
        xT.append([T]+x.tolist())
        print(f"T={np.round(T,3)}", x)
    xT = pd.DataFrame(xT, columns=["Texp"]+paramsMERkey)
    xT.to_csv(dataFolder+"test_MertonCalibration.csv", index=False)

def test_CalibrateMertonJumpModelToCallPricePrx():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    x = CalibrateModelToOptionPrice(k,T,mid,MertonJumpCharFunc,paramsMERval,paramsMERkey,bounds=paramsMERbnd,w=w,optionType="call")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsMERkey)
    x.to_csv(dataFolder+"test_MertonCalibrationPrx.csv", index=False)

def test_CalibrateMertonModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVol(k,T,iv,MertonJumpCharFunc,paramsMERval,paramsMERkey,bounds=paramsMERbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsMERkey)
    x.to_csv(dataFolder+"test_MertonCalibrationIv.csv", index=False)

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

def test_ImpVolFromMertonJumpCalibrationPrx():
    cal = pd.read_csv(dataFolder+"test_MertonCalibrationPrx.csv")
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
    PlotImpliedVol(dfnew, dataFolder+"test_MertonImpliedVolPrx.png")

def test_ImpVolFromMertonJumpIvCalibration():
    cal = pd.read_csv(dataFolder+"test_MertonCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsMERkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(MertonJumpCharFunc(**params),FFT=True)
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_MertonImpliedVolIv.png")

#### SVJ #######################################################################

def test_CalibrateSVJModelToCallPricePrx():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    x = CalibrateModelToOptionPrice(k,T,mid,SVJCharFunc,paramsSVJval,paramsSVJkey,bounds=paramsSVJbnd,w=w,optionType="call")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsSVJkey)
    x.to_csv(dataFolder+"test_SVJCalibrationPrx.csv", index=False)

def test_CalibrateSVJModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVol(k,T,iv,SVJCharFunc,paramsSVJval,paramsSVJkey,bounds=paramsSVJbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsSVJkey)
    x.to_csv(dataFolder+"test_SVJCalibrationIv.csv", index=False)

def test_ImpVolFromSVJIvCalibration():
    cal = pd.read_csv(dataFolder+"test_SVJCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsSVJkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(SVJCharFunc(**params),FFT=True)
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_SVJImpliedVolIv.png")

#### VGamma ####################################################################

def test_CalibrateVGModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVol(k,T,iv,VarianceGammaCharFunc,paramsVGval,paramsVGkey,bounds=paramsVGbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsVGkey)
    x.to_csv(dataFolder+"test_VGCalibrationIv.csv", index=False)

def test_ImpVolFromVGIvCalibration():
    cal = pd.read_csv(dataFolder+"test_VGCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsVGkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(VarianceGammaCharFunc(**params),FFT=True)
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_VGImpliedVolIv.png")

#### rHeston ###################################################################

def test_CalibrateRHPMModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVol(k,T,iv,rHestonPoorMansCharFunc,paramsRHPMval,paramsRHPMkey,bounds=paramsRHPMbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsRHPMkey)
    x.to_csv(dataFolder+"test_RHPMCalibrationIv.csv", index=False)

def test_ImpVolFromRHPMIvCalibration():
    cal = pd.read_csv(dataFolder+"test_RHPMCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsRHPMkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(rHestonPoorMansCharFunc(**params),FFT=True)
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_RHPMImpliedVolIv.png")

#### Speed Test ################################################################

def test_CalibrationSpeed():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    def unit(logStrike, maturity):
        # charFunc = HestonCharFunc(**paramsBCC)
        # impVolFunc = CharFuncImpliedVol(charFunc, optionType="call", FFT=True, inversionMethod="Newton")
        charFunc = HestonCharFunc(**paramsBCC, curry=True)
        impVolFunc = CharFuncImpliedVol(charFunc, optionType="call", FFT=True, useGlobal=True, curryCharFunc=True, inversionMethod="Newton")
        # impVolFunc = CharFuncImpliedVol(charFunc, optionType="call", FFT=True, useGlobal=True, curryCharFunc=True, inversionMethod="Interp")
        impVol = np.concatenate([impVolFunc(logStrike[maturity==T], T) for T in np.unique(maturity)], axis=None) # most costly
    from time import time
    t0 = time(); unit(k,T); t1 = time()
    print(f"unit() takes {round(t1-t0,4)}s") # 0.40s = 0.25s (FFT price) + 0.15s (BS inversion)

def test_CharFuncSpeed():
    alpha = 2
    N = 2**16
    B = 4000
    maturity = 1
    du = B/N
    u = np.arange(N)*du
    w = np.arange(N)
    w = 3+(-1)**(w+1)
    w[0] = 1; w[N-1] = 1
    dk = 2*np.pi/B
    b = N*dk/2
    k = -b+np.arange(N)*dk
    charFunc = HestonCharFunc(**paramsBCC)
    def modCharFunc(u, maturity):
        return charFunc(u-(alpha+1)*1j, maturity) / (alpha**2+alpha-u**2+1j*(2*alpha+1)*u)
    def unit():
        I = w * np.exp(1j*b*u) * modCharFunc(u, maturity) * du/3
    from time import time
    t0 = time(); unit(); t1 = time()
    print(f"unit() takes {round(t1-t0,4)}s")

if __name__ == '__main__':
    # test_BlackScholesImpVol()
    # test_BlackScholesImpVolInterp()
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
    # test_HestonSmileFFTForVariousDates()
    # test_HestonSmileLewis()
    # test_CalibrateHestonModelToCallPrice()
    # test_CalibrateHestonModelToCallPricePrx()
    # test_CalibrateHestonModelToImpVol()
    # test_ImpVolFromHestonCalibration()
    # test_ImpVolFromHestonCalibrationPrx()
    # test_ImpVolFromHestonIvCalibration()
    #### Merton ####
    # test_MertonJumpSmile()
    # test_MertonJumpSmileSensitivity()
    # test_FitShortDatedMertonSmile()
    # test_CalibrateMertonJumpModelToCallPrice()
    # test_CalibrateMertonJumpModelToCallPricePrx()
    # test_CalibrateMertonModelToImpVol()
    # test_ImpVolFromMertonJumpCalibration()
    # test_ImpVolFromMertonJumpCalibrationPrx()
    # test_ImpVolFromMertonJumpIvCalibration()
    #### SVJ ####
    # test_CalibrateSVJModelToCallPricePrx()
    # test_CalibrateSVJModelToImpVol()
    # test_ImpVolFromSVJIvCalibration()
    #### VGamma ####
    # test_CalibrateVGModelToImpVol()
    # test_ImpVolFromVGIvCalibration()
    #### rHeston ####
    # test_CalibrateRHPMModelToImpVol()
    # test_ImpVolFromRHPMIvCalibration()
    #### Speed Test ####
    test_CalibrationSpeed()
    # test_CharFuncSpeed()
