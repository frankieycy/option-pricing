import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
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

def test_BlackScholesImpVolRational():
    # n = {
    #     "01" : "-0.068098378725",
    #     "10" : "+0.440639436211",
    #     "02" : "-0.263473754689",
    #     "11" : "-5.792537721792",
    #     "20" : "-5.267481008429",
    #     "03" : "+4.714393825758",
    #     "12" : "+3.529944137559",
    #     "21" : "-23.636495876611",
    #     "30" : "-9.020361771283",
    #     "04" : "+14.749084301452",
    #     "13" : "-32.570660102526",
    #     "22" : "+76.398155779133",
    #     "31" : "+41.855161781749",
    #     "40" : "-12.150611865704",
    # }
    # m = {
    #     "01" : "+6.268456292246",
    #     "10" : "-6.284840445036",
    #     "02" : "+30.068281276567",
    #     "11" : "-11.780036995036",
    #     "20" : "-2.310966989723",
    #     "03" : "-11.473184324152",
    #     "12" : "-230.101682610568",
    #     "21" : "+86.127219899668",
    #     "30" : "+3.730181294225",
    #     "04" : "-13.954993561151",
    #     "13" : "+261.950288864225",
    #     "22" : "+20.090690444187",
    #     "31" : "-50.117067019539",
    #     "40" : "+13.723711519422",
    # }
    # nFmla = ""; mFmla = "1"
    # terms = 0
    # for i in range(5):
    #     for j in range(5):
    #         idx = str(i)+str(j)
    #         if i+j>=1 and i+j<=4:
    #             nFmla += n[idx]+("*x" if i==1 else f"*x**{i}" if i>1 else "")+("*c" if j==2 else f"*c**{round(0.5*j,1)}" if j>0 else "")
    #             mFmla += m[idx]+("*x" if i==1 else f"*x**{i}" if i>1 else "")+("*c" if j==2 else f"*c**{round(0.5*j,1)}" if j>0 else "")
    #             terms += 1
    # print("(%s)/(%s)"%(nFmla,mFmla))
    # print(terms)
    vol = np.array([0.23,0.20,0.18,0.15,0.12])
    strike = np.array([0.9,1.0,1.1,1.2,1.5])
    price = BlackScholesFormula(1,strike,1,0,vol,"call")
    impVol = BlackScholesImpliedVol(1,strike,1,0,price,"call",method="Rational")
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

def test_CalcFwdVarCurve2005():
    df = pd.read_csv("spxVols20050509.csv")
    df = df.drop(df.columns[0], axis=1).dropna()
    Texp = df["Texp"].unique()
    curveVS = CalcSwapCurve(df,VarianceSwapFormula)
    curveFV = CalcFwdVarCurve(curveVS)
    fvMid = curveFV["mid"]
    fvFunc = FwdVarCurveFunc(Texp,fvMid,"spline")
    T = np.linspace(0,2,500)
    print(curveFV)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(Texp, fvMid, c='k', s=5)
    plt.plot(T, [fvFunc(t) for t in T], 'k--', lw=0.5)
    plt.title("Forward Variance Curve (SPX 20050509)")
    plt.xlabel("maturity")
    plt.ylabel("forward variance")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_FwdVarCurve2005.png")
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

def test_HestonCOSFormula():
    impVolFunc = CharFuncImpliedVol(HestonCharFunc(**paramsBCC),optionType="call",formulaType="COS")
    k = np.arange(-0.4,0.4,0.02)
    iv = impVolFunc(k,1)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(k, 100*iv, c='k', s=5)
    plt.title("Heston 1-Year Smile (BCC Params)")
    plt.xlabel("log-strike")
    plt.ylabel("implied vol (%)")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_HestonSmileBCC_COS.png")
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

def test_HestonSkewLewis():
    atmSkewFunc = LewisCharFuncImpliedSkewAtm(HestonCharFunc(**paramsBCC))
    skew = lambda t: np.array([atmSkewFunc(t) for t in T]).reshape(-1)
    T = np.arange(0.02,5,0.02)
    sk = np.abs(skew(T))
    fig = plt.figure(figsize=(6,4))
    plt.scatter(T, sk, c='k', s=1)
    plt.title("Heston Skew (BCC Params)")
    plt.xlabel("maturity")
    plt.ylabel("atm skew")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_HestonSkewBCC.png")
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
    # df = pd.read_csv("spxVols20170424_tiny.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    # w = 1/(df["Ask"]-df["Bid"]).to_numpy()
    iv = df[["Bid","Ask"]]
    # x = CalibrateModelToImpliedVol(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call")
    # x = CalibrateModelToImpliedVol(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Newton")
    # x = CalibrateModelToImpliedVol(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    # x = CalibrateModelToImpliedVol(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Interp",useGlobal=True,curryCharFunc=True)
    # x = CalibrateModelToImpliedVolFast(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Bisection",useGlobal=True,curryCharFunc=True)
    # x = CalibrateModelToImpliedVolFast(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Bisection",useGlobal=True,curryCharFunc=True,formulaType="COSAdpt")
    # x = CalibrateModelToImpliedVolFast(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Rational",useGlobal=True,curryCharFunc=True,formulaType="COS")
    # x = CalibrateModelToImpliedVolFast(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Bisection",useGlobal=True,curryCharFunc=True,formulaType="COS",optMethod="Evolution")
    x = CalibrateModelToImpliedVolFast(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Bisection",useGlobal=True,curryCharFunc=True,formulaType="COS")
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
    # df = pd.read_csv("spxVols20170424_tiny.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsBCCkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),FFT=True)
    # impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),FFT=True,inversionMethod="Newton")
    # impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),FFT=True,inversionMethod="Interp")
    # impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),FFT=True,inversionMethod="Rational")
    # impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),optionType="call",formulaType="COS")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_HestonImpliedVolIv.png")

def test_ImpVolFromHestonIvCalibrationCOS():
    # Adaptive grid for cosFmla:
    # T<0.005:  a=-5, b=5, N=6000
    # T<0.50:   a=-3, b=3, N=2000
    # else:     a=-5, b=5, N=1000
    cal = pd.read_csv(dataFolder+"test_HestonCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsBCCkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),optionType="call",formulaType="COS")
    # impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),optionType="call",formulaType="COSAdpt")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_HestonImpliedVolIvCOS.png")

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
    # x = CalibrateModelToImpliedVol(k,T,iv,MertonJumpCharFunc,paramsMERval,paramsMERkey,bounds=paramsMERbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    x = CalibrateModelToImpliedVolFast(k,T,iv,MertonJumpCharFunc,paramsMERval,paramsMERkey,bounds=paramsMERbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
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

def test_SVJSkewLewis():
    atmSkewFunc = LewisCharFuncImpliedSkewAtm(SVJCharFunc(**paramsSVJ))
    skew = lambda t: np.array([atmSkewFunc(t) for t in T]).reshape(-1)
    T = np.arange(0.02,5,0.02)
    sk = np.abs(skew(T))
    fig = plt.figure(figsize=(6,4))
    plt.scatter(T, sk, c='k', s=1)
    plt.title("SVJ Skew")
    plt.xlabel("maturity")
    plt.ylabel("atm skew")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_SVJSkew.png")
    plt.close()

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
    # x = CalibrateModelToImpliedVol(k,T,iv,SVJCharFunc,paramsSVJval,paramsSVJkey,bounds=paramsSVJbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    # x = CalibrateModelToImpliedVolFast(k,T,iv,SVJCharFunc,paramsSVJval,paramsSVJkey,bounds=paramsSVJbnd,w=w,optionType="call",inversionMethod="Bisection",useGlobal=True,curryCharFunc=True,optMethod="Evolution")
    x = CalibrateModelToImpliedVolFast(k,T,iv,SVJCharFunc,paramsSVJval,paramsSVJkey,bounds=paramsSVJbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
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

#### SVJJ #######################################################################

def test_SVJJSkewLewis():
    atmSkewFunc = LewisCharFuncImpliedSkewAtm(SVJJCharFunc(**paramsSVJJ),optionType="call",formulaType="COS",inversionMethod="Newton",useGlobal=True)
    skew = lambda t: np.array([atmSkewFunc(t) for t in tqdm(T)]).reshape(-1)
    T = np.arange(0.02,5,0.02)
    sk = np.abs(skew(T))
    fig = plt.figure(figsize=(6,4))
    plt.scatter(T, sk, c='k', s=1)
    plt.title("SVJJ Skew")
    plt.xlabel("maturity")
    plt.ylabel("atm skew")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_SVJJSkew.png")
    plt.close()

def test_CalibrateSVJJModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,SVJJCharFunc,paramsSVJJval,paramsSVJJkey,bounds=paramsSVJJbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsSVJJkey)
    x.to_csv(dataFolder+"test_SVJJCalibrationIv.csv", index=False)

def test_ImpVolFromSVJJIvCalibration():
    cal = pd.read_csv(dataFolder+"test_SVJJCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsSVJJkey].iloc[0].to_dict()
    # impVolFunc = CharFuncImpliedVol(SVJJCharFunc(**params),FFT=True)
    impVolFunc = CharFuncImpliedVol(SVJJCharFunc(**params),optionType="call",formulaType="COS")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_SVJJImpliedVolIv.png")

#### VGamma ####################################################################

def test_CalibrateVGModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    # x = CalibrateModelToImpliedVol(k,T,iv,VarianceGammaCharFunc,paramsVGval,paramsVGkey,bounds=paramsVGbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    # x = CalibrateModelToImpliedVolFast(k,T,iv,VarianceGammaCharFunc,paramsVGval,paramsVGkey,bounds=paramsVGbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,optMethod="Evolution")
    x = CalibrateModelToImpliedVolFast(k,T,iv,VarianceGammaCharFunc,paramsVGval,paramsVGkey,bounds=paramsVGbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
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

def test_CalibrateVGLModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,VarianceGammaLevyCharFunc,paramsVGLval,paramsVGLkey,bounds=paramsVGLbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsVGLkey)
    x.to_csv(dataFolder+"test_VGLCalibrationIv.csv", index=False)

def test_ImpVolFromVGLIvCalibration():
    cal = pd.read_csv(dataFolder+"test_VGLCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsVGLkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(VarianceGammaLevyCharFunc(**params),optionType="call",formulaType="COS")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_VGLImpliedVolIv.png")

#### CGMY ######################################################################

def test_CGMYSmile_COS():
    impVolFunc = CharFuncImpliedVol(CGMYCharFunc(**paramsCGMY),optionType="call",formulaType="COS")
    k = np.arange(-0.4,0.4,0.02)
    iv = impVolFunc(k,1)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(k, 100*iv, c='k', s=5)
    plt.title("CGMY 1-Year Smile (CGMY Params)")
    plt.xlabel("log-strike")
    plt.ylabel("implied vol (%)")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_CGMYSmile_COS.png")
    plt.close()

def test_CalibrateCGMYModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,CGMYCharFunc,paramsCGMYval,paramsCGMYkey,bounds=paramsCGMYbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsCGMYkey)
    x.to_csv(dataFolder+"test_CGMYCalibrationIv.csv", index=False)

def test_ImpVolFromCGMYIvCalibration():
    cal = pd.read_csv(dataFolder+"test_CGMYCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsCGMYkey].iloc[0].to_dict()
    # impVolFunc = CharFuncImpliedVol(CGMYCharFunc(**params),FFT=True)
    impVolFunc = CharFuncImpliedVol(CGMYCharFunc(**params),optionType="call",formulaType="COS")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_CGMYImpliedVolIv.png")

def test_CalibrateECGMYModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,eCGMYCharFunc,paramsECGMYval,paramsECGMYkey,bounds=paramsECGMYbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsECGMYkey)
    x.to_csv(dataFolder+"test_eCGMYCalibrationIv.csv", index=False)

def test_ImpVolFromECGMYIvCalibration():
    cal = pd.read_csv(dataFolder+"test_eCGMYCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsECGMYkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(eCGMYCharFunc(**params),optionType="call",formulaType="COS")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_eCGMYImpliedVolIv.png")

def test_CalibratePNCGMYModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,pnCGMYCharFunc,paramsPNCGMYval,paramsPNCGMYkey,bounds=paramsPNCGMYbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsPNCGMYkey)
    x.to_csv(dataFolder+"test_pnCGMYCalibrationIv.csv", index=False)

def test_ImpVolFromPNCGMYIvCalibration():
    cal = pd.read_csv(dataFolder+"test_pnCGMYCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsPNCGMYkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(pnCGMYCharFunc(**params),optionType="call",formulaType="COS")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_pnCGMYImpliedVolIv.png")

#### NIG #######################################################################

def test_NIGSmile_COS():
    impVolFunc = CharFuncImpliedVol(NIGCharFunc(**paramsNIG),optionType="call",formulaType="COS")
    k = np.arange(-0.4,0.4,0.02)
    iv = impVolFunc(k,1)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(k, 100*iv, c='k', s=5)
    plt.title("NIG 1-Year Smile (NIG Params)")
    plt.xlabel("log-strike")
    plt.ylabel("implied vol (%)")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_NIGSmile_COS.png")
    plt.close()

def test_CalibrateNIGModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,NIGCharFunc,paramsNIGval,paramsNIGkey,bounds=paramsNIGbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsNIGkey)
    x.to_csv(dataFolder+"test_NIGCalibrationIv.csv", index=False)

def test_ImpVolFromNIGIvCalibration():
    cal = pd.read_csv(dataFolder+"test_NIGCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsNIGkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(NIGCharFunc(**params),optionType="call",formulaType="COS")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_NIGImpliedVolIv.png")

#### SA ########################################################################

def test_CalibrateVGSAModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,VGSACharFunc,paramsVGSAval,paramsVGSAkey,bounds=paramsVGSAbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS",a=-1,b=1)
    # x = CalibrateModelToImpliedVolFast(k,T,iv,VGSACharFunc,paramsVGSAval,paramsVGSAkey,bounds=paramsVGSAbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS",optMethod="Evolution",a=-1,b=1)
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsVGSAkey)
    x.to_csv(dataFolder+"test_VGSACalibrationIv.csv", index=False)

def test_ImpVolFromVGSAIvCalibration():
    # Price calculations are unstable for large maturities
    cal = pd.read_csv(dataFolder+"test_VGSACalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsVGSAkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(VGSACharFunc(**params),optionType="call",formulaType="COS",a=-1,b=1)
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_VGSAImpliedVolIv.png")

def test_CalibrateCGMYSAModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,CGMYSACharFunc,paramsCGMYSAval,paramsCGMYSAkey,bounds=paramsCGMYSAbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS",a=-3,b=3)
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsCGMYSAkey)
    x.to_csv(dataFolder+"test_CGMYSACalibrationIv.csv", index=False)

def test_ImpVolFromCGMYSAIvCalibration():
    # Price calculations are unstable for large maturities
    cal = pd.read_csv(dataFolder+"test_CGMYSACalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsCGMYSAkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(CGMYSACharFunc(**params),optionType="call",formulaType="COS",a=-3,b=3)
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_CGMYSAImpliedVolIv.png")

def test_CalibrateNIGSAModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,NIGSACharFunc,paramsNIGSAval,paramsNIGSAkey,bounds=paramsNIGSAbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsNIGSAkey)
    x.to_csv(dataFolder+"test_NIGSACalibrationIv.csv", index=False)

def test_ImpVolFromNIGSAIvCalibration():
    cal = pd.read_csv(dataFolder+"test_NIGSACalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsNIGSAkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(NIGSACharFunc(**params),optionType="call",formulaType="COS")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_NIGSAImpliedVolIv.png")

#### rHeston ###################################################################

def test_CalibrateRHPMModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    # x = CalibrateModelToImpliedVol(k,T,iv,rHestonPoorMansCharFunc,paramsRHPMval,paramsRHPMkey,bounds=paramsRHPMbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    x = CalibrateModelToImpliedVolFast(k,T,iv,rHestonPoorMansCharFunc,paramsRHPMval,paramsRHPMkey,bounds=paramsRHPMbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
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

def test_CalibrateRHPMMModelToImpVol():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,rHestonPoorMansModCharFunc,paramsRHPMMval,paramsRHPMMkey,bounds=paramsRHPMMbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True)
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsRHPMMkey)
    x.to_csv(dataFolder+"test_RHPMMCalibrationIv.csv", index=False)

def test_ImpVolFromRHPMMIvCalibration():
    cal = pd.read_csv(dataFolder+"test_RHPMMCalibrationIv.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsRHPMMkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(rHestonPoorMansModCharFunc(**params),FFT=True)
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_RHPMMImpliedVolIv.png")

def test_CalibrateRHPModelToImpVol():
    # df = pd.read_csv("spxVols20050509.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1).dropna()
    T = df["Texp"]

    Texp = df["Texp"].unique()
    curveVS = CalcSwapCurve(df,VarianceSwapFormula)
    curveFV = CalcFwdVarCurve(curveVS)
    fvMid = curveFV["mid"]
    fvFunc = FwdVarCurveFunc(Texp,fvMid,"const")
    # fvFunc = FwdVarCurveFunc(Texp,fvMid,"spline")

    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.2)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k,T,iv,rHestonPadeCharFunc,paramsRHPval,paramsRHPkey,bounds=paramsRHPbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS",kwargsCF={"fvFunc":fvFunc},optMethod="Evolution")
    # x = CalibrateModelToImpliedVolFast(k,T,iv,rHestonPadeCharFunc,paramsRHPval,paramsRHPkey,bounds=paramsRHPbnd,w=w,optionType="call",inversionMethod="Newton",useGlobal=True,curryCharFunc=True,formulaType="COS",kwargsCF={"fvFunc":fvFunc,"dhPade":dhPade44},optMethod="Evolution")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsRHPkey)
    x.to_csv(dataFolder+"test_RHPCalibrationIv.csv", index=False)

def test_ImpVolFromRHPIvCalibration():
    cal = pd.read_csv(dataFolder+"test_RHPCalibrationIv.csv")
    # df = pd.read_csv("spxVols20050509.csv")
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1).dropna()
    Texp = df["Texp"].unique()

    curveVS = CalcSwapCurve(df,VarianceSwapFormula)
    curveFV = CalcFwdVarCurve(curveVS)
    fvMid = curveFV["mid"]
    fvFunc = FwdVarCurveFunc(Texp,fvMid,"const")
    # fvFunc = FwdVarCurveFunc(Texp,fvMid,"spline")

    dfnew = list()
    params = cal[paramsRHPkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(rHestonPadeCharFunc(**params,fvFunc=fvFunc),optionType="call",formulaType="COS")
    # impVolFunc = CharFuncImpliedVol(rHestonPadeCharFunc(**params,fvFunc=fvFunc,dhPade=dhPade44),optionType="call",formulaType="COS")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_RHPImpliedVolIv.png")

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
    print(f"unit() takes {round(t1-t0,4)}s") # 0.38s

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

#### Calibration Results #######################################################

def test_PlotCalibratedAtmVolAndSkew():
    models = {
        "Merton": {"CF": MertonJumpCharFunc,      "params": paramsMERkey},
        "Heston": {"CF": HestonCharFunc,          "params": paramsBCCkey},
        "VG":     {"CF": VarianceGammaCharFunc,   "params": paramsVGkey},
        "CGMY":   {"CF": CGMYCharFunc,            "params": paramsCGMYkey},
        "SVJ":    {"CF": SVJCharFunc,             "params": paramsSVJkey},
        "SVJJ":   {"CF": SVJJCharFunc,            "params": paramsSVJJkey},
        # "RHPM":   {"CF": rHestonPoorMansCharFunc, "params": paramsRHPMkey},
        # "RHP":    {"CF": rHestonPadeCharFunc,     "params": paramsRHPkey},
    }

    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    ivDict = CalcAtmVolAndSkew(df)
    Texp = ivDict['Texp']
    mktIv = ivDict['atmVol']
    mktSk = ivDict['atmSkew']
    T = np.linspace(Texp.min(), Texp.max(), 100)

    if "RHP" in models:
        Texp = df["Texp"].unique()
        curveVS = CalcSwapCurve(df,VarianceSwapFormula)
        curveFV = CalcFwdVarCurve(curveVS)
        fvMid = curveFV["mid"]
        fvFunc = FwdVarCurveFunc(Texp,fvMid,"const")

    fig = plt.figure(figsize=(6,4))
    plt.scatter(Texp, 100*mktIv, c='k', s=10, zorder=1)
    plt.title("ATM Vol")
    plt.xlabel("maturity")
    plt.ylabel("vol (%)")
    plt.xlim(0.002,2.7)
    plt.ylim(6,18)

    for model in models.keys():
        cal = pd.read_csv(dataFolder+f"Calibration/test_{model}CalibrationIv.csv")
        params = cal[models[model]["params"]].iloc[0].to_dict()
        if model == "RHP":
            atmVolFunc = CharFuncImpliedVolAtm(models[model]["CF"](**params,fvFunc=fvFunc),optionType="call",formulaType="COS",useGlobal=True)
        else:
            atmVolFunc = CharFuncImpliedVolAtm(models[model]["CF"](**params),optionType="call",formulaType="COS",useGlobal=True)
        vol = lambda t: np.array([atmVolFunc(t) for t in tqdm(T)]).reshape(-1)
        plt.plot(T, 100*vol(T), label=model, zorder=0)

    fig.tight_layout()
    plt.legend()
    plt.savefig(dataFolder+"calibration_atmVol.png")
    plt.close()

    fig = plt.figure(figsize=(6,4))
    plt.scatter(Texp, np.abs(mktSk), c='k', s=10, zorder=1)
    plt.title("ATM Skew")
    plt.xlabel("maturity")
    plt.ylabel("skew")
    plt.xlim(0.002,2.7)
    plt.ylim(0,4)

    for model in models.keys():
        cal = pd.read_csv(dataFolder+f"Calibration/test_{model}CalibrationIv.csv")
        params = cal[models[model]["params"]].iloc[0].to_dict()
        if model == "RHP":
            atmSkewFunc = LewisCharFuncImpliedSkewAtm(models[model]["CF"](**params,fvFunc=fvFunc),optionType="call",formulaType="COS",useGlobal=True)
        else:
            atmSkewFunc = LewisCharFuncImpliedSkewAtm(models[model]["CF"](**params),optionType="call",formulaType="COS",useGlobal=True)
        skew = lambda t: np.array([atmSkewFunc(t) for t in tqdm(T)]).reshape(-1)
        plt.plot(T, np.abs(skew(T)), label=model, zorder=0)

    fig.tight_layout()
    plt.legend()
    plt.savefig(dataFolder+"calibration_atmSkew.png")
    plt.close()

def test_PlotAtmSkewPowerLawFit():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    ivDict = CalcAtmVolAndSkew(df)
    Texp = ivDict['Texp']
    mktIv = ivDict['atmVol']
    mktSk = ivDict['atmSkew']
    T = np.linspace(Texp.min(), Texp.max(), 100)

    mktSkAbs = np.abs(mktSk)
    fitFunc = lambda A,H,T: A/T**(0.5-H)
    def objective(params):
        return np.sum((fitFunc(*params,Texp)-mktSkAbs)**2)
    opt = minimize(objective, x0=(0.5,0.1), bounds=((0,10),(0,0.5)))
    print("Optimization output:", opt, sep="\n")

    A,H = opt.x
    fig = plt.figure(figsize=(6,4))
    plt.scatter(Texp, mktSkAbs, c='k', s=10, zorder=1)
    plt.plot(T, fitFunc(A,H,T), 'r--', zorder=0, label=r'$A/\tau^{1/2-H}$')
    plt.title("ATM Skew Power-Law Fit: $A=%.3f, H=%.3f$" % (A,H))
    plt.xlabel("maturity")
    plt.ylabel("skew")
    plt.xlim(0.002,2.7)
    plt.ylim(0,3)
    fig.tight_layout()
    plt.legend()
    plt.savefig(dataFolder+"calibration_atmSkewPowLawFit.png")
    plt.close()

def test_SpeedProfile():
    models = {
        "Merton": {
            "CF": MertonJumpCharFunc,
            "paramsVal": paramsMERval,
            "paramsKey": paramsMERkey,
            "paramsBnd": paramsMERbnd,
        },
        "Heston": {
            "CF": HestonCharFunc,
            "paramsVal": paramsBCCval,
            "paramsKey": paramsBCCkey,
            "paramsBnd": paramsBCCbnd,
        },
        "VG": {
            "CF": VarianceGammaCharFunc,
            "paramsVal": paramsVGval,
            "paramsKey": paramsVGkey,
            "paramsBnd": paramsVGbnd,
        },
        "CGMY": {
            "CF": CGMYCharFunc,
            "paramsVal": paramsCGMYval,
            "paramsKey": paramsCGMYkey,
            "paramsBnd": paramsCGMYbnd,
        },
        "SVJ": {
            "CF": SVJCharFunc,
            "paramsVal": paramsSVJval,
            "paramsKey": paramsSVJkey,
            "paramsBnd": paramsSVJbnd,
        },
    }

    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.1)
    iv = df[["Bid","Ask"]]

    for model in models.keys():
        startTime = time()
        CalibrateModelToImpliedVolFast(k,T,iv,models[model]["CF"],models[model]["paramsVal"],models[model]["paramsKey"],bounds=models[model]["paramsBnd"],w=w,optionType="call",inversionMethod="Bisection",useGlobal=True,curryCharFunc=True,formulaType="COS")
        endTime = time()
        print(f"{model}: {endTime-startTime}s")

#### Plot Implied Vol Surface ##################################################

def test_PlotImpliedVolSurface():
    run = [1,2]

    models = {
        "Merton": {"CF": MertonJumpCharFunc,      "params": paramsMERkey},
        "Heston": {"CF": HestonCharFunc,          "params": paramsBCCkey},
        "VG":     {"CF": VarianceGammaCharFunc,   "params": paramsVGkey},
        "CGMY":   {"CF": CGMYCharFunc,            "params": paramsCGMYkey},
        "NIG":    {"CF": NIGCharFunc,             "params": paramsNIGkey},
        "SVJ":    {"CF": SVJCharFunc,             "params": paramsSVJkey},
        "SVJJ":   {"CF": SVJJCharFunc,            "params": paramsSVJJkey},
        "RHPM":   {"CF": rHestonPoorMansCharFunc, "params": paramsRHPMkey},
        # "VGSA":   {"CF": VGSACharFunc,            "params": paramsVGSAkey},
        # "CGMYSA": {"CF": CGMYSACharFunc,          "params": paramsCGMYSAkey},
        # "NIGSA":  {"CF": NIGSACharFunc,           "params": paramsNIGSAkey},
    }

    k = np.arange(-0.3,0.3,0.01)
    T = np.arange(0.1,2.1,0.1)
    X,Y = np.meshgrid(k,T)

    if 1 in run:
        for model in models.keys():
            cal = pd.read_csv(dataFolder+f"Calibration/test_{model}CalibrationIv.csv")
            params = cal[models[model]["params"]].iloc[0].to_dict()
            impVolFunc = CharFuncImpliedVol(models[model]["CF"](**params),optionType="call",formulaType="COS",N=6000)
            Z = np.array([impVolFunc(k,t) for t in T])
            Z[Z<1e-8] = np.nan
            df = pd.DataFrame(np.array([X,Y,Z]).reshape(3,-1).T,columns=["Log-strike","Texp","IV"])
            df.to_csv(dataFolder+f"Implied Vol Surface/IVS_{model}.csv",index=False)

    if 2 in run:
        for model in models.keys():
            df = pd.read_csv(dataFolder+f"Implied Vol Surface/IVS_{model}.csv")
            PlotImpliedVolSurface(df,dataFolder+f"Implied Vol Surface/IVS_{model}.png",model)

#### Plot Local Vol Surface ##################################################

def test_PlotLocalVolSurface():
    run = [1,2]

    models = {
        "Merton": {"CF": MertonJumpCharFunc,      "params": paramsMERkey},
        "Heston": {"CF": HestonCharFunc,          "params": paramsBCCkey},
        "VG":     {"CF": VarianceGammaCharFunc,   "params": paramsVGkey},
        "CGMY":   {"CF": CGMYCharFunc,            "params": paramsCGMYkey},
        "NIG":    {"CF": NIGCharFunc,             "params": paramsNIGkey},
        "SVJ":    {"CF": SVJCharFunc,             "params": paramsSVJkey},
        "SVJJ":   {"CF": SVJJCharFunc,            "params": paramsSVJJkey},
        "RHPM":   {"CF": rHestonPoorMansCharFunc, "params": paramsRHPMkey},
        "VGSA":   {"CF": VGSACharFunc,            "params": paramsVGSAkey},
        "CGMYSA": {"CF": CGMYSACharFunc,          "params": paramsCGMYSAkey},
        "NIGSA":  {"CF": NIGSACharFunc,           "params": paramsNIGSAkey},
    }

    if 1 in run:
        for model in models.keys():
            df = pd.read_csv(dataFolder+f"Implied Vol Surface/IVS_{model}.csv")
            lv = CalcLocalVolSurface(df)
            lv.to_csv(dataFolder+f"Local Vol Surface/LVS_{model}.csv",index=False)

    if 2 in run:
        for model in models.keys():
            df = pd.read_csv(dataFolder+f"Local Vol Surface/LVS_{model}.csv")
            df = df[(df['Texp']>=0.5)&(df['Texp']<=1.5)]
            PlotLocalVolSurface(df,dataFolder+f"Local Vol Surface/LVS_{model}.png",model)

#### Results Check #############################################################

def test_CalibrateHestonModelToImpVol2005():
    df = pd.read_csv("spxVols20050509.csv")
    df = df.drop(df.columns[0], axis=1).dropna()
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()
    iv = df[["Bid","Ask"]]
    # x = CalibrateModelToImpliedVolFast(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Bisection",useGlobal=True,curryCharFunc=True,formulaType="COS")
    x = CalibrateModelToImpliedVolFast(k,T,iv,HestonCharFunc,paramsBCCval,paramsBCCkey,bounds=paramsBCCbnd,w=w,optionType="call",inversionMethod="Bisection",useGlobal=True,curryCharFunc=True,formulaType="COS",optMethod="Evolution")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsBCCkey)
    x.to_csv(dataFolder+"test_HestonCalibrationIv.csv", index=False)

def test_ImpVolFromHestonIvCalibration2005():
    cal = pd.read_csv(dataFolder+"test_HestonCalibrationIv.csv")
    df = pd.read_csv("spxVols20050509.csv").dropna()
    df = df.drop(df.columns[0], axis=1)
    Texp = df["Texp"].unique()
    dfnew = list()
    params = cal[paramsBCCkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params),optionType="call",formulaType="COS")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, dataFolder+"test_HestonImpliedVolIv.png")

def test_CalibrateModels2005():
    models = {
        "Merton": {
            "CF": MertonJumpCharFunc,
            "paramsVal": paramsMERval,
            "paramsKey": paramsMERkey,
            "paramsBnd": paramsMERbnd,
        },
        "Heston": {
            "CF": HestonCharFunc,
            "paramsVal": paramsBCCval,
            "paramsKey": paramsBCCkey,
            "paramsBnd": paramsBCCbnd,
        },
        "RHPM": {
            "CF": rHestonPoorMansCharFunc,
            "paramsVal": paramsRHPMval,
            "paramsKey": paramsRHPMkey,
            "paramsBnd": paramsRHPMbnd,
        },
        "RHPMM": {
            "CF": rHestonPoorMansModCharFunc,
            "paramsVal": paramsRHPMMval,
            "paramsKey": paramsRHPMMkey,
            "paramsBnd": paramsRHPMMbnd,
        },
        "VG": {
            "CF": VarianceGammaCharFunc,
            "paramsVal": paramsVGval,
            "paramsKey": paramsVGkey,
            "paramsBnd": paramsVGbnd,
        },
        "SVJ": {
            "CF": SVJCharFunc,
            "paramsVal": paramsSVJval,
            "paramsKey": paramsSVJkey,
            "paramsBnd": paramsSVJbnd,
        },
        # "NIG": {
        #     "CF": NIGCharFunc,
        #     "paramsVal": paramsNIGval,
        #     "paramsKey": paramsNIGkey,
        #     "paramsBnd": paramsNIGbnd,
        # },
        # "CGMY": {
        #     "CF": CGMYCharFunc,
        #     "paramsVal": paramsCGMYval,
        #     "paramsKey": paramsCGMYkey,
        #     "paramsBnd": paramsCGMYbnd,
        # },
        # "VGSA": {
        #     "CF": VGSACharFunc,
        #     "paramsVal": paramsVGSAval,
        #     "paramsKey": paramsVGSAkey,
        #     "paramsBnd": paramsVGSAbnd,
        # },
        # "NIGSA": {
        #     "CF": NIGSACharFunc,
        #     "paramsVal": paramsNIGSAval,
        #     "paramsKey": paramsNIGSAkey,
        #     "paramsBnd": paramsNIGSAbnd,
        # },
        # "CGMYSA": {
        #     "CF": CGMYSACharFunc,
        #     "paramsVal": paramsCGMYSAval,
        #     "paramsKey": paramsCGMYSAkey,
        #     "paramsBnd": paramsCGMYSAbnd,
        # },
    }

    df = pd.read_csv("spxVols20050509.csv")
    df = df.drop(df.columns[0], axis=1).dropna()
    T = df["Texp"]; Texp = T.unique()
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    mid = (df["CallMid"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k,scale=0.2)
    iv = df[["Bid","Ask"]]

    for model in models.keys():
        x = CalibrateModelToImpliedVolFast(k,T,iv,models[model]["CF"],models[model]["paramsVal"],models[model]["paramsKey"],bounds=models[model]["paramsBnd"],w=w,optionType="call",inversionMethod="Bisection",useGlobal=True,curryCharFunc=True,formulaType="COS",optMethod="Evolution")
        impVolFunc = CharFuncImpliedVol(models[model]["CF"](*x),optionType="call",formulaType="COS")
        x = pd.DataFrame(x.reshape(1,-1), columns=models[model]["paramsKey"])
        x.to_csv(dataFolder+f"Calibration-2005/test_{model}CalibrationIv.csv", index=False)

        dfnew = list()
        for t in Texp:
            dfT = df[df["Texp"]==t].copy()
            kT = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
            dfT["Fit"] = impVolFunc(kT,t)
            dfnew.append(dfT)
        dfnew = pd.concat(dfnew)
        PlotImpliedVol(dfnew, dataFolder+f"Calibration-2005/test_{model}ImpliedVolIv.png")

if __name__ == '__main__':
    # test_BlackScholesImpVol()
    # test_BlackScholesImpVolInterp()
    # test_BlackScholesImpVolRational()
    # test_PlotImpliedVol()
    # test_VarianceSwapFormula()
    # test_CalcSwapCurve()
    # test_LevSwapCurve()
    # test_CalcFwdVarCurve()
    # test_CalcFwdVarCurve2005()
    #### Heston ####
    # test_HestonSmile()
    # test_HestonSmileSensitivity()
    # test_HestonSmileFFT()
    # test_ShortDatedHestonSmileFFT()
    # test_HestonSmileFFTForVariousDates()
    # test_HestonCOSFormula()
    # test_HestonSmileLewis()
    # test_HestonSkewLewis()
    # test_CalibrateHestonModelToCallPrice()
    # test_CalibrateHestonModelToCallPricePrx()
    # test_CalibrateHestonModelToImpVol()
    # test_ImpVolFromHestonCalibration()
    # test_ImpVolFromHestonCalibrationPrx()
    # test_ImpVolFromHestonIvCalibration()
    # test_ImpVolFromHestonIvCalibrationCOS()
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
    # test_SVJSkewLewis()
    # test_CalibrateSVJModelToCallPricePrx()
    # test_CalibrateSVJModelToImpVol()
    # test_ImpVolFromSVJIvCalibration()
    #### SVJJ ####
    # test_SVJJSkewLewis()
    # test_CalibrateSVJJModelToImpVol()
    # test_ImpVolFromSVJJIvCalibration()
    #### VGamma ####
    # test_CalibrateVGModelToImpVol()
    # test_ImpVolFromVGIvCalibration()
    # test_CalibrateVGLModelToImpVol()
    # test_ImpVolFromVGLIvCalibration()
    #### CGMY ####
    # test_CGMYSmile_COS()
    # test_CalibrateCGMYModelToImpVol()
    # test_ImpVolFromCGMYIvCalibration()
    # test_CalibrateECGMYModelToImpVol()
    # test_ImpVolFromECGMYIvCalibration()
    # test_CalibratePNCGMYModelToImpVol()
    # test_ImpVolFromPNCGMYIvCalibration()
    #### NIG ####
    # test_NIGSmile_COS()
    # test_CalibrateNIGModelToImpVol()
    # test_ImpVolFromNIGIvCalibration()
    #### SA ####
    # test_CalibrateVGSAModelToImpVol()
    # test_ImpVolFromVGSAIvCalibration()
    # test_CalibrateCGMYSAModelToImpVol()
    # test_ImpVolFromCGMYSAIvCalibration()
    # test_CalibrateNIGSAModelToImpVol()
    # test_ImpVolFromNIGSAIvCalibration()
    #### rHeston ####
    # test_CalibrateRHPMModelToImpVol()
    # test_ImpVolFromRHPMIvCalibration()
    # test_CalibrateRHPMMModelToImpVol()
    # test_ImpVolFromRHPMMIvCalibration()
    test_CalibrateRHPModelToImpVol()
    test_ImpVolFromRHPIvCalibration()
    #### Speed Test ####
    # test_CalibrationSpeed()
    # test_CharFuncSpeed()
    #### Calibration Results ####
    # test_PlotCalibratedAtmVolAndSkew()
    # test_PlotAtmSkewPowerLawFit()
    # test_SpeedProfile()
    #### Plot IVS ####
    # test_PlotImpliedVolSurface()
    #### Plot LVS ####
    # test_PlotLocalVolSurface()
    #### Check ####
    # test_CalibrateHestonModelToImpVol2005()
    # test_ImpVolFromHestonIvCalibration2005()
    # test_CalibrateModels2005()
