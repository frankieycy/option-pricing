import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from option import *
from pricer import *
from params import *
from svi import *
from american import *
plt.switch_backend("Agg")

dataFolder = "test/"

#### Options Chain #############################################################

def test_GenerateYfinOptionsChainDataset():
    GenerateYfinOptionsChainDataset(dataFolder+"spxOptions20220414.csv")

def test_StandardizeOptionsChainDataset():
    df = pd.read_csv('data-futu/option_chain_US.SPY_2022-04-14.csv')
    print(StandardizeOptionsChainDataset(df,'2022-04-14').head())

def test_GenerateImpVolDatasetFromStdDf():
    code = "SPY"
    df = pd.read_csv(f'data-futu/option_chain_US.{code}_2022-04-14.csv')
    df = StandardizeOptionsChainDataset(df,'2022-04-14')
    # ivdf = GenerateImpVolDatasetFromStdDf(df)
    # ivdf.to_csv(f'{code.lower()}Vols20220414.csv',index=False)
    ivdf = GenerateImpVolDatasetFromStdDf(df,volCorrection='delta')
    ivdf.to_csv(f'{code.lower()}Vols20220414_corr.csv',index=False)

def test_TermStructure():
    code = "SPY"
    snap = {
        "SPY": 437.790,
        "QQQ": 338.430,
    }
    df = pd.read_csv(f'{code.lower()}Vols20220414.csv')
    df = df[df['Texp']>0.25]
    S = snap[code]
    T = df['Texp']
    F = df['Fwd']
    PV = df['PV']
    r = -np.log(PV)/T
    y = np.log(F/S)/T
    d = r-y
    for x in ["r","y","d"]:
        if x == "r":
            ts = r
            title = "Risk-Free Rate"
        elif x == "y":
            ts = y
            title = "Yield"
        elif x == "d":
            ts = d
            title = "Dividend Rate"
        fig = plt.figure(figsize=(6,4))
        plt.scatter(T,ts,c='k',s=20)
        plt.xlabel('maturity')
        plt.ylabel(f'${x}$')
        plt.title(f'{code} {title}')
        fig.tight_layout()
        plt.savefig(dataFolder+f"test_TermStruct_{code}{x}.png")
        plt.close()

#### Black-Scholes #############################################################

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

def test_PlotImpliedVol2019():
    PlotImpliedVol(pd.read_csv("spxVols20191220.csv").dropna(), dataFolder+"test_SPXimpliedvol2019.png")
    PlotImpliedVol(pd.read_csv("vixVols20191220.csv").dropna(), dataFolder+"test_VIXimpliedvol2019.png")

def test_PlotImpliedVolSPY2022():
    # PlotImpliedVol(pd.read_csv("spyVols20220414.csv").dropna(), dataFolder+"test_SPYimpliedvol2022.png")
    PlotImpliedVol(pd.read_csv("spyVols20220414_corr.csv").dropna(), dataFolder+"test_SPYimpliedvol2022_corr.png")

def test_PlotImpliedVolQQQ2022():
    # PlotImpliedVol(pd.read_csv("qqqVols20220414.csv").dropna(), dataFolder+"test_QQQimpliedvol2022.png")
    PlotImpliedVol(pd.read_csv("qqqVols20220414_corr.csv").dropna(), dataFolder+"test_QQQimpliedvol2022_corr.png")

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
    # curveFV = CalcFwdVarCurve(curveVS)
    curveFV = CalcFwdVarCurve(curveVS,eps=0.003)
    fvMid = curveFV["mid"]
    fvFunc = FwdVarCurveFunc(Texp,fvMid)
    # fvFuncSmth0 = SmoothFwdVarCurveFunc(Texp,curveVS["mid"])
    fvFuncSmth1 = SmoothFwdVarCurveFunc(Texp,curveVS["mid"],eps=0.003)
    T = np.linspace(0,3,1000)
    print(curveFV)
    fig = plt.figure(figsize=(6,4))
    # plt.scatter(Texp, fvMid, c='k', s=5)
    # plt.plot(T, fvFunc(T), 'k', lw=1)
    # # plt.plot(T, fvFuncSmth0(T), 'r--', lw=1)
    # # plt.plot(T, fvFuncSmth1(T), 'r', lw=1)
    plt.plot(T, fvFunc(T), 'k', lw=1, label='step')
    plt.plot(T, fvFuncSmth1(T), 'r', lw=1, label='smoothed')
    plt.title("Forward Variance Curve (SPX 20170424)")
    plt.xlabel("maturity")
    plt.ylabel("forward variance")
    plt.legend()
    fig.tight_layout()
    plt.savefig(dataFolder+"test_FwdVarCurve.png")
    plt.close()

def test_CalcFwdVarCurve2005():
    df = pd.read_csv("spxVols20050509.csv")
    df = df.drop(df.columns[0], axis=1).dropna()
    Texp = df["Texp"].unique()
    curveVS = CalcSwapCurve(df,VarianceSwapFormula)
    curveFV = CalcFwdVarCurve(curveVS)
    # curveFV = CalcFwdVarCurve(curveVS,eps=0.003)
    fvMid = curveFV["mid"]
    fvFunc = FwdVarCurveFunc(Texp,fvMid)
    # fvFunc = FwdVarCurveFunc(Texp,fvMid,"spline")
    # fvFuncSmth0 = SmoothFwdVarCurveFunc(Texp,curveVS["mid"])
    fvFuncSmth1 = SmoothFwdVarCurveFunc(Texp,curveVS["mid"],eps=0.003)
    T = np.linspace(0,2,1000)
    print(curveFV)
    fig = plt.figure(figsize=(6,4))
    # plt.scatter(Texp, fvMid, c='k', s=5)
    # plt.plot(T, fvFunc(T), 'k', lw=1)
    # # plt.plot(T, fvFuncSmth0(T), 'r--', lw=1)
    # # plt.plot(T, fvFuncSmth1(T), 'r', lw=1)
    plt.plot(T, fvFunc(T), 'k', lw=1, label='step')
    plt.plot(T, fvFuncSmth1(T), 'r', lw=1, label='smoothed')
    plt.title("Forward Variance Curve (SPX 20050509)")
    plt.xlabel("maturity")
    plt.ylabel("forward variance")
    plt.legend()
    fig.tight_layout()
    plt.savefig(dataFolder+"test_FwdVarCurve2005.png")
    plt.close()

def test_VswpPriceCompare():
    from scipy.integrate import quad_vec
    df = pd.read_csv("spxVols20170424.csv").dropna()
    # df = df.drop(df.columns[0], axis=1)
    # df = pd.read_csv("spxVols20050509.csv").dropna()
    Texp = df["Texp"].unique()
    curveVS = CalcSwapCurve(df,VarianceSwapFormula)
    # curveFV = CalcFwdVarCurve(curveVS)
    curveFV = CalcFwdVarCurve(curveVS,eps=0.003)
    fvMid = curveFV["mid"]
    fvFunc = FwdVarCurveFunc(Texp,fvMid)
    # fvFuncSmth0 = SmoothFwdVarCurveFunc(Texp,curveVS["mid"])
    fvFuncSmth1 = SmoothFwdVarCurveFunc(Texp,curveVS["mid"],eps=0.003)

    vswp = [quad_vec(fvFunc,0,tau)[0]/tau for tau in Texp]
    # vswpSmth0 = [quad_vec(fvFuncSmth0,0,tau)[0]/tau for tau in Texp]
    vswpSmth1 = [quad_vec(fvFuncSmth1,0,tau)[0]/tau for tau in Texp]

    # print(curveFV)
    fig = plt.figure(figsize=(6,4))
    # plt.scatter(Texp, curveVS["mid"], c='k', s=5)
    # plt.plot(Texp, vswp, 'k', lw=1)
    # plt.plot(Texp, vswpSmth0, 'r--', lw=1)
    # plt.plot(Texp, vswpSmth1, 'r', lw=1)
    plt.scatter(Texp, 100*np.sqrt(curveVS["mid"]), c='grey', s=20, label='SPX-implied')
    plt.plot(Texp, 100*np.sqrt(vswp), c='k', lw=1, label='step')
    plt.plot(Texp, 100*np.sqrt(vswpSmth1), c='r', lw=1, label='smoothed')
    plt.title("Variance Swap Curve (SPX 20170424)")
    plt.xlabel("maturity")
    plt.ylabel("swap price (% vol)")
    plt.legend()
    fig.tight_layout()
    plt.savefig(dataFolder+"test_VswpPriceCompare.png")
    plt.close()

def test_FwdVswp2019():
    run = 4
    df = pd.read_csv("spxVols20191220.csv").dropna()
    Texp = df["Texp"].unique()
    curveVS = CalcSwapCurve(df,VarianceSwapFormula)
    curveFV = CalcFwdVarCurve(curveVS,eps=0.003)
    fvMid = curveFV["mid"]
    fvFunc = FwdVarCurveFunc(Texp,fvMid)
    fvFuncSmth0 = SmoothFwdVarCurveFunc(Texp,curveVS["mid"])
    fvFuncSmth1 = SmoothFwdVarCurveFunc(Texp,curveVS["mid"],eps=0.003)
    T = np.linspace(0,2,1000)

    if run == 1: # Swap curve
        vsMid = curveVS["mid"]
        print(curveVS)
        fig = plt.figure(figsize=(6,4))
        plt.plot(Texp, vsMid, c='r', label="variance swap")
        plt.title("Swap Curve (SPX 20191220)")
        plt.xlabel("maturity")
        plt.ylabel("swap price")
        plt.legend()
        fig.tight_layout()
        plt.savefig(dataFolder+"test_SwapCurve2019.png")
        plt.close()

    elif run == 2: # Fwd var curve
        print(curveFV)
        fig = plt.figure(figsize=(6,4))
        plt.scatter(Texp, fvMid, c='k', s=5)
        plt.plot(T, fvFunc(T), 'k', lw=1, label='sprd-smoothed vswp')
        plt.plot(T, fvFuncSmth0(T), 'r--', lw=1, label='FW-smoothed vswp')
        plt.plot(T, fvFuncSmth1(T), 'r', lw=1, label='FW/sprd-smoothed vswp')
        plt.title("Forward Variance Curve (SPX 20191220)")
        plt.xlabel("maturity")
        plt.ylabel("forward variance")
        plt.legend()
        fig.tight_layout()
        plt.savefig(dataFolder+"test_FwdVarCurve2019.png")
        plt.close()

    elif run == 3: # SPX Fwd 1m-vswp curve
        print(curveFV)
        fig = plt.figure(figsize=(6,4))
        # plt.plot(T, FwdVarSwapFunc(fvFunc)(T), 'k', lw=1, label='sprd-smoothed vswp')
        # plt.plot(T, FwdVarSwapFunc(fvFuncSmth0)(T), 'r--', lw=1, label='FW-smoothed vswp')
        # plt.plot(T, FwdVarSwapFunc(fvFuncSmth1)(T), 'r', lw=1, label='FW/sprd-smoothed vswp')
        # plt.plot(T, FwdVarSwapFunc(fvFuncSmth1)(T), 'k', lw=1, label='FW/sprd-smoothed vswp')
        plt.plot(T, 100*np.sqrt(FwdVarSwapFunc(fvFuncSmth1)(T)), 'k', lw=1, label='FW/sprd-smoothed vswp')
        plt.title("Forward 1-month Variance Swap Curve (SPX 20191220)")
        plt.xlabel("maturity")
        plt.ylabel("swap price (% vol)")
        plt.legend()
        fig.tight_layout()
        plt.savefig(dataFolder+"test_FwdVswpCurve2019.png")
        plt.close()

    elif run == 4: # VIX Fwd 1m-vswp curve
        dfvix = pd.read_csv("vixVols20191220.csv").dropna()
        Texpvix = dfvix["Texp"].unique()
        curveVSvix = CalcVIXSwapCurve(dfvix)
        print(curveVSvix)

        fig = plt.figure(figsize=(6,4))
        plt.scatter(Texpvix, 100*np.sqrt(FwdVarSwapFunc(fvFuncSmth1)(Texpvix)), c='b', s=20, label='SPX')
        plt.scatter(Texpvix, 100*np.sqrt(curveVSvix["mid"]), c='r', s=20, label='VIX')
        plt.plot(Texpvix, 100*np.sqrt(curveVSvix["mid"])-0.5, 'r--', lw=1)
        plt.plot(Texpvix, 100*np.sqrt(curveVSvix["mid"])+0.5, 'r--', lw=1)
        plt.title("Forward 1-month Variance Swap Curve")
        plt.xlabel("maturity")
        plt.ylabel("swap price (% vol)")
        plt.legend()
        fig.tight_layout()
        plt.savefig(dataFolder+"test_VixFwdVswpCurve2019.png")
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
        # "RHP":    {"CF": rHestonPadeCharFunc,     "params": paramsRHPkey},
        # "RHPM":   {"CF": rHestonPoorMansCharFunc, "params": paramsRHPMkey},
        # "VGSA":   {"CF": VGSACharFunc,            "params": paramsVGSAkey},
        # "CGMYSA": {"CF": CGMYSACharFunc,          "params": paramsCGMYSAkey},
        # "NIGSA":  {"CF": NIGSACharFunc,           "params": paramsNIGSAkey},
    }

    if "RHP" in models:
        df = pd.read_csv("spxVols20170424.csv")
        df = df.drop(df.columns[0], axis=1)
        Texp = df["Texp"].unique()
        curveVS = CalcSwapCurve(df,VarianceSwapFormula)
        curveFV = CalcFwdVarCurve(curveVS)
        fvMid = curveFV["mid"]
        fvFunc = FwdVarCurveFunc(Texp,fvMid,"const")

    k = np.arange(-0.3,0.3,0.01)
    T = np.arange(0.1,2.1,0.1)
    X,Y = np.meshgrid(k,T)

    if 1 in run:
        for model in models.keys():
            cal = pd.read_csv(dataFolder+f"Calibration/test_{model}CalibrationIv.csv")
            params = cal[models[model]["params"]].iloc[0].to_dict()
            if model == "RHP":
                impVolFunc = CharFuncImpliedVol(models[model]["CF"](**params,fvFunc=fvFunc),optionType="call",formulaType="COS",N=6000)
            else:
                impVolFunc = CharFuncImpliedVol(models[model]["CF"](**params),optionType="call",formulaType="COS",N=6000)
            Z = np.array([impVolFunc(k,t) for t in T])
            Z[Z<1e-8] = np.nan
            df = pd.DataFrame(np.array([X,Y,Z]).reshape(3,-1).T,columns=["Log-strike","Texp","IV"])
            df.to_csv(dataFolder+f"Implied Vol Surface/IVS_{model}.csv",index=False)

    if 2 in run:
        for model in models.keys():
            df = pd.read_csv(dataFolder+f"Implied Vol Surface/IVS_{model}.csv")
            PlotImpliedVolSurface(df,dataFolder+f"Implied Vol Surface/IVS_{model}.png",model)

#### Plot Local Vol Surface ####################################################

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
        # "RHP": {
        #     "CF": rHestonPadeCharFunc,
        #     "paramsVal": paramsRHPval,
        #     "paramsKey": paramsRHPkey,
        #     "paramsBnd": paramsRHPbnd,
        # },
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

    if "RHP" in models:
        Texp = df["Texp"].unique()
        curveVS = CalcSwapCurve(df,VarianceSwapFormula)
        curveFV = CalcFwdVarCurve(curveVS)
        # curveFV = CalcFwdVarCurve(curveVS,eps=0.003)
        fvMid = curveFV["mid"]
        fvFunc = FwdVarCurveFunc(Texp,fvMid,"const")
        # fvFunc = SmoothFwdVarCurveFunc(Texp,curveVS["mid"])
        # fvFunc = SmoothFwdVarCurveFunc(Texp,curveVS["mid"],eps=0.003)

    for model in models.keys():
        if model == "RHP":
            x = CalibrateModelToImpliedVolFast(k,T,iv,models[model]["CF"],models[model]["paramsVal"],models[model]["paramsKey"],bounds=models[model]["paramsBnd"],w=w,optionType="call",inversionMethod="Bisection",useGlobal=True,curryCharFunc=True,formulaType="COS",optMethod="Evolution",kwargsCF={"fvFunc":fvFunc})
            impVolFunc = CharFuncImpliedVol(models[model]["CF"](*x,fvFunc=fvFunc),optionType="call",formulaType="COS")
        else:
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

#### SVI #######################################################################

def test_svi():
    sviParams1 = {'a': 0.04, 'b': 0.4, 'sig': 0.1, 'rho': -0.4, 'm': 0.1}
    sviParams2 = {'a': 0.04, 'b': 0.8, 'sig': 0.1, 'rho': -0.4, 'm': 0.1}
    k = np.arange(-0.5,0.5,0.01)
    w1 = svi(**sviParams1)(k)
    w2 = svi(**sviParams2)(k)
    d1 = sviDensity(**sviParams1)(k)
    d2 = sviDensity(**sviParams2)(k)

    fig = plt.figure(figsize=(6,4))
    plt.plot(k, w1, 'k')
    plt.plot(k, w2, 'k--')
    plt.title("SVI Parametrization")
    plt.xlabel("log-strike")
    plt.ylabel("total implied var")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_svi.png")
    plt.close()

    fig = plt.figure(figsize=(6,4))
    plt.plot(k, d1, 'k')
    plt.plot(k, d2, 'k--')
    plt.title("SVI Density")
    plt.xlabel("log-strike")
    plt.ylabel("density")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_sviDensity.png")
    plt.close()

def test_sviCross():
    sviParams1 = {'a': 1.8, 'b': 0.8, 'sig': 0, 'rho': -0.5, 'm': 0}
    sviParams2 = {'a': 1.0, 'b': 1.0, 'sig': 1.0, 'rho': -0.5, 'm': 0}
    k = np.arange(-2,2,0.01)
    w1 = svi(**sviParams1)(k)
    w2 = svi(**sviParams2)(k)

    sviCrx = sviCrossing(sviParams1, sviParams2)
    roots = sviCrx['roots']
    cross = sviCrx['cross']

    fig = plt.figure(figsize=(6,4))
    plt.plot(k, w1, 'k')
    plt.plot(k, w2, 'k--')
    plt.title(f"SVI Crossing: roots={np.round(roots,2)} crossedness={np.round(cross,2)}")
    plt.xlabel("log-strike")
    plt.ylabel("total implied var")
    fig.tight_layout()
    plt.savefig(dataFolder+"test_sviCross.png")
    plt.close()

def test_sviArb():
    sviParams1 = {'a': 1.8, 'b': 0.8, 'sig': 0, 'rho': -0.5, 'm': 0}
    sviParams2 = {'a': 1.0, 'b': 1.0, 'sig': 1.0, 'rho': -0.5, 'm': 0}
    print('calendar loss:', CalendarArbLoss(sviParams1, sviParams2))

    sviParamsVogt = {'a': -0.0410, 'b': 0.1331, 'sig': 0.4153, 'rho': 0.3060, 'm': 0.3586}
    print('butterfly loss:', ButterflyArbLoss(sviParamsVogt))

    k = np.arange(-1.5,1.5,0.01)
    d = sviDensityFactor(**sviParamsVogt)(k)
    fig = plt.figure(figsize=(6,4))
    plt.plot(k, d, 'k')
    plt.title("SVI Density Factor")
    plt.xlabel("log-strike")
    plt.ylabel("density factor $g(k)$")
    plt.grid()
    fig.tight_layout()
    plt.savefig(dataFolder+"test_sviDensityVogt.png")
    plt.close()

def test_FitSimpleSVI():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)

    # fit = FitSimpleSVI(df)
    # fit.to_csv(dataFolder+"fit_SimpleSVI.csv")
    # print(fit)

    fit = pd.read_csv(dataFolder+"fit_SimpleSVI.csv", index_col=0)

    Texp = df["Texp"].unique()
    dfnew = list()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        w = svi(**fit.loc[T].to_dict())(k)
        dfT["Fit"] = np.sqrt(w/T)
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)

    PlotImpliedVol(dfnew, dataFolder+"test_FitSimpleSVI.png", ncol=7)
    # PlotTotalVar(dfnew, dataFolder+"test_FitSimpleSVIw.png", xlim=[-0.2,0.2], ylim=[0,0.004]) # Arbitrage everywhere!

def test_FitArbFreeSimpleSVI():
    # Best performance with high stability!
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)

    guess = pd.read_csv(dataFolder+"fit_SimpleSVI.csv", index_col=0)

    # fit = FitArbFreeSimpleSVI(df,guess)
    # fit.to_csv(dataFolder+"fit_ArbFreeSimpleSVI.csv")
    # print(fit)

    fit = pd.read_csv(dataFolder+"fit_ArbFreeSimpleSVI.csv", index_col=0)

    Texp = df["Texp"].unique()
    dfnew = list()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        w = svi(**fit.loc[T].to_dict())(k)
        dfT["Fit"] = np.sqrt(w/T)
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)

    PlotImpliedVol(dfnew, dataFolder+"test_FitArbFreeSimpleSVI.png", ncol=7)
    # PlotTotalVar(dfnew, dataFolder+"test_FitArbFreeSimpleSVIw.png", xlim=[-0.2,0.2], ylim=[0,0.004]) # No arbitrage!

def test_FitSqrtSVI():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)

    # fit = FitSqrtSVI(df)
    # fit.to_csv(dataFolder+"fit_SqrtSVI.csv")
    # print(fit)

    fit = pd.read_csv(dataFolder+"fit_SqrtSVI.csv", index_col=0)

    Texp = df["Texp"].unique()
    dfnew = list()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        w = svi(**fit.loc[T].to_dict())(k)
        dfT["Fit"] = np.sqrt(w/T)
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)

    PlotImpliedVol(dfnew, dataFolder+"test_FitSqrtSVI.png", ncol=7)
    # PlotTotalVar(dfnew, dataFolder+"test_FitSqrtSVIw.png", xlim=[-0.2,0.2], ylim=[0,0.004])

def test_FitSurfaceSVI():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)

    # fit = FitSurfaceSVI(df,skewKernel='PowerLaw')
    # # fit = FitSurfaceSVI(df,skewKernel='Heston')
    # fit.to_csv(dataFolder+"fit_SurfaceSVI.csv")
    # print(fit)

    fit = pd.read_csv(dataFolder+"fit_SurfaceSVI.csv", index_col=0)

    Texp = df["Texp"].unique()
    dfnew = list()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        w = svi(**fit.loc[T].to_dict())(k)
        dfT["Fit"] = np.sqrt(w/T)
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)

    PlotImpliedVol(dfnew, dataFolder+"test_FitSurfaceSVI.png", ncol=7)
    # PlotTotalVar(dfnew, dataFolder+"test_FitSurfaceSVIw.png", xlim=[-0.2,0.2], ylim=[0,0.004])

def test_FitExtendedSurfaceSVI():
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)

    fit = FitExtendedSurfaceSVI(df)
    fit.to_csv(dataFolder+"fit_ExtSurfaceSVI.csv")
    print(fit)

    # fit = pd.read_csv(dataFolder+"fit_ExtSurfaceSVI.csv", index_col=0)

    Texp = df["Texp"].unique()
    dfnew = list()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        w = svi(**fit.loc[T].to_dict())(k)
        dfT["Fit"] = np.sqrt(w/T)
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)

    PlotImpliedVol(dfnew, dataFolder+"test_FitExtSurfaceSVI.png", ncol=7)
    PlotTotalVar(dfnew, dataFolder+"test_FitExtSurfaceSVIw.png", xlim=[-0.2,0.2], ylim=[0,0.004])

def test_FitArbFreeSimpleSVIWithSqrtSeed():
    # Using sqrt-SVI as seed gives unstable params! e.g. abrupt change in rho
    df = pd.read_csv("spxVols20170424.csv")
    df = df.drop(df.columns[0], axis=1)

    # fit = FitArbFreeSimpleSVIWithSqrtSeed(df,Tcut=1)
    # fit.to_csv(dataFolder+"fit_ArbFreeSimpleSVI_SqrtSeed.csv")
    # print(fit)

    fit = pd.read_csv(dataFolder+"fit_ArbFreeSimpleSVI_SqrtSeed.csv", index_col=0)

    Texp = df["Texp"].unique()
    dfnew = list()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        w = svi(**fit.loc[T].to_dict())(k)
        dfT["Fit"] = np.sqrt(w/T)
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)

    PlotImpliedVol(dfnew, dataFolder+"test_FitArbFreeSimpleSVI_SqrtSeed.png", ncol=7)
    # PlotTotalVar(dfnew, dataFolder+"test_FitArbFreeSimpleSVIw_SqrtSeed.png", xlim=[-0.2,0.2], ylim=[0,0.004])

def test_SVIVolSurface():
    fit = pd.read_csv(dataFolder+"fit_ArbFreeSimpleSVI.csv", index_col=0)
    ivFunc = SVIVolSurface(fit)
    k = np.arange(-0.5,0.55,0.05)
    T = np.arange(0.1,2.1,0.1)
    X,Y = np.meshgrid(k,T)
    Z = ivFunc(k,T)
    iv = pd.DataFrame(np.array([X,Y,Z]).reshape(3,-1).T,columns=["Log-strike","Texp","IV"])
    iv.to_csv(dataFolder+'IVS_ArbFreeSimpleSVI.csv',index=False)
    PlotImpliedVolSurface(iv,dataFolder+f"IVS_ArbFreeSimpleSVI.png")

def test_SVIVolSurface2005():
    run = [1,2]

    if 1 in run:
        df = pd.read_csv("spxVols20050509.csv")
        df = df.dropna()

        guess = pd.read_csv(dataFolder+"fit_SimpleSVI2005.csv", index_col=0)

        fit = FitArbFreeSimpleSVI(df,guess)
        fit.to_csv(dataFolder+"fit_ArbFreeSimpleSVI2005.csv")
        print(fit)

        Texp = df["Texp"].unique()
        dfnew = list()
        for T in Texp:
            dfT = df[df["Texp"]==T].copy()
            k = np.log(dfT["Strike"]/dfT["Fwd"])
            w = svi(**fit.loc[T].to_dict())(k)
            dfT["Fit"] = np.sqrt(w/T)
            dfnew.append(dfT)
        dfnew = pd.concat(dfnew)

        PlotImpliedVol(dfnew, dataFolder+"test_FitArbFreeSimpleSVI2005.png", ncol=4)
        PlotTotalVar(dfnew, dataFolder+"test_FitArbFreeSimpleSVIw2005.png", xlim=[-0.2,0.2], ylim=[0,0.004]) # No arbitrage!

    if 2 in run:
        fit = pd.read_csv(dataFolder+"fit_ArbFreeSimpleSVI2005.csv", index_col=0)

        ivFunc = SVIVolSurface(fit)
        k = np.arange(-0.5,0.52,0.02)
        T = np.arange(0.05,2.05,0.05)
        X,Y = np.meshgrid(k,T)
        Z = ivFunc(k,T)
        iv = pd.DataFrame(np.array([X,Y,Z]).reshape(3,-1).T,columns=["Log-strike","Texp","IV"])
        iv.to_csv(dataFolder+'IVS_ArbFreeSimpleSVI2005.csv',index=False)
        PlotImpliedVolSurface(iv,dataFolder+f"IVS_ArbFreeSimpleSVI2005.png")

    if 3 in run:
        iv = pd.read_csv(dataFolder+'IVS_ArbFreeSimpleSVI2005.csv')
        PlotImpliedVolSurface(iv,dataFolder+f"IVS_ArbFreeSimpleSVI2005.png",surfaceOnly=True)

#### Am Option #################################################################

def test_PriceAmericanOption():
    # Test combinations: (put/call) (K/n) (with/no early-ex)
    # Typically converge with n=5000 for T=1 (check other T to obtain optimal dt!)
    np.set_printoptions(precision=7, suppress=True, linewidth=np.inf)
    run = [1,2,3,4,5,6,7,8]
    if 1 in run:
        K = np.arange(0.1,2.1,0.1)
        r = 0.05
        D = np.exp(-r)
        F = np.exp(r)
        n = 1000
        print('Am call for various K, no early-ex')
        print(PriceAmericanOption_vec(1,F,K,1,r,0.2,'call',n))
        print(D*BlackScholesFormula(F,K,1,0,0.2,'call'))
        print()
    if 2 in run:
        K = np.arange(0.1,2.1,0.1)
        r = 0.05
        D = np.exp(-r)
        F = np.exp(r)
        print('Am call for various n, no early-ex')
        print(D*BlackScholesFormula(F,K,1,0,0.2,'call'))
        print('-----------------------------------')
        for n in np.arange(1000,6000,1000):
            print(PriceAmericanOption_vec(1,F,K,1,r,0.2,'call',n))
        print()
    if 3 in run:
        K = np.arange(0.1,2.1,0.1)
        r = 0
        q = 0.05
        D = np.exp(-r)
        F = np.exp(r-q)
        n = 1000
        print('Am call for various K, with early-ex')
        print(PriceAmericanOption_vec(1,F,K,1,r,0.2,'call',n))
        print(D*BlackScholesFormula(F,K,1,0,0.2,'call'))
        print()
    if 4 in run:
        K = np.arange(0.1,2.1,0.1)
        r = 0
        q = 0.05
        D = np.exp(-r)
        F = np.exp(r-q)
        print('Am call for various n, with early-ex')
        print(D*BlackScholesFormula(F,K,1,0,0.2,'call'))
        print('-----------------------------------')
        for n in np.arange(1000,6000,1000):
            print(PriceAmericanOption_vec(1,F,K,1,r,0.2,'call',n))
        print()
    if 5 in run:
        K = np.arange(0.1,2.1,0.1)
        r = 0
        D = np.exp(-r)
        F = 1
        n = 1000
        print('Am put for various K, no early-ex')
        print(PriceAmericanOption_vec(1,F,K,1,r,0.2,'put',n))
        print(D*BlackScholesFormula(F,K,1,0,0.2,'put'))
        print()
    if 6 in run:
        K = np.arange(0.1,2.1,0.1)
        r = 0
        D = np.exp(-r)
        F = 1
        print('Am put for various n, no early-ex')
        print(D*BlackScholesFormula(F,K,1,0,0.2,'put'))
        print('-----------------------------------')
        for n in np.arange(1000,6000,1000):
            print(PriceAmericanOption_vec(1,F,K,1,r,0.2,'put',n))
        print()
    if 7 in run:
        K = np.arange(0.1,2.1,0.1)
        r = 0.05
        D = np.exp(-r)
        F = np.exp(r)
        n = 1000
        print('Am put for various K, with early-ex')
        print(PriceAmericanOption_vec(1,F,K,1,r,0.2,'put',n))
        print(D*BlackScholesFormula(F,K,1,0,0.2,'put'))
        print()
    if 8 in run:
        K = np.arange(0.1,2.1,0.1)
        r = 0.05
        D = np.exp(-r)
        F = np.exp(r)
        print('Am put for various n, with early-ex')
        print(D*BlackScholesFormula(F,K,1,0,0.2,'put'))
        print('-----------------------------------')
        for n in np.arange(1000,6000,1000):
            print(PriceAmericanOption_vec(1,F,K,1,r,0.2,'put',n))
        print()

def test_AmPrxConvergence():
    # Consider OTM options where pricing error is magnified due to small premium
    # Convergence at n=2^11~2000 for all T! (thus variable/adaptive dt)
    np.set_printoptions(precision=7, suppress=True, linewidth=np.inf)
    run = [3,4]
    K = np.arange(0.1,2.1,0.1)
    N = 2**np.arange(6,14)
    print(K)
    if 1 in run:
        r = 0
        q = 0.05
        D = np.exp(-r)
        F = np.exp(r-q)
        print('Am call for various n, with early-ex')
        print(D*BlackScholesFormula(F,K,1,0,0.2,'call'))
        print('-----------------------------------')
        for n in N:
            print(PriceAmericanOption_vec(1,F,K,1,r,0.2,'call',n))
    if 2 in run:
        r = 0.05
        D = np.exp(-r)
        F = np.exp(r)
        print('Am put for various n, with early-ex')
        print(D*BlackScholesFormula(F,K,1,0,0.2,'put'))
        print('-----------------------------------')
        for n in N:
            print(PriceAmericanOption_vec(1,F,K,1,r,0.2,'put',n))
    if 3 in run:
        T = 2**np.arange(-6,3,dtype='float')
        print('Am call convergence for various T, with early-ex')
        for t in T:
            r = 0
            q = 0.05
            D = np.exp(-r*t)
            F = np.exp((r-q)*t)
            print(f'T={t}')
            print(D*BlackScholesFormula(F,K,t,0,0.2,'call'))
            print('-----------------------------------')
            for n in N:
                print(PriceAmericanOption_vec(1,F,K,t,r,0.2,'call',n))
            print()
    if 4 in run:
        T = 2**np.arange(-6,3,dtype='float')
        print('Am put convergence for various T, with early-ex')
        for t in T:
            r = 0.05
            D = np.exp(-r*t)
            F = np.exp(r*t)
            print(f'T={t}')
            print(D*BlackScholesFormula(F,K,t,0,0.2,'put'))
            print('-----------------------------------')
            for n in N:
                print(PriceAmericanOption_vec(1,F,K,t,r,0.2,'put',n))
            print()

def test_AmPrxForVariousImpVol():
    # Variation of price with implied vol
    # (1) Can a unique flat vol be backed out?
    # (2) What K give unique flat vols?
    # Criterion: P >= (intrinsic value)*(1+delta), delta = 1%
    np.set_printoptions(precision=7, suppress=True, linewidth=np.inf)
    run = [1,2]
    K = np.arange(0.1,2.05,0.05)
    sig = np.arange(0.05,0.65,0.05)
    print(sig)
    if 1 in run:
        r = 0
        q = 0.05
        D = np.exp(-r)
        F = np.exp(r-q)
        print('Am call for various sig, with early-ex')
        for k in K:
            print(f'K=%.2f'%k, PriceAmericanOption_vec(1,F,k,1,r,sig,'call',2000))
    if 2 in run:
        r = 0.05
        D = np.exp(-r)
        F = np.exp(r)
        print('Am put for various sig, with early-ex')
        for k in K:
            print(f'K=%.2f'%k, PriceAmericanOption_vec(1,F,k,1,r,sig,'put',2000))

def test_AmericanOptionImpliedVol():
    # Convergence at n=2^10~1000 ATM!
    np.set_printoptions(precision=7, suppress=True, linewidth=np.inf)
    run = [3,4]
    N = 2**np.arange(6,14)
    K = np.arange(0.1,2.05,0.05)
    # print(N)
    if 1 in run:
        r = 0
        q = 0.05
        D = np.exp(-r)
        F = np.exp(r-q)
        print('Am call vol for various n, with early-ex')
        P0 = PriceAmericanOption(1,F,1,1,r,0.2,'call',8000)
        iv = AmericanOptionImpliedVol_vec(1,F,1,1,r,P0,'call',N)
        print(iv)
    if 2 in run:
        r = 0
        D = np.exp(-r)
        F = np.exp(r)
        print('Am put vol for various n, with early-ex')
        P0 = PriceAmericanOption(1,F,1,1,r,0.2,'put',8000)
        iv = AmericanOptionImpliedVol_vec(1,F,1,1,r,P0,'put',N)
        print(iv)
    if 3 in run:
        r = 0
        q = 0.05
        D = np.exp(-r)
        F = np.exp(r-q)
        P0 = PriceAmericanOption_vec(1,F,K,1,r,0.2,'call',8000)
        iv = AmericanOptionImpliedVol_vec(1,F,K,1,r,P0,'call',1000)
        print('Am call vol for various K, with early-ex')
        print(P0)
        print(iv)
    if 4 in run:
        r = 0.05
        D = np.exp(-r)
        F = np.exp(r)
        P0 = PriceAmericanOption_vec(1,F,K,1,r,0.2,'put',8000)
        iv = AmericanOptionImpliedVol_vec(1,F,K,1,r,P0,'put',1000)
        print('Am put vol for various K, with early-ex')
        print(P0)
        print(iv)

def test_AmericanOptionImpliedForwardAndRate():
    run = 1
    if run == 1:
        S = 437.79
        K = 435
        T = 1.7341269841269842
        Cm = (54.17+58.48)/2
        Pm = (42.55+46.86)/2
        AmericanOptionImpliedForwardAndRate(S,K,T,Cm,Pm)
    elif run == 2:
        S = 437.79
        K = 438
        T = 0.011904762
        Cm = (1.98+2.01)/2
        Pm = (2.34+2.35)/2
        AmericanOptionImpliedForwardAndRate(S,K,T,Cm,Pm)
    elif run == 3:
        S = 437.79
        K = 438
        T = 0.099206349
        Cm = (10.25+10.38)/2
        Pm = (10.37+10.51)/2
        AmericanOptionImpliedForwardAndRate(S,K,T,Cm,Pm)

def test_SPYAmOptionImpFwdAndRate():
    S = 437.79
    df = pd.read_csv('data-futu/option_chain_US.SPY_2022-04-14.csv')
    df = StandardizeOptionsChainDataset(df,'2022-04-14')
    implied = dict()
    Texp = df["Texp"].unique()
    print(f"Texp={Texp}")
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        dfTc = dfT[dfT['Put/Call']=='Call']
        dfTp = dfT[dfT['Put/Call']=='Put']
        Kc = dfTc['Strike']
        Kp = dfTp['Strike']
        K0 = Kc[Kc.isin(Kp)] # common strikes
        dfTc = dfTc[Kc.isin(K0)]
        dfTp = dfTp[Kp.isin(K0)]
        if len(K0) > 0:
            ntm = (K0-S).abs().argmin()
            K = K0.iloc[ntm]
            *_, Cb, Ca = dfTc.iloc[ntm]
            *_, Pb, Pa = dfTp.iloc[ntm]
            Cm = (Cb+Ca)/2
            Pm = (Pb+Pa)/2
            print('-----------------------------------------------------------')
            print(f"T={T} S={S} K={K} Cm={Cm} Pm={Pm}")
            implied[T] = AmericanOptionImpliedForwardAndRate(S,K,T,Cm,Pm,lossType="Dividend",iterLog=True,useGlobal=True)
            print(f"implied: {implied[T]}")
            print('-----------------------------------------------------------')
    print(implied)

def test_SPYAmOptionPlotImpDivAndRate():
    # Weird behavior: loss minimized around r=q, and decreases for larger q...
    S = 437.79
    df = pd.read_csv('data-futu/option_chain_US.SPY_2022-04-14.csv')
    df = StandardizeOptionsChainDataset(df,'2022-04-14')
    Texp = df["Texp"].unique()
    print(Texp)

    T = Texp[5]
    dfT = df[df["Texp"]==T].copy()
    dfTc = dfT[dfT['Put/Call']=='Call']
    dfTp = dfT[dfT['Put/Call']=='Put']
    Kc = dfTc['Strike']
    Kp = dfTp['Strike']
    K0 = Kc[Kc.isin(Kp)] # common strikes
    dfTc = dfTc[Kc.isin(K0)]
    dfTp = dfTp[Kp.isin(K0)]
    if len(K0) > 0:
        ntm = (K0-S).abs().argmin()
        K = K0.iloc[ntm]
        *_, Cb, Ca = dfTc.iloc[ntm]
        *_, Pb, Pa = dfTp.iloc[ntm]
        Cm = (Cb+Ca)/2
        Pm = (Pb+Pa)/2
        print(f"T={T} S={S} K={K} Cm={Cm} Pm={Pm}")

        def loss(params):
            q, r = params
            D = np.exp(-r*T)
            F = S*np.exp((r-q)*T)
            sigC = AmericanOptionImpliedVol(S, F, K, T, r, Cm, 'call', 500)
            sigP = AmericanOptionImpliedVol(S, F, K, T, r, Pm, 'put', 500)
            Cbs = BlackScholesFormula(F, K, T, 0, sigC, 'call')
            Pbs = BlackScholesFormula(F, K, T, 0, sigP, 'put')
            Fi = (Cbs-Pbs) + K
            qi = r-np.log(Fi/S)/T
            # loss = 10000*(q-qi)**2
            loss = 100*(q-qi)
            print(f"params: {params} loss: {loss}")
            print(f"  r={r} q={q} qi={qi} F={F} Fi={Fi} sigC={sigC} sigP={sigP}")
            return loss

        loss_vec = np.vectorize(loss)
        fig = plt.figure(figsize=(6,4))
        #### (1) grid (r,q) varying both
        # for r in np.arange(0,0.11,0.01):
        #     Q = np.arange(0,0.11,0.01)
        #     L = [loss((q,r)) for q in Q]
        #     plt.scatter(Q,L,s=20,label='$r=%.2f$'%r)
        #### (2) r=q varying q
        # Q = np.arange(0,0.41,0.01)
        # L = [loss((q,q)) for q in Q]
        #### (3) fixed r varying q
        Q = np.arange(0.00,0.02,0.0002)
        L = [loss((q,0.01)) for q in Q]
        plt.scatter(Q,L,c='k',s=20)
        plt.xlabel('div $q$')
        plt.ylabel('loss')
        plt.title(f'T={T}')
        plt.ylim([-10,10])
        # plt.legend()
        fig.tight_layout()
        plt.savefig(dataFolder+f"test_lossForVariousDivAndRate.png")
        plt.close()

def test_DeAmericanizedOptionsChainDataset():
    S = 437.79
    df = pd.read_csv('data-futu/option_chain_US.SPY_2022-04-14.csv')
    df = StandardizeOptionsChainDataset(df,'2022-04-14')
    df = DeAmericanizedOptionsChainDataset(df,S,400,useGlobal=True)
    ivdf = GenerateImpVolDatasetFromStdDf(df,volCorrection='delta')
    df.to_csv(dataFolder+f'spyPrxs20220414_deam.csv',index=False)
    ivdf.to_csv(dataFolder+f'spyVols20220414_deam.csv',index=False)
    # PlotImpliedVol(pd.read_csv(dataFolder+"spyVols20220414_deam.csv").dropna(), dataFolder+"test_SPYimpliedvol2022_deam.png", ncol=7)

if __name__ == '__main__':
    #### Options Chain ####
    # test_GenerateYfinOptionsChainDataset()
    # test_StandardizeOptionsChainDataset()
    # test_GenerateImpVolDatasetFromStdDf()
    # test_TermStructure()
    #### Black-Scholes ####
    # test_BlackScholesImpVol()
    # test_BlackScholesImpVolInterp()
    # test_BlackScholesImpVolRational()
    # test_PlotImpliedVol()
    # test_PlotImpliedVol2019()
    # test_PlotImpliedVolSPY2022()
    # test_PlotImpliedVolQQQ2022()
    #### Var Curve ####
    # test_VarianceSwapFormula()
    # test_CalcSwapCurve()
    # test_LevSwapCurve()
    # test_CalcFwdVarCurve()
    # test_CalcFwdVarCurve2005()
    # test_VswpPriceCompare()
    # test_FwdVswp2019()
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
    # test_CalibrateRHPModelToImpVol()
    # test_ImpVolFromRHPIvCalibration()
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
    #### SVI ####
    # test_svi()
    # test_sviCross()
    # test_sviArb()
    # test_FitSimpleSVI()
    # test_FitArbFreeSimpleSVI()
    # test_FitSqrtSVI()
    # test_FitSurfaceSVI()
    # test_FitExtendedSurfaceSVI()
    # test_FitArbFreeSimpleSVIWithSqrtSeed()
    # test_SVIVolSurface()
    # test_SVIVolSurface2005()
    #### Am Option ####
    # test_PriceAmericanOption()
    # test_AmPrxConvergence()
    # test_AmPrxForVariousImpVol()
    # test_AmericanOptionImpliedVol()
    # test_AmericanOptionImpliedForwardAndRate()
    # test_SPYAmOptionImpFwdAndRate()
    test_SPYAmOptionPlotImpDivAndRate()
    # test_DeAmericanizedOptionsChainDataset()
