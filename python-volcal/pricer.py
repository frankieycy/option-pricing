import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.integrate import quad
from scipy.interpolate import splrep, splev, pchip, interp1d
plt.switch_backend("Agg")

def BlackScholesFormulaCall(currentPrice, strike, maturity, riskFreeRate, impliedVol):
    # Black Scholes formula for call
    logMoneyness = np.log(currentPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    riskFreeRateFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol
    price = currentPrice * norm.cdf(d1) - riskFreeRateFactor * strike * norm.cdf(d2)
    return price

def BlackScholesFormulaPut(currentPrice, strike, maturity, riskFreeRate, impliedVol):
    # Black Scholes formula for put
    logMoneyness = np.log(currentPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    riskFreeRateFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol
    price = riskFreeRateFactor * strike * norm.cdf(-d2) - currentPrice * norm.cdf(-d1)
    return price

def BlackScholesImpliedVolCall(currentPrice, strike, maturity, riskFreeRate, price):
    # Black Scholes implied volatility for call
    nStrikes = len(strike) if isinstance(strike, np.ndarray) else 1
    impVol0 = np.repeat(1e-10, nStrikes)
    impVol1 = np.repeat(10., nStrikes)
    price0 = BlackScholesFormulaCall(currentPrice, strike, maturity, riskFreeRate, impVol0)
    price1 = BlackScholesFormulaCall(currentPrice, strike, maturity, riskFreeRate, impVol1)
    while np.mean(impVol1-impVol0) > 1e-10:
        impVol2 = (impVol0+impVol1)/2
        price2 = BlackScholesFormulaCall(currentPrice, strike, maturity, riskFreeRate, impVol2)
        price0 += (price2<price)*(price2-price0)
        impVol0 += (price2<price)*(impVol2-impVol0)
        price1 += (price2>=price)*(price2-price1)
        impVol1 += (price2>=price)*(impVol2-impVol1)
    return impVol2

def BlackScholesImpliedVolPut(currentPrice, strike, maturity, riskFreeRate, price):
    # Black Scholes implied volatility for put
    nStrikes = len(strike) if isinstance(strike, np.ndarray) or isinstance(strike, list) else 1
    impVol0 = np.repeat(1e-10, nStrikes)
    impVol1 = np.repeat(10., nStrikes)
    price0 = BlackScholesFormulaPut(currentPrice, strike, maturity, riskFreeRate, impVol0)
    price1 = BlackScholesFormulaPut(currentPrice, strike, maturity, riskFreeRate, impVol1)
    while np.mean(impVol1-impVol0) > 1e-10:
        impVol2 = (impVol0+impVol1)/2
        price2 = BlackScholesFormulaPut(currentPrice, strike, maturity, riskFreeRate, impVol2)
        price0 += (price2<price)*(price2-price0)
        impVol0 += (price2<price)*(impVol2-impVol0)
        price1 += (price2>=price)*(price2-price1)
        impVol1 += (price2>=price)*(impVol2-impVol1)
    return impVol2

def BlackScholesImpliedVolOTM(currentPrice, strike, maturity, riskFreeRate, price):
    # Black Scholes implied volatility for OTM option
    forwardPrice = currentPrice*np.exp(riskFreeRate*maturity)
    impVol = BlackScholesImpliedVolCall(currentPrice, strike, maturity, riskFreeRate, price) if strike > forwardPrice else \
        BlackScholesImpliedVolPut(currentPrice, strike, maturity, riskFreeRate, price)
    return impVol

def LewisFormulaOTM(charFunc, logStrike, maturity):
    # Lewis formula for OTM option
    integrand = lambda u: np.real(np.exp(-1j*u*logStrike) * charFunc(u-1j/2, maturity) / (u**2+.25))
    logStrikeMinus = (logStrike<0)*logStrike
    price = np.exp(logStrikeMinus) - np.exp(logStrike/2)/np.pi * quad(integrand, 0, np.inf)[0]
    return price

def LewisFFTFormulaOTM(charFunc, logStrike, maturity, interp="cubic", N=2**12, B=200):
    # Lewis FFT formula for OTM option
    du = B/N
    u = np.arange(N)*du
    w = np.arange(N)
    w = 3+(-1)**(w+1)
    w[0] = 1; w[N-1] = 1
    dk = 2*np.pi/B
    b = N*dk/2
    k = -b+np.arange(N)*dk
    I = w * np.exp(1j*b*u) * charFunc(u-1j/2, maturity) / (u**2+0.25) * du/3
    Ifft = np.real(fft(I))
    spline = interp1d(k, Ifft, kind=interp)
    logStrikeMinus = (logStrike<0)*logStrike
    price = np.exp(logStrikeMinus) - np.exp(logStrike/2)/np.pi * spline(logStrike)
    return price

def CharFuncImpliedVol(charFunc, FFT=False):
    # Implied volatility for OTM option priced with charFunc
    LewisFormula = LewisFFTFormulaOTM if FFT else LewisFormulaOTM
    def impVolFunc(logStrike, maturity):
        return BlackScholesImpliedVolOTM(1, np.exp(logStrike), maturity, 0, LewisFormula(charFunc, logStrike, maturity))
    return impVolFunc

def HestonCharFunc(meanRevRate, correlation, volOfVol, meanVol, currentVol, riskFreeRate=0):
    # Characteristic function for Heston model
    def charFunc(u, maturity):
        alpha = -u**2/2-1j*u/2
        beta = meanRevRate-correlation*volOfVol*1j*u
        gamma = volOfVol**2/2
        d = np.sqrt(beta**2-4*alpha*gamma)
        rp = (beta+d)/(2*gamma)
        rm = (beta-d)/(2*gamma)
        g = rm/rp
        D = rm*(1-np.exp(-d*maturity))/(1-g*np.exp(-d*maturity))
        C = meanRevRate*(rm*maturity-2/volOfVol**2*np.log((1-g*np.exp(-d*maturity))/(1-g)))
        return np.exp(1j*u*riskFreeRate*maturity+C*meanVol+D*currentVol)
    return charFunc

def MertonJumpCharFunc():
    # Characteristic function for Merton-Jump model
    def charFunc(u, maturity):
        pass
    return charFunc

def VarianceGammaCharFunc():
    # Characteristic function for Variance-Gamma model
    def charFunc(u, maturity):
        pass
    return charFunc

def plotImpliedVol(df, figname=None, ncol=6):
    # Plot bid-ask implied volatilities based on df
    if not figname:
        figname = "impliedvol.png"
    Texp = df["Texp"].unique()
    Nexp = len(Texp)
    nrow = int(np.ceil(Nexp/ncol))
    fig, ax = plt.subplots(nrow,ncol,figsize=(15,10))
    for i in range(nrow*ncol):
        ix,iy = i//ncol,i%ncol
        if i < Nexp:
            T = Texp[i]
            dfT = df[df["Texp"]==T]
            k = np.log(dfT["Strike"]/dfT["Fwd"])
            bid = dfT["Bid"]
            ask = dfT["Ask"]
            ax[ix,iy].scatter(k,bid,c='r',s=5)
            ax[ix,iy].scatter(k,ask,c='b',s=5)
            ax[ix,iy].set_title(rf"$T={np.round(T,3)}$")
            ax[ix,iy].set_xlabel("log-strike")
            ax[ix,iy].set_ylabel("implied vol")
        else:
            ax[ix,iy].axis("off")
    fig.tight_layout()
    plt.savefig(figname)
    plt.close()

def VarianceSwapFormula(logStrike, maturity, impliedVol, showPlot=False):
    # Fukasawa robust variance swap formula
    logStrike,impliedVol = np.array(logStrike),np.array(impliedVol)
    totalImpVol = impliedVol*np.sqrt(maturity)
    d2 = -logStrike/totalImpVol-totalImpVol/2
    y = norm.cdf(d2)
    ord = np.argsort(y)
    ymin,ymax = np.min(y),np.max(y)
    y,d2,logStrike,totalImpVol = y[ord],d2[ord],logStrike[ord],totalImpVol[ord]

    if showPlot:
        yint = np.linspace(ymin,ymax,200)
        fig = plt.figure(figsize=(6,4))
        plt.scatter(y, totalImpVol, c='k', s=5)
        plt.title(rf"$T={np.round(maturity,3)}$")
        plt.xlabel("$y=N(d_2)$")
        plt.ylabel("total implied vol")
        fig.tight_layout()
        plt.savefig("test_totalImpVolVsY.png")
        plt.close()

    # tck = splrep(y, totalImpVol**2, s=0)
    # intTotalImpVar = lambda x: splev(x, tck, der=0)
    pch = pchip(y, totalImpVol**2)
    intTotalImpVar = lambda x: pch(x)

    areaMid = quad(intTotalImpVar, ymin, ymax, limit=1000)[0]
    areaMin = totalImpVol[0]**2*norm.cdf(d2[0])
    areaMax = totalImpVol[-1]**2*norm.cdf(-d2[-1])
    price = areaMin + areaMid + areaMax
    return price

def GammaSwapFormula(logStrike, maturity, impliedVol):
    # Fukasawa robust gamma swap formula
    logStrike,impliedVol = np.array(logStrike),np.array(impliedVol)
    totalImpVol = impliedVol*np.sqrt(maturity)
    d1 = -logStrike/totalImpVol+totalImpVol/2
    y = norm.cdf(d1)
    ord = np.argsort(y)
    ymin,ymax = np.min(y),np.max(y)
    y,d1,logStrike,totalImpVol = y[ord],d1[ord],logStrike[ord],totalImpVol[ord]

    # tck = splrep(y, totalImpVol**2, s=0)
    # intTotalImpVar = lambda x: splev(x, tck, der=0)
    pch = pchip(y, totalImpVol**2)
    intTotalImpVar = lambda x: pch(x)

    areaMid = quad(intTotalImpVar, ymin, ymax, limit=1000)[0]
    areaMin = totalImpVol[0]**2*norm.cdf(d1[0])
    areaMax = totalImpVol[-1]**2*norm.cdf(-d1[-1])
    price = areaMin + areaMid + areaMax
    return price

def LeverageSwapFormula(logStrike, maturity, impliedVol):
    # Fukasawa robust leverage swap formula
    return GammaSwapFormula(logStrike, maturity, impliedVol) - VarianceSwapFormula(logStrike, maturity, impliedVol)

def calcSwapCurve(df, swapFormula):
    # Calculate swap curves based on implied volatilities in df
    Texp = df["Texp"].unique()
    Nexp = len(Texp)
    curve = {c: list() for c in ["bid","mid","ask"]}
    for T in Texp:
        dfT = df[df["Texp"]==T]
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        bid = dfT["Bid"]
        ask = dfT["Ask"]
        mid = (bid+ask)/2
        curve["bid"].append(swapFormula(k,T,bid)/T)
        curve["mid"].append(swapFormula(k,T,mid)/T)
        curve["ask"].append(swapFormula(k,T,ask)/T)
    curve = pd.DataFrame(curve)
    curve["Texp"] = Texp
    curve = curve[["Texp","bid","mid","ask"]]
    return curve

def calcFwdVarCurve(curveVS):
    # Calculate forward variance curve based on VS curve
    Texp = curveVS["Texp"]
    diffTexp = curveVS["Texp"].diff()
    price = curveVS[["bid","mid","ask"]].multiply(Texp,axis=0)
    curve = price.diff()
    curve = curve.div(diffTexp,axis=0)
    curve.iloc[0] = price.iloc[0]/Texp.iloc[0]
    curve["Texp"] = Texp
    curve = curve[["Texp","bid","mid","ask"]]
    return curve

def FwdVarCurveFunc(maturity, fwdVar, fitType="const"):
    # Smooth out forward variance curve
    Texp = maturity
    Nexp = len(Texp)
    curveFunc = None
    if fitType == "const":
        curveFunc = lambda t: fwdVar[min(sum(Texp<t),Nexp-1)]
    return curveFunc
