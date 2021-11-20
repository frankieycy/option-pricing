import re
import math
import subprocess
import cvxpy as cvx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.stats import norm
from mpl_toolkits import mplot3d
from yahoo_fin import stock_info, options
from util import *
# plt.switch_backend("Agg")

LOG = True

onDate = getPrevBDay()
maxMaturity = "2022-12-31"
stockList = ["SPY"]
# stockList = stock_info.tickers_dow()

exeFolder = "../exe/"
dataFolder = "data/"
plotFolder = "assets/"

def logMessage(msg):
    if LOG:
        logMsg = getCurrentTime()+" [LOG] "
        if type(msg) is list:
            for m in msg: logMsg += str(m)
        else: logMsg += str(msg)
        print(logMsg)

def calcRiskFreeRate(discountRate, maturity):
    riskFreeRate = (-252/maturity)*np.log(1-maturity*discountRate/360)
    return riskFreeRate

def getDateFromContractName(name):
    return name[-15:-9]

def printPricerVariablesToCsvFile(fileName, pricerVariables):
    file = open(fileName,"w")
    for var in pricerVariables:
        if type(var) is str: sub = "%s"
        elif type(var) is float: sub = "%.10f"
        file.write(("%s,"+sub+"\n")%(var,pricerVariables[var]))
    file.close()

def printOptionChainsToCsvFile(fileName, optionChains, optionPriceType="mid"):
    file = open(fileName,"w")
    optionDates = optionChains.keys()
    for date in optionDates:
        daysFromToday = bDaysBetween(onDate,date)
        maturity = daysFromToday/252
        for putCall in optionChains[date]:
            if putCall == "calls": type = "Call"
            elif putCall == "puts": type = "Put"
            for idx,row in optionChains[date][putCall].iterrows():
                name, strike, price = row["Contract Name"], row["Strike"], 0
                if optionPriceType == "mid":
                    price = (row["Bid"]+row["Ask"])/2
                elif optionPriceType == "last":
                    price = row["Last Price"]
                elif optionPriceType == "arb-free":
                    price = row["Arb-Free Price"]
                file.write("%s,European,%s,%.10f,%.10f,%.10f\n"%
                    (name,type,strike,maturity,price))
    file.close()

def arbitrageFreeSmoothing(optionChains, pricerVariables, weightType="uniform", penalty=1e-7, verbose=False):
    S, r, q = pricerVariables["currentPrice"], pricerVariables["riskFreeRate"], pricerVariables["dividendYield"]
    nOptionDates = len(optionChains)
    optionDates = list(optionChains.keys())
    optionMaturities = {date: bDaysBetween(onDate,date)/252 for date in optionDates}
    rateFactor = {date: np.exp(r*optionMaturities[date]) for date in optionDates}
    yieldFactor = {date: np.exp(q*optionMaturities[date]) for date in optionDates}
    driftFactor = {date: np.exp((r-q)*optionMaturities[date]) for date in optionDates}
    splineParameters = {date: {} for date in optionDates}
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    def splinePrice(K, maturity, putCall):
        a_spl = splineParameters[maturity][putCall]['a']
        b_spl = splineParameters[maturity][putCall]['b']
        c_spl = splineParameters[maturity][putCall]['c']
        d_spl = splineParameters[maturity][putCall]['d']
        u_spl = splineParameters[maturity][putCall]['u']
        j = np.argmax(K<u_spl)-1
        return max(a_spl[j]+b_spl[j]*(K-u_spl[j])+c_spl[j]*(K-u_spl[j])**2+d_spl[j]*(K-u_spl[j])**3,0)

    for idx, maturity in enumerate(optionDates[::-1]):
        logMessage("===================================================================")
        logMessage(["calibrating arbitrage-free price surface on ",maturity])
        logMessage("===================================================================")
        nextMaturity = optionDates[nOptionDates-idx] if idx>0 else None
        for putCall in ["puts","calls"]:
            chain = optionChains[maturity][putCall]
            mid = (chain["Bid"]+chain["Ask"])/2
            u = chain["Strike"]
            mid,u = mid.values,u.values

            n = len(u)
            h = u[1:]-u[:-1] # dim n-1
            hi = 1/h

            if weightType == "uniform":
                w = np.ones(n)/n
            elif weightType == "bid-ask":
                w = (chain["Ask"]+chain["Bid"])/(chain["Ask"]-chain["Bid"]).values
                w /= sum(w)

            Ru = Rd = h[1:-1]/6
            Rm = (h[:-1]+h[1:])/3
            R = np.diag(Rm,0)+np.diag(Ru,1)+np.diag(Rd,-1)

            Q = np.zeros((n,n-2))
            i = np.arange(n-2)
            Q[i,i] = hi[:-1]
            Q[i+1,i] = -hi[:-1]-hi[1:]
            Q[i+2,i] = hi[1:]

            W = np.diag(w)
            A = np.vstack((Q,-R.T))
            B = block_diag(W,penalty*R)
            x = cvx.Variable(2*n-2)
            y = np.concatenate([w*mid,np.zeros(n-2)])

            objective = cvx.Minimize(-y@x+.5*cvx.quad_form(x,B))
            constraints = []
            if putCall == "puts":
                constraints += [
                    A.T@x == 0,
                    #### strike arbitrage
                    x[n:] >= 0,
                    #### derivative bound
                    (x[1]-x[0])/h[0]-h[0]/6*x[n] >= 0,
                    (x[n-1]-x[n-2])/h[-1]+h[-1]/6*x[-1] <= 1/rateFactor[maturity],
                    #### price bound
                    x[n-1] <= u[-1]/rateFactor[maturity],
                    x[n-1] >= u[-1]/rateFactor[maturity]-S/yieldFactor[maturity],
                    x[0] >= 0
                ]
                #### calendar arbitrage
                # if idx > 0:
                #     for i in range(n):
                #         fwdStrike = u[i]*driftFactor[nextMaturity]/driftFactor[maturity]
                #         bnd = splinePrice(fwdStrike, nextMaturity, putCall)*yieldFactor[nextMaturity]/yieldFactor[maturity]
                #         constraints += [x[i] <= bnd]
            elif putCall == "calls":
                constraints += [
                    A.T@x == 0,
                    #### strike arbitrage
                    x[n:] >= 0,
                    #### derivative bound
                    (x[1]-x[0])/h[0]-h[0]/6*x[n] >= -1/rateFactor[maturity],
                    (x[n-1]-x[n-2])/h[-1]+h[-1]/6*x[-1] <= 0,
                    #### price bound
                    x[0] <= S/yieldFactor[maturity],
                    x[0] >= S/yieldFactor[maturity]-u[0]/rateFactor[maturity],
                    x[n-1] >= 0
                ]
                #### calendar arbitrage
                # if idx > 0:
                #     for i in range(n):
                #         fwdStrike = u[i]*driftFactor[nextMaturity]/driftFactor[maturity]
                #         bnd = splinePrice(fwdStrike, nextMaturity, putCall)*yieldFactor[nextMaturity]/yieldFactor[maturity]
                #         constraints += [x[i] <= bnd]

            prob = cvx.Problem(objective, constraints)
            try:
                prob.solve(verbose=verbose); x = x.value
                chain["Arb-Free Price"] = x[:n]

                g = x[:n] # dim n
                gamma = np.concatenate([[0],x[n:],[0]]) # dim n
                a = np.concatenate([[0],g[:-1],[0]])
                b = np.concatenate([[0],(g[1:]-g[:-1])/h-h/6*(2*gamma[:-1]+gamma[1:]),[0]])
                c = np.concatenate([[0],gamma[:-1]/2,[0]])
                d = np.concatenate([[0],(gamma[1:]-gamma[:-1])/(6*h),[0]])
                b[0], b[-1] = b[1], (g[-1]-g[-2])/h[-1]+h[-1]/6*(gamma[-3]+2*gamma[-1])
                a[0], a[-1] = g[0]-b[0]*u[0], g[-1]
                u = np.concatenate([[0],u])

                splineParameters[maturity][putCall] = {
                    'a': np.copy(a),
                    'b': np.copy(b),
                    'c': np.copy(c),
                    'd': np.copy(d),
                    'u': np.copy(u),
                }
            except Exception:
                chain["Arb-Free Price"] = mid # midprice fallback
                splineParameters[maturity][putCall] = {}

    return optionChains

def generateImpliedVolSurfaceInputFiles(stock, optionPriceType="mid", zeroBond="^IRX", zeroBondMaturity=85):
    logMessage(["starting generateImpliedVolSurfaceInputFiles on stock ",stock,", optionPriceType ",optionPriceType,
        ", zeroBond ",zeroBond," of zeroBondMaturity ",zeroBondMaturity," days"])
    currentPrice = stock_info.get_data(stock,onDate).iloc[0]["close"]
    discountRate = stock_info.get_data(zeroBond,onDate).iloc[0]["close"]
    try:
        quote_table = stock_info.get_quote_table(stock)
        if "Forward Dividend & Yield" in quote_table: # single name
            dividendYield = quote_table["Forward Dividend & Yield"]
            dividendYield = float(re.sub("[()%]","",dividendYield.split()[1]))/100
        elif "Yield" in quote_table: # index ETF
            dividendYield = quote_table["Yield"]
            dividendYield = float(re.sub("[%]","",dividendYield))/100
    except: dividendYield = 0

    riskFreeRate = calcRiskFreeRate(discountRate,zeroBondMaturity)
    pricerVariables = {
        "stockName": stock,
        "currentPrice": currentPrice,
        "riskFreeRate": riskFreeRate,
        "dividendYield": dividendYield,
    }

    optionDates = options.get_expiration_dates(stock)
    if maxMaturity: optionDates = [date for date in optionDates if bDaysBetween(date,maxMaturity)>0]
    optionChains = {}
    for date in optionDates:
        optionChains[date] = options.get_options_chain(stock,date)
    if optionPriceType == "arb-free":
        optionChains = arbitrageFreeSmoothing(optionChains, pricerVariables)

    makeDirectory(dataFolder)
    printPricerVariablesToCsvFile(dataFolder+"pricer_var.csv",pricerVariables)
    printOptionChainsToCsvFile(dataFolder+"option_data.csv",optionChains,optionPriceType)
    logMessage(["ending generateImpliedVolSurfaceInputFiles with ",
        "pricerVariables ",pricerVariables,", ",
        "optionDates ",optionDates])

def smoother(y, box_pts):
    y_smooth = y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth

def plotImpliedVolSurface(stock, fileName, figName, smooth=False, plot="scatter", angle=[20,80]):
    logMessage(["starting plotImpliedVolSurface on stock ",stock,
        ", fileName ",fileName,", figName ",figName])
    impVolSurface = np.loadtxt(fileName,delimiter=",",usecols=(3,4,5))
    strike   = impVolSurface[:,0]
    maturity = impVolSurface[:,1]
    impVol   = impVolSurface[:,2]
    idx      = impVol<1
    strike   = strike[idx]
    maturity = maturity[idx]
    impVol   = impVol[idx]
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection="3d")
    if smooth:
        impVolSmooth = []
        for m in np.unique(maturity):
            maturityIdx = np.argwhere(maturity==m).flatten()
            box_pts = math.ceil(len(maturityIdx)/10)
            impVolSmooth.append(smoother(impVol[maturityIdx],box_pts))
        impVol = np.concatenate(impVolSmooth)
    if plot=="scatter": ax.scatter3D(strike,maturity,impVol,c="k",marker=".")
    elif plot=="trisurf":
        surf = ax.plot_trisurf(strike,maturity,impVol,cmap="binary",linewidth=1)
        cbar = fig.colorbar(surf,shrink=.4,aspect=15,pad=0,orientation="horizontal")
    ax.set_title("Option implied vol surface of "+stock+" on "+onDate)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (Year)")
    ax.set_zlabel("Implied vol")
    ax.set_zlim(0,1)
    ax.view_init(angle[0],angle[1])
    fig.tight_layout()
    plt.savefig(figName)
    plt.close()
    logMessage("ending plotImpliedVolSurface")

def callExecutable(name):
    logMessage(["starting callExecutable on name ",name])
    proc = subprocess.Popen(name)
    proc.wait()
    logMessage("ending callExecutable")

def impliedVolSurfaceGenerator(stock):
    logMessage(["starting impliedVolSurfaceGenerator on stock ",stock,
        ", onDate ",onDate])
    try:
        generateImpliedVolSurfaceInputFiles(stock)
        callExecutable("./"+
            exeFolder+"genImpliedVolSurface")
        makeDirectory(plotFolder)
        plotImpliedVolSurface(stock,
            dataFolder+"option_vol.csv",
            plotFolder+"option_vol_"+
                stock+"_"+onDate+".png",
            smooth=True,plot="trisurf")
    except Exception: pass
    logMessage("ending impliedVolSurfaceGenerator")

def optionChainsWithGreeksGenerator(stock):
    logMessage(["starting optionChainsWithGreeksGenerator on stock ",stock,
        ", onDate ",onDate])
    dataCols = ["Contract Name","Type","Put/Call","Strike","Maturity (Year)","Market Price"]
    grkCols = ["Contract Name","Type","Put/Call","Strike","Maturity (Year)","Implied Vol",
        "Delta","Gamma","Vega","Rho","Theta"]
    try:
        callExecutable("./"+
            exeFolder+"genGreeksFromImpVolFile")
        data = pd.read_csv(dataFolder+"option_data.csv",header=None)
        grk  = pd.read_csv(dataFolder+"option_grk.csv",header=None)
        data.columns = dataCols
        grk.columns = grkCols
        data = data.set_index("Contract Name")
        grk = grk.set_index("Contract Name")
        for var in ["Implied Vol","Delta","Gamma","Vega","Rho","Theta"]:
            data[var] = grk[var]
        data.to_csv(dataFolder+"option_chain"+"_"
            +stock+"_"+onDate+".csv")
    except Exception: pass
    logMessage("ending optionChainsWithGreeksGenerator")

def main():
    for stock in stockList:
        impliedVolSurfaceGenerator(stock)
        optionChainsWithGreeksGenerator(stock)

if __name__ == "__main__":
    # main()
    generateImpliedVolSurfaceInputFiles("SPY","arb-free")
