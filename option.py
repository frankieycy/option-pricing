import re
import math
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from yahoo_fin import stock_info, options
from util import *
plt.switch_backend('Agg')

LOG = True

onDate = getPrevBDay()
stockList = ["IBM","JPM","DIS"]
# stockList = stock_info.tickers_dow()

exeFolder = "exe/"
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

def printOptionChainsToCsvFile(fileName, optionChains):
    file = open(fileName,"w")
    optionDates = optionChains.keys()
    for date in optionDates:
        daysFromToday = bDaysBetween(onDate,date)
        maturity = daysFromToday/252
        for putCall in optionChains[date]:
            if putCall == "calls": type = "Call"
            elif putCall == "puts": type = "Put"
            for idx,row in optionChains[date][putCall].iterrows():
                name = row["Contract Name"]
                strike = row["Strike"]
                marketPrice = row["Last Price"]
                file.write("%s,European,%s,%.10f,%.10f,%.10f\n"%
                    (name,type,strike,maturity,marketPrice))
    file.close()

def generateImpliedVolSurfaceInputFiles(stock, zeroBond="^IRX", zeroBondMaturity=85):
    logMessage(["starting generateImpliedVolSurfaceInputFiles on stock ",stock,
        ", zeroBond ",zeroBond," of zeroBondMaturity ",zeroBondMaturity," days"])
    currentPrice = stock_info.get_data(stock,onDate).iloc[0]["close"]
    discountRate = stock_info.get_data(zeroBond,onDate).iloc[0]["close"]
    try:
        dividendYield = stock_info.get_quote_table(stock)["Forward Dividend & Yield"]
        dividendYield = float(re.sub("[()%]","",dividendYield.split()[1]))/100
    except: dividendYield = 0

    riskFreeRate = calcRiskFreeRate(discountRate,zeroBondMaturity)
    pricerVariables = {
        "stockName": stock,
        "currentPrice": currentPrice,
        "riskFreeRate": riskFreeRate,
        "dividendYield": dividendYield,
    }

    optionDates = options.get_expiration_dates(stock)
    optionChains = {}
    for date in optionDates:
        optionChains[date] = options.get_options_chain(stock,date)

    makeDirectory(dataFolder)
    printPricerVariablesToCsvFile(dataFolder+"pricer_var.csv",pricerVariables)
    printOptionChainsToCsvFile(dataFolder+"option_data.csv",optionChains)
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
    dataCols = ['Contract Name','Type','Put/Call','Strike','Maturity (Year)','Market Price']
    grkCols = ['Contract Name','Type','Put/Call','Strike','Maturity (Year)','Implied Vol',
        'Delta','Gamma','Vega','Rho','Theta']
    try:
        callExecutable("./"+
            exeFolder+"genGreeksFromImpVolFile")
        data = pd.read_csv(dataFolder+"option_data.csv",header=None)
        grk  = pd.read_csv(dataFolder+"option_grk.csv",header=None)
        data.columns = dataCols
        grk.columns = grkCols
        data = data.set_index('Contract Name')
        grk = grk.set_index('Contract Name')
        for var in ['Implied Vol','Delta','Gamma','Vega','Rho','Theta']:
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
    main()
