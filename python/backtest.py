import math
import re, glob
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from util import *
plt.switch_backend("Agg")

LOG = True

exeFolder = "../exe/"
dataFolder = "../data-VolArbStrat/"
plotFolder = "../plot-VolArbStrat/"

BacktestData = {
    "labels": [
        "simPrice",
        "stratCash",
        "stratNStock",
        "stratModPrice",
        "stratModValue",
        "stratGrkDelta",
        "stratGrkGamma",
        "stratGrkVega",
        "stratGrkRho",
        "stratGrkTheta"
    ],
    "hLabels": [
        "stratNOption",
        "stratHModPrice"
    ],
    "fullLabels": {
        "simPrice":         "Simulated Price Path",
        "stratCash":        "Strategy Cash Position",
        "stratNStock":      "Strategy Stock Position",
        "stratModPrice":    "Strategy Option Model Price",
        "stratModValue":    "Strategy Total Model Value",
        "stratGrkDelta":    "Strategy Delta",
        "stratGrkGamma":    "Strategy Gamma",
        "stratGrkVega":     "Strategy Vega",
        "stratGrkRho":      "Strategy Rho",
        "stratGrkTheta":    "Strategy Theta",
        "stratNOption":     "Strategy Hedge Option Position",
        "stratHModPrice":   "Strategy Hedge Option Model Price"
    }
}

BacktestStats = {}

def logMessage(msg):
    if LOG:
        logMsg = getCurrentTime()+" [LOG] "
        if type(msg) is list:
            for m in msg: logMsg += str(m)
        else: logMsg += str(msg)
        print(logMsg)

def callExecutable(args):
    logMessage(["starting callExecutable on args ",args])
    proc = subprocess.run(args)
    logMessage("ending callExecutable")

def loadBacktestResults(name="backtest"):
    logMessage(["starting loadBacktestResults on name ",name])
    for label in BacktestData["labels"]:
        file = dataFolder+name+"-"+label+".csv"
        df = pd.read_csv(file,header=None)
        BacktestData[label] = df
    for label in BacktestData["hLabels"]:
        fileName = dataFolder+name+"-"+label+"-*.csv"
        fileList = glob.glob(fileName)
        fileList.sort(key=lambda f:int(re.split("[-.]",f)[-2]))
        BacktestData[label] = []
        for file in fileList:
            df = pd.read_csv(file,header=None)
            BacktestData[label].append(df)
    logMessage("ending loadBacktestResults")

def headBacktestResults(show=10):
    for label in BacktestData["labels"]:
        print("==== BacktestData[%s] ====="%label)
        print(BacktestData[label].head(show))
    for label in BacktestData["hLabels"]:
        print("==== BacktestData[%s] ====="%label)
        for df in BacktestData[label]:
            print(df.head(show))

def plotBacktestResults(figName, maxNumPaths=3):
    logMessage(["starting plotBacktestResults on figName ",figName,", maxNumPaths ",maxNumPaths])
    n = len(BacktestData["labels"])
    fig,ax = plt.subplots(math.ceil(n/2),2,figsize=(10,3*math.ceil(n/2)))
    for idx,label in enumerate(BacktestData["labels"]):
        for i in range(min(maxNumPaths,BacktestData[label].shape[1])):
            ax[idx//2,idx%2].plot(BacktestData[label].iloc[:-1,i],linewidth=1,label="Sim-%d"%i)
        ax[idx//2,idx%2].set_xlim([0,BacktestData[label].shape[0]-2])
        ax[idx//2,idx%2].set_xticklabels([])
        ax[idx//2,idx%2].set_title(BacktestData["fullLabels"][label])
        ax[idx//2,idx%2].grid()
    ax[0,0].legend()
    fig.tight_layout()
    plt.savefig(figName)
    plt.close()
    logMessage("ending plotBacktestResults")

def plotBacktestHResults(figName, maxNumPaths=3):
    logMessage(["starting plotBacktestHResults on figName ",figName,", maxNumPaths ",maxNumPaths])
    n = len(BacktestData["hLabels"])*len(BacktestData[BacktestData["hLabels"][0]])
    fig,ax = plt.subplots(math.ceil(n/2),2,figsize=(10,3*math.ceil(n/2)))
    for idx,label in enumerate(BacktestData["hLabels"]):
        for k,df in enumerate(BacktestData[label]):
            m = idx*len(BacktestData[BacktestData["hLabels"][0]])+k
            for i in range(min(maxNumPaths,df.shape[1])):
                ax[m//2,m%2].plot(df.iloc[:-1,i],linewidth=1,label="Sim-%d"%i)
            ax[m//2,m%2].set_xlim([0,df.shape[0]-2])
            ax[m//2,m%2].set_xticklabels([])
            ax[m//2,m%2].set_title(BacktestData["fullLabels"][label]+"-"+str(k))
            ax[m//2,m%2].grid()
    ax[0,0].legend()
    fig.tight_layout()
    plt.savefig(figName)
    plt.close()
    logMessage("ending plotBacktestHResults")

def calcBacktestStats():
    logMessage("starting calcBacktestStats")
    stratPNL = BacktestData["stratModValue"].iloc[-1,:]
    BacktestStats.update({
        "Min":      np.min(stratPNL),
        "25%":      np.percentile(stratPNL,25),
        "Median":   np.median(stratPNL),
        "75%":      np.percentile(stratPNL,75),
        "Max":      np.max(stratPNL),
        "Mean":     np.mean(stratPNL),
        "Std Dev":  np.std(stratPNL),
        "Skew":     skew(stratPNL),
        "Ex Kurt":  kurtosis(stratPNL),
        "95% VaR":  -np.percentile(stratPNL,5),
        "99% VaR":  -np.percentile(stratPNL,1),
    })
    logMessage("ending calcBacktestStats")

def reportBacktest(name="backtest", maxNumPaths=3):
    loadBacktestResults(name)
    plotBacktestResults(plotFolder+name+"-report.png",maxNumPaths)
    if BacktestData[BacktestData["hLabels"][0]]:
        plotBacktestHResults(plotFolder+name+"-reportH.png",maxNumPaths)
    calcBacktestStats()
    print(BacktestStats)

def plotStratStatsAgainstVar(fileName, figName, var, title=""):
    df = pd.read_csv(fileName)
    x = df[var]
    fig = plt.figure()
    for stat in ["Mean","Std Dev","Min","Max"]:
        plt.scatter(x,df[stat],s=10)
        plt.plot(x,df[stat],linewidth=2,label=stat)
    plt.xlim([np.min(x),np.max(x)])
    plt.xlabel(var)
    if title: plt.title(title)
    plt.legend()
    plt.grid()
    fig.tight_layout()
    fig.savefig(figName)
    plt.close()

def main():
    # sigHedgeList = np.linspace(0.05,0.9,18)
    # VolArbStratStats = {stat: [] for stat in [
    #     "Min","25%","Median","75%","Max","Mean","Std Dev","Skew","Ex Kurt","95% VaR","99% VaR"
    # ]}
    # [makeDirectory(dir) for dir in [dataFolder,plotFolder]]
    # for sigHedge in sigHedgeList:
    #     callExecutable([
    #         "./"+exeFolder+"backtestVolArbStrat",
    #         "%.2f"%sigHedge])
    #     reportBacktest("sigHedge=%.2f"%sigHedge,maxNumPaths=5)
    #     for stat in VolArbStratStats:
    #         VolArbStratStats[stat].append(BacktestStats[stat])
    # VolArbStratStats["sigHedge"] = sigHedgeList
    # pd.DataFrame.from_dict(VolArbStratStats).round(6).to_csv(
    #     dataFolder+"VolArbStratStats.csv",index=False)
    plotStratStatsAgainstVar(
        dataFolder+"VolArbStratStats.csv",
        plotFolder+"VolArbStratStats.png",
        "sigHedge","Simulated Vol Arb Payoff vs Hedging Vol\n"
        "sigAct = %.1f, sigImp = %.1f"%(0.2,0.4))

if __name__=="__main__":
    main()
