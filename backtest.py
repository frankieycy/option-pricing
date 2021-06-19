import math
import re, glob
import numpy as np
import matplotlib.pyplot as plt
from util import *
plt.switch_backend("Agg")

LOG = True

dataFolder = "test-0619/"
plotFolder = "test-0619/"

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
    ]
}

def logMessage(msg):
    if LOG:
        logMsg = getCurrentTime()+" [LOG] "
        if type(msg) is list:
            for m in msg: logMsg += str(m)
        else: logMsg += str(msg)
        print(logMsg)

def loadBacktestResults(name="backtest", perSim=False):
    if perSim:
        fileName = dataFolder+name+"-*.csv"
        fileList = glob.glob(fileName)
        fileList.sort(key=lambda f:int(re.split("[-.]",f)[-2]))
        for idx,file in enumerate(fileList):
            df = pd.read_csv(file)
            cols = df.columns
            for col in cols:
                label = col.split("-")[0]
                pass # TO DO
    else:
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

def headBacktestResults(show=10):
    for label in BacktestData["labels"]:
        print("==== BacktestData[%s] ====="%label)
        print(BacktestData[label].head(show))
    for label in BacktestData["hLabels"]:
        print("==== BacktestData[%s] ====="%label)
        for df in BacktestData[label]:
            print(df.head(show))

def plotBacktestResults(figName, maxNumPaths=3):
    n = len(BacktestData["labels"])
    fig,ax = plt.subplots(math.ceil(n/2),2,figsize=(10,15))
    for idx,label in enumerate(BacktestData["labels"]):
        for i in range(min(maxNumPaths,BacktestData[label].shape[1])):
            ax[idx//2,idx%2].plot(BacktestData[label].iloc[:-1,i],linewidth=1,label="Sim-%d"%i)
        ax[idx//2,idx%2].set_xlim([0,BacktestData[label].shape[0]-2])
        ax[idx//2,idx%2].set_xticklabels([])
        ax[idx//2,idx%2].set_title(label)
        ax[idx//2,idx%2].grid()
    ax[0,0].legend()
    fig.tight_layout()
    plt.savefig(figName)
    plt.close()

def plotBacktestHResults(figName, maxNumPaths=3):
    pass

def calcBacktestStats():
    pass

def reportBacktest(name="backtest", perSim=False, maxNumPaths=3):
    loadBacktestResults(name,perSim)
    plotBacktestResults(plotFolder+name+"-report.png",maxNumPaths)
    plotBacktestHResults(plotFolder+name+"-reportH.png",maxNumPaths)
    calcBacktestStats()

def main():
    reportBacktest()

if __name__=="__main__":
    main()
