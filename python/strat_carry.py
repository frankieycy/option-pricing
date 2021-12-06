import cvxpy as cvx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yahoo_fin import stock_info, options
from util import *
import option as o

stratParams = {
    "stock":                        "SPY",
    "maturityRange":                ["2022-01-21","2022-01-21"],
    "logMoneyness":                 [-0.2,0.2],
    "wealthConstr":                 100,
    "tCostPerOrder":                0,
    "volumeLowerBnd":               200,
    "carryPeriod":                  1/252,
    "deltaBnd":                     [-1,1],
    "vegaBnd":                      [-100,100],
    "pricerVarCsv":                 f"{o.dataFolder}pricer_var.csv",
    "yfinRawOptionChainsCsv":       f"{o.dataFolder}yfin_option_chain_SPY_{o.onDate}.csv",
    "calcOptionChainsWithGrkCsv":   f"{o.dataFolder}option_chain_SPY_{o.onDate}.csv",
    "combinedOptionChainsCsv":      f"{o.dataFolder}comb_option_chain_SPY_{o.onDate}.csv",
}

def prepareInputFiles():
    stock = stratParams["stock"]
    yfinName = stratParams["yfinRawOptionChainsCsv"]
    makeDirectory(o.dataFolder)
    o.impliedVolSurfaceGenerator(stock,optionPriceType="mid")
    o.optionChainsWithGreeksGenerator(stock)
    o.downloadOptionChainsToCsvFile(yfinName,stock)

def appendGrkToYfinRawDownload():
    yfinCols = ["Maturity","Put/Call","Contract Name","Last Trade Date","Strike","Last Price",
        "Bid","Ask","Change","% Change","Volume","Open Interest","Implied Vol"]
    grkCols = ["Contract Name","Type","Put/Call","Strike","Maturity (Year)","Mid Price","Implied Vol",
        "Delta","Gamma","Vega","Rho","Theta"]
    yfinName = stratParams["yfinRawOptionChainsCsv"]
    calcName = stratParams["calcOptionChainsWithGrkCsv"]
    yfinChain = pd.read_csv(yfinName)
    calcChain = pd.read_csv(calcName)
    yfinChain.columns, calcChain.columns = yfinCols, grkCols
    yfinChain = yfinChain.set_index("Contract Name")
    calcChain = calcChain.set_index("Contract Name")
    combChain = yfinChain.copy()
    for var in ["Implied Vol","Delta","Gamma","Vega","Rho","Theta"]:
        combChain[var] = calcChain[var]
    combChain.to_csv(stratParams["combinedOptionChainsCsv"], index=False)
    return combChain

def filterCombinedOptionChains(combChain):
    pricerVar = pd.read_csv(stratParams["pricerVarCsv"], header=None)
    S = float(pricerVar.iloc[1,1])
    K0,K1 = S*np.exp(stratParams["logMoneyness"])
    M0,M1 = pd.to_datetime(stratParams["maturityRange"])
    V0 = stratParams["volumeLowerBnd"]
    filCombChain = combChain.dropna()
    maturity = pd.to_datetime(filCombChain["Maturity"])
    filCombChain = filCombChain[(maturity>=M0) & (maturity<=M1)]
    strike = filCombChain["Strike"]
    filCombChain = filCombChain[(strike>=K0) & (strike<=K1)]
    volume = pd.to_numeric(filCombChain["Volume"],errors='coerce').fillna(0)
    filCombChain = filCombChain[(volume>=V0)]
    return filCombChain

def carryStrat(combChain):
    nOptions = len(combChain)
    pricerVar = pd.read_csv(stratParams["pricerVarCsv"], header=None)
    S   = float(pricerVar.iloc[1,1])
    W   = stratParams["wealthConstr"]
    C   = stratParams["tCostPerOrder"]
    V0  = stratParams["volumeLowerBnd"]
    dt  = stratParams["carryPeriod"]
    Delta0,Delta1 = stratParams["deltaBnd"]
    Vega0,Vega1 = stratParams["vegaBnd"]
    n = cvx.Variable(nOptions,integer=True)
    z = cvx.Variable(nOptions,boolean=True)
    theta   = combChain["Theta"].to_numpy()
    delta   = combChain["Delta"].to_numpy()
    gamma   = combChain["Gamma"].to_numpy()
    vega    = combChain["Vega"].to_numpy()
    sigma   = combChain["Implied Vol"].to_numpy()
    bids    = combChain["Bid"].to_numpy()
    carryVal = theta + .5 * sigma**2 * S**2 * gamma
    objective = cvx.Maximize(dt * carryVal @ n - 2 * C * sum(z))
    constraints = [
        # delta @ n >= Delta0,
        # delta @ n <= Delta1,
        # vega @ n >= Vega0,
        # vega @ n <= Vega1,
        bids @ n <= W,
        n <= z * W / bids,
        n >= 0,
    ]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver='ECOS_BB',verbose=True,max_iters=int(1e2))
    n = np.round(n.value.T,0)
    z = z.value.T
    activeIdx = np.argwhere(n).flatten()
    summary = {
        "nOptions":             nOptions,
        "optimalStratValue":    prob.value,
        "optimalCarryValue":    dt * carryVal @ n,
        "totTCost":             2 * C * sum(z),
        "stratDelta":           delta @ n,
        "stratVega":            vega @ n,
        "position":             {combChain.index[i]: n[i] for i in activeIdx},
        "optimalAllocation":    n,
    }
    return summary

def main():
    # prepareInputFiles()
    combChain = appendGrkToYfinRawDownload()
    combChain = filterCombinedOptionChains(combChain)
    summary = carryStrat(combChain)
    print(summary)

if __name__ == "__main__":
    main()
