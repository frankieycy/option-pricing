import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from yahoo_fin import stock_info, options

stock = "IBM"
zeroBond = "^IRX"
zeroBondMaturity = 85

def bDaysBetween(d1, d2):
    return len(pd.bdate_range(d1,d2))

def calcRiskFreeRate(discountRate, maturity):
    riskFreeRate = (-252/maturity)*np.log(1-maturity*discountRate/360)
    return riskFreeRate

def printPricerVariablesToCsvFile(fileName, pricerVariables):
    file = open(fileName,"w")
    for var in pricerVariables:
        if type(var) is str: sub = "%s"
        elif type(var) is float: sub = "%.10f"
        file.write(("%s,"+sub+"\n")%(var,pricerVariables[var]))
    file.close()

def printOptionChainsToCsvFile(fileName, optionChains):
    file = open(fileName,"w")
    for date in optionDates:
        daysFromToday = bDaysBetween(todayDate,date)
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

todayDate = datetime.today().strftime("%Y-%m-%d")
currentPrice = stock_info.get_data(stock,todayDate).iloc[0]["close"]
discountRate = stock_info.get_data(zeroBond,todayDate).iloc[0]["close"]
dividendYield = stock_info.get_quote_table(stock)["Forward Dividend & Yield"]
dividendYield = float(re.sub("[()%]","",dividendYield.split()[1]))/100

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

printPricerVariablesToCsvFile("pricer_var.csv",pricerVariables)
printOptionChainsToCsvFile("option_data.csv",optionChains)
