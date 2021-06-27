import time
import pandas as pd
from util import *
from futu import *

LOG = True

quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)

onDate = getRecentBDay()
codeList = ["US.IBM","US.JPM","US.DIS"]
expirationDateEnd = "2021-06-30"

dataFolder = "bkts_data/"

optionChainData = {} # master data dict
optionChainDataFrames = {}
optionHistDataFrames = {}

optionChainDataCols = {
    "code":                 "Contract Name",
    "option_area_type":     "Type",
    "sec_status":           "Status",
    "expiry_date_distance": "Maturity (Day)",
    "contract_size":        "Contract Size",
    "strike_price":         "Strike",
    "last_price":           "Last Price",
    "open_price":           "Open Price",
    "high_price":           "High Price",
    "low_price":            "Low Price",
    "prev_close_price":     "Close Price",
    "open_interest":        "Open Interest",
    "volume":               "Volume",
    "turnover":             "Turnover",
    "implied_volatility":   "Implied Vol",
    "premium":              "Premium",
    "delta":                "Delta",
    "gamma":                "Gamma",
    "vega":                 "Vega",
    "theta":                "Theta",
    "rho":                  "Rho"
}
optionChainDataCsvCols = [
    "Contract Name",
    "Type",
    "Put/Call",
    "Status",
    "Maturity (Day)",
    "Maturity (BDay)",
    "Maturity (Year)",
    "Contract Size",
    "Strike",
    "Last Price",
    "Open Price",
    "High Price",
    "Low Price",
    "Close Price",
    "Open Interest",
    "Volume",
    "Turnover",
    "Implied Vol",
    "Premium",
    "Delta",
    "Gamma",
    "Vega",
    "Theta",
    "Rho"
]
optionHistDataCols = {
    "time_key":     "Time",
    "open":         "Open Price",
    "close":        "Close Price",
    "high":         "High Price",
    "low":          "Low Price",
    "volume":       "Volume",
    "turnover":     "Turnover",
    "last_close":   "Last Close Price"
}
optionHistDataCsvCols = [
    "Contract Name",
    "Date",
    "Time",
    "Open Price",
    "Close Price",
    "High Price",
    "Low Price",
    "Volume",
    "Turnover",
    "Last Close Price"
]

def logMessage(msg):
    if LOG:
        logMsg = getCurrentTime()+" [LOG] "
        if type(msg) is list:
            for m in msg: logMsg += str(m)
        else: logMsg += str(msg)
        print(logMsg)

def getPutCallFromContractName(name):
    if "P" in name[-10:]: return "Put"
    else: return "Call"

def getMatBDaysFromContractName(name):
    putCall = getPutCallFromContractName(name)
    matDate = name.split("P" if putCall == "Put" else "C")[-2][-6:]
    matDate = "20"+matDate[:2]+"-"+matDate[2:4]+"-"+matDate[4:]
    return bDaysFromToday(matDate)

################################################################################

def initOptionChain():
    logMessage("initializing option chain")
    for code in codeList:
        optionChainData[code] = {}
        ret1, data1 = quote_ctx.get_option_expiration_date(code=code)
        if ret1 == RET_OK:
            expiration_date_list = data1["strike_time"].values.tolist()
            for date in expiration_date_list:
                if expirationDateEnd:
                    if bDaysBetween(date, expirationDateEnd) == 0: continue
                optionChainData[code][date] = {}
                ret2, data2 = quote_ctx.get_option_chain(code=code, start=date, end=date, option_type=OptionType.CALL)
                time.sleep(3)
                if ret2 == RET_OK:
                    optionChainData[code][date]["codes"] = data2["code"].values.tolist()
                    # print(data2["code"].values.tolist())
                    ret3, data3 = quote_ctx.get_option_chain(code=code, start=date, end=date, option_type=OptionType.PUT)
                    time.sleep(3)
                    if ret3 == RET_OK:
                        optionChainData[code][date]["codes"] += data3["code"].values.tolist()
                        # print(data3["code"].values.tolist())
                    else:
                        print("error:", data3)
                else:
                    print("error:", data2)
        else:
            print("error:", data1)
    # print(optionChainData)

def fillOptionChain():
    logMessage("filling option chain")
    for code in optionChainData:
        for date in optionChainData[code]:
            optionCodeList = optionChainData[code][date]["codes"]
            print("current subscription status:", quote_ctx.query_subscription())
            ret_sub, err_message = quote_ctx.subscribe(optionCodeList, [SubType.QUOTE], subscribe_push=False)
            if ret_sub == RET_OK:
                print("subscription success. current subscription status:", quote_ctx.query_subscription())
                ret, data = quote_ctx.get_stock_quote(optionCodeList)
                if ret == RET_OK:
                    logMessage(["filling option chain: ",code,", ",date])
                    optionChainData[code][date]["data"] = data[list(optionChainDataCols.keys())]
                else:
                    print("error:", data)
                time.sleep(60)
            else:
                print("subscription failed", err_message)
            ret_unsub, err_message_unsub = quote_ctx.unsubscribe_all()
            if ret_unsub == RET_OK:
                print("unsubscription success. current subscription status:", quote_ctx.query_subscription())
            else:
                print("unsubscription failed", err_message_unsub)

def getOptionChainAsDataFrame():
    logMessage("getting option chain as dataframe")
    for code in optionChainData:
        optionChainDataFrames[code] = pd.DataFrame()
        for date in optionChainData[code]:
            optionChainDataFrames[code] = optionChainDataFrames[code].append(optionChainData[code][date]["data"])
        optionChainDataFrames[code].columns = [optionChainDataCols[col] for col in optionChainDataFrames[code].columns]
        optionChainDataFrames[code]["Maturity (BDay)"] = optionChainDataFrames[code]["Contract Name"].apply(getMatBDaysFromContractName)
        optionChainDataFrames[code]["Maturity (Year)"] = optionChainDataFrames[code]["Maturity (BDay)"]/252
        optionChainDataFrames[code]["Put/Call"] = optionChainDataFrames[code]["Contract Name"].apply(getPutCallFromContractName)
        optionChainDataFrames[code] = optionChainDataFrames[code][optionChainDataCsvCols]

def saveOptionChainToCsvFile():
    logMessage("saving option chain to csv file")
    for code in optionChainDataFrames:
        fileName = dataFolder+"option_chain_"+code+"_"+onDate+".csv"
        optionChainDataFrames[code].to_csv(fileName, index=False)

################################################################################

def fillOptionHistPrice():
    logMessage("filling option historical price")
    for code in optionChainData:
        for date in optionChainData[code]:
            optionChainData[code][date]["hist"] = {}
            optionCodeList = optionChainData[code][date]["codes"]
            print("current subscription status:", quote_ctx.query_subscription())
            ret_sub, err_message = quote_ctx.subscribe(optionCodeList, [SubType.K_DAY], subscribe_push=False)
            if ret_sub == RET_OK:
                print("subscription success. current subscription status:", quote_ctx.query_subscription())
                for optionCode in optionCodeList:
                    ret, data = quote_ctx.get_cur_kline(optionCode, num=1000)
                    time.sleep(0.5)
                    if ret == RET_OK:
                        logMessage(["filling option historical price: ",code,", ",date,", ",optionCode])
                        optionChainData[code][date]["hist"][optionCode] = data[list(optionHistDataCols.keys())]
                    else:
                        print('error:', data)
                time.sleep(60)
            else:
                print("subscription failed", err_message)
            ret_unsub, err_message_unsub = quote_ctx.unsubscribe_all()
            if ret_unsub == RET_OK:
                print("unsubscription success. current subscription status:", quote_ctx.query_subscription())
            else:
                print("unsubscription failed", err_message_unsub)

def getOptionHistPriceAsDataFrame():
    logMessage("getting option historical price as dataframe")
    for code in optionChainData:
        optionHistDataFrames[code] = pd.DataFrame()
        for date in optionChainData[code]:
            for optionCode in optionChainData[code][date]["hist"]:
                histPrice = optionChainData[code][date]["hist"][optionCode]
                histPrice.columns = [optionHistDataCols[col] for col in histPrice.columns]
                histPrice["Contract Name"] = optionCode
                histPrice["Date"] = histPrice["Time"].apply(lambda x: x.split()[0])
                histPrice = histPrice[optionHistDataCsvCols]
                optionHistDataFrames[code] = optionHistDataFrames[code].append(histPrice)

def saveOptionHistPriceToCsvFile():
    logMessage("saving option historical price to csv file")
    for code in optionHistDataFrames:
        fileName = dataFolder+"option_hist_"+code+"_"+onDate+".csv"
        optionHistDataFrames[code].to_csv(fileName, index=False)

################################################################################

def closeConnection():
    quote_ctx.close()

def optionChainSnapshot():
    makeDirectory(dataFolder)
    initOptionChain()
    fillOptionChain()
    getOptionChainAsDataFrame()
    saveOptionChainToCsvFile()
    closeConnection()

def optionHistPricePull():
    makeDirectory(dataFolder)
    initOptionChain()
    fillOptionHistPrice()
    getOptionHistPriceAsDataFrame()
    saveOptionHistPriceToCsvFile()
    closeConnection()

def main():
    optionHistPricePull()

if __name__ == "__main__":
    main()
