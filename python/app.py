import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from option import *

calcHistory = []
optionChainsDfDict = {}

def generateTable(df):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col, id="dense-th") for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col], id="dense-td") for col in df.columns
            ]) for i in range(len(df))
        ])
    ])

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css", "assets/style.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(id="container", children=[
        html.Div("Stock Option Dashboard", id="title"),
        html.Div(["Stock input (csv format): ",
            dcc.Input(id="stock-input", value="", type="text", debounce=True)
        ]),
        dcc.Checklist(id="output-options",
            options=[
                {"label": "Display option chains", "value": "display-option-chains"},
                {"label": "Display implied vol table", "value": "display-implied-vol-table"},
                {"label": "Display Greeks curves", "value": "display-greeks-curves"},
            ],
            value=[]
        ),
        dcc.Checklist(id="option-chains-maturities", value=[]),
        html.Div(id="stock-output"),
        html.Div(id="calc-history"),
        html.Div(id="option-chains"),
        html.Div(id="implied-vol-table"),
        html.Div(id="greeks-curve-plots"),
        html.Div(id="imp-vol-plots"),
    ])
])

@app.callback(
    Output("stock-output", "children"),
    Output("calc-history", "children"),
    Output("option-chains", "children"),
    Output("option-chains-maturities", "options"),
    Output("imp-vol-plots", "children"),
    Input("stock-input", "value"),
    Input("output-options", "value"),
    Input("option-chains-maturities", "value"))
def updateOutputs(stockInput, outputOptions, maturityOptions):
    stockInput = stockInput.replace(" ","")
    stockList = stockInput.split(",")
    stockList = list(dict.fromkeys(stockList))
    validStockList = []
    impVolFileList = []
    optionChainsFileList = []
    if stockList[0]:
        for stock in stockList:
            impVolFile = "option_vol_"+stock+"_"+onDate+".png"
            optionChainsFile = "option_chain_"+stock+"_"+onDate+".csv"
            if not isInDirectory(impVolFile, plotFolder) or \
                not isInDirectory(optionChainsFile, dataFolder):
                impliedVolSurfaceGenerator(stock)
                optionChainsWithGreeksGenerator(stock)
                pricerVariables = pd.read_csv(dataFolder+"pricer_var.csv", header=None)
                pricerVariables = pricerVariables.set_index(0).T
                calcHistory.append(pricerVariables)
            if isInDirectory(impVolFile, plotFolder) and \
                isInDirectory(optionChainsFile, dataFolder):
                validStockList.append(stock)
                impVolFileList.append(plotFolder+impVolFile)
                optionChainsFileList.append(dataFolder+optionChainsFile)
    #### stock-output ##########################################################
    stockOutputDiv = html.Div(["Ouput results on stocks: %s on %s"%(
        ", ".join(validStockList) if validStockList else "Null", onDate
    )])
    #### calc-history ##########################################################
    if calcHistory:
        calcHistoryDf = pd.concat(calcHistory)
        calcHistoryTable = generateTable(calcHistoryDf)
    calcHistoryDiv = html.Div([
        "Pricer variable input history: ",
        calcHistoryTable if calcHistory else "Null",
        html.Br()
    ])
    #### option-chains #########################################################
    #### option-chains-maturities ##############################################
    displayOptionChains = ("display-option-chains" in outputOptions)
    optionChainsDfList = []
    optionChainsTableList = []
    datesList = []
    datesStocksDict = {}
    if displayOptionChains:
        for file in optionChainsFileList:
            if file not in optionChainsDfDict:
                df = pd.read_csv(file)
                for var in ["Maturity (Year)","Implied Vol",
                    "Delta","Gamma","Vega","Rho","Theta"]:
                    df[var] = df[var].round(4)
                optionChainsDfDict[file] = df
            else: df = optionChainsDfDict[file]
            names = df["Contract Name"]
            dates = names.apply(getDateFromContractName)
            uniqueDates = list(dict.fromkeys(dates))
            datesList += uniqueDates
            stock = file.split("_")[-2]
            for date in uniqueDates:
                if date in datesStocksDict:
                    datesStocksDict[date].append(stock)
                else: datesStocksDict[date] = [stock]
            df = df.loc[dates.isin(maturityOptions)]
            if len(df)>0: optionChainsDfList.append(df)
        datesList = list(dict.fromkeys(datesList))
        datesList.sort()
        optionChainsTableList = [generateTable(df) for df in optionChainsDfList]
    optionChainsDiv = html.Div(optionChainsTableList+
        ([html.Br()] if optionChainsTableList else []))
    maturityList = [{"label":date+"-"+",".join(datesStocksDict[date]),"value":date} for date in datesList]
    #### imp-vol-plots #########################################################
    impVolPlots = [html.Img(src=file) for file in impVolFileList]
    impVolPlotsDiv = html.Div(["Implied vol surface plots: "]+
        ([html.Br()] if impVolPlots else [])+
        (impVolPlots if impVolPlots else ["Null"]))
    return stockOutputDiv, calcHistoryDiv, optionChainsDiv, maturityList, impVolPlotsDiv

if __name__ == "__main__":
    app.run_server(debug=True)
