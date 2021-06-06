import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from option import *

calcHistory = []

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

containerStyle = {
    "padding": "0 30px",
    "font-size": ".8em",
    "line-height": "200%"
}

imageStyle = {
    "margin": "0",
    "width": "300px",
    "border-style": "solid",
    "border-width": "1px",
    "border-color": "#EEEEEE"
}

def generateTable(df):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(len(df))
        ])
    ])

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.Div("Stock Option Dashboard", style={
            "font-size": "1.2em", "font-weight": "bold"
        }),
        html.Div(["Stock input (csv format): ",
            dcc.Input(id="stock-input", value="IBM", type="text", debounce=True)
        ]),
        html.Div(id="stock-output"),
        html.Div(id="calc-history"),
        html.Div(id="imp-vol-plots"),
    ], id="container", style=containerStyle)
])

@app.callback(
    Output("stock-output", "children"),
    Output("calc-history", "children"),
    Output("imp-vol-plots", "children"),
    Input("stock-input", "value"))
def updateOutputs(stockInput):
    stockInput = stockInput.replace(" ","")
    stockList = stockInput.split(",")
    stockList = list(dict.fromkeys(stockList))
    impVolFileList = []
    validStockList = []
    if stockList[0]:
        for stock in stockList:
            file = "option_vol_"+stock+"_"+onDate+".png"
            if not isInDirectory(file, plotFolder):
                impliedVolSurfaceGenerator(stock)
                pricerVariables = pd.read_csv(dataFolder+"pricer_var.csv", header=None)
                pricerVariables = pricerVariables.set_index(0).T
                calcHistory.append(pricerVariables)
            if isInDirectory(file, plotFolder):
                validStockList.append(stock)
                impVolFileList.append(plotFolder+file)
    stockOutputMsg = "Ouput results on stocks: %s on %s"%(
        ", ".join(validStockList) if validStockList else "Null", onDate
    )
    calcHistoryTable = ''
    if calcHistory: calcHistoryTable = generateTable(pd.concat(calcHistory))
    calcHistoryDiv = html.Div([
        "Pricer variable input history: ",
        calcHistoryTable if calcHistory else "Null",
        html.Br()
    ])
    htmlImgList = [html.Img(src=file, style=imageStyle) for file in impVolFileList]
    htmlImgDiv = html.Div(["Implied vol surface plots: "]+
        ([html.Br()]+htmlImgList if htmlImgList else ["Null"])
    )
    return stockOutputMsg, calcHistoryDiv, htmlImgDiv

if __name__ == "__main__":
    app.run_server(debug=True)
