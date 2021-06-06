import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from option import *

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H4(children="Implied Vol Dashboard"),
    html.Div(["Stocks: ",
        dcc.Input(id="stock-input", value="IBM", type="text")
    ]),
    html.Div(id="imp-vol-plots"),
])

@app.callback(
    Output(component_id="imp-vol-plots", component_property="children"),
    Input(component_id="stock-input", component_property="value")
)
def updateImpVolPlots(stockInput):
    stockInput.replace(" ","")
    stockList = stockInput.split(",")
    impVolFileList = []
    for stock in stockList:
        file = "option_vol_"+stock+"_"+onDate+".png"
        if not isInDirectory(file,plotFolder):
            impliedVolSurfaceGenerator(stock)
        impVolFileList.append(plotFolder+file)
    htmlImgList = [html.Img(src=file) for file in impVolFileList]
    return htmlImgList

if __name__ == "__main__":
    app.run_server(debug=True)
