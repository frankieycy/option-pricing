## Option Pricing Library

* last update: 210613
* an object-oriented option pricing library that
    1. supports efficient valuation of vanilla & exotic contracts and Greeks calculation via binomial tree, numerical integration, PDE schemes and Monte Carlo simulation, accommodating various stock price and interest rate stochastic processes
    1. streamlines market data to compute and construct implied vol surface with live tabulation as well as visualize Greeks vs strike price curves on an interactive Python dashboard using Dash and Plotly package
    1. offers backtest engines for realistic simulation of dynamic delta/gamma/theta-neutral option hedging strategies riding on market-calibrated underlying price dynamics and modellable vol surface accounting for transaction cost

## To-do List

* option.cpp commenting
* ok, organize library files into lib/
* ok, printToCsvFile: support header
* ok, Black-Scholes PDE solver (calc matrix of price surface)
* ok, pricer demo examples on a vanilla call (put in demo/)
* ok, backtest functions in Pricer for option hedging strategies
* ok, implement DGT-neutral strategy
* vol surface modeling
* vol arb research on DGT-neutral strategy
* stochastic process object (called Process) and simulation
* deterministic (non-constant) interest rate extension
* deterministic (non-constant) volatility extension
* stochastic interest rate extension
* stochastic volatility extension
* (app.py) visualization of implied vol surface in table format
* (app.py) visualization of Greeks vs strike
* broker API wrappers for market quotes
