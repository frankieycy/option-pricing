## Option Pricing Library

* last update: 210703
* an object-oriented option pricing library that
    1. supports efficient valuation of vanilla & exotic contracts and Greeks calculation via binomial tree, numerical integration, PDE schemes and Monte Carlo simulation, accommodating various stock price and interest rate stochastic processes
    1. streamlines market data to compute and construct implied vol surface with live tabulation as well as visualize Greeks vs strike price curves on an interactive Python dashboard using Dash and Plotly package
    1. offers robust backtest engines for realistic simulation of dynamic delta/gamma/theta-neutral option hedging strategies riding on market-calibrated underlying price dynamics and modellable/interpolated vol surface accounting for transaction cost
    1. provides access to live market quotes by wrappers around broker API to identify opportunities with proprietary vol arbitrage-based trading signals on at-the-money options to speculate on mispriced implied vol by a dynamic delta-hedged position

## To-do List

* last, option.cpp commenting
* last, use const pointer args for class methods
* ok, organize library files into lib/
* ok, printToCsvFile: support header
* ok, Black-Scholes PDE solver (calc matrix of price surface)
* ok, pricer demo examples on a vanilla call (put in demo/)
* ok, backtest functions in Pricer for option hedging strategies
* ok, implement DGT-neutral strategy
* ok, vol surface modeling
* ok, vol arb research on delta-hedged strategy
* ok, broker API wrappers for market quotes
* ok, antithetic variate, control variate
* visualize Girsanov change of measure
* American option pricing with Monte Carlo
* ok, Exotic option pricing with Monte Carlo
* stochastic vol model: jump diffusion, Heston
* Heston model construct imp vol surface
* deterministic (non-constant) interest rate extension (PDE)
* deterministic (non-constant) volatility extension (PDE)
* (app.py) visualization of implied vol surface in table format
* (app.py) visualization of Greeks vs strike
