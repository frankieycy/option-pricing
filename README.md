## Option Pricing Library

* last update: 210608
* an object-oriented option pricing library that
    1. supports efficient valuation of vanilla & exotic contracts and Greeks calculation via numerical integration, Monte Carlo simulation and binomial tree, accomodating various price & interest rate stochastic processes
    1. streamlines market data to construct implied vol surface with live visualization on Python dashboard (via app.py)

## To-do List

* option.cpp commenting
* organize library files into lib/
* ok, printToCsvFile: support header
* ok, Black-Scholes PDE solver (calc matrix of price surface)
* pricer demo examples on a vanilla call (put in demo/)
    1. generatePriceSurface
    1. Closed Form
    1. Binomial Tree
    1. Monte Carlo
    1. Num Integration
    1. PDE Solver
* stochastic process object (called Process) and simulation
* deterministic (non-constant) interest rate extension
* deterministic (non-constant) volatility extension
* stochastic interest rate extension
* stochastic volatility extension
* (app.py) visualization of implied vol surface in table format
* (app.py) visualization of Greeks vs strike
