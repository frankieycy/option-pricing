## To-Do List (2021 Summer)

* last, option.cpp commenting
* last, use const pointer args for class methods
* last, header files and makefile
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
* ok, Exotic option pricing with Monte Carlo
* ok, stochastic vol model: Heston
* ok, Heston model construct imp vol surface
* ok, jump-diffusion model: Merton
* ok, stochastic vol model: GARCH
* ok, local vol model: CEV
* boost FFT & MC efficiency
* model param calibration
* visualize Girsanov change of measure
* American option pricing with Monte Carlo
* facade design pattern: optionlib.cpp (wrapper)
* deterministic (non-constant) interest rate extension (PDE)
* deterministic (non-constant) volatility extension (PDE)
* (app.py) visualization of implied vol surface in table format
* (app.py) visualization of Greeks vs strike

## To-Do List (2021 Fall - 2022 Spring)

* ok, implement arb-free price surface smoothing (Fengler)
* ok, interpolate price surface via splines, calibrate IVS
* ok, fix FFT pricer stability issues (B,N params for short/long exp)
    - ok, check Lewis/CarrMadan w/wo FFT results converge
    - ok, calibration to price vs. imp vol
* BS inversion: Scipy fsolve
* BS inversion: Jackel, Glau Chebychev IV interpolation
* ok, calibrate stochastic vol models (Merton/VGamma/Heston/rHeston)
    - ok, Heston
    - ok, Merton
    - ok, VGamma
    - ok, SVJ
    - ok, SVJJ
    - ok, CGMY
    - ok, rHeston: poor man's approx
    - ok, rHeston: Pade approx
    - x, rHeston: Riccati solver
* back out local vol from model/market IVS
    - ok, model IVS
    - market IVS
* SVI parametrization with no-arb constraints
* pricing under execution costs
* joint calibration with VIX derivs
* fast Monte-Carlo for exotics pricing
