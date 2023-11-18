## Option Pricing Library

* last update: 220104
* C++ option pricing library that implements the classical models
* Python vol-surface calibration libraty
* Miscellaneous quantitative research on volatility

## Single-Asset Option Pricing Examples

For pricing of option on single underlying asset via Monte-Carlo simulation, the following serves as a template. Concrete examples can be found under calc/.
```
#include "option.cpp" // The pricing lib
/**** Config-related ********/
int n = ...; // Num of time steps, e.g. 500
int m = ...; // Num of MC sims, e.g. 1e4
/**** Option-related ********/
string Type             = ...;
string PutCall          = ...;
double Strike           = ...;
double Maturity         = ...;
vector<double> Params   = ...;
vector<string> Nature   = ...;
/**** Stock-related *********/
double CurrPrice        = ...;
double DivYield         = ...;
double Drift            = ...;
double Vol              = ...;
/**** Market-related ********/
double RiskFreeRate     = ...;
/**** Declare objects *******/
Option option           = Option(Type,PutCall,Strike,Maturity,Params,Nature);
Stock  stock            = Stock(CurrPrice,DivYield,Drift,Vol);
Market market           = Market(RiskFreeRate,stock);
Pricer pricer           = Pricer(option,market);
SimConfig config        = SimConfig(Maturity,n);
/**** Calc price ************/
double mcPrice          = pricer.MonteCarloPricer(config,m);
```

For vanilla European options, `Nature` and `Params` can be left empty.
```
Nature = {};
Params = {};
```

However, for exotic options, they are generally required, as described case by case below.

**Technical Note on Monte-Carlo.** In the simulations, asset prices drift under the risk-neutral measure, not the physical measure. In practice, a risk-neutral market is created and assets have Drift equal to RiskFreeRate minus DivYield, or mu=r-q. With the simulated price paths, option payoffs can computed and averaged, eventually discounted to give the present value. Other pricing methods aside from Monte-Carlo are supported, including: 1. Black-Scholes closed form, 2. binomial tree, 3. numerical integration and 4. PDE solver. However, they have their limitations and are restricted to certain option types, e.g. simple closed form price is not always available (except for few examples like vanilla, digital and Margrabe), binomial tree does not easily support path-dependence. Monte-Carlo serves as a generic brute-force method and applies to a wide class of non-early exercisable options, despite disadvantages like: 1. slow in convergence of O(n^-0.5) and statistical error, 2. slow in computation. Early exercise can be handled via some techniques (which estimate continuation value), and will be incorporated in future development.

##### 1. Asian Option

Asian option has the average of historical prices as its underlying. The averaging can be arithmetic or geometric, while in practice, arithmetic is more common.
```
Nature = {"Arithmetic"} or {"Geometric"};
Params = {};
```

##### 2. Barrier Option

Barrier option has the terminal asset price as its underlying but only if, for an In, the barrier is hit or if, for an Out, the barrier is not hit. For example, for a Down-and-Out, rebate is paid out if historically the barrier (lower than strike) was crossed. Rebate can be zero, meaning the option goes worthless on such occurrence. If otherwise, the usual vanilla payoff is paid.
```
Barrier = ...; // Has to be consistent with Up/Down setting
Rebate  = ...; // e.g. 0
Nature  = {"Up-and-In"} or {"Up-and-Out"} or {"Down-and-In"} or {"Down-and-Out"};
Params  = {Barrier, Rebate};
```

##### 3. Lookback Option

Lookback option has the terminal asset price as its underlying but is struck at the historical min or max. For a Lookback call, max(S-Smin,0) is paid; for a Lookback put, max(Smax-S,0) is paid. `Nature` and `Params` can be left empty.
```
Nature = {};
Params = {};
```

##### 4. Chooser Option

Chooser option allows the holder to decide the Put/Call identity at a specific choice time, paying the vanilla payoff upon the decision. On the choice time t, the holder chooses a call if C(t)>P(t), or by the put-call parity: S(t)>K*e^-r(T-t), and vice versa. A Chooser at time t is the sum of a call of strike K, maturity T and a put struck at the discounted strike expiring immediately.
```
ChTime = ...; // Choice time
PutCall = "";
Nature = {};
Params = {ChTime};
```

## Multi-Asset Option Pricing Examples

For pricing of option on multiple underlying assets via Monte-Carlo simulation, the following serves as a template. Additionally, all underlying stocks and the correlation matrix (of their underlying Brownian motions) have to be specified. Concrete examples can be found under calc/.
```
#include "option.cpp" // The pricing lib
/**** Config-related ********/
int n = ...; // Num of time steps
int m = ...; // Num of MC sims
/**** Option-related ********/
string Type             = ...;
string PutCall          = ...;
double Strike           = ...;
double Maturity         = ...;
vector<double> Params   = ...;
vector<string> Nature   = ...;
/**** Stock-related *********/
int nStk = ...; // Num of stocks
vector<double> CurrPrices = ...;
vector<double> DivYields  = ...;
vector<double> Drifts     = ...;
vector<double> Vols       = ...;
vector<Stock>  stocks;
for(int i=0; i<nStk; i++){
    double CurrPrice    = CurrPrices[i];
    double DivYield     = DivYields[i];
    double Drift        = Drifts[i];
    double Vol          = Vols[i];
    Stock stock = Stock(CurrPrice,DivYield,Drift,Vol);
    stocks.push_back(stock);
}
matrix corMatrix;
corMatrix = matrix(...); // Double array arguments
/**** Market-related ********/
double RiskFreeRate     = ...;
/**** Declare objects *******/
Option option           = Option(Type,PutCall,Strike,Maturity,Params,Nature);
Market market           = Market(RiskFreeRate,NULL_STOCK,stocks,corMatrix);
Pricer pricer           = Pricer(option,market);
SimConfig config        = SimConfig(Maturity,n);
/**** Calc price ************/
double mcPrice          = pricer.MonteCarloPricer(config,m);
```

##### 1. Basket Option

Basket option has the average of terminal asset prices of the basket as its underlying. The averaging can be a simple even-out or weighted. If weighted, specify `Params` as the respective weights.
```
Nature = {};
Params = ...; // Weights for each stock
```

##### 2. Margrabe Option

Margrabe option gives the holder the right to exchange two assets at maturity. For a call, the second asset is exchanged for the first, hence payoff max(S1-S2,0); for a put, the first asset is exchanged for the second, hence payoff max(S2-S1,0). `Nature` and `Params` can be left empty.
```
Nature = {};
Params = {};
```

##### 3. Rainbow Option

Rainbow option has the max or min of terminal asset prices of the basket as its underlying, struck at the strike price. The max/min nature is specified via `Nature`. Alternatively, Best Rainbow pays the max between asset prices and strike, which effectively is the lump sum of cash payout and a call on max.
```
Nature = {"Max"} or {"Min"} or {"Best"};
Params = {};
```

## Other Price Dynamics

Asset prices may admit dynamics other than the usual constant vol drift-diffusion process, such as local vol or stochastic vol process. This is easily accommodated by modifying the Stock object initialization, and the following serves as a template.

##### 1. Heston Model
##### 2. Jump-Diffusion Model

## Dynamic Hedging Simulation
## Real Stock Implied Vol
## Arbitrage-free Smoothing of Price Surface (Fengler)
## Backing out Local Vol from Implied Vol
## Calibration of Stochastic Vol Models
##### Heston Model
##### Merton Jump-Diffusion Model
##### Variance-Gamma Model
##### Rough-Heston Model
