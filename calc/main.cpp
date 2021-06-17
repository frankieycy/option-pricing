#include "option.cpp"
using namespace std;

int main() {
    Option option       = Option("European","Call",100,1);
    Stock  stock        = Stock(100,0,0.05,0.1);
    Market market       = Market(0.02,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,50);
    /**** price ***************************************************************/
    // pricer.calcPrice("Closed Form");
    // pricer.calcPrice("Binomial Tree",config);
    // pricer.calcPrice("Monte Carlo",config,1000);
    // pricer.calcPrice("Num Integration");
    // pricer.calcPrice("PDE Solver",config,0,1000);
    /**** greeks **************************************************************/
    // matrix S0; S0.setRange(80,121);
    // cout << pricer.varyGreekWithVariable("currentPrice",S0,"Delta") << endl;
    /**** price surface *******************************************************/
    // matrix S0; S0.setRange(80,121); S0.printToCsvFile("test_stock.csv");
    // matrix T; T.setRange(1,0,20,true); T.printToCsvFile("test_term.csv");
    // pricer.generatePriceSurface(S0,T).printToCsvFile("test_option.csv");
    /**** implied vol *********************************************************/
    // cout << pricer.calcImpliedVolatility(10) << endl;
    /**** implied vol dashboard ***********************************************/
    // Pricer pricer;
    // pricer.setVariablesFromFile("pricer_var.csv");
    // pricer.generateImpliedVolSurfaceFromFile("option_data.csv","option_vol.csv");
    /**** strat backtest ******************************************************/
    // pricer.runBacktest(config,50,"simple-delta",1).printToCsvFiles();
    // Option hOption = Option("European","Call",100,2);
    // pricer.runBacktest(config,3,"simple-delta-gamma",1,0,{hOption}).printToCsvFiles(true);
    Option hOption0 = Option("European","Call",110,2);
    Option hOption1 = Option("European","Call",90,2);
    pricer.runBacktest(config,3,"simple-delta-gamma-theta",1,0,{hOption0,hOption1}).printToCsvFiles(true);
    return 0;
}