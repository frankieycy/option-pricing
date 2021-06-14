#include "option.cpp"
using namespace std;

int main() {
    Option option       = Option("European","Call",100,1);
    Stock  stock        = Stock(100,0,0.05,0.1);
    Market market       = Market(0.02,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,100);
    pricer.runBacktest(config,10,"simple-delta");
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
    // pricer.setVariablesFromFile("pricer_var.csv"); pricer.saveAsOriginal();
    // pricer.generateImpliedVolSurfaceFromFile("option_data.csv","option_vol.csv");
    return 0;
}
