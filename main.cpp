#include "option.cpp"
using namespace std;

int main() {
    Option option       = Option("European","Call",100,1);
    Stock  stock        = Stock(100,0,0.05,0.1);
    Market market       = Market(0.02,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,200);
    /**** price ***************************************************************/
    // pricer.calcPrice("Closed Form");
    // pricer.calcPrice("Binomial Tree",config);
    // pricer.calcPrice("Monte Carlo",config,1000);
    // pricer.calcPrice("Num Integration");
    /**** greeks **************************************************************/
    // matrix<double> S0; S0.setRange(80,120);
    // cout << pricer.varyGreekWithVariable("currentPrice",S0,"Delta") << endl;
    /**** price surface *******************************************************/
    matrix<double> S0; S0.setRange(80,121); S0.printToCsvFile("test_stock.csv");
    matrix<double> T; T.setRange(1,0,20,true); T.printToCsvFile("test_term.csv");
    pricer.generatePriceSurface(S0,T).printToCsvFile("test_option.csv");
    return 0;
}
