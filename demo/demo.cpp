#include "lib/option.cpp"
using namespace std;

int main() {
    /**** define objects ******************************************************/
    Option    option = Option("European","Call",100,1);
    Stock     stock  = Stock(100,0,0.05,0.1);
    Market    market = Market(0.02,stock);
    Pricer    pricer = Pricer(option,market);
    SimConfig config = SimConfig(1,100);
    /**** price ***************************************************************/
    pricer.calcPrice("Closed Form");
    pricer.calcPrice("Binomial Tree",config);
    pricer.calcPrice("Monte Carlo",config,1000);
    pricer.calcPrice("Num Integration");
    pricer.calcPrice("PDE Solver",config,0,1000);
    /**** greeks **************************************************************/
    matrix<double> S0; S0.setRange(80,121); S0.printToCsvFile("out-stock.csv");
    pricer.varyGreekWithVariable("currentPrice",S0,"Delta").printToCsvFile("out-delta.csv");
    /**** price surface *******************************************************/
    matrix<double> T; T.setRange(1,0,20,true); T.printToCsvFile("out-term.csv");
    pricer.generatePriceSurface(S0,T).printToCsvFile("out-option.csv");
    /**** implied vol *********************************************************/
    double mktPrice = 10;
    cout << "mkt-implied vol: " << pricer.calcImpliedVolatility(mktPrice) << endl;
    return 0;
}
