#include "option.cpp"
using namespace std;

int main() {
    /**** Heston model ********************************************************/
    Option option       = Option("European","Call",100,1);
    Stock  stock        = Stock(100,0,0.05,0.2,{5,0.04,0.6,-0.4},"Heston");
    Market market       = Market(0.02,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,200);
    double bsPrice      = pricer.BlackScholesClosedForm();
    double mcPrice      = pricer.MonteCarloPricer(config,1e4);
    cout << "Black-Scholes Price (Lognormal): " << bsPrice << endl;
    cout << "Monte-Carlo Price (Heston): " << mcPrice << endl;
    // vector<matrix> result = stock.simulatePriceWithFullCalc(config,10);
    // result[0].printToCsvFile("test_price.csv");
    // result[1].printToCsvFile("test_vol.csv");
    // result[2].printToCsvFile("test_var.csv");
    return 0;
}
