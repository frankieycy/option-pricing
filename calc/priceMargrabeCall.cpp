#include "option.cpp"
using namespace std;

int main() {
    /**** Margrabe ************************************************************/
    Option option       = Option("Margrabe","Call",0,1);
    Stock  stock1       = Stock(100,0,0.05,0.2);
    Stock  stock2       = Stock(100,0,0.08,0.3);
    matrix corMatrix    = matrix({{1,0.3},{0.3,1}});
    Market market       = Market(0.02,NULL_STOCK,{stock1,stock2},corMatrix);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,200);
    double bsPrice      = pricer.BlackScholesClosedForm();
    double mcPrice      = pricer.MultiStockMonteCarloPricer(config,1e4);
    cout << "Option Details: " << option << endl;
    cout << "Black-Scholes Price: " << bsPrice << endl;
    cout << "Monte-Carlo Price: " << mcPrice << endl;
    return 0;
}
