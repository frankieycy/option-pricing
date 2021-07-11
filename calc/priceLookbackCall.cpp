#include "option.cpp"
using namespace std;

int main() {
    /**** Lookback ************************************************************/
    Option option       = Option("Lookback","Call",100,1);
    Stock  stock        = Stock(100,0,0.05,0.2);
    Market market       = Market(0.02,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,200);
    double mcPrice      = pricer.MonteCarloPricer(config,1e4);
    cout << "Option Details: "      << option << endl;
    cout << "Monte-Carlo Price: "   << mcPrice << endl;
    return 0;
}
