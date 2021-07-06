#include "option.cpp"
using namespace std;

int main() {
    /**** Digital *************************************************************/
    Option option       = Option("Barrier","Call",100,1,{120,0},{"Up-and-Out"});
    // Option option       = Option("Barrier","Call",100,1,{90,0},{"Down-and-In"});
    Stock  stock        = Stock(100,0,0.05,0.2);
    Market market       = Market(0.1,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,200);
    double mcPrice      = pricer.MonteCarloPricer(config,1e4);
    double pdePrice     = pricer.BlackScholesPDESolver(config,1e3);
    cout << "Option Details: "      << option   << endl;
    cout << "Monte-Carlo Price: "   << mcPrice  << endl;
    cout << "PDE Price: "           << pdePrice << endl;
    return 0;
}
