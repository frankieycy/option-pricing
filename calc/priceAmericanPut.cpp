#include "option.cpp"
using namespace std;

int main() {
    /**** American ************************************************************/
    Option option       = Option("American","Put",100,1);
    Stock  stock        = Stock(100,0,0.05,0.2);
    Market market       = Market(0.1,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,200);
    double btPrice      = pricer.BinomialTreePricer(config);
    double pdePrice     = pricer.BlackScholesPDESolver(config,1e3);
    cout << "Option Details: "          << option   << endl;
    cout << "Binomial-Tree Price: "     << btPrice  << endl;
    cout << "PDE Price: "               << pdePrice << endl;
    return 0;
}
