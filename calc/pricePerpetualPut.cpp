#include "option.cpp"
using namespace std;

int main() {
    /**** Perpetual ***********************************************************/
    Option option       = Option("American","Put",100,INF);
    Stock  stock        = Stock(100,0,0.05,0.2);
    Market market       = Market(0.1,stock);
    Pricer pricer       = Pricer(option,market);
    double bsPrice      = pricer.BlackScholesClosedForm();
    cout << "Option Details: "          << option   << endl;
    cout << "Black-Scholes Price: "     << bsPrice  << endl;
    return 0;
}
