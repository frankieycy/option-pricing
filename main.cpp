#include "option.cpp"
using namespace std;

int main() {
    Option option       = Option("European","Call",100,1);
    Stock  stock        = Stock(100,0,0.05,0.1);
    Market market       = Market(0.02,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,100);
    cout << pricer.BlackScholesClosedForm() << endl;
    cout << pricer.BinomialTreePricer(config) << endl;
    cout << pricer.MonteCarloPricer(config,500) << endl;
    return 0;
}
