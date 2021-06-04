#include "option.cpp"
using namespace std;

int main() {
    Option option       = Option("European","Call",100,1);
    Stock  stock        = Stock(100,0,0.05,0.1);
    Market market       = Market(0.02,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,100);
    // cout << pricer.calcPrice("Closed Form") << endl;
    // cout << pricer.calcPrice("Binomial Tree",config) << endl;
    // cout << pricer.calcPrice("Monte Carlo",config,500) << endl;
    matrix<double> S0; S0.setRange(80,120);
    cout << pricer.varyGreekWithVariable("currentPrice",S0,"Delta") << endl;
    return 0;
}
