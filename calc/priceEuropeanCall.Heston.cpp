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
    double fiPrice      = pricer.FourierInversionPricer(1e4);
    cout << "Black-Scholes Price (Lognormal): " << bsPrice << endl;
    cout << "Monte-Carlo Price (Heston): " << mcPrice << endl;
    cout << "Fourier-Inversion Price (Heston): " << fiPrice << endl;
    return 0;
}
