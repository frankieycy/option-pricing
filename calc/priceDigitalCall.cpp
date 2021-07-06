#include "option.cpp"
using namespace std;

int main() {
    /**** Digital *************************************************************/
    Option option       = Option("Digital","Call",100,1);
    Stock  stock        = Stock(100,0,0.05,0.2);
    Market market       = Market(0.02,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,200);
    double bsPrice      = pricer.BlackScholesClosedForm();
    double niPrice      = pricer.NumIntegrationPricer();
    double btPrice      = pricer.BinomialTreePricer(config);
    double mcPrice      = pricer.MonteCarloPricer(config,1e4);
    double pdePrice     = pricer.BlackScholesPDESolver(config,1e3);
    double fiPrice      = pricer.FourierInversionPricer(1e4,1e3);
    cout << "Option Details: "          << option   << endl;
    cout << "Black-Scholes Price: "     << bsPrice  << endl;
    cout << "Num-Integration Price: "   << niPrice  << endl;
    cout << "Binomial-Tree Price: "     << btPrice  << endl;
    cout << "Monte-Carlo Price: "       << mcPrice  << endl;
    cout << "PDE Price: "               << pdePrice << endl;
    cout << "Fourier-Inversion Price: " << fiPrice  << endl;
    return 0;
}
