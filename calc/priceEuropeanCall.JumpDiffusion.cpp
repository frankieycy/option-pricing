#include "option.cpp"
using namespace std;

int main() {
    /**** Jump-diffusion model ************************************************/
    Option option       = Option("European","Call",100,1);
    Stock  stock        = Stock(100,0,0.05,0.2,{1,-0.05,0.1},"jump-diffusion");
    Market market       = Market(0.1,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,200);
    double mcPrice      = pricer.MonteCarloPricer(config,1e4);
    double fiPrice      = pricer.FourierInversionPricer(1e4);
    cout << "Monte-Carlo Price (Jump-Diffusion): " << mcPrice << endl;
    cout << "Fourier-Inversion Price (Jump-Diffusion): " << fiPrice << endl;
    return 0;
}
