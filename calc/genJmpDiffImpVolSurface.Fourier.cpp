#include "option.cpp"
using namespace std;

int main() {
    /**** Merton model imp vol ************************************************/
    string dataFolder = "data/";
    int m = 4e3; // num of steps
    double K0 = 50, K1 = 120, dK = 5;
    double T0 = 0.2, T1 = 1.2, dT = 0.2;
    double r        = 0.02;
    double S0       = 100;
    double q        = 0;
    double mu       = 0.05;
    double sig      = 0.2;
    double lamJ     = 2;    // jump intensity
    double muJ      = -0.1; // jump mean
    double sigJ     = 0.2;  // jump s.d.
    Stock  stock    = Stock(S0,q,mu,sig,{lamJ,muJ,sigJ},"jump-diffusion","MER");
    Market market   = Market(r,stock);
    vector<string> PutCallSet;
    vector<double> StrikeSet;
    vector<double> MaturitySet;
    PutCallSet = {"Put","Call"};
    for(double K=K0; K<=K1; K+=dK) StrikeSet.push_back(K);
    for(double T=T0; T<=T1; T+=dT) MaturitySet.push_back(T);
    ofstream f1; f1.open(dataFolder+"merton_config.json");  // input data
    ofstream f2; f2.open(dataFolder+"merton_price.csv");    // pricing data
    f1 << "{";
    for(double T:MaturitySet){
        for(string PC:PutCallSet){
            for(double K:StrikeSet){
                ostringstream oss;
                oss << "MER" << fixed << setprecision(2) << T << ((PC=="Call")?"C":"P") << K;
                string name = oss.str();
                Option option       = Option("European",PC,K,T,{},{},name);
                Pricer pricer       = Pricer(option,market);
                double bsPrice      = pricer.BlackScholesClosedForm();
                double fiPrice      = pricer.FourierInversionPricer(m,INF,"Lewis");
                cout << "Contract Name: " << name << endl;
                cout << "Black-Scholes Price (Lognormal): " << bsPrice << endl;
                cout << "Fourier-Inversion Price (Merton): " << fiPrice << endl;
                f1 << "\"" << name << "\":" << pricer << ",";
                f2 << name << "," << "European" << "," << PC << "," << K << "," << T << "," << fiPrice << endl;
            }
        }
    }
    f1 << "\"\":{}}"; // dummy
    f1.close();
    f2.close();
    return 0;
}
