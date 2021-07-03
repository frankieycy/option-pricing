#include "option.cpp"
using namespace std;

int main() {
    /**** Heston model imp vol ************************************************/
    string dataFolder = "data/";
    int m = 1e4; // num of sims
    double dt = 0.005; // step size
    double K0 = 50, K1 = 200, dK = 5;
    double T0 = 0.25, T1 = 2, dT = 0.25;
    double r        = 0.02;
    double S0       = 100;
    double q        = 0;
    double mu       = 0.05;
    double sig      = 0.2;
    double kappa    = 5;    // reversion rate
    double theta    = 0.04; // long run var
    double zeta     = 0.6;  // vol of vol
    double rho      = -0.4; // correlation
    Stock  stock    = Stock(S0,q,mu,sig,{kappa,theta,zeta,rho},"Heston");
    Market market   = Market(r,stock);
    vector<string> PutCallSet;
    vector<double> StrikeSet;
    vector<double> MaturitySet;
    PutCallSet = {"Put","Call"};
    for(double K=K0; K<=K1; K+=dK) StrikeSet.push_back(K);
    for(double T=T0; T<=T1; T+=dT) MaturitySet.push_back(T);
    ofstream f1; f1.open(dataFolder+"heston_config.json");
    ofstream f2; f2.open(dataFolder+"heston_price.csv");
    f1 << "{";
    for(double T:MaturitySet){
        for(string PC:PutCallSet){
            for(double K:StrikeSet){
                int n = T/dt;
                ostringstream oss;
                oss << "HES" << fixed << setprecision(2) << T << ((PC=="Call")?"C":"P") << K;
                string name = oss.str();
                SimConfig config    = SimConfig(T,n);
                Option option       = Option("European",PC,K,T,{},"",name);
                Pricer pricer       = Pricer(option,market);
                double bsPrice      = pricer.BlackScholesClosedForm();
                double mcPrice      = pricer.MonteCarloPricer(config,m);
                cout << "Contract Name: " << name << endl;
                cout << "Black-Scholes Price (Lognormal): " << bsPrice << endl;
                cout << "Monte-Carlo Price (Heston): " << mcPrice << endl;
                f1 << "\"" << name << "\":" << pricer << ",";
                f2 << name << "," << "European" << "," << PC << "," << K << "," << T << "," << mcPrice << endl;
            }
        }
    }
    f1 << "\"\":{}}"; // dummy
    f1.close();
    f2.close();
    return 0;
}
