#include "option.cpp"
using namespace std;

int main() {
    /**** DNN price dataset ***************************************************/
    srand(time(NULL));
    string dataFolder = "dnn_data/";
    int numSamples = 1e2;
    double M0 = 0.8, M1 = 1.2;          // moneyness
    double T0 = 4E-3, T1 = 3;           // maturity
    double r0 = 0.01, r1 = 0.03;        // risk-free rate
    double q0 = 0, q1 = 0.03;           // dividend yeild
    double sig0 = 0.05, sig1 = 0.50;    // volatility
    double kappa0 = 0.2, kappa1 = 2;    // reversion rate
    double theta0 = 0.01, theta1 = 0.2; // long run var
    double zeta0 = 0.01, zeta1 = 0.2;   // vol of vol
    double rho0 = -0.9, rho1 = -0.1;    // correlation
    ofstream f1; f1.open(dataFolder+"EuropeanCall_GBM.csv"); // pricing data
    f1 << "index,M,T,r,q,sig,bsPrice" << endl;
    for(int i=0; i<numSamples; i++){
        double S0 = 100;
        double M = uniformRand(M0,M1);
        double K = S0/M;
        double T = uniformRand(T0,T1);
        double r = uniformRand(r0,r1);
        double q = uniformRand(q0,q1);
        double mu = r-q;
        double sig = uniformRand(sig0,sig1);
        Stock stock = Stock(S0,q,mu,sig);
        Market market = Market(r,stock);
        Option option = Option("European","Call",K,T);
        Pricer pricer = Pricer(option,market);
        double bsPrice = pricer.BlackScholesClosedForm();
        f1 << "GBM-" << i << "," << M << "," << T << "," << r << "," << q << "," << sig << "," << bsPrice << endl;
        cout << getCurrentTime() << " [LOG] finish generating sample: GBM-" << i << endl;
    }
    f1.close();
    ofstream f2; f2.open(dataFolder+"EuropeanCall_HES.csv"); // pricing data
    f2 << "index,M,T,r,q,sig,kappa,theta,zeta,rho,fiPrice" << endl;
    for(int i=0; i<numSamples; i++){
        double S0 = 100;
        double M = uniformRand(M0,M1);
        double K = S0/M;
        double T = uniformRand(T0,T1);
        double r = uniformRand(r0,r1);
        double q = uniformRand(q0,q1);
        double mu = r-q;
        double sig = uniformRand(sig0,sig1);
        double kappa = uniformRand(kappa0,kappa1);
        double theta = uniformRand(theta0,theta1);
        double zeta = uniformRand(zeta0,zeta1);
        double rho = uniformRand(rho0,rho1);
        Stock stock = Stock(S0,q,mu,sig,{kappa,theta,zeta,rho});
        Market market = Market(r,stock);
        Option option = Option("European","Call",K,T);
        Pricer pricer = Pricer(option,market);
        double fiPrice = pricer.FourierInversionPricer(8192,INF,"FFT");
        f2 << "HES-" << i << ","  << M << "," << T << "," << r << "," << q << "," << sig << "," << kappa << "," << theta << "," << zeta << "," << rho << "," << fiPrice << endl;
        cout << getCurrentTime() << " [LOG] finish generating sample: HES-" << i << endl;
    }
    f2.close();
    return 0;
}
