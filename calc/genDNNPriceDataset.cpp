#include "option.cpp"
using namespace std;

int main() {
    /**** DNN price dataset ***************************************************/
    srand(time(NULL));
    string dataFolder = "dnn_data/";
    int numSamples = 1e5;
    vector<bool> calcSwitch = {
        /* GBM   */ false,
        /* HES   */ false,
        /* MER   */ false,
    };
    string PC = "Call";
    double M0 = 0.8, M1 = 1.2;          // moneyness
    double T0 = 4e-3, T1 = 3;           // maturity
    double r0 = 0.01, r1 = 0.03;        // risk-free rate
    double q0 = 0, q1 = 0.03;           // dividend yeild
    double sig0 = 0.05, sig1 = 0.5;     // volatility
    if(calcSwitch[0]){
        ofstream f1; f1.open(dataFolder+"European"+PC+"_GBM."+to_string(numSamples)+"smpl.csv"); // pricing data
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
            Option option = Option("European",PC,K,T);
            Pricer pricer = Pricer(option,market);
            double bsPrice = pricer.BlackScholesClosedForm()/K;
            cout << "GBM-" << i << " Option Details: " << option << "; Stock Details: " << stock << endl;
            f1 << fixed << setprecision(6) << "GBM-" << i << "," << M << "," << T << "," << r << "," << q << "," << sig << "," << bsPrice << endl;
            cout << getCurrentTime() << " [LOG] finish generating sample: GBM-" << i << endl;
        }
        f1.close();
    }
    if(calcSwitch[1]){
        double kappa0 = 0.2, kappa1 = 2;    // reversion rate
        double theta0 = 0.01, theta1 = 0.2; // long run var
        double zeta0 = 0.05, zeta1 = 0.5;   // vol of var
        double rho0 = -0.9, rho1 = -0.1;    // correlation
        ofstream f2; f2.open(dataFolder+"European"+PC+"_HES."+to_string(numSamples)+"smpl.csv"); // pricing data
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
            Stock stock = Stock(S0,q,mu,sig,{kappa,theta,zeta,rho},"Heston");
            Market market = Market(r,stock);
            Option option = Option("European",PC,K,T);
            Pricer pricer = Pricer(option,market);
            double fiPrice = pricer.FourierInversionPricer(8e3,INF,"Lewis")/K;
            cout << "HES-" << i << " Option Details: " << option << "; Stock Details: " << stock << endl;
            f2 << fixed << setprecision(6) << "HES-" << i << ","  << M << "," << T << "," << r << "," << q << "," << sig << "," << kappa << "," << theta << "," << zeta << "," << rho << "," << fiPrice << endl;
            cout << getCurrentTime() << " [LOG] finish generating sample: HES-" << i << endl;
        }
        f2.close();
    }
    if(calcSwitch[2]){
        double lamJ0 = 0.1, lamJ1 = 5;      // jump intensity
        double muJ0 = -0.2, muJ1 = 0.2;     // jump mean
        double sigJ0 = 0.05, sigJ1 = 0.3;   // jump s.d.
        ofstream f3; f3.open(dataFolder+"European"+PC+"_MER."+to_string(numSamples)+"smpl.csv"); // pricing data
        f3 << "index,M,T,r,q,sig,lamJ,muJ,sigJ,fiPrice" << endl;
        for(int i=0; i<numSamples; i++){
            double S0 = 100;
            double M = uniformRand(M0,M1);
            double K = S0/M;
            double T = uniformRand(T0,T1);
            double r = uniformRand(r0,r1);
            double q = uniformRand(q0,q1);
            double mu = r-q;
            double sig = uniformRand(sig0,sig1);
            double lamJ = uniformRand(lamJ0,lamJ1);
            double muJ = uniformRand(muJ0,muJ1);
            double sigJ = uniformRand(sigJ0,sigJ1);
            Stock stock = Stock(S0,q,mu,sig,{lamJ,muJ,sigJ},"jump-diffusion");
            Market market = Market(r,stock);
            Option option = Option("European",PC,K,T);
            Pricer pricer = Pricer(option,market);
            double fiPrice = pricer.FourierInversionPricer(8e3,INF,"Lewis")/K;
            cout << "MER-" << i << " Option Details: " << option << "; Stock Details: " << stock << endl;
            f3 << fixed << setprecision(6) << "MER-" << i << ","  << M << "," << T << "," << r << "," << q << "," << sig << "," << lamJ << "," << muJ << "," << sigJ << "," << fiPrice << endl;
            cout << getCurrentTime() << " [LOG] finish generating sample: MER-" << i << endl;
        }
        f3.close();
    }
    return 0;
}
