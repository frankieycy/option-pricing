#include "option.cpp"
using namespace std;

int main() {
    /**** Convergence check ***************************************************/
    int n0 = 1e3;
    int n1 = 1e4;
    int dn = 1e3;
    int numSamples = 20;
    vector<bool> calcSwitch = {
        /* HES */ false,
        /* MER */ false
    };
    string dataFolder = "data-conv/";
    if(calcSwitch[0]){
        double kappa0 = 0.2, kappa1 = 2;    // reversion rate
        double theta0 = 0.01, theta1 = 0.2; // long run var
        double zeta0 = 0.05, zeta1 = 0.5;   // vol of var
        double rho0 = -0.9, rho1 = -0.1;    // correlation
        ofstream f1; f1.open(dataFolder+"EuropeanPut_HES.Random.csv");
        Option option = Option("European","Put",80,1);
        f1 << "steps,config,fiPrice" << endl;
        for(int i=0; i<numSamples; i++){
            double kappa = uniformRand(kappa0,kappa1);
            double theta = uniformRand(theta0,theta1);
            double zeta = uniformRand(zeta0,zeta1);
            double rho = uniformRand(rho0,rho1);
            Stock  stock  = Stock(100,0,0.05,0.2,{kappa,theta,zeta,rho},"Heston");
            Market market = Market(0.02,stock);
            Pricer pricer = Pricer(option,market);
            for(int n=n0; n<=n1; n+=dn){
                double fiPrice = pricer.FourierInversionPricer(n,INF,"Lewis");
                cout << "Fourier-Inversion Price (Heston): " << fiPrice << endl;
                f1 << n << "," << fixed << setprecision(6) << kappa << "|" << theta << "|" << zeta << "|" << rho << "," << fiPrice << endl;
            }
        }
        f1.close();
    }
    if(calcSwitch[1]){
        double lamJ0 = 0.1, lamJ1 = 5;      // jump intensity
        double muJ0 = -0.2, muJ1 = 0.2;     // jump mean
        double sigJ0 = 0.05, sigJ1 = 0.3;   // jump s.d.
        ofstream f2; f2.open(dataFolder+"EuropeanPut_MER.Random.csv");
        Option option = Option("European","Put",80,1);
        f2 << "steps,config,fiPrice" << endl;
        for(int i=0; i<numSamples; i++){
            double lamJ = uniformRand(lamJ0,lamJ1);
            double muJ = uniformRand(muJ0,muJ1);
            double sigJ = uniformRand(sigJ0,sigJ1);
            Stock  stock  = Stock(100,0,0.05,0.2,{lamJ,muJ,sigJ},"jump-diffusion");
            Market market = Market(0.02,stock);
            Pricer pricer = Pricer(option,market);
            for(int n=n0; n<=n1; n+=dn){
                double fiPrice = pricer.FourierInversionPricer(n,INF,"Lewis");
                cout << "Fourier-Inversion Price (Merton): " << fiPrice << endl;
                f2 << n << "," << fixed << setprecision(6) << lamJ << "|" << muJ << "|" << sigJ << "," << fiPrice << endl;
            }
        }
        f2.close();
    }
    return 0;
}
