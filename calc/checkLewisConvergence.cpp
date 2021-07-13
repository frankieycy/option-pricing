#include "option.cpp"
using namespace std;

int main() {
    /**** Convergence check ***************************************************/
    int n0 = 1e3;
    int n1 = 2e4;
    int dn = 1e3;
    vector<bool> calcSwitch = {
        /* HES */ false,
        /* MER */ true
    };
    string dataFolder = "data-conv/";
    if(calcSwitch[0]){
        ofstream f1; f1.open(dataFolder+"EuropeanPut_HES.csv");
        Option option = Option("European","Put",80,1);
        Stock  stock  = Stock(100,0,0.05,0.2,{5,0.04,0.6,-0.4},"Heston");
        Market market = Market(0.02,stock);
        Pricer pricer = Pricer(option,market);
        f1 << "steps,fiPrice" << endl;
        for(int n=n0; n<=n1; n+=dn){
            double fiPrice = pricer.FourierInversionPricer(n,INF,"Lewis");
            cout << "Fourier-Inversion Price (Heston): " << fiPrice << endl;
            f1 << n << "," << fixed << setprecision(6) << fiPrice << endl;
        }
        f1.close();
    }
    if(calcSwitch[1]){
        ofstream f2; f2.open(dataFolder+"EuropeanPut_MER.csv");
        Option option = Option("European","Put",80,1);
        Stock  stock  = Stock(100,0,0.05,0.2,{2,-0.1,0.2},"jump-diffusion");
        Market market = Market(0.02,stock);
        Pricer pricer = Pricer(option,market);
        f2 << "steps,fiPrice" << endl;
        for(int n=n0; n<=n1; n+=dn){
            double fiPrice = pricer.FourierInversionPricer(n,INF,"Lewis");
            cout << "Fourier-Inversion Price (Merton): " << fiPrice << endl;
            f2 << n << "," << fixed << setprecision(6) << fiPrice << endl;
        }
        f2.close();
    }
    return 0;
}
