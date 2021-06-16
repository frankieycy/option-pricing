#include "option.cpp"
using namespace std;

int main() {
    string dataFolder = "data/";
    Pricer pricer;
    pricer.setVariablesFromFile(
        dataFolder+"pricer_var.csv");
    pricer.generateImpliedVolSurfaceFromFile(
        dataFolder+"option_data.csv",
        dataFolder+"option_vol.csv");
    return 0;
}
