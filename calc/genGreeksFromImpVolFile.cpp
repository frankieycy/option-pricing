#include "lib/option.cpp"
using namespace std;

int main() {
    string dataFolder = "data/";
    Pricer pricer;
    pricer.setVariablesFromFile(
        dataFolder+"pricer_var.csv");
    pricer.saveAsOriginal();
    pricer.generateGreeksFromImpliedVolFile(
        dataFolder+"option_vol.csv",
        dataFolder+"option_grk.csv");
    return 0;
}
