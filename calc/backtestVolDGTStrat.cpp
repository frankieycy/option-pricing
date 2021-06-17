#include "option.cpp"
using namespace std;

int main(){
    /**** vol DGT-neutral strat ***********************************************/
    Option option       = Option("European","Call",100,1);
    Option hOption0     = Option("European","Call",80,2);
    Option hOption1     = Option("European","Call",120,2);
    Stock  stock        = Stock(100,0,0.05,0.1);
    Market market       = Market(0.02,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,50);
    auto impVolFunc0 = [](double x){return mathFunc(x,"exponential",{0.03,5.65,1});}; // 20% --- 5%
    auto impVolFunc1 = [](double x){return mathFunc(x,"const",{0.1});}; // 10% --- 10%
    vector<matrix> impVolSurfaceSet = pricer.modelImpliedVolSurface(SimConfig(2,100),20,impVolFunc0,impVolFunc1,1,1e-3);
    Backtest backtest = pricer.runBacktest(config,50,"vol-delta-gamma-theta",1,0,{hOption0,hOption1},impVolSurfaceSet);
    impVolSurfaceSet[0].printToCsvFile("vol_gridX.csv");
    impVolSurfaceSet[1].printToCsvFile("vol_gridT.csv");
    impVolSurfaceSet[2].printToCsvFile("vol_surface.csv");
    backtest.printToCsvFiles();
    return 0;
}
