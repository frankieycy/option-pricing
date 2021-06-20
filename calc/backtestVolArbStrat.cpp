#include "option.cpp"
using namespace std;

int main(int argc, char **argv){ // pass in sigHedge
    /**** vol arb strat *******************************************************/
    string dataFolder = "data-VolArbStrat/";
    double sigAct     = 0.2;
    double sigImp     = 0.4;
    double sigHedge   = stod(argv[1]);
    Option option     = Option("European","Call",100,1);
    Stock  stock      = Stock(100,0,0.1,sigAct);
    Market market     = Market(0.05,stock);
    Pricer pricer     = Pricer(option,market);
    SimConfig config  = SimConfig(1,50);
    Backtest backtest = pricer.runBacktest(config,200,"mkt-delta-hedgingVol",1,sigImp,0,{sigHedge});
    backtest.printToCsvFiles(false,
        dataFolder+"sigHedge="+argv[1]);
    return 0;
}
