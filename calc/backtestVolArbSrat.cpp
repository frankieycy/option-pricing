#include "option.cpp"
using namespace std;

int main(){
    /**** vol arb strat *******************************************************/
    Option option       = Option("European","Call",100,1);
    Stock  stock        = Stock(100,0,0.1,0.2);
    Market market       = Market(0.05,stock);
    Pricer pricer       = Pricer(option,market);
    SimConfig config    = SimConfig(1,100);
    Backtest backtest = pricer.runBacktest(config,500,"mkt-delta",1,0.3);
    backtest.printToCsvFiles();
    return 0;
}
