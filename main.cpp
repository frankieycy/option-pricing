#include "option.cpp"
using namespace std;

int main() {
    Option option   = Option("European", "Call", 100, 1);
    Stock  stock    = Stock(100, 0, 0.05, 0.1);
    Market market   = Market(0.05, stock);
    Pricer pricer   = Pricer(option, market);
    cout << pricer << endl;
    return 0;
}
