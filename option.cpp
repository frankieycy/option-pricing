#ifndef OPTION
#define OPTION
#include "util.cpp"
#include "matrix.cpp"
using namespace std;

const set<string> OPTION_TYPES{
    "European", "American", "Digital"
};

const set<string> PUT_CALL{
    "Put", "Call"
};

struct SimConfig{
    int iters;
    double endTime, dt;
    SimConfig(double t, int n):endTime(t),iters(n),dt(t/n){}
};

class Option{
private:
    string type, putCall;
    double strike, maturity;
public:
    /**** constructors ****/
    Option(){};
    Option(string type, string putCall, double strike, double maturity);
    Option(const Option& option);
    /**** accessors ****/
    string getType() const {return type;}
    string getPutCall() const {return putCall;}
    double getStrike() const {return strike;}
    double getMaturity() const {return maturity;}
    string getOptionAsJSON() const;
    friend ostream& operator<<(ostream& out, const Option& option);
    /**** main ****/
    bool checkParams() const;
};

class Stock{
private:
    double currentPrice, dividendYield, driftRate, volatility;
    matrix<double> simPriceMatrix;
public:
    /**** constructors ****/
    Stock(){};
    Stock(double currentPrice, double dividendYield, double driftRate, double volatility);
    /**** accessors ****/
    double getCurrentPrice() const {return currentPrice;}
    double getDividendYield() const {return dividendYield;}
    double getVolatility() const {return volatility;}
    matrix<double> getSimPriceMatrix() const {return simPriceMatrix;}
    string getStockAsJSON() const;
    friend ostream& operator<<(ostream& out, const Stock& stock);
    /**** main ****/
    bool checkParams() const;
    matrix<double> simulatePrice(SimConfig config, int numSim);
};

class Market{
private:
    double riskFreeRate;
    Stock stock;
public:
    /**** constructors ****/
    Market(){};
    Market(double riskFreeRate, const Stock& stock);
    /**** accessors ****/
    Stock getStock() const {return stock;}
    double getRiskFreeRate() const {return riskFreeRate;}
    string getMarketAsJSON() const;
    friend ostream& operator<<(ostream& out, const Market& market);
    /**** main ****/
};

class Pricer{
private:
    Option option;
    Market market;
    double price;
public:
    /**** constructors ****/
    Pricer(){};
    Pricer(const Option& option, const Market& market);
    /**** accessors ****/
    Option getOption() const {return option;}
    Market getMarket() const {return market;}
    string getPricerAsJSON() const;
    friend ostream& operator<<(ostream& out, const Pricer& pricer);
    /**** main ****/
    double calcCallFromParity();
    double calcPutFromParity();
    double calcImpliedVolatility();
    double BlackScholesClosedForm();
    double BlackScholesPDESolver();
    double MonteCarloPricer();
};

/******************************************************************************/

Option::Option(string type, string putCall, double strike, double maturity){
    this->type = type;
    this->putCall = putCall;
    this->strike = strike;
    this->maturity = maturity;
    assert(checkParams());
}

Option::Option(const Option& option){
    this->type = option.type;
    this->putCall = option.putCall;
    this->strike = option.strike;
    this->maturity = option.maturity;
}

string Option::getOptionAsJSON() const {
    ostringstream oss;
    oss << "{" <<
    "'type':'"      << type     << "'," <<
    "'putCall':'"   << putCall  << "'," <<
    "'strike':"     << strike   << ","  <<
    "'maturity':"   << maturity <<
    "}";
    return oss.str();
}

bool Option::checkParams() const {
    return
    OPTION_TYPES.find(type) != OPTION_TYPES.end() &&
    PUT_CALL.find(putCall) != PUT_CALL.end() &&
    strike >= 0 && maturity >= 0;
}

/******************************************************************************/

Stock::Stock(double currentPrice, double dividendYield, double driftRate, double volatility){
    this->currentPrice = currentPrice;
    this->dividendYield = dividendYield;
    this->driftRate = driftRate;
    this->volatility = volatility;
    assert(checkParams());
}

string Stock::getStockAsJSON() const {
    ostringstream oss;
    oss << "{" <<
    "'currentPrice':"   << currentPrice     << "," <<
    "'dividendYield':"  << dividendYield    << "," <<
    "'driftRate':"      << driftRate        << "," <<
    "'volatility':"     << volatility       <<
    "}";
    return oss.str();
}

bool Stock::checkParams() const {
    return currentPrice >= 0 && dividendYield >=0 && volatility >= 0;
}

matrix<double> Stock::simulatePrice(SimConfig config, int numSim=1){
    return simPriceMatrix;
}

/******************************************************************************/

Market::Market(double riskFreeRate, const Stock& stock){
    this->riskFreeRate = riskFreeRate;
    this->stock = stock;
}

string Market::getMarketAsJSON() const {
    ostringstream oss;
    oss << "{" <<
    "'riskFreeRate':"   << riskFreeRate << "," <<
    "'stock':"          << stock        <<
    "}";
    return oss.str();
}

/******************************************************************************/

Pricer::Pricer(const Option& option, const Market& market){
    this->price = NAN;
    this->option = option;
    this->market = market;
}

string Pricer::getPricerAsJSON() const {
    ostringstream oss;
    oss << "{" <<
    "'option':"     << option   << "," <<
    "'market':"     << market   << "," <<
    "'price':"      << price    <<
    "}";
    return oss.str();
}

/******************************************************************************/

ostream& operator<<(ostream& out, const Option& option){
    out << option.getOptionAsJSON();
    return out;
}

ostream& operator<<(ostream& out, const Stock& stock){
    out << stock.getStockAsJSON();
    return out;
}

ostream& operator<<(ostream& out, const Market& market){
    out << market.getMarketAsJSON();
    return out;
}

ostream& operator<<(ostream& out, const Pricer& pricer){
    out << pricer.getPricerAsJSON();
    return out;
}

#endif
