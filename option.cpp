#ifndef OPTION
#define OPTION
#include "util.cpp"
#include "matrix.cpp"
using namespace std;

const set<string> OPTION_TYPES{
    "European", "American", "Digital", "Asian"
};

const set<string> PUT_CALL{
    "Put", "Call"
};

struct SimConfig{
    int iters;
    double endTime, stepSize;
    SimConfig(double t, int n):endTime(t),iters(n),stepSize(t/n){}
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
    string getAsJSON() const;
    /**** main ****/
    bool checkParams() const;
    double calcPayoff(double stockPrice=0, matrix<double> priceSeries=NULL_VECTOR);
    matrix<double> calcPayoffs(matrix<double> stockPriceVector=NULL_VECTOR, matrix<double> priceMatrix=NULL_MATRIX);
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Option& option);
};

class Stock{
private:
    double currentPrice, dividendYield, driftRate, volatility;
    matrix<double> simTimeVector, simPriceMatrix, binomialPriceTree;
public:
    /**** constructors ****/
    Stock(){};
    Stock(double currentPrice, double dividendYield, double driftRate, double volatility);
    /**** accessors ****/
    double getCurrentPrice() const {return currentPrice;}
    double getDividendYield() const {return dividendYield;}
    double getDriftRate() const {return driftRate;}
    double getVolatility() const {return volatility;}
    matrix<double> getSimTimeVector() const {return simTimeVector;}
    matrix<double> getSimPriceMatrix() const {return simPriceMatrix;}
    matrix<double> getBinomialPriceTree() const {return binomialPriceTree;}
    string getAsJSON() const;
    /**** mutators ****/
    double setDriftRate(double driftRate);
    /**** main ****/
    bool checkParams() const;
    matrix<double> simulatePrice(const SimConfig& config, int numSim=1);
    matrix<double> generatePriceTree(const SimConfig& config);
    matrix<double> generatePriceMatrixFromTree();
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Stock& stock);
};

class Market{
private:
    Stock stock;
    double riskFreeRate;
public:
    /**** constructors ****/
    Market(){};
    Market(double riskFreeRate, const Stock& stock);
    /**** accessors ****/
    Stock getStock() const {return stock;}
    double getRiskFreeRate() const {return riskFreeRate;}
    string getAsJSON() const;
    /**** main ****/
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Market& market);
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
    string getAsJSON() const;
    /**** main ****/
    double calcImpliedVolatility();
    double BlackScholesClosedForm();
    double BlackScholesPDESolver();
    double BinomialTreePricer(const SimConfig& config);
    double MonteCarloPricer(const SimConfig& config, int numSim);
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Pricer& pricer);
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

string Option::getAsJSON() const {
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
    OPTION_TYPES.find(type)!=OPTION_TYPES.end() &&
    PUT_CALL.find(putCall)!=PUT_CALL.end() &&
    strike>=0 && maturity>=0;
}

double Option::calcPayoff(double stockPrice, matrix<double> priceSeries){
    double S;
    if(type=="European"||type=="American"){
        if(priceSeries.isEmpty()) S = stockPrice;
        else S = priceSeries.getLastEntry();
        if(putCall=="Put") return max(strike-S,0.);
        else if(putCall=="Call") return max(S-strike,0.);
    }else if(type=="Digital"){
        if(priceSeries.isEmpty()) S = stockPrice;
        else S = priceSeries.getLastEntry();
        if(putCall=="Put") return (S<strike)?1.:0.;
        else if(putCall=="Call") return (S>strike)?1.:0.;
    }else if(type=="Asian"){
        if(priceSeries.isEmpty()) return NAN;
        else S = priceSeries.getRow(0).mean();
        if(putCall=="Put") return max(strike-S,0.);
        else if(putCall=="Call") return max(S-strike,0.);
    }
    return NAN;
}

matrix<double> Option::calcPayoffs(matrix<double> stockPriceVector, matrix<double> priceMatrix){
    matrix<double> S;
    if(type=="European" || type=="American"){
        if(priceMatrix.isEmpty()) S = stockPriceVector;
        else S = priceMatrix.getLastRow();
        if(putCall == "Put") return (strike-S).maxWith(0.);
        else if(putCall == "Call") return (S-strike).maxWith(0.);
    }else if(type=="Digital"){
        if(priceMatrix.isEmpty()) S = stockPriceVector;
        else S = priceMatrix.getLastRow();
        if(putCall == "Put") return (S<strike);
        else if(putCall == "Call") return (S>strike);
    }else if(type=="Asian"){
        if(priceMatrix.isEmpty()) return NULL_VECTOR;
        else S = priceMatrix.mean(2);
        if(putCall == "Put") return (strike-S).maxWith(0.);
        else if(putCall == "Call") return (S-strike).maxWith(0.);
    }
    return NULL_VECTOR;
}

/******************************************************************************/

Stock::Stock(double currentPrice, double dividendYield, double driftRate, double volatility){
    this->currentPrice = currentPrice;
    this->dividendYield = dividendYield;
    this->driftRate = driftRate;
    this->volatility = volatility;
    assert(checkParams());
}

string Stock::getAsJSON() const {
    ostringstream oss;
    oss << "{" <<
    "'currentPrice':"   << currentPrice     << "," <<
    "'dividendYield':"  << dividendYield    << "," <<
    "'driftRate':"      << driftRate        << "," <<
    "'volatility':"     << volatility       <<
    "}";
    return oss.str();
}

double Stock::setDriftRate(double driftRate){
    this->driftRate = driftRate;
    return driftRate;
}

bool Stock::checkParams() const {
    return currentPrice>=0 && dividendYield>=0 && volatility>=0;
}

matrix<double> Stock::simulatePrice(const SimConfig& config, int numSim){
    double n = config.iters;
    double dt = config.stepSize;
    double sqrt_dt = sqrt(dt);
    matrix<double> randomVector(1,numSim);
    matrix<double> simPriceVector(1,numSim,currentPrice);
    simTimeVector = matrix<double>(1,n);
    simPriceMatrix = matrix<double>(n,numSim);
    simPriceMatrix.setRow(0,simPriceVector);
    simTimeVector.setEntry(0,0,0);
    for(int i=1; i<n; i++){
        randomVector.setNormalRand();
        simPriceVector += simPriceVector*(driftRate*dt+volatility*sqrt_dt*randomVector);
        simPriceMatrix.setRow(i,simPriceVector);
        simTimeVector.setEntry(0,i,i*dt);
    }
    return simPriceMatrix;
}

matrix<double> Stock::generatePriceTree(const SimConfig& config){
    double n = config.iters;
    double dt = config.stepSize;
    double sqrt_dt = sqrt(dt);
    double u = exp(volatility*sqrt_dt), d = 1/u;
    simTimeVector = matrix<double>(1,n);
    binomialPriceTree = matrix<double>(n,n);
    binomialPriceTree.setEntry(0,0,currentPrice);
    simTimeVector.setEntry(0,0,0);
    for(int i=1; i<n; i++){
        for(int j=0; j<i; j++)
            binomialPriceTree.setEntry(i,j,binomialPriceTree.getEntry(i-1,j)*d);
        binomialPriceTree.setEntry(i,i,binomialPriceTree.getEntry(i-1,i-1)*u);
        simTimeVector.setEntry(0,i,i*dt);
    }
    return binomialPriceTree;
}

/******************************************************************************/

Market::Market(double riskFreeRate, const Stock& stock){
    this->riskFreeRate = riskFreeRate;
    this->stock = stock;
}

string Market::getAsJSON() const {
    ostringstream oss;
    oss << "{" <<
    "'riskFreeRate':"   << riskFreeRate << "," <<
    "'stock':"          << stock        <<
    "}";
    return oss.str();
}

/******************************************************************************/

Pricer::Pricer(const Option& option, const Market& market){
    this->option = option;
    this->market = market;
    this->price = NAN;
}

string Pricer::getAsJSON() const {
    ostringstream oss;
    oss << "{" <<
    "'option':"     << option   << "," <<
    "'market':"     << market   << "," <<
    "'price':"      << price    <<
    "}";
    return oss.str();
}

double Pricer::BlackScholesClosedForm(){
    if(option.getType()=="European"){
        Stock stock = market.getStock();
        double K   = option.getStrike();
        double T   = option.getMaturity();
        double r   = market.getRiskFreeRate();
        double S0  = stock.getCurrentPrice();
        double q   = stock.getDividendYield();
        double sig = stock.getVolatility();
        double d1  = (log(S0/K)+(r-q+sig*sig/2)*T)/(sig*sqrt(T));
        double d2  = d1-sig*sqrt(T);
        if(option.getPutCall()=="Call")
            price = S0*exp(-q*T)*normalCDF(d1)-K*exp(-r*T)*normalCDF(d2);
        else if(option.getPutCall()=="Put")
            price = K*exp(-r*T)*normalCDF(-d2)-S0*exp(-q*T)*normalCDF(-d1);
    }
    return price;
}

double Pricer::BinomialTreePricer(const SimConfig& config){
    Stock stock = market.getStock();
    double n = config.iters;
    double dt = config.stepSize;
    double sqrt_dt = sqrt(dt);
    double r = market.getRiskFreeRate();
    double q = stock.getDividendYield();
    double sig = stock.getVolatility();
    double u = exp(sig*sqrt_dt), d = 1/u;
    double qu = (exp((r-q)*dt)-d)/(u-d), qd = 1-qu;
    stock.setDriftRate(r);
    stock.generatePriceTree(config);
    matrix<double> optionBinomialTree(n,n);
    matrix<double> payoffs = option.calcPayoffs(stock.getBinomialPriceTree().getLastRow());
    optionBinomialTree.setRow(n-1,payoffs);
    for(int i=n-2; i>=0; i--){
        for(int j=0; j<i+1; j++)
            optionBinomialTree.setEntry(i,j,max(
                exp(-r*dt)*(qu*optionBinomialTree.getEntry(i+1,j+1)+qd*optionBinomialTree.getEntry(i+1,j)),
                (option.getType()=="American")?option.calcPayoff(stock.getBinomialPriceTree().getEntry(i,j)):0.
            ));
    }
    // cout << stock.getBinomialPriceTree().print() << endl;
    // cout << optionBinomialTree.print() << endl;
    price = optionBinomialTree.getEntry(0,0);
    return price;
}

double Pricer::MonteCarloPricer(const SimConfig& config, int numSim){
    Stock stock = market.getStock();
    double r = market.getRiskFreeRate();
    double T = option.getMaturity();
    stock.setDriftRate(r);
    stock.simulatePrice(config,numSim);
    matrix<double> payoffs = option.calcPayoffs(NULL_VECTOR,stock.getSimPriceMatrix());
    price = exp(-r*T)*payoffs.mean();
    return price;
}

/******************************************************************************/

ostream& operator<<(ostream& out, const Option& option){
    out << option.getAsJSON();
    return out;
}

ostream& operator<<(ostream& out, const Stock& stock){
    out << stock.getAsJSON();
    return out;
}

ostream& operator<<(ostream& out, const Market& market){
    out << market.getAsJSON();
    return out;
}

ostream& operator<<(ostream& out, const Pricer& pricer){
    out << pricer.getAsJSON();
    return out;
}

#endif
