#ifndef OPTION
#define OPTION
#include "util.cpp"
#include "matrix.cpp"
using namespace std;

#define GUI true
#define LOG true

inline void logMessage(string msg){if(LOG) cout << getCurrentTime() << " [LOG] " << msg << endl;}

/**** global variables ********************************************************/

const set<string> OPTION_TYPES{
    "European", "American", "Digital", "Asian"
};
const set<string> EARLY_EX_OPTIONS{
    "American"
};
const set<string> PATH_DEP_OPTIONS{
    "Asian"
};
const set<string> PUT_CALL{
    "Put", "Call"
};

/**** class declarations ******************************************************/

class SimConfig{
public:
    int iters;
    double endTime, stepSize;
    SimConfig(double t=0, int n=1):endTime(t),iters(n),stepSize(t/n){}
    bool isEmpty() const {return endTime==0;}
    string getAsJson() const;
    friend ostream& operator<<(ostream& out, const SimConfig& config);
};

const SimConfig NULL_CONFIG;

class Process{}; // TO DO

class Option{
private:
    string name, type, putCall;
    double strike, maturity;
public:
    /**** constructors ****/
    Option(){};
    Option(string type, string putCall, double strike, double maturity, string name="unnamed");
    Option(const Option& option);
    /**** accessors ****/
    bool canEarlyExercise() const;
    bool isPathDependent() const;
    string getName() const {return name;}
    string getType() const {return type;}
    string getPutCall() const {return putCall;}
    double getStrike() const {return strike;}
    double getMaturity() const {return maturity;}
    string getAsJson() const;
    /**** mutators ****/
    string setName(string name);
    double setStrike(double strike);
    double setMaturity(double maturity);
    /**** main ****/
    bool checkParams() const;
    double calcPayoff(double stockPrice=0, matrix<double> priceSeries=NULL_VECTOR);
    matrix<double> calcPayoffs(matrix<double> stockPriceVector=NULL_VECTOR, matrix<double> priceMatrix=NULL_MATRIX);
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Option& option);
};

class Stock{
private:
    string name;
    double currentPrice, dividendYield, driftRate, volatility;
    matrix<double> simTimeVector, simPriceMatrix, binomialPriceTree;
public:
    /**** constructors ****/
    Stock(){};
    Stock(double currentPrice, double dividendYield, double driftRate, double volatility, string name="unnamed");
    Stock(const Stock& stock);
    /**** accessors ****/
    string getName() const {return name;}
    double getCurrentPrice() const {return currentPrice;}
    double getDividendYield() const {return dividendYield;}
    double getDriftRate() const {return driftRate;}
    double getVolatility() const {return volatility;}
    matrix<double> getSimTimeVector() const {return simTimeVector;}
    matrix<double> getSimPriceMatrix() const {return simPriceMatrix;}
    matrix<double> getBinomialPriceTree() const {return binomialPriceTree;}
    string getAsJson() const;
    /**** mutators ****/
    string setName(string name);
    double setCurrentPrice(double currentPrice);
    double setDividendYield(double dividendYield);
    double setDriftRate(double driftRate);
    double setVolatility(double volatility);
    double estDriftRateFromPrice(matrix<double> priceSeries, double dt, string method="simple");
    double estVolatilityFromPrice(matrix<double> priceSeries, double dt, string method="simple");
    /**** main ****/
    bool checkParams() const;
    double calcLognormalPrice(double z, double time);
    matrix<double> calcLognormalPriceVector(matrix<double> z, double time);
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
    Market(const Market& market);
    /**** accessors ****/
    Stock getStock() const {return stock;}
    double getRiskFreeRate() const {return riskFreeRate;}
    string getAsJson() const;
    /**** mutators ****/
    double setRiskFreeRate(double riskFreeRate);
    Stock setStock(const Stock& stock);
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Market& market);
};

class Pricer{
private:
    Option option, option_orig;
    Market market, market_orig;
    double price;
public:
    /**** constructors ****/
    Pricer(){};
    Pricer(const Option& option, const Market& market);
    /**** accessors ****/
    Option getOption() const {return option;}
    Market getMarket() const {return market;}
    double getPrice() const {return price;}
    string getAsJson() const;
    double getVariable(string var) const;
    /**** mutators ****/
    double setVariable(string var, double v);
    string setStringVariable(string var, string v);
    Pricer setVariablesFromFile(string file);
    Pricer resetOriginal();
    Pricer saveAsOriginal();
    /**** main ****/
    double BlackScholesClosedForm();
    double BinomialTreePricer(const SimConfig& config);
    double MonteCarloPricer(const SimConfig& config, int numSim);
    double NumIntegrationPricer(double z=5, double dz=1e-3);
    double BlackScholesPDESolver(const SimConfig& config, int numSpace, string method="implicit");
    double calcPrice(string method="Closed Form", const SimConfig& config=NULL_CONFIG,
        int numSim=0, int numSpace=0);
    matrix<double> varyPriceWithVariable(string var, matrix<double> varVector,
        string method="Closed Form", const SimConfig& config=NULL_CONFIG, int numSim=0);
    double ClosedFormGreek(string var, int derivOrder=1);
    double FiniteDifferenceGreek(string var, int derivOrder=1, string method="Closed Form",
        const SimConfig& config=NULL_CONFIG, int numSim=0, double eps=1e-5);
    double calcGreek(string greekName, string greekMethod="Closed Form", string method="Closed Form",
        const SimConfig& config=NULL_CONFIG, int numSim=0, double eps=1e-5);
    matrix<double> varyGreekWithVariable(string var, matrix<double> varVector,
        string greekName, string greekMethod="Closed Form", string method="Closed Form",
        const SimConfig& config=NULL_CONFIG, int numSim=0, double eps=1e-5);
    matrix<double> generatePriceSurface(matrix<double> stockPriceVector, matrix<double> optionTermVector,
        string method="Closed Form", const SimConfig& config=NULL_CONFIG, int numSim=0);
    bool satisfyPriceBounds(double optionMarketPrice);
    double calcImpliedVolatility(double optionMarketPrice, double vol0=5, double eps=1e-5);
    void generateImpliedVolSurfaceFromFile(string input, string file, double vol0=5, double eps=1e-5);
    void generateGreeksFromImpliedVolFile(string input, string file);
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Pricer& pricer);
};

class Backtester{
private:
    string name;
    Option option;
    Market market;
public:
};

/**** class functions *********************************************************/
//### SimConfig class ##########################################################

string SimConfig::getAsJson() const {
    ostringstream oss;
    oss << "{" <<
    "\"iters\":"    << iters    << "," <<
    "\"endTime\":"  << endTime  << "," <<
    "\"stepSize\":" << stepSize <<
    "}";
    return oss.str();
}

//### Option class #############################################################

Option::Option(string type, string putCall, double strike, double maturity, string name){
    this->type = type;
    this->putCall = putCall;
    this->strike = strike;
    this->maturity = maturity;
    this->name = name;
    assert(checkParams());
}

Option::Option(const Option& option){
    this->type = option.type;
    this->putCall = option.putCall;
    this->strike = option.strike;
    this->maturity = option.maturity;
    this->name = option.name;
}

bool Option::canEarlyExercise() const {
    return EARLY_EX_OPTIONS.find(type)!=EARLY_EX_OPTIONS.end();
}

bool Option::isPathDependent() const {
    return PATH_DEP_OPTIONS.find(type)!=PATH_DEP_OPTIONS.end();
}

string Option::getAsJson() const {
    ostringstream oss;
    oss << "{" <<
    "\"name\":\""      << name     << "\"," <<
    "\"type\":\""      << type     << "\"," <<
    "\"putCall\":\""   << putCall  << "\"," <<
    "\"strike\":"      << strike   << ","  <<
    "\"maturity\":"    << maturity <<
    "}";
    return oss.str();
}

string Option::setName(string name){
    this->name = name;
    return name;
}

double Option::setStrike(double strike){
    this->strike = strike;
    return strike;
}

double Option::setMaturity(double maturity){
    this->maturity = maturity;
    return maturity;
}

bool Option::checkParams() const {
    return
    OPTION_TYPES.find(type)!=OPTION_TYPES.end() &&
    PUT_CALL.find(putCall)!=PUT_CALL.end() &&
    strike>=0 && maturity>=0;
}

double Option::calcPayoff(double stockPrice, matrix<double> priceSeries){
    double S;
    if(type=="European" || type=="American"){
        if(priceSeries.isEmpty()) S = stockPrice;
        else S = priceSeries.getLastEntry();
        if(putCall=="Put") return max(strike-S,0.);
        else if(putCall=="Call") return max(S-strike,0.);
    }else if(type=="Digital"){
        if(priceSeries.isEmpty()) S = stockPrice;
        else S = priceSeries.getLastEntry();
        if(putCall=="Put") return (S<strike);
        else if(putCall=="Call") return (S>strike);
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
        if(putCall=="Put") return (strike-S).maxWith(0.);
        else if(putCall=="Call") return (S-strike).maxWith(0.);
    }else if(type=="Digital"){
        if(priceMatrix.isEmpty()) S = stockPriceVector;
        else S = priceMatrix.getLastRow();
        if(putCall=="Put") return (S<strike);
        else if(putCall=="Call") return (S>strike);
    }else if(type=="Asian"){
        if(priceMatrix.isEmpty()) return NULL_VECTOR;
        else S = priceMatrix.mean(2);
        if(putCall=="Put") return (strike-S).maxWith(0.);
        else if(putCall=="Call") return (S-strike).maxWith(0.);
    }
    return NULL_VECTOR;
}

//### Stock class ##############################################################

Stock::Stock(double currentPrice, double dividendYield, double driftRate, double volatility, string name){
    this->currentPrice = currentPrice;
    this->dividendYield = dividendYield;
    this->driftRate = driftRate;
    this->volatility = volatility;
    this->name = name;
    assert(checkParams());
}

Stock::Stock(const Stock& stock){
    this->currentPrice = stock.currentPrice;
    this->dividendYield = stock.dividendYield;
    this->driftRate = stock.driftRate;
    this->volatility = stock.volatility;
    this->name = stock.name;
}

string Stock::getAsJson() const {
    ostringstream oss;
    oss << "{" <<
    "\"name\":\""         << name             << "\"," <<
    "\"currentPrice\":"   << currentPrice     << "," <<
    "\"dividendYield\":"  << dividendYield    << "," <<
    "\"driftRate\":"      << driftRate        << "," <<
    "\"volatility\":"     << volatility       <<
    "}";
    return oss.str();
}

string Stock::setName(string name){
    this->name = name;
    return name;
}

double Stock::setCurrentPrice(double currentPrice){
    this->currentPrice = currentPrice;
    return currentPrice;
}

double Stock::setDividendYield(double dividendYield){
    this->dividendYield = dividendYield;
    return dividendYield;
}

double Stock::setDriftRate(double driftRate){
    this->driftRate = driftRate;
    return driftRate;
}

double Stock::setVolatility(double volatility){
    this->volatility = volatility;
    return volatility;
}

double Stock::estDriftRateFromPrice(matrix<double> priceSeries, double dt, string method){
    if(method=="simple"){
        matrix<double> returnSeries;
        returnSeries =
            (priceSeries.submatrix(1,-1,"col")-priceSeries.submatrix(0,-2,"col"))
            /priceSeries.submatrix(0,-2,"col");
        driftRate = returnSeries.mean()/dt;
    }
    return driftRate;
}

double Stock::estVolatilityFromPrice(matrix<double> priceSeries, double dt, string method){
    if(method=="simple"){
        matrix<double> returnSeries;
        returnSeries =
            (priceSeries.submatrix(1,-1,"col")-priceSeries.submatrix(0,-2,"col"))
            /priceSeries.submatrix(0,-2,"col");
        volatility = sqrt(returnSeries.var()/dt);
    }
    return volatility;
}

bool Stock::checkParams() const {
    return currentPrice>=0 && dividendYield>=0 && volatility>=0;
}

double Stock::calcLognormalPrice(double z, double time){
    double S = currentPrice*exp(
        (driftRate-volatility*volatility/2)*time+volatility*sqrt(time)*z
    );
    return S;
}

matrix<double> Stock::calcLognormalPriceVector(matrix<double> z, double time){
    int n = z.getCols();
    matrix<double> S(1,n);
    for(int i=0; i<n; i++) S.setEntry(0,i,calcLognormalPrice(z.getEntry(0,i),time));
    return S;
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

//### Market class #############################################################

Market::Market(double riskFreeRate, const Stock& stock){
    this->riskFreeRate = riskFreeRate;
    this->stock = stock;
}

Market::Market(const Market& market){
    this->riskFreeRate = market.riskFreeRate;
    this->stock = market.stock;
}

string Market::getAsJson() const {
    ostringstream oss;
    oss << "{" <<
    "\"riskFreeRate\":"   << riskFreeRate << "," <<
    "\"stock\":"          << stock        <<
    "}";
    return oss.str();
}

double Market::setRiskFreeRate(double riskFreeRate){
    this->riskFreeRate = riskFreeRate;
    return riskFreeRate;
}

Stock Market::setStock(const Stock& stock){
    this->stock = stock;
    return stock;
}

//### Pricer class #############################################################

Pricer::Pricer(const Option& option, const Market& market){
    this->option = option; this->option_orig = option;
    this->market = market; this->market_orig = market;
    this->price = NAN;
}

string Pricer::getAsJson() const {
    ostringstream oss;
    oss << "{" <<
    "\"option\":"     << option   << "," <<
    "\"market\":"     << market   <<
    "}";
    return oss.str();
}

double Pricer::getVariable(string var) const {
    double v = NAN;
    if(var=="currentPrice"){
        v = market.getStock().getCurrentPrice();
    }else if(var=="dividendYield"){
        v = market.getStock().getDividendYield();
    }else if(var=="volatility"){
        v = market.getStock().getVolatility();
    }else if(var=="riskFreeRate"){
        v = market.getRiskFreeRate();
    }else if(var=="strike"){
        v = option.getStrike();
    }else if(var=="maturity"){
        v = option.getMaturity();
    }
    return v;
}

double Pricer::setVariable(string var, double v){
    if(var=="currentPrice"){
        Stock tmpStock = market.getStock();
        tmpStock.setCurrentPrice(v);
        market.setStock(tmpStock);
    }else if(var=="dividendYield"){
        Stock tmpStock = market.getStock();
        tmpStock.setDividendYield(v);
        market.setStock(tmpStock);
    }else if(var=="volatility"){
        Stock tmpStock = market.getStock();
        tmpStock.setVolatility(v);
        market.setStock(tmpStock);
    }else if(var=="riskFreeRate"){
        market.setRiskFreeRate(v);
    }else if(var=="strike"){
        option.setStrike(v);
    }else if(var=="maturity"){
        option.setMaturity(v);
    }
    return v;
}

string Pricer::setStringVariable(string var, string v){
    if(var=="stockName"){
        Stock tmpStock = market.getStock();
        tmpStock.setName(v);
        market.setStock(tmpStock);
    }else if(var=="optionName"){
        option.setName(v);
    }
    return v;
}

Pricer Pricer::setVariablesFromFile(string file){
    string var, val;
    ifstream f(file);
    while(getline(f,var,',')){
        getline(f,val);
        double v;
        if(isDouble(val,v)) setVariable(var,v);
        else setStringVariable(var,val);
    }
    f.close();
    return *this;
}

Pricer Pricer::resetOriginal(){
    option = option_orig;
    market = market_orig;
    price = NAN;
    return *this;
}

Pricer Pricer::saveAsOriginal(){
    option_orig = option;
    market_orig = market;
    return *this;
}

double Pricer::BlackScholesClosedForm(){
    logMessage("starting calculation BlackScholesClosedForm");
    if(option.getType()=="European"){
        double K   = getVariable("strike");
        double T   = getVariable("maturity");
        double r   = getVariable("riskFreeRate");
        double S0  = getVariable("currentPrice");
        double q   = getVariable("dividendYield");
        double sig = getVariable("volatility");
        double d1  = (log(S0/K)+(r-q+sig*sig/2)*T)/(sig*sqrt(T));
        double d2  = d1-sig*sqrt(T);
        if(option.getPutCall()=="Call")
            price = S0*exp(-q*T)*normalCDF(d1)-K*exp(-r*T)*normalCDF(d2);
        else if(option.getPutCall()=="Put")
            price = K*exp(-r*T)*normalCDF(-d2)-S0*exp(-q*T)*normalCDF(-d1);
    }
    logMessage("ending calculation BlackScholesClosedForm, return "+to_string(price));
    return price;
}

double Pricer::BinomialTreePricer(const SimConfig& config){
    logMessage("starting calculation BinomialTreePricer on config "+to_string(config));
    Stock stock = market.getStock();
    double n = config.iters;
    double dt = config.stepSize;
    double sqrt_dt = sqrt(dt);
    double r = getVariable("riskFreeRate");
    double q = getVariable("dividendYield");
    double sig = getVariable("volatility");
    double u = exp(sig*sqrt_dt), d = 1/u;
    double qu = (exp((r-q)*dt)-d)/(u-d), qd = 1-qu;
    stock.setDriftRate(r);
    stock.generatePriceTree(config);
    matrix<double> optionBinomialTree(n,n);
    if(!option.isPathDependent()){
        matrix<double> payoffs = option.calcPayoffs(stock.getBinomialPriceTree().getLastRow());
        optionBinomialTree.setRow(n-1,payoffs);
        for(int i=n-2; i>=0; i--){
            for(int j=0; j<i+1; j++)
                optionBinomialTree.setEntry(i,j,max(
                    exp(-r*dt)*(qu*optionBinomialTree.getEntry(i+1,j+1)+qd*optionBinomialTree.getEntry(i+1,j)),
                    (option.canEarlyExercise())?option.calcPayoff(stock.getBinomialPriceTree().getEntry(i,j)):0.
                ));
        }
        // cout << stock.getBinomialPriceTree().print() << endl;
        // cout << optionBinomialTree.print() << endl;
        price = optionBinomialTree.getEntry(0,0);
    }
    logMessage("ending calculation BinomialTreePricer, return "+to_string(price));
    return price;
}

double Pricer::MonteCarloPricer(const SimConfig& config, int numSim){
    logMessage("starting calculation MonteCarloPricer on config "+to_string(config)+", numSim "+to_string(numSim));
    Stock stock = market.getStock();
    double r = getVariable("riskFreeRate");
    double T = getVariable("maturity");
    double err = NAN;
    stock.setDriftRate(r);
    stock.simulatePrice(config,numSim);
    if(!option.canEarlyExercise()){
        matrix<double> payoffs = option.calcPayoffs(NULL_VECTOR,stock.getSimPriceMatrix());
        price = exp(-r*T)*payoffs.mean();
        err = exp(-r*T)*payoffs.stdev()/sqrt(numSim);
    }
    logMessage("ending calculation MonteCarloPricer, return "+to_string(price)+" with errror "+to_string(err));
    return price;
}

double Pricer::NumIntegrationPricer(double z, double dz){
    logMessage("starting calculation NumIntegrationPricer on z "+to_string(z)+", dz "+to_string(dz));
    Stock stock = market.getStock();
    double r = getVariable("riskFreeRate");
    double T = getVariable("maturity");
    stock.setDriftRate(r);
    int n = static_cast<int>(z/dz);
    matrix<double> z0; z0.setRange(-z,z,2*n);
    matrix<double> S = stock.calcLognormalPriceVector(z0,T);
    if(!option.canEarlyExercise() && !option.isPathDependent()){
        matrix<double> payoffs = option.calcPayoffs(S);
        matrix<double> probs = z0.apply(stdNormalPDF)*dz;
        price = exp(-r*T)*(probs*payoffs).sum();
    }
    logMessage("ending calculation NumIntegrationPricer, return "+to_string(price));
    return price;
}

double Pricer::BlackScholesPDESolver(const SimConfig& config, int numSpace, string method){
    logMessage("starting calculation BlackScholesPDESolver on config "+to_string(config)+", numSpace "+to_string(numSpace));
    Stock stock = market.getStock();
    double K = getVariable("strike");
    double T = getVariable("maturity");
    double r = getVariable("riskFreeRate");
    double S0 = getVariable("currentPrice");
    double q = getVariable("dividendYield");
    double sig = getVariable("volatility");
    int n = config.iters;
    int m = numSpace;
    double dt = config.stepSize;
    double x0 = log(K/3), x1 = log(3*K);
    double dx = (x1-x0)/m, dx2 = dx*dx;
    double sig2 = sig*sig;
    matrix<double> priceMatrix(n+1,m+1);
    matrix<double> timeGrids; timeGrids.setRange(0,T,n,true);
    matrix<double> spaceGrids; spaceGrids.setRange(x0,x1,m,true);
    // cout << D.print() << endl;
    if(option.getType()=="European"){
        matrix<double> payoffs = option.calcPayoffs(spaceGrids.apply(exp));
        matrix<double> bdryCondition0(1,n+1), bdryCondition1(1,n+1), v, u(m-1,1);
        if(option.getPutCall()=="Call"){
            bdryCondition1 = exp(x1)-K*(-r*(T-timeGrids)).apply(exp);
        }else if(option.getPutCall()=="Put"){
            bdryCondition0 = K*(-r*(T-timeGrids)).apply(exp);
        }
        priceMatrix.setRow(n,payoffs);
        priceMatrix.setCol(0,bdryCondition0);
        priceMatrix.setCol(m,bdryCondition1);
        // cout << priceMatrix.print() << endl;
        v = priceMatrix.submatrix(n,n+1,1,m).transpose();
        if(method=="implicit"){
            double a = +(r-q-sig2/2)*dt/(2*dx)-sig2/2*dt/dx2;
            double b = 1+r*dt+sig2*dt/dx2;
            double c = -(r-q-sig2/2)*dt/(2*dx)-sig2/2*dt/dx2;
            matrix<double> D(m-1,m-1);
            D.setDiags(vector<double>{a,b,c},vector<int>{-1,0,1});
            D = D.inverse();
            for(int i=n-1; i>=0; i--){
                double u0 = a*priceMatrix.getEntry(i,0);
                double u1 = c*priceMatrix.getEntry(i,m);
                u.setEntry(0,0,u0);
                u.setEntry(m-2,0,u1);
                v = D.dot(v-u);
                priceMatrix.setSubmatrix(i,i+1,1,m,v.transpose());
            }
        }else if(method=="explicit"){
            double a = -(r-q-sig2/2)*dt/(2*dx)+sig2/2*dt/dx2;
            double b = 1-r*dt-sig2*dt/dx2;
            double c = +(r-q-sig2/2)*dt/(2*dx)+sig2/2*dt/dx2;
            matrix<double> D(m-1,m-1);
            D.setDiags(vector<double>{a,b,c},vector<int>{-1,0,1});
            for(int i=n-1; i>=0; i--){
                double u0 = a*priceMatrix.getEntry(i+1,0);
                double u1 = c*priceMatrix.getEntry(i+1,m);
                u.setEntry(0,0,u0);
                u.setEntry(m-2,0,u1);
                v = D.dot(v)+u;
                priceMatrix.setSubmatrix(i,i+1,1,m,v.transpose());
            }
        }
        // cout << priceMatrix.print() << endl;
        double x = log(S0);
        int idx = (x-spaceGrids).apply(abs).minIdx()[1];
        price = priceMatrix.getEntry(0,idx);
    }
    logMessage("ending calculation BlackScholesPDESolver, return "+to_string(price));
    return price;
}

double Pricer::calcPrice(string method, const SimConfig& config, int numSim, int numSpace){
    if(GUI) cout << "calculating option price with " << method << " pricer";
    if(method=="Closed Form"){
        if(GUI) cout << endl;
        price = BlackScholesClosedForm();
    }else if(method=="Binomial Tree"){
        if(GUI) cout << " on config " << config << endl;
        price = BinomialTreePricer(config);
    }else if(method=="Monte Carlo"){
        if(GUI) cout << " on config " << config << ", numSim " << numSim << endl;
        price = MonteCarloPricer(config,numSim);
    }else if(method=="Num Integration"){
        if(GUI) cout << endl;
        price = NumIntegrationPricer();
    }else if(method=="PDE Solver"){
        if(GUI) cout << " on config " << config << ", numSpace " << numSpace << endl;
        price = BlackScholesPDESolver(config,numSpace);
    }
    return price;
}

matrix<double> Pricer::varyPriceWithVariable(string var, matrix<double> varVector,
    string method, const SimConfig& config, int numSim){
    int n = varVector.getCols();
    matrix<double> optionPriceVector(1,n);
    for(int i=0; i<n; i++){
        double v = varVector.getEntry(0,i);
        setVariable(var,v);
        price = calcPrice(method,config,numSim);
        optionPriceVector.setEntry(0,i,price);
    }
    resetOriginal();
    return optionPriceVector;
}

double Pricer::ClosedFormGreek(string var, int derivOrder){
    logMessage("starting calculation ClosedFormGreek on var "+var+", derivOrder "+to_string(derivOrder));
    double greek = NAN;
    if(option.getType()=="European"){
        double K   = getVariable("strike");
        double T   = getVariable("maturity");
        double r   = getVariable("riskFreeRate");
        double S0  = getVariable("currentPrice");
        double q   = getVariable("dividendYield");
        double sig = getVariable("volatility");
        double sqrt_T = sqrt(T);
        double d1  = (log(S0/K)+(r-q+sig*sig/2)*T)/(sig*sqrt_T);
        double d2  = d1-sig*sqrt_T;
        if(option.getPutCall()=="Call"){
            if(var=="currentPrice" && derivOrder==1)
                greek = normalCDF(d1);
            else if(var=="currentPrice" && derivOrder==2)
                greek = normalPDF(d1)/(S0*sig*sqrt_T);
            else if(var=="volatility" && derivOrder==1)
                greek = S0*normalPDF(d1)*sqrt_T;
            else if(var=="riskFreeRate" && derivOrder==1)
                greek = K*T*exp(-r*T)*normalCDF(d2);
            else if(var=="time" && derivOrder==1)
                greek = -S0*normalPDF(d1)*sig/(2*sqrt_T)-r*K*exp(-r*T)*normalCDF(d2);
        }else if(option.getPutCall()=="Put"){
            if(var=="currentPrice" && derivOrder==1)
                greek = normalCDF(d1)-1;
            else if(var=="currentPrice" && derivOrder==2)
                greek = normalPDF(d1)/(S0*sig*sqrt_T);
            else if(var=="volatility" && derivOrder==1)
                greek = S0*normalPDF(d1)*sqrt_T;
            else if(var=="riskFreeRate" && derivOrder==1)
                greek = -K*T*exp(-r*T)*normalCDF(-d2);
            else if(var=="time" && derivOrder==1)
                greek = -S0*normalPDF(d1)*sig/(2*sqrt_T)+r*K*exp(-r*T)*normalCDF(-d2);
        }
    }
    logMessage("ending calculation ClosedFormGreek, return "+to_string(greek));
    return greek;
}

double Pricer::FiniteDifferenceGreek(string var, int derivOrder, string method,
    const SimConfig& config, int numSim, double eps){
    logMessage("starting calculation FiniteDifferenceGreek on var "+var+", derivOrder "+to_string(derivOrder)+
        ", method "+method+", config "+to_string(config)+", numSim "+to_string(numSim)+", eps "+to_string(eps));
    double greek = NAN;
    double v,dv,v_pos,v_neg,price_pos,price_neg;
    v = getVariable(var);
    dv = v*eps;
    v_pos = v+dv;
    v_neg = v-dv;
    price = calcPrice(method,config,numSim);
    setVariable(var,v_pos);
    price_pos = calcPrice(method,config,numSim);
    setVariable(var,v_neg);
    price_neg = calcPrice(method,config,numSim);
    switch(derivOrder){
        case 1: greek = (price_pos-price_neg)/(2*dv); break;
        case 2: greek = (price_pos-2*price+price_neg)/(dv*dv); break;
    }
    resetOriginal();
    logMessage("ending calculation FiniteDifferenceGreek, return "+to_string(greek));
    return greek;
}

double Pricer::calcGreek(string greekName, string greekMethod, string method,
    const SimConfig& config, int numSim, double eps){
    if(GUI) cout << "calculating option " << greekName << " with " << method << " calculator" << endl;
    double greek = NAN;
    string var; int derivOrder;
    if(greekName=="Delta"){
        var = "currentPrice";
        derivOrder = 1;
    }else if(greekName=="Gamma"){
        var = "currentPrice";
        derivOrder = 2;
    }else if(greekName=="Vega"){
        var = "volatility";
        derivOrder = 1;
    }else if(greekName=="Rho"){
        var = "riskFreeRate";
        derivOrder = 1;
    }else if(greekName=="Theta"){
        var = "time";
        derivOrder = 1;
    }
    if(greekMethod=="Closed Form")
        greek = ClosedFormGreek(var,derivOrder);
    else if(greekMethod=="Finite Difference")
        greek = FiniteDifferenceGreek(var,derivOrder,method,config,numSim,eps);
    return greek;
}

matrix<double> Pricer::varyGreekWithVariable(string var, matrix<double> varVector, string greekName,
    string greekMethod, string method, const SimConfig& config, int numSim, double eps){
    int n = varVector.getCols();
    double greek;
    matrix<double> optionGreekVector(1,n);
    for(int i=0; i<n; i++){
        double v = varVector.getEntry(0,i);
        setVariable(var,v);
        greek = calcGreek(greekName,greekMethod,method,config,numSim,eps);
        optionGreekVector.setEntry(0,i,greek);
    }
    resetOriginal();
    return optionGreekVector;
}

matrix<double> Pricer::generatePriceSurface(matrix<double> stockPriceVector, matrix<double> optionTermVector,
    string method, const SimConfig& config, int numSim){
    if(GUI) cout << "generating option price surface with " << method << " pricer" << endl;
    int m = optionTermVector.getCols();
    int n = stockPriceVector.getCols();
    matrix<double> priceSurface(m,n);
    for(int i=0; i<m; i++){
        double term = optionTermVector.getEntry(0,i);
        setVariable("maturity",term);
        priceSurface.setRow(i,
            varyPriceWithVariable("currentPrice",stockPriceVector,method,config,numSim)
        );
    }
    resetOriginal();
    return priceSurface;
}

bool Pricer::satisfyPriceBounds(double optionMarketPrice){
    if(option.getType()=="European"){
        double K   = getVariable("strike");
        double T   = getVariable("maturity");
        double r   = getVariable("riskFreeRate");
        double S0  = getVariable("currentPrice");
        double q   = getVariable("dividendYield");
        if(option.getPutCall()=="Call")
            return (optionMarketPrice<S0*exp(-q*T)) &&
                (optionMarketPrice>max(S0*exp(-q*T)-K*exp(-r*T),0.));
        else if(option.getPutCall()=="Put")
            return (optionMarketPrice<K*exp(-r*T)) &&
                (optionMarketPrice>max(K*exp(-r*T)-S0*exp(-q*T),0.));
    }
    return false;
}

double Pricer::calcImpliedVolatility(double optionMarketPrice, double vol0, double eps){
    if(GUI) cout << "calculating option implied vol on optionMarketPrice " << optionMarketPrice << endl;
    double impliedVol = NAN;
    double impliedVol0, impliedVol1;
    double err = 1;
    if(option.getType()=="European"){
        impliedVol0 = 0;
        impliedVol1 = vol0;
        if(satisfyPriceBounds(optionMarketPrice)){
            while(err>eps){
                impliedVol = (impliedVol0+impliedVol1)/2;
                setVariable("volatility",impliedVol);
                price = calcPrice("Closed Form");
                if(price>optionMarketPrice) impliedVol1 = impliedVol;
                else if(price<optionMarketPrice) impliedVol0 = impliedVol;
                err = abs(price-optionMarketPrice)/optionMarketPrice;
                if(impliedVol==vol0) break;
            }
        }
    }
    resetOriginal();
    return impliedVol;
}

void Pricer::generateImpliedVolSurfaceFromFile(string input, string file, double vol0, double eps){
    string name, type, putCall;
    string strikeStr, maturityStr, optionMarketPriceStr;
    double strike, maturity, optionMarketPrice;
    double impliedVol;
    ifstream fi(input);
    ofstream fo; fo.open(file);
    while(getline(fi,name,',')){
        getline(fi,type,',');
        getline(fi,putCall,',');
        getline(fi,strikeStr,',');
        getline(fi,maturityStr,',');
        getline(fi,optionMarketPriceStr);
        strike = stod(strikeStr);
        maturity = stod(maturityStr);
        optionMarketPrice = stod(optionMarketPriceStr);
        option = Option(type,putCall,strike,maturity,name);
        // cout << option << endl;
        impliedVol = calcImpliedVolatility(optionMarketPrice,vol0,eps);
        if(!isnan(impliedVol)) fo << name << "," << type << "," << putCall << "," <<
            strike << "," << maturity << "," << impliedVol << endl;
    }
    fi.close();
    fo.close();
    resetOriginal();
}

void Pricer::generateGreeksFromImpliedVolFile(string input, string file){
    string name, type, putCall;
    string strikeStr, maturityStr, impliedVolStr;
    double strike, maturity, impliedVol;
    double Delta, Gamma, Vega, Rho, Theta;
    ifstream fi(input);
    ofstream fo; fo.open(file);
    while(getline(fi,name,',')){
        getline(fi,type,',');
        getline(fi,putCall,',');
        getline(fi,strikeStr,',');
        getline(fi,maturityStr,',');
        getline(fi,impliedVolStr);
        strike = stod(strikeStr);
        maturity = stod(maturityStr);
        impliedVol = stod(impliedVolStr);
        setVariable("volatility",impliedVol);
        option = Option(type,putCall,strike,maturity,name);
        // cout << option << endl;
        Delta = calcGreek("Delta");
        Gamma = calcGreek("Gamma");
        Vega = calcGreek("Vega");
        Rho = calcGreek("Rho");
        Theta = calcGreek("Theta");
        fo << name << "," << type << "," << putCall << "," <<
            strike << "," << maturity << "," << impliedVol << "," << Delta << "," <<
            Gamma << "," << Vega << "," << Rho << "," << Theta << endl;
    }
    fi.close();
    fo.close();
    resetOriginal();
}

//### operators ################################################################

ostream& operator<<(ostream& out, const SimConfig& config){
    out << config.getAsJson();
    return out;
}

ostream& operator<<(ostream& out, const Option& option){
    out << option.getAsJson();
    return out;
}

ostream& operator<<(ostream& out, const Stock& stock){
    out << stock.getAsJson();
    return out;
}

ostream& operator<<(ostream& out, const Market& market){
    out << market.getAsJson();
    return out;
}

ostream& operator<<(ostream& out, const Pricer& pricer){
    out << pricer.getAsJson();
    return out;
}

#endif
