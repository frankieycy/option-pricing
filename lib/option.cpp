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
    "European", "Digital", "American", "Bermudan", "Asian", "Barrier", "Lookback",
    "Margrabe", "Basket", "Rainbow", "Chooser", "Shout"
};
const set<string> EARLY_EX_OPTIONS{
    "American", "Bermudan"
};
const set<string> PATH_DEP_OPTIONS{
    "Asian", "Barrier", "Lookback", "Chooser", "Shout"
};
const set<string> PUT_CALL{
    "Put", "Call", ""
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

class Option{
private:
    string name, type, putCall, nature;
    double strike, discStrike, maturity;
    vector<double> params;
public:
    /**** constructors ****/
    Option(){};
    Option(string type, string putCall, double strike, double maturity,
        vector<double> params={}, string nature="", string name="unnamed");
    Option(const Option& option);
    /**** accessors ****/
    bool canEarlyExercise() const;
    bool isPathDependent() const;
    string getName() const {return name;}
    string getNature() const {return nature;}
    string getType() const {return type;}
    string getPutCall() const {return putCall;}
    double getStrike() const {return strike;}
    double getDiscStrike() const {return discStrike;}
    double getMaturity() const {return maturity;}
    vector<double> getParams() const {return params;}
    string getAsJson() const;
    /**** mutators ****/
    string setName(string name);
    string setNature(string nature);
    string setType(string type);
    double setStrike(double strike);
    double setDiscStrike(double discStrike);
    double setMaturity(double maturity);
    vector<double> setParams(vector<double> params);
    /**** main ****/
    bool checkParams() const;
    double calcPayoff(double stockPrice=0, matrix priceSeries=NULL_VECTOR,
        vector<matrix> priceSeriesSet={}, matrix timeVector=NULL_VECTOR);
    matrix calcPayoffs(matrix stockPriceVector=NULL_VECTOR, matrix priceMatrix=NULL_MATRIX,
        vector<matrix> priceMatrixSet={}, matrix timeVector=NULL_VECTOR);
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Option& option);
};

class Stock{
private:
    string name, dynamics;
    double currentPrice, dividendYield, driftRate, volatility;
    matrix simTimeVector, simPriceMatrix, binomialPriceTree;
    vector<double> dynParams;
public:
    /**** constructors ****/
    Stock(){};
    Stock(double currentPrice, double dividendYield, double driftRate, double volatility,
        vector<double> dynParams={}, string dynamics="lognormal", string name="unnamed");
    Stock(const Stock& stock);
    /**** accessors ****/
    string getName() const {return name;}
    string getDynamics() const {return dynamics;}
    double getCurrentPrice() const {return currentPrice;}
    double getDividendYield() const {return dividendYield;}
    double getDriftRate() const {return driftRate;}
    double getVolatility() const {return volatility;}
    matrix getSimTimeVector() const {return simTimeVector;}
    matrix getSimPriceMatrix() const {return simPriceMatrix;}
    matrix getBinomialPriceTree() const {return binomialPriceTree;}
    vector<double> getDynParams() const {return dynParams;}
    string getAsJson() const;
    /**** mutators ****/
    string setName(string name);
    string setDynamics(string dynamics);
    double setCurrentPrice(double currentPrice);
    double setDividendYield(double dividendYield);
    double setDriftRate(double driftRate);
    double setVolatility(double volatility);
    vector<double> setDynParams(vector<double> dynParams);
    matrix setSimTimeVector(matrix simTimeVector);
    matrix setSimPriceMatrix(matrix simPriceMatrix);
    double estDriftRateFromPrice(matrix priceSeries, double dt, string method="simple");
    double estVolatilityFromPrice(matrix priceSeries, double dt, string method="simple");
    /**** main ****/
    bool checkParams() const;
    double calcLognormalPrice(double z, double time);
    matrix calcLognormalPriceVector(matrix z, double time);
    matrix simulatePrice(const SimConfig& config, int numSim=1, matrix randomMatrix=NULL_MATRIX);
    vector<matrix> simulatePriceWithFullCalc(const SimConfig& config, int numSim=1, matrix randomMatrix=NULL_MATRIX);
    matrix bootstrapPrice(matrix priceSeries, const SimConfig& config, int numSim=1);
    matrix generatePriceTree(const SimConfig& config);
    matrix generatePriceMatrixFromTree();
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Stock& stock);
};

const Stock NULL_STOCK;

class Market{
private:
    double riskFreeRate;
    Stock stock;
    vector<Stock> stocks;
    matrix corMatrix;
public:
    /**** constructors ****/
    Market(){};
    Market(double riskFreeRate, const Stock& stock, vector<Stock> stocks={}, matrix corMatrix=NULL_MATRIX);
    Market(const Market& market);
    /**** accessors ****/
    double getRiskFreeRate() const {return riskFreeRate;}
    Stock getStock(int i=-1) const {return i<0?stock:stocks[i];}
    vector<Stock> getStocks() const {return stocks;}
    matrix getCorMatrix() const {return corMatrix;}
    string getAsJson() const;
    /**** mutators ****/
    double setRiskFreeRate(double riskFreeRate);
    Stock setStock(const Stock& stock, int i=-1);
    vector<Stock> setStocks(vector<Stock> stocks);
    matrix setCorMatrix(matrix corMatrix);
    /**** main ****/
    vector<matrix> simulateCorrelatedPrices(const SimConfig& config, int numSim=1, vector<matrix> randomMatrixSet={});
    vector<vector<matrix>> simulateCorrelatedPricesWithFullCalc(const SimConfig& config, int numSim=1, vector<matrix> randomMatrixSet={});
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Market& market);
};

class Backtest{
public:
    vector<string> labels;
    vector<matrix> results;
    vector<string> hLabels;
    vector<vector<matrix>> hResults;
    Backtest(
        vector<matrix> results,
        vector<vector<matrix>> hResults
    );
    void printToCsvFiles(
        bool perSim=false,
        string name="backtest"
    );
};

class Pricer{
private:
    Option option, option_orig;
    Market market, market_orig;
    double price;
public:
    vector<double> tmp; // tmp variable log
    /**** constructors ****/
    Pricer(){};
    Pricer(const Option& option, const Market& market);
    /**** accessors ****/
    Option getOption() const {return option;}
    Market getMarket() const {return market;}
    double getPrice() const {return price;}
    string getAsJson() const;
    double getVariable(string var, int i=-1, int j=-1) const;
    /**** mutators ****/
    double setVariable(string var, double v, int i=-1);
    string setStringVariable(string var, string v);
    Pricer setVariablesFromFile(string file);
    Pricer resetOriginal();
    Pricer saveAsOriginal();
    /**** main ****/
    double BlackScholesClosedForm();
    double BinomialTreePricer(const SimConfig& config);
    double MonteCarloPricer(const SimConfig& config, int numSim, string method="simple");
    double MultiStockMonteCarloPricer(const SimConfig& config, int numSim, string method="simple");
    double NumIntegrationPricer(double z=5, double dz=1e-3);
    double BlackScholesPDESolver(const SimConfig& config, int numSpace, string method="implicit");
    double calcPrice(string method="Closed Form", const SimConfig& config=NULL_CONFIG,
        int numSim=0, int numSpace=0);
    matrix varyPriceWithVariable(string var, matrix varVector,
        string method="Closed Form", const SimConfig& config=NULL_CONFIG, int numSim=0);
    double ClosedFormGreek(string var, int derivOrder=1);
    double FiniteDifferenceGreek(string var, int derivOrder=1, string method="Closed Form",
        const SimConfig& config=NULL_CONFIG, int numSim=0, double eps=1e-5);
    double calcGreek(string greekName, string greekMethod="Closed Form", string method="Closed Form",
        const SimConfig& config=NULL_CONFIG, int numSim=0, double eps=1e-5);
    matrix varyGreekWithVariable(string var, matrix varVector,
        string greekName, string greekMethod="Closed Form", string method="Closed Form",
        const SimConfig& config=NULL_CONFIG, int numSim=0, double eps=1e-5);
    matrix generatePriceSurface(matrix stockPriceVector, matrix optionTermVector,
        string method="Closed Form", const SimConfig& config=NULL_CONFIG, int numSim=0);
    bool satisfyPriceBounds(double optionMarketPrice);
    double calcImpliedVolatility(double optionMarketPrice, double vol0=5, double eps=1e-5);
    void generateImpliedVolSurfaceFromFile(string input, string file, double vol0=5, double eps=1e-5);
    void generateGreeksFromImpliedVolFile(string input, string file);
    vector<matrix> modelImpliedVolSurface(const SimConfig& config, int numSpace,
        const function<double(double)>& impVolFunc0, const function<double(double)>& impVolFunc1,
        double lambdaT, double eps=1e-5);
    vector<matrix> modelImpliedVolSurfaceFromFile(string input, const SimConfig& config, int numSpace); // TO DO
    Backtest runBacktest(const SimConfig& config, int numSim=1,
        string strategy="simple-delta", int hedgeFreq=1, double mktImpVol=0, double mktPrice=0,
        vector<double> stratParams={}, vector<Option> hOptions={}, vector<matrix> impVolSurfaceSet={},
        string simPriceMethod="model", matrix stockPriceSeries=NULL_VECTOR);
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const Pricer& pricer);
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

Option::Option(string type, string putCall, double strike, double maturity, vector<double> params, string nature, string name){
    this->type = type;
    this->putCall = putCall;
    this->strike = strike;
    this->maturity = maturity;
    this->params = params;
    this->nature = nature;
    this->name = name;
    assert(checkParams());
}

Option::Option(const Option& option){
    this->type = option.type;
    this->putCall = option.putCall;
    this->strike = option.strike;
    this->maturity = option.maturity;
    this->params = option.params;
    this->nature = option.nature;
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
    "\"nature\":\""    << nature   << "\"," <<
    "\"putCall\":\""   << putCall  << "\"," <<
    "\"strike\":"      << strike   << ","   <<
    "\"maturity\":"    << maturity << ","   <<
    "\"params\":"      << params   <<
    "}";
    return oss.str();
}

string Option::setName(string name){
    this->name = name;
    return name;
}

string Option::setNature(string nature){
    this->nature = nature;
    return nature;
}

string Option::setType(string type){
    this->type = type;
    return type;
}

double Option::setStrike(double strike){
    this->strike = strike;
    return strike;
}

double Option::setDiscStrike(double discStrike){
    this->discStrike = discStrike;
    return discStrike;
}

double Option::setMaturity(double maturity){
    this->maturity = maturity;
    return maturity;
}

vector<double> Option::setParams(vector<double> params){
    this->params = params;
    return params;
}

bool Option::checkParams() const {
    return
    OPTION_TYPES.find(type)!=OPTION_TYPES.end() &&
    PUT_CALL.find(putCall)!=PUT_CALL.end() &&
    strike>=0 && maturity>=0;
}

double Option::calcPayoff(double stockPrice, matrix priceSeries, vector<matrix> priceSeriesSet, matrix timeVector){
    // case by case
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
        else S = priceSeries.getRow(0).mean(nature);
        if(putCall=="Put") return max(strike-S,0.);
        else if(putCall=="Call") return max(S-strike,0.);
    }else if(type=="Barrier"){
        if(priceSeries.isEmpty()) return NAN;
        else S = priceSeries.getLastEntry();
        double barrier = params[0];
        double rebate = params[1];
        bool triggered =
            (nature=="Up-and-In" && priceSeries.getMax()>barrier) ||
            (nature=="Up-and-Out" && priceSeries.getMax()<barrier) ||
            (nature=="Down-and-In" && priceSeries.getMin()<barrier) ||
            (nature=="Down-and-Out" && priceSeries.getMin()>barrier);
        if(triggered){
            if(putCall=="Put") return max(strike-S,0.);
            else if(putCall=="Call") return max(S-strike,0.);
        }else return rebate;
    }else if(type=="Lookback"){
        if(priceSeries.isEmpty()) return NAN;
        else S = priceSeries.getLastEntry();
        if(putCall=="Put"){
            double Smax = priceSeries.getMax();
            return max(Smax-S,0.);
        }else if(putCall=="Call"){
            double Smin = priceSeries.getMin();
            return max(S-Smin,0.);
        }
    }else if(type=="Margrabe"){
        if(priceSeriesSet.empty()) return NAN;
        double S0 = priceSeriesSet[0].getLastEntry();
        double S1 = priceSeriesSet[1].getLastEntry();
        if(putCall=="Put") return max(S1-S0,0.);
        else if(putCall=="Call") return max(S0-S1,0.);
    }else if(type=="Basket"){
        if(priceSeriesSet.empty()) return NAN;
        int n = priceSeriesSet.size();
        matrix Sset(1,n);
        for(int i=0; i<n; i++) Sset.setEntry(0,i,priceSeriesSet[i].getLastEntry());
        double S;
        if(params.size()) S = Sset.wmean(params); // weighted average
        else S = Sset.mean(); // simple average
        if(putCall=="Put") return max(strike-S,0.);
        else if(putCall=="Call") return max(S-strike,0.);
    }else if(type=="Rainbow"){
        if(priceSeriesSet.empty()) return NAN;
        int n = priceSeriesSet.size();
        matrix Sset(1,n);
        for(int i=0; i<n; i++) Sset.setEntry(0,i,priceSeriesSet[i].getLastEntry());
        double S;
        if(nature=="Best"){
            S = Sset.getMax();
            return max(S,strike); // Best of assets or cash
        }else if(nature=="Max") S = Sset.getMax(); // Put/Call on max
        else if(nature=="Min") S = Sset.getMin(); // Put/Call on min
        if(putCall=="Put") return max(strike-S,0.);
        else if(putCall=="Call") return max(S-strike,0.);
    }else if(type=="Chooser"){
        if(priceSeries.isEmpty()) return NAN;
        else S = priceSeries.getLastEntry();
        double chTime = params[0];
        vector<int> chTimeIdx = (timeVector-chTime).apply(abs).minIdx();
        string chPutCall;
        if(priceSeries.getEntry(chTimeIdx)<discStrike) chPutCall = "Put";
        else chPutCall = "Call";
        if(chPutCall=="Put") return max(strike-S,0.);
        else if(chPutCall=="Call") return max(S-strike,0.);
    }
    return NAN;
}

matrix Option::calcPayoffs(matrix stockPriceVector, matrix priceMatrix, vector<matrix> priceMatrixSet, matrix timeVector){
    matrix S;
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
        else S = priceMatrix.mean(2,nature);
        if(putCall=="Put") return (strike-S).maxWith(0.);
        else if(putCall=="Call") return (S-strike).maxWith(0.);
    }else if(type=="Barrier" || type=="Lookback" || type=="Chooser"){ // generic single-stock
        if(priceMatrix.isEmpty()) return NULL_VECTOR;
        int n = priceMatrix.getCols();
        matrix V(1,n);
        for(int i=0; i<n; i++) V.setEntry(0,i,calcPayoff(0,priceMatrix.getCol(i),{},timeVector));
        return V;
    }else if(type=="Margrabe" || type=="Basket" || type=="Rainbow"){ // generic multi-stock
        if(priceMatrixSet.empty()) return NULL_VECTOR;
        int n = priceMatrixSet[0].getCols();
        int m = priceMatrixSet.size();
        matrix V(1,n);
        for(int i=0; i<n; i++){
            vector<matrix> priceSeriesSet;
            for(int j=0; j<m; j++) priceSeriesSet.push_back(priceMatrixSet[j].getCol(i));
            V.setEntry(0,i,calcPayoff(0,NULL_VECTOR,priceSeriesSet));
        }
        return V;
    }
    return NULL_VECTOR;
}

//### Stock class ##############################################################

Stock::Stock(double currentPrice, double dividendYield, double driftRate, double volatility,
    vector<double> dynParams, string dynamics, string name){
    this->currentPrice = currentPrice;
    this->dividendYield = dividendYield;
    this->driftRate = driftRate;
    this->volatility = volatility;
    this->dynParams = dynParams;
    this->dynamics = dynamics;
    this->name = name;
    assert(checkParams());
}

Stock::Stock(const Stock& stock){
    this->currentPrice = stock.currentPrice;
    this->dividendYield = stock.dividendYield;
    this->driftRate = stock.driftRate;
    this->volatility = stock.volatility;
    this->dynParams = stock.dynParams;
    this->dynamics = stock.dynamics;
    this->name = stock.name;
}

string Stock::getAsJson() const {
    ostringstream oss;
    oss << "{" <<
    "\"name\":\""         << name             << "\"," <<
    "\"dynamics\":\""     << dynamics         << "\"," <<
    "\"currentPrice\":"   << currentPrice     << "," <<
    "\"dividendYield\":"  << dividendYield    << "," <<
    "\"driftRate\":"      << driftRate        << "," <<
    "\"volatility\":"     << volatility       << "," <<
    "\"dynParams\":"      << dynParams        <<
    "}";
    return oss.str();
}

string Stock::setName(string name){
    this->name = name;
    return name;
}

string Stock::setDynamics(string dynamics){
    this->dynamics = dynamics;
    return dynamics;
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

vector<double> Stock::setDynParams(vector<double> dynParams){
    this->dynParams = dynParams;
    return dynParams;
}

matrix Stock::setSimTimeVector(matrix simTimeVector){
    this->simTimeVector = simTimeVector;
    return simTimeVector;
}

matrix Stock::setSimPriceMatrix(matrix simPriceMatrix){
    this->simPriceMatrix = simPriceMatrix;
    return simPriceMatrix;
}

double Stock::estDriftRateFromPrice(matrix priceSeries, double dt, string method){
    if(method=="simple"){
        matrix returnSeries;
        returnSeries =
            (priceSeries.submatrix(1,-1,"col")-priceSeries.submatrix(0,-2,"col"))
            /priceSeries.submatrix(0,-2,"col");
        driftRate = returnSeries.mean()/dt;
    }
    return driftRate;
}

double Stock::estVolatilityFromPrice(matrix priceSeries, double dt, string method){
    if(method=="simple"){
        matrix returnSeries;
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

matrix Stock::calcLognormalPriceVector(matrix z, double time){
    int n = z.getCols();
    matrix S(1,n);
    for(int i=0; i<n; i++) S.setEntry(0,i,calcLognormalPrice(z.getEntry(0,i),time));
    return S;
}

matrix Stock::simulatePrice(const SimConfig& config, int numSim, matrix randomMatrix){
    vector<matrix> fullCalc = simulatePriceWithFullCalc(config,numSim,randomMatrix);
    return fullCalc[0]; // simPriceMatrix
}

vector<matrix> Stock::simulatePriceWithFullCalc(const SimConfig& config, int numSim, matrix randomMatrix){
    int n = config.iters;
    double dt = config.stepSize;
    double sqrt_dt = sqrt(dt);
    matrix randomVector(1,numSim);
    matrix simPriceVector(1,numSim,currentPrice);
    simTimeVector = matrix(1,n+1);
    simPriceMatrix = matrix(n+1,numSim);
    simPriceMatrix.setRow(0,simPriceVector);
    simTimeVector.setEntry(0,0,0);
    if(dynamics=="lognormal"){
        for(int i=1; i<n+1; i++){
            if(randomMatrix.isEmpty()) randomVector.setNormalRand();
            else randomVector = randomMatrix.getRow(i);
            simPriceVector += simPriceVector*(driftRate*dt+volatility*sqrt_dt*randomVector);
            simPriceMatrix.setRow(i,simPriceVector);
            simTimeVector.setEntry(0,i,i*dt);
        }
    }else if(dynamics=="jump-diffusion"){
        // TO DO
    }else if(dynamics=="Heston"){
        double sig0             = volatility;
        double reversionRate    = dynParams[0];
        double longRunVar       = dynParams[1];
        double volOfVol         = dynParams[2];
        double brownianCor0     = dynParams[3];
        double brownianCor1     = sqrt(1-brownianCor0*brownianCor0);
        matrix volRandomVector(1,numSim);
        matrix currentVol(1,numSim,sig0), currentVar(1,numSim,sig0*sig0);
        matrix simVolMatrix(n+1,numSim), simVarMatrix(n+1,numSim);
        simVolMatrix.setRow(0,currentVol);
        simVarMatrix.setRow(0,currentVar);
        assert(2*reversionRate*longRunVar>volOfVol*volOfVol); // Feller condition
        for(int i=1; i<n+1; i++){
            if(randomMatrix.isEmpty()) randomVector.setNormalRand();
            else randomVector = randomMatrix.getRow(i);
            volRandomVector.setNormalRand();
            volRandomVector = brownianCor0*randomVector+brownianCor1*volRandomVector;
            // currentVar += reversionRate*(longRunVar-currentVar.maxWith(0.))*dt+volOfVol*currentVar.maxWith(0.).apply(sqrt)*sqrt_dt*volRandomVector;
            currentVar += reversionRate*(longRunVar-currentVar)*dt+volOfVol*currentVar.apply(sqrt)*sqrt_dt*volRandomVector;
            currentVar  = currentVar.apply(abs);
            currentVol  = currentVar.apply(sqrt);
            simPriceVector += simPriceVector*(driftRate*dt+currentVol*sqrt_dt*randomVector);
            simPriceMatrix.setRow(i,simPriceVector);
            simVolMatrix.setRow(i,currentVol);
            simVarMatrix.setRow(i,currentVar);
            simTimeVector.setEntry(0,i,i*dt);
        }
        return {simPriceMatrix,simVolMatrix,simVarMatrix};
    }
    return {simPriceMatrix};
}

matrix Stock::bootstrapPrice(matrix priceSeries, const SimConfig& config, int numSim){
    int n = config.iters;
    double dt = config.stepSize;
    matrix simPriceVector(1,numSim,currentPrice);
    matrix returnSeries, bootReturnSeries;
    simTimeVector = matrix(1,n+1);
    simPriceMatrix = matrix(n+1,numSim);
    simPriceMatrix.setRow(0,simPriceVector);
    simTimeVector.setEntry(0,0,0);
    returnSeries =
        (priceSeries.submatrix(1,-1,"col")-priceSeries.submatrix(0,-2,"col"))
        /priceSeries.submatrix(0,-2,"col");
    for(int i=1; i<n+1; i++){
        bootReturnSeries = returnSeries.sample(numSim,true);
        simPriceVector += simPriceVector*bootReturnSeries;
        simPriceMatrix.setRow(i,simPriceVector);
        simTimeVector.setEntry(0,i,i*dt);
    }
    return simPriceMatrix;
}

matrix Stock::generatePriceTree(const SimConfig& config){
    int n = config.iters;
    double dt = config.stepSize;
    double sqrt_dt = sqrt(dt);
    double u = exp(volatility*sqrt_dt), d = 1/u;
    simTimeVector = matrix(1,n);
    binomialPriceTree = matrix(n,n);
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

Market::Market(double riskFreeRate, const Stock& stock, vector<Stock> stocks, matrix corMatrix){
    this->riskFreeRate = riskFreeRate;
    this->stock = stock;
    this->stocks = stocks;
    this->corMatrix = corMatrix;
}

Market::Market(const Market& market){
    this->riskFreeRate = market.riskFreeRate;
    this->stock = market.stock;
    this->stocks = market.stocks;
    this->corMatrix = market.corMatrix;
}

string Market::getAsJson() const {
    ostringstream oss;
    oss << "{" <<
    "\"riskFreeRate\":"   << riskFreeRate << "," <<
    "\"stock\":"          << stock        << "," <<
    "\"stocks\":"         << stocks       << "," <<
    "\"corMatrix\":"      << corMatrix    <<
    "}";
    return oss.str();
}

double Market::setRiskFreeRate(double riskFreeRate){
    this->riskFreeRate = riskFreeRate;
    return riskFreeRate;
}

Stock Market::setStock(const Stock& stock, int i){
    if(i<0) this->stock = stock;
    else this->stocks[i] = stock;
    return stock;
}

vector<Stock> Market::setStocks(vector<Stock> stocks){
    this->stocks = stocks;
    return stocks;
}

matrix Market::setCorMatrix(matrix corMatrix){
    this->corMatrix = corMatrix;
    return corMatrix;
}

vector<matrix> Market::simulateCorrelatedPrices(const SimConfig& config, int numSim, vector<matrix> randomMatrixSet){
    vector<vector<matrix>> fullCalc = simulateCorrelatedPricesWithFullCalc(config,numSim,randomMatrixSet);
    return fullCalc[0]; // simPriceMatrixSet
}

vector<vector<matrix>> Market::simulateCorrelatedPricesWithFullCalc(const SimConfig& config, int numSim, vector<matrix> randomMatrixSet){
    int n = config.iters;
    int m = stocks.size();
    double dt = config.stepSize;
    double sqrt_dt = sqrt(dt);
    matrix simTimeVector(1,n+1);
    vector<matrix> randomVectorSet;
    vector<matrix> simPriceVectorSet;
    vector<matrix> simPriceMatrixSet;
    simTimeVector.setEntry(0,0,0);
    for(auto stock:stocks){
        double S = stock.getCurrentPrice();
        matrix randomVector(1,numSim);
        matrix simPriceVector(1,numSim,S);
        matrix simPriceMatrix(n+1,numSim);
        simPriceMatrix.setRow(0,simPriceVector);
        randomVectorSet.push_back(randomVector);
        simPriceVectorSet.push_back(simPriceVector);
        simPriceMatrixSet.push_back(simPriceMatrix);
    }
    matrix corFactor = corMatrix.chol(); // Choleskey decomposition
    string dynamics = stocks[0].getDynamics();
    if(dynamics=="lognormal"){
        for(int i=1; i<n+1; i++){
            matrix iidRandomMatrix(m,numSim);
            if(randomMatrixSet.empty()) iidRandomMatrix.setNormalRand();
            else for(int j=0; j<m; j++) iidRandomMatrix.setRow(j,randomMatrixSet[j].getRow(i));
            matrix corRandomMatrix = corFactor.dot(iidRandomMatrix);
            for(int j=0; j<m; j++){
                double driftRate = stocks[j].getDriftRate();
                double volatility = stocks[j].getVolatility();
                randomVectorSet[j] = corRandomMatrix.getRow(j);
                simPriceVectorSet[j] += simPriceVectorSet[j]*(driftRate*dt+volatility*sqrt_dt*randomVectorSet[j]);
                simPriceMatrixSet[j].setRow(i,simPriceVectorSet[j]);
            }
            simTimeVector.setEntry(0,i,i*dt);
        }
        for(int j=0; j<m; j++){
            stocks[j].setSimTimeVector(simTimeVector);
            stocks[j].setSimPriceMatrix(simPriceMatrixSet[j]);
        }
    }else if(dynamics=="jump-diffusion"){
        // TO DO
    }else if(dynamics=="Heston"){
        // TO DO
    }
    return {simPriceMatrixSet};
}

//### Backtest class ###########################################################

Backtest::Backtest(vector<matrix> results, vector<vector<matrix>> hResults):
    results(results),hResults(hResults){
    labels = {
        "simPrice",
        "stratCash",
        "stratNStock",
        "stratModPrice",
        "stratModValue",
        "stratGrkDelta",
        "stratGrkGamma",
        "stratGrkVega",
        "stratGrkRho",
        "stratGrkTheta"
    };
    hLabels = {
        "stratNOption",
        "stratHModPrice"
    };
}

void Backtest::printToCsvFiles(bool perSim, string name){
    int a = results.size();
    int b = hResults.size();
    int n = hResults[0].size();
    if(perSim){
        int iters = results[0].getRows();
        int numSim = results[0].getCols();
        string header = joinStr(labels);
        for(int i=0; i<b; i++)
            for(int j=0; j<n; j++)
                header += ","+hLabels[i]+"-"+to_string(j);
        for(int k=0; k<numSim; k++){
            matrix result(iters,a+b*n);
            for(int i=0; i<a; i++)
                result.setCol(i,results[i].getCol(k));
            for(int i=0; i<b; i++)
                for(int j=0; j<n; j++)
                    result.setCol(a+i*n+j,hResults[i][j].getCol(k));
            result.printToCsvFile(
                name+"-"+to_string(k)+".csv",
                header
            );
        }
    }else{
        for(int i=0; i<a; i++)
            results[i].printToCsvFile(
                name+"-"+labels[i]+".csv"
            );
        for(int i=0; i<b; i++)
            for(int j=0; j<n; j++)
                hResults[i][j].printToCsvFile(
                    name+"-"+hLabels[i]+"-"+to_string(j)+".csv"
                );
    }
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
    "\"market\":"     << market   << "," <<
    "\"price\":"      << price    <<
    "}";
    return oss.str();
}

double Pricer::getVariable(string var, int i, int j) const {
    double v = NAN;
    if(var=="currentPrice"){
        Stock stock = market.getStock(i);
        v = stock.getCurrentPrice();
    }else if(var=="driftRate"){
        Stock stock = market.getStock(i);
        v = stock.getDriftRate();
    }else if(var=="dividendYield"){
        Stock stock = market.getStock(i);
        v = stock.getDividendYield();
    }else if(var=="volatility"){
        Stock stock = market.getStock(i);
        v = stock.getVolatility();
    }else if(var=="correlation"){
        v = market.getCorMatrix().getEntry(i,j);
    }else if(var=="riskFreeRate"){
        v = market.getRiskFreeRate();
    }else if(var=="strike"){
        v = option.getStrike();
    }else if(var=="maturity"){
        v = option.getMaturity();
    }
    return v;
}

double Pricer::setVariable(string var, double v, int i){
    if(var=="currentPrice"){
        Stock tmpStock = market.getStock(i);
        tmpStock.setCurrentPrice(v);
        market.setStock(tmpStock,i);
    }else if(var=="dividendYield"){
        Stock tmpStock = market.getStock(i);
        tmpStock.setDividendYield(v);
        market.setStock(tmpStock,i);
    }else if(var=="volatility"){
        Stock tmpStock = market.getStock(i);
        tmpStock.setVolatility(v);
        market.setStock(tmpStock,i);
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
    }else if(var=="optionType"){
        option.setType(v);
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
    }else if(option.getType()=="Margrabe"){
        double T   = getVariable("maturity");
        double r   = getVariable("riskFreeRate");
        double S0  = getVariable("currentPrice",0);
        double S1  = getVariable("currentPrice",1);
        double q0  = getVariable("dividendYield",0);
        double q1  = getVariable("dividendYield",1);
        double sig0 = getVariable("volatility",0);
        double sig1 = getVariable("volatility",1);
        double rho = getVariable("correlation",0,1);
        double sig = sqrt(sig0*sig0+sig1*sig1-2*rho*sig0*sig1);
        double d0  = (log(S0/S1)+(q1-q0+sig*sig/2)*T)/(sig*sqrt(T));
        double d1  = d0-sig*sqrt(T);
        if(option.getPutCall()=="Call")
            price = S0*exp(-q0*T)*normalCDF(d0)-S1*exp(-q1*T)*normalCDF(d1);
        else if(option.getPutCall()=="Put")
            price = S1*exp(-q1*T)*normalCDF(d1)-S0*exp(-q0*T)*normalCDF(d0);
    }
    logMessage("ending calculation BlackScholesClosedForm, return "+to_string(price));
    return price;
}

double Pricer::BinomialTreePricer(const SimConfig& config){
    logMessage("starting calculation BinomialTreePricer on config "+to_string(config));
    Stock stock = market.getStock();
    int n = config.iters;
    double dt = config.stepSize;
    double sqrt_dt = sqrt(dt);
    double r = getVariable("riskFreeRate");
    double q = getVariable("dividendYield");
    double sig = getVariable("volatility");
    double u = exp(sig*sqrt_dt), d = 1/u;
    double qu = (exp((r-q)*dt)-d)/(u-d), qd = 1-qu;
    stock.setDriftRate(r-q);
    stock.generatePriceTree(config);
    matrix optionBinomialTree(n,n);
    if(!option.isPathDependent()){
        matrix payoffs = option.calcPayoffs(stock.getBinomialPriceTree().getLastRow());
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

double Pricer::MonteCarloPricer(const SimConfig& config, int numSim, string method){
    logMessage("starting calculation MonteCarloPricer on config "+to_string(config)+", numSim "+to_string(numSim));
    int n = config.iters;
    Stock stock = market.getStock();
    double r = getVariable("riskFreeRate");
    double q = getVariable("dividendYield");
    double T = getVariable("maturity");
    double err = NAN;
    matrix simPriceMatrix, simTimeVector;
    stock.setDriftRate(r-q);
    // handle exceptions
    string optionType = option.getType();
    if(optionType=="Chooser"){
        double K = option.getStrike();
        double t = option.getParams()[0];
        option.setDiscStrike(K*exp(-r*(T-t)));
    }
    if(method=="simple"){
        simPriceMatrix = stock.simulatePrice(config,numSim);
        simTimeVector = stock.getSimTimeVector();
        if(!option.canEarlyExercise()){
            matrix payoffs = option.calcPayoffs(NULL_VECTOR,simPriceMatrix,{},simTimeVector);
            price = exp(-r*T)*payoffs.mean();
            err = exp(-r*T)*payoffs.stdev()/sqrt(numSim);
        }else{
            // Longstaff-Schwartz algorithm
        }
    }else if(method=="antithetic variates"){
        matrix simPriceMatrix0, simPriceMatrix1;
        matrix randomMatrix0(n+1,numSim), randomMatrix1(n+1,numSim);
        randomMatrix0.setNormalRand(); randomMatrix1 = -randomMatrix0;
        simPriceMatrix0 = stock.simulatePrice(config,numSim,randomMatrix0);
        simPriceMatrix1 = stock.simulatePrice(config,numSim,randomMatrix1);
        simTimeVector = stock.getSimTimeVector();
        if(!option.canEarlyExercise()){
            matrix payoffs0 = option.calcPayoffs(NULL_VECTOR,simPriceMatrix0,{},simTimeVector);
            matrix payoffs1 = option.calcPayoffs(NULL_VECTOR,simPriceMatrix1,{},simTimeVector);
            matrix payoffs = (payoffs0+payoffs1)/2;
            price = exp(-r*T)*payoffs.mean();
            err = exp(-r*T)*payoffs.stdev()/sqrt(numSim);
        }
    }else if(method=="control variates"){
        simPriceMatrix = stock.simulatePrice(config,numSim);
        simTimeVector = stock.getSimTimeVector();
        if(!option.canEarlyExercise()){
            matrix payoffs = option.calcPayoffs(NULL_VECTOR,simPriceMatrix,{},simTimeVector);
            price = exp(-r*T)*payoffs.mean();
            err = exp(-r*T)*payoffs.stdev()/sqrt(numSim);
        }
        Pricer refPricer(option,market);
        refPricer.setStringVariable("optionType","European");
        double refPriceMonte = refPricer.MonteCarloPricer(config,numSim);
        double refPriceClosed = refPricer.BlackScholesClosedForm();
        double refErr = refPricer.tmp[0];
        price += refPriceClosed-refPriceMonte;
        err = sqrt(err*err+refErr*refErr); // assume uncorrelated MC estimates
    }
    tmp = {err};
    logMessage("ending calculation MonteCarloPricer, return "+to_string(price)+" with error "+to_string(err));
    return price;
}

double Pricer::MultiStockMonteCarloPricer(const SimConfig& config, int numSim, string method){
    logMessage("starting calculation MonteCarloPricer on config "+to_string(config)+", numSim "+to_string(numSim));
    int n = config.iters;
    double r = getVariable("riskFreeRate");
    double T = getVariable("maturity");
    Market rnMarket(market); // risk-neutral market
    vector<Stock> stocks = rnMarket.getStocks();
    for(auto& stock:stocks){
        double q = stock.getDividendYield();
        stock.setDriftRate(r-q);
    }
    rnMarket.setStocks(stocks);
    double err = NAN;
    vector<matrix> simPriceMatrixSet;
    if(method=="simple"){
        simPriceMatrixSet = rnMarket.simulateCorrelatedPrices(config,numSim);
        if(!option.canEarlyExercise()){
            matrix payoffs = option.calcPayoffs(NULL_VECTOR,NULL_MATRIX,simPriceMatrixSet);
            price = exp(-r*T)*payoffs.mean();
            err = exp(-r*T)*payoffs.stdev()/sqrt(numSim);
        }
    }else if(method=="antithetic variates"){
        // TO DO
    }else if(method=="control variates"){
        // TO DO
    }
    tmp = {err};
    logMessage("ending calculation MonteCarloPricer, return "+to_string(price)+" with error "+to_string(err));
    return price;
}

double Pricer::NumIntegrationPricer(double z, double dz){
    logMessage("starting calculation NumIntegrationPricer on z "+to_string(z)+", dz "+to_string(dz));
    Stock stock = market.getStock();
    double r = getVariable("riskFreeRate");
    double q = getVariable("dividendYield");
    double T = getVariable("maturity");
    stock.setDriftRate(r-q);
    int n = static_cast<int>(z/dz);
    matrix z0; z0.setRange(-z,z,2*n);
    matrix S = stock.calcLognormalPriceVector(z0,T);
    if(!option.canEarlyExercise() && !option.isPathDependent()){
        matrix payoffs = option.calcPayoffs(S);
        matrix probs = z0.apply(stdNormalPDF)*dz;
        price = exp(-r*T)*(probs*payoffs).sum();
    }
    logMessage("ending calculation NumIntegrationPricer, return "+to_string(price));
    return price;
}

double Pricer::BlackScholesPDESolver(const SimConfig& config, int numSpace, string method){
    logMessage("starting calculation BlackScholesPDESolver on config "+to_string(config)+
        ", numSpace "+to_string(numSpace)+", method "+method);
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
    matrix priceMatrix(n+1,m+1);
    matrix timeGrids; timeGrids.setRange(0,T,n,true);
    matrix spaceGrids; spaceGrids.setRange(x0,x1,m,true);
    // cout << D.print() << endl;
    if(option.getType()=="European"){
        matrix payoffs = option.calcPayoffs(spaceGrids.apply(exp));
        matrix bdryCondition0(1,n+1), bdryCondition1(1,n+1), v, u(m-1,1);
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
            matrix D(m-1,m-1);
            D.setDiags({a,b,c},{-1,0,1});
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
            matrix D(m-1,m-1);
            D.setDiags({a,b,c},{-1,0,1});
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
        vector<int> idx = (x-spaceGrids).apply(abs).minIdx();
        price = priceMatrix.getEntry(idx);
    }
    logMessage("ending calculation BlackScholesPDESolver, return "+to_string(price));
    return price;
}

double Pricer::calcPrice(string method, const SimConfig& config, int numSim, int numSpace){
    if(method=="Closed Form"){
        price = BlackScholesClosedForm();
    }else if(method=="Binomial Tree"){
        price = BinomialTreePricer(config);
    }else if(method=="Monte Carlo"){
        price = MonteCarloPricer(config,numSim);
    }else if(method=="Num Integration"){
        price = NumIntegrationPricer();
    }else if(method=="PDE Solver"){
        price = BlackScholesPDESolver(config,numSpace);
    }
    return price;
}

matrix Pricer::varyPriceWithVariable(string var, matrix varVector,
    string method, const SimConfig& config, int numSim){
    saveAsOriginal();
    int n = varVector.getCols();
    matrix optionPriceVector(1,n);
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
    saveAsOriginal();
    double greek = NAN;
    if(option.getType()=="European"){
        if(getVariable("maturity")==0)
            setVariable("maturity",1e-5);
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
    resetOriginal();
    logMessage("ending calculation ClosedFormGreek, return "+to_string(greek));
    return greek;
}

double Pricer::FiniteDifferenceGreek(string var, int derivOrder, string method,
    const SimConfig& config, int numSim, double eps){
    logMessage("starting calculation FiniteDifferenceGreek on var "+var+", derivOrder "+to_string(derivOrder)+
        ", method "+method+", config "+to_string(config)+", numSim "+to_string(numSim)+", eps "+to_string(eps));
    saveAsOriginal();
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

matrix Pricer::varyGreekWithVariable(string var, matrix varVector, string greekName,
    string greekMethod, string method, const SimConfig& config, int numSim, double eps){
    saveAsOriginal();
    int n = varVector.getCols();
    double greek;
    matrix optionGreekVector(1,n);
    for(int i=0; i<n; i++){
        double v = varVector.getEntry(0,i);
        setVariable(var,v);
        greek = calcGreek(greekName,greekMethod,method,config,numSim,eps);
        optionGreekVector.setEntry(0,i,greek);
    }
    resetOriginal();
    return optionGreekVector;
}

matrix Pricer::generatePriceSurface(matrix stockPriceVector, matrix optionTermVector,
    string method, const SimConfig& config, int numSim){
    saveAsOriginal();
    int m = optionTermVector.getCols();
    int n = stockPriceVector.getCols();
    matrix priceSurface(m,n);
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
    saveAsOriginal();
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
    saveAsOriginal();
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
        option = Option(type,putCall,strike,maturity,{},"",name);
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
    saveAsOriginal();
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
        option = Option(type,putCall,strike,maturity,{},"",name);
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

vector<matrix> Pricer::modelImpliedVolSurface(const SimConfig& config, int numSpace,
    const function<double(double)>& impVolFunc0, const function<double(double)>& impVolFunc1,
    double lambdaT, double eps){
    double K = getVariable("strike");
    double T = config.endTime;
    int n = config.iters;
    int m = numSpace;
    double dt = config.stepSize;
    double dt2 = dt*dt;
    double x0 = log(K/3), x1 = log(3*K);
    double dx = (x1-x0)/m;
    double dx2 = dx*dx;
    matrix impVolSurface(n+1,m+1);
    matrix timeGrids; timeGrids.setRange(0,T,n,true);
    matrix spaceGrids; spaceGrids.setRange(x0,x1,m,true);
    matrix initCondition, termCondition, bdryCondition0, bdryCondition1;
    double sig0, sig1, expFactor = exp(-T/lambdaT);
    initCondition = spaceGrids.apply(impVolFunc0);
    termCondition = spaceGrids.apply(impVolFunc1);
    sig0 = initCondition.getFirstEntry();
    sig1 = termCondition.getFirstEntry();
    bdryCondition0 = timeGrids.apply([sig0,sig1,expFactor,lambdaT](double x){
        return mathFunc(x,"exponential",{
            (sig1-sig0*expFactor)/(1-expFactor),(sig0-sig1)/(1-expFactor),lambdaT
        });
    });
    sig0 = initCondition.getLastEntry();
    sig1 = termCondition.getLastEntry();
    bdryCondition1 = timeGrids.apply([sig0,sig1,expFactor,lambdaT](double x){
        return mathFunc(x,"exponential",{
            (sig1-sig0*expFactor)/(1-expFactor),(sig0-sig1)/(1-expFactor),lambdaT
        });
    });
    impVolSurface.setRow(0,initCondition);
    impVolSurface.setRow(n,termCondition);
    impVolSurface.setCol(0,bdryCondition0);
    impVolSurface.setCol(m,bdryCondition1);
    double err = 1;
    while(err>eps){
        matrix impVolSurfacePrev = impVolSurface;
        for(int i=1; i<n; i++)
            for(int j=1; j<m; j++){
                double avg =
                    (impVolSurfacePrev.getEntry(i-1,j)+
                     impVolSurfacePrev.getEntry(i+1,j)+
                     impVolSurfacePrev.getEntry(i,j-1)+
                     impVolSurfacePrev.getEntry(i,j+1))/4;
                impVolSurface.setEntry(i,j,avg);
            }
        err = (impVolSurface-impVolSurfacePrev).sum()/impVolSurfacePrev.sum();
    }
    // cout << impVolSurface.print() << endl;
    vector<matrix> impVolSurfaceSet{
        spaceGrids,
        timeGrids,
        impVolSurface
    };
    return impVolSurfaceSet;
}

Backtest Pricer::runBacktest(const SimConfig& config, int numSim,
    string strategy, int hedgeFreq, double mktImpVol, double mktPrice,
    vector<double> stratParams, vector<Option> hOptions, vector<matrix> impVolSurfaceSet,
    string simPriceMethod, matrix stockPriceSeries){
    if(GUI) cout << "running backtest for strategy: " << strategy << endl;
    Stock stock = market.getStock();
    double K = getVariable("strike");
    double T = getVariable("maturity");
    double r = getVariable("riskFreeRate");
    double S0 = getVariable("currentPrice");
    double q = getVariable("dividendYield");
    double sig = getVariable("volatility");
    double sig2 = sig*sig;
    double dt = config.stepSize;
    double riskFreeRateFactor = exp(r*dt);
    double dividendYieldFactor = exp(q*dt);
    double Delta, Gamma, Vega, Rho, Theta;
    bool flatImpVolSurface = impVolSurfaceSet.size()==0;
    if(simPriceMethod=="bootstrap")
        stock.bootstrapPrice(stockPriceSeries,config,numSim);
    else stock.simulatePrice(config,numSim);
    int n = config.iters;
    matrix
        stratCashMatrix(n+1,numSim),
        stratNStockMatrix(n+1,numSim),
        stratModPriceMatrix(n+1,numSim),
        stratModValueMatrix(n+1,numSim),
        stratGrkDelta(n+1,numSim),
        stratGrkGamma(n+1,numSim),
        stratGrkVega(n+1,numSim),
        stratGrkRho(n+1,numSim),
        stratGrkTheta(n+1,numSim);
    vector<matrix>
        stratNOptions,
        stratHModPrices;
    matrix simPriceMatrix = stock.getSimPriceMatrix();
    if(strategy=="simple-delta" || strategy=="mkt-delta"){
        if(strategy=="mkt-delta"){
            double sigImp;
            if(mktImpVol>0) sigImp = mktImpVol;
            else sigImp = calcImpliedVolatility(mktPrice);
            setVariable("volatility",sigImp);
        }
        for(int i=0; i<numSim; i++){
            setVariable("currentPrice",S0);
            setVariable("maturity",T);
            double modPrice = calcPrice("Closed Form");
            double nStock = calcGreek("Delta");
            double cash = modPrice;
            cash -= nStock*S0;
            double value = cash+nStock*S0-modPrice;
            stratCashMatrix.setEntry(0,i,cash);
            stratNStockMatrix.setEntry(0,i,nStock);
            stratModPriceMatrix.setEntry(0,i,modPrice);
            stratModValueMatrix.setEntry(0,i,value);
            Delta = calcGreek("Delta");
            Gamma = calcGreek("Gamma");
            Vega = calcGreek("Vega");
            Rho = calcGreek("Rho");
            Theta = calcGreek("Theta");
            stratGrkDelta.setEntry(0,i,nStock-Delta);
            stratGrkGamma.setEntry(0,i,-Gamma);
            stratGrkVega.setEntry(0,i,-Vega);
            stratGrkRho.setEntry(0,i,-Rho);
            stratGrkTheta.setEntry(0,i,r*cash+q*nStock*S0-Theta-sig2*S0*S0/2*Gamma);
            for(int t=1; t<n; t++){
                double S = simPriceMatrix.getEntry(t,i);
                double nStockPrev = nStock;
                setVariable("currentPrice",S);
                setVariable("maturity",T-t*dt);
                modPrice = calcPrice("Closed Form");
                if(t%hedgeFreq==0){
                    nStock = calcGreek("Delta");
                    cash = cash*riskFreeRateFactor
                        +nStock*S*(dividendYieldFactor-1)
                        -(nStock-nStockPrev)*S;
                }else{
                    cash = cash*riskFreeRateFactor
                        +nStockPrev*S*(dividendYieldFactor-1);
                }
                value = cash+nStock*S-modPrice;
                stratCashMatrix.setEntry(t,i,cash);
                stratNStockMatrix.setEntry(t,i,nStock);
                stratModPriceMatrix.setEntry(t,i,modPrice);
                stratModValueMatrix.setEntry(t,i,value);
                Delta = calcGreek("Delta");
                Gamma = calcGreek("Gamma");
                Vega = calcGreek("Vega");
                Rho = calcGreek("Rho");
                Theta = calcGreek("Theta");
                stratGrkDelta.setEntry(t,i,nStock-Delta);
                stratGrkGamma.setEntry(t,i,-Gamma);
                stratGrkVega.setEntry(t,i,-Vega);
                stratGrkRho.setEntry(t,i,-Rho);
                stratGrkTheta.setEntry(t,i,r*cash+q*nStock*S-Theta-sig2*S*S/2*Gamma);
            }
            double S1 = simPriceMatrix.getEntry(n,i);
            double nStockPrev = nStock;
            setVariable("currentPrice",S1);
            setVariable("maturity",0);
            modPrice = option.calcPayoff(S1);
            nStock = 0;
            cash = cash*riskFreeRateFactor+nStockPrev*S1;
            value = cash-modPrice;
            stratCashMatrix.setEntry(n,i,cash);
            stratNStockMatrix.setEntry(n,i,nStock);
            stratModPriceMatrix.setEntry(n,i,modPrice);
            stratModValueMatrix.setEntry(n,i,value);
            stratGrkDelta.setEntry(n,i,0);
            stratGrkGamma.setEntry(n,i,0);
            stratGrkVega.setEntry(n,i,0);
            stratGrkRho.setEntry(n,i,0);
            stratGrkTheta.setEntry(n,i,r*cash);
            if(GUI) cout << "[" << getCurrentTime() << "] simulation " <<
                i+1 << "/" << numSim << " completes" << endl;
        }
        // cout << stratModValueMatrix.print() << endl;
    }else if(strategy=="mkt-delta-hedgingVol"){
        double hedgingVol = stratParams[0];
        Pricer hPricer(*this);
        hPricer.setVariable("volatility",hedgingVol);
        double sigImp;
        if(mktImpVol>0) sigImp = mktImpVol;
        else sigImp = calcImpliedVolatility(mktPrice);
        setVariable("volatility",sigImp);
        for(int i=0; i<numSim; i++){
            setVariable("currentPrice",S0);
            setVariable("maturity",T);
            hPricer.setVariable("currentPrice",S0);
            hPricer.setVariable("maturity",T);
            double modPrice = calcPrice("Closed Form");
            double nStock = hPricer.calcGreek("Delta");
            double cash = modPrice;
            cash -= nStock*S0;
            double value = cash+nStock*S0-modPrice;
            stratCashMatrix.setEntry(0,i,cash);
            stratNStockMatrix.setEntry(0,i,nStock);
            stratModPriceMatrix.setEntry(0,i,modPrice);
            stratModValueMatrix.setEntry(0,i,value);
            Delta = calcGreek("Delta");
            Gamma = calcGreek("Gamma");
            Vega = calcGreek("Vega");
            Rho = calcGreek("Rho");
            Theta = calcGreek("Theta");
            stratGrkDelta.setEntry(0,i,nStock-Delta);
            stratGrkGamma.setEntry(0,i,-Gamma);
            stratGrkVega.setEntry(0,i,-Vega);
            stratGrkRho.setEntry(0,i,-Rho);
            stratGrkTheta.setEntry(0,i,r*cash+q*nStock*S0-Theta-sig2*S0*S0/2*Gamma);
            for(int t=1; t<n; t++){
                double S = simPriceMatrix.getEntry(t,i);
                double nStockPrev = nStock;
                setVariable("currentPrice",S);
                setVariable("maturity",T-t*dt);
                hPricer.setVariable("currentPrice",S);
                hPricer.setVariable("maturity",T-t*dt);
                modPrice = calcPrice("Closed Form");
                if(t%hedgeFreq==0){
                    nStock = hPricer.calcGreek("Delta");
                    cash = cash*riskFreeRateFactor
                        +nStock*S*(dividendYieldFactor-1)
                        -(nStock-nStockPrev)*S;
                }else{
                    cash = cash*riskFreeRateFactor
                        +nStockPrev*S*(dividendYieldFactor-1);
                }
                value = cash+nStock*S-modPrice;
                stratCashMatrix.setEntry(t,i,cash);
                stratNStockMatrix.setEntry(t,i,nStock);
                stratModPriceMatrix.setEntry(t,i,modPrice);
                stratModValueMatrix.setEntry(t,i,value);
                Delta = calcGreek("Delta");
                Gamma = calcGreek("Gamma");
                Vega = calcGreek("Vega");
                Rho = calcGreek("Rho");
                Theta = calcGreek("Theta");
                stratGrkDelta.setEntry(t,i,nStock-Delta);
                stratGrkGamma.setEntry(t,i,-Gamma);
                stratGrkVega.setEntry(t,i,-Vega);
                stratGrkRho.setEntry(t,i,-Rho);
                stratGrkTheta.setEntry(t,i,r*cash+q*nStock*S-Theta-sig2*S*S/2*Gamma);
            }
            double S1 = simPriceMatrix.getEntry(n,i);
            double nStockPrev = nStock;
            setVariable("currentPrice",S1);
            setVariable("maturity",0);
            hPricer.setVariable("currentPrice",S1);
            hPricer.setVariable("maturity",0);
            modPrice = option.calcPayoff(S1);
            nStock = 0;
            cash = cash*riskFreeRateFactor+nStockPrev*S1;
            value = cash-modPrice;
            stratCashMatrix.setEntry(n,i,cash);
            stratNStockMatrix.setEntry(n,i,nStock);
            stratModPriceMatrix.setEntry(n,i,modPrice);
            stratModValueMatrix.setEntry(n,i,value);
            stratGrkDelta.setEntry(n,i,0);
            stratGrkGamma.setEntry(n,i,0);
            stratGrkVega.setEntry(n,i,0);
            stratGrkRho.setEntry(n,i,0);
            stratGrkTheta.setEntry(n,i,r*cash);
            if(GUI) cout << "[" << getCurrentTime() << "] simulation " <<
                i+1 << "/" << numSim << " completes" << endl;
        }
    }else if(strategy=="simple-delta-gamma" || strategy=="mkt-delta-gamma"){
        double hDelta, hGamma, hVega, hRho, hTheta;
        stratNOptions.push_back(matrix(n+1,numSim));
        stratHModPrices.push_back(matrix(n+1,numSim));
        Option hOption = hOptions[0];
        Pricer hPricer(hOption,market);
        double Th = hPricer.getVariable("maturity");
        double O0 = hPricer.calcPrice("Closed Form");
        if(strategy=="mkt-delta-gamma"){
            double sigImp;
            if(mktImpVol>0) sigImp = mktImpVol;
            else sigImp = calcImpliedVolatility(mktPrice);
            setVariable("volatility",sigImp);
            hPricer.setVariable("volatility",sigImp);
        }
        for(int i=0; i<numSim; i++){
            setVariable("currentPrice",S0);
            setVariable("maturity",T);
            hPricer.setVariable("currentPrice",S0);
            hPricer.setVariable("maturity",Th);
            double modPrice = calcPrice("Closed Form");
            double nOption = calcGreek("Gamma")
                /hPricer.calcGreek("Gamma");
            double nStock = calcGreek("Delta")
                -nOption*hPricer.calcGreek("Delta");
            double cash = modPrice;
            cash -= nStock*S0+nOption*O0;
            double value = cash+nStock*S0+nOption*O0-modPrice;
            stratCashMatrix.setEntry(0,i,cash);
            stratNStockMatrix.setEntry(0,i,nStock);
            stratModPriceMatrix.setEntry(0,i,modPrice);
            stratModValueMatrix.setEntry(0,i,value);
            Delta = calcGreek("Delta"); hDelta = hPricer.calcGreek("Delta");
            Gamma = calcGreek("Gamma"); hGamma = hPricer.calcGreek("Gamma");
            Vega = calcGreek("Vega"); hVega = hPricer.calcGreek("Vega");
            Rho = calcGreek("Rho"); hRho = hPricer.calcGreek("Rho");
            Theta = calcGreek("Theta"); hTheta = hPricer.calcGreek("Theta");
            stratGrkDelta.setEntry(0,i,nStock+nOption*hDelta-Delta);
            stratGrkGamma.setEntry(0,i,nOption*hGamma-Gamma);
            stratGrkVega.setEntry(0,i,nOption*hVega-Vega);
            stratGrkRho.setEntry(0,i,nOption*hRho-Rho);
            stratGrkTheta.setEntry(0,i,r*cash+q*nStock*S0+nOption*hTheta-Theta);
            stratNOptions[0].setEntry(0,i,nOption);
            stratHModPrices[0].setEntry(0,i,O0);
            for(int t=1; t<n; t++){
                double S = simPriceMatrix.getEntry(t,i);
                double nStockPrev = nStock;
                double nOptionPrev = nOption;
                setVariable("currentPrice",S);
                setVariable("maturity",T-t*dt);
                hPricer.setVariable("currentPrice",S);
                hPricer.setVariable("maturity",Th-t*dt);
                double O = hPricer.calcPrice("Closed Form");
                modPrice = calcPrice("Closed Form");
                if(t%hedgeFreq==0){
                    nOption = calcGreek("Gamma")
                        /hPricer.calcGreek("Gamma");
                    nStock = calcGreek("Delta")
                        -nOption*hPricer.calcGreek("Delta");
                    cash = cash*riskFreeRateFactor
                        +nStock*S*(dividendYieldFactor-1)
                        -(nStock-nStockPrev)*S
                        -(nOption-nOptionPrev)*O;
                }else{
                    cash = cash*riskFreeRateFactor
                        +nStockPrev*S*(dividendYieldFactor-1);
                }
                value = cash+nStock*S+nOption*O-modPrice;
                stratCashMatrix.setEntry(t,i,cash);
                stratNStockMatrix.setEntry(t,i,nStock);
                stratModPriceMatrix.setEntry(t,i,modPrice);
                stratModValueMatrix.setEntry(t,i,value);
                Delta = calcGreek("Delta"); hDelta = hPricer.calcGreek("Delta");
                Gamma = calcGreek("Gamma"); hGamma = hPricer.calcGreek("Gamma");
                Vega = calcGreek("Vega"); hVega = hPricer.calcGreek("Vega");
                Rho = calcGreek("Rho"); hRho = hPricer.calcGreek("Rho");
                Theta = calcGreek("Theta"); hTheta = hPricer.calcGreek("Theta");
                stratGrkDelta.setEntry(t,i,nStock+nOption*hDelta-Delta);
                stratGrkGamma.setEntry(t,i,nOption*hGamma-Gamma);
                stratGrkVega.setEntry(t,i,nOption*hVega-Vega);
                stratGrkRho.setEntry(t,i,nOption*hRho-Rho);
                stratGrkTheta.setEntry(t,i,r*cash+q*nStock*S+nOption*hTheta-Theta);
                stratNOptions[0].setEntry(t,i,nOption);
                stratHModPrices[0].setEntry(t,i,O);
            }
            double S1 = simPriceMatrix.getEntry(n,i);
            double nStockPrev = nStock;
            double nOptionPrev = nOption;
            setVariable("currentPrice",S1);
            setVariable("maturity",0);
            hPricer.setVariable("currentPrice",S1);
            hPricer.setVariable("maturity",Th-n*dt);
            double O1 = hPricer.calcPrice("Closed Form");
            modPrice = option.calcPayoff(S1);
            nStock = 0;
            nOption = 0;
            cash = cash*riskFreeRateFactor+nStockPrev*S1+nOptionPrev*O1;
            value = cash-modPrice;
            stratCashMatrix.setEntry(n,i,cash);
            stratNStockMatrix.setEntry(n,i,nStock);
            stratModPriceMatrix.setEntry(n,i,modPrice);
            stratModValueMatrix.setEntry(n,i,value);
            stratGrkDelta.setEntry(n,i,0);
            stratGrkGamma.setEntry(n,i,0);
            stratGrkVega.setEntry(n,i,0);
            stratGrkRho.setEntry(n,i,0);
            stratGrkTheta.setEntry(n,i,r*cash);
            stratNOptions[0].setEntry(n,i,nOption);
            stratHModPrices[0].setEntry(n,i,O1);
            if(GUI) cout << "[" << getCurrentTime() << "] simulation " <<
                i+1 << "/" << numSim << " completes" << endl;
        }
    }else if(strategy=="simple-delta-gamma-theta" || strategy=="mkt-delta-gamma-theta"){
        double hDelta0, hGamma0, hVega0, hRho0, hTheta0;
        double hDelta1, hGamma1, hVega1, hRho1, hTheta1;
        for(int i=0; i<2; i++){
            stratNOptions.push_back(matrix(n+1,numSim));
            stratHModPrices.push_back(matrix(n+1,numSim));
        }
        Option hOption0 = hOptions[0], hOption1 = hOptions[1];
        Pricer hPricer0(hOption0,market), hPricer1(hOption1,market);
        double Th0 = hPricer0.getVariable("maturity");
        double Th1 = hPricer1.getVariable("maturity");
        double O00 = hPricer0.calcPrice("Closed Form");
        double O10 = hPricer1.calcPrice("Closed Form");
        if(strategy=="mkt-delta-gamma-theta"){
            double sigImp;
            if(mktImpVol>0) sigImp = mktImpVol;
            else sigImp = calcImpliedVolatility(mktPrice);
            setVariable("volatility",sigImp);
            hPricer0.setVariable("volatility",sigImp);
            hPricer1.setVariable("volatility",sigImp);
        }
        for(int i=0; i<numSim; i++){
            setVariable("currentPrice",S0);
            setVariable("maturity",T);
            hPricer0.setVariable("currentPrice",S0);
            hPricer0.setVariable("maturity",Th0);
            hPricer1.setVariable("currentPrice",S0);
            hPricer1.setVariable("maturity",Th1);
            double modPrice = calcPrice("Closed Form");
            double tmpM[2][2]
                = {{hPricer0.calcGreek("Theta"),hPricer1.calcGreek("Theta")},
                    hPricer0.calcGreek("Gamma"),hPricer1.calcGreek("Gamma")};
            double tmpV[2] = {calcGreek("Theta"),calcGreek("Gamma")};
            matrix nOptions = matrix(tmpM).inverse().dot(matrix(tmpV).transpose());
            double nOption0 = nOptions.getEntry(0,0),
                   nOption1 = nOptions.getEntry(1,0);
            double nStock = calcGreek("Delta")
                -nOption0*hPricer0.calcGreek("Delta")
                -nOption1*hPricer1.calcGreek("Delta");
            double cash = modPrice;
            cash -= nStock*S0+nOption0*O00+nOption1*O10;
            double value = cash+nStock*S0+nOption0*O00+nOption1*O10-modPrice;
            stratCashMatrix.setEntry(0,i,cash);
            stratNStockMatrix.setEntry(0,i,nStock);
            stratModPriceMatrix.setEntry(0,i,modPrice);
            stratModValueMatrix.setEntry(0,i,value);
            Delta = calcGreek("Delta"); hDelta0 = hPricer0.calcGreek("Delta"); hDelta1 = hPricer1.calcGreek("Delta");
            Gamma = calcGreek("Gamma"); hGamma0 = hPricer0.calcGreek("Gamma"); hGamma1 = hPricer1.calcGreek("Gamma");
            Vega = calcGreek("Vega"); hVega0 = hPricer0.calcGreek("Vega"); hVega1 = hPricer1.calcGreek("Vega");
            Rho = calcGreek("Rho"); hRho0 = hPricer0.calcGreek("Rho"); hRho1 = hPricer1.calcGreek("Rho");
            Theta = calcGreek("Theta"); hTheta0 = hPricer0.calcGreek("Theta"); hTheta1 = hPricer1.calcGreek("Theta");
            stratGrkDelta.setEntry(0,i,nStock+nOption0*hDelta0+nOption1*hDelta1-Delta);
            stratGrkGamma.setEntry(0,i,nOption0*hGamma0+nOption1*hGamma1-Gamma);
            stratGrkVega.setEntry(0,i,nOption0*hVega0+nOption1*hVega1-Vega);
            stratGrkRho.setEntry(0,i,nOption0*hRho0+nOption1*hRho1-Rho);
            stratGrkTheta.setEntry(0,i,r*cash+q*nStock*S0);
            stratNOptions[0].setEntry(0,i,nOption0);
            stratNOptions[1].setEntry(0,i,nOption1);
            stratHModPrices[0].setEntry(0,i,O00);
            stratHModPrices[1].setEntry(0,i,O10);
            for(int t=1; t<n; t++){
                double S = simPriceMatrix.getEntry(t,i);
                double nStockPrev = nStock;
                double nOptionPrev0 = nOption0;
                double nOptionPrev1 = nOption1;
                setVariable("currentPrice",S);
                setVariable("maturity",T-t*dt);
                hPricer0.setVariable("currentPrice",S);
                hPricer0.setVariable("maturity",Th0-t*dt);
                hPricer1.setVariable("currentPrice",S);
                hPricer1.setVariable("maturity",Th1-t*dt);
                double O0 = hPricer0.calcPrice("Closed Form");
                double O1 = hPricer1.calcPrice("Closed Form");
                modPrice = calcPrice("Closed Form");
                if(t%hedgeFreq==0){
                    double tmpM[2][2]
                        = {{hPricer0.calcGreek("Theta"),hPricer1.calcGreek("Theta")},
                            hPricer0.calcGreek("Gamma"),hPricer1.calcGreek("Gamma")};
                    double tmpV[2] = {calcGreek("Theta"),calcGreek("Gamma")};
                    nOptions = matrix(tmpM).inverse().dot(matrix(tmpV).transpose());
                    nOption0 = nOptions.getEntry(0,0),
                    nOption1 = nOptions.getEntry(1,0);
                    nStock = calcGreek("Delta")
                        -nOption0*hPricer0.calcGreek("Delta")
                        -nOption1*hPricer1.calcGreek("Delta");
                    cash = cash*riskFreeRateFactor
                        +nStock*S*(dividendYieldFactor-1)
                        -(nStock-nStockPrev)*S
                        -(nOption0-nOptionPrev0)*O0
                        -(nOption1-nOptionPrev1)*O1;
                }else{
                    cash = cash*riskFreeRateFactor
                        +nStockPrev*S*(dividendYieldFactor-1);
                }
                value = cash+nStock*S+nOption0*O0+nOption1*O1-modPrice;
                stratCashMatrix.setEntry(t,i,cash);
                stratNStockMatrix.setEntry(t,i,nStock);
                stratModPriceMatrix.setEntry(t,i,modPrice);
                stratModValueMatrix.setEntry(t,i,value);
                Delta = calcGreek("Delta"); hDelta0 = hPricer0.calcGreek("Delta"); hDelta1 = hPricer1.calcGreek("Delta");
                Gamma = calcGreek("Gamma"); hGamma0 = hPricer0.calcGreek("Gamma"); hGamma1 = hPricer1.calcGreek("Gamma");
                Vega = calcGreek("Vega"); hVega0 = hPricer0.calcGreek("Vega"); hVega1 = hPricer1.calcGreek("Vega");
                Rho = calcGreek("Rho"); hRho0 = hPricer0.calcGreek("Rho"); hRho1 = hPricer1.calcGreek("Rho");
                Theta = calcGreek("Theta"); hTheta0 = hPricer0.calcGreek("Theta"); hTheta1 = hPricer1.calcGreek("Theta");
                stratGrkDelta.setEntry(t,i,nStock+nOption0*hDelta0+nOption1*hDelta1-Delta);
                stratGrkGamma.setEntry(t,i,nOption0*hGamma0+nOption1*hGamma1-Gamma);
                stratGrkVega.setEntry(t,i,nOption0*hVega0+nOption1*hVega1-Vega);
                stratGrkRho.setEntry(t,i,nOption0*hRho0+nOption1*hRho1-Rho);
                stratGrkTheta.setEntry(t,i,r*cash+q*nStock*S);
                stratNOptions[0].setEntry(t,i,nOption0);
                stratNOptions[1].setEntry(t,i,nOption1);
                stratHModPrices[0].setEntry(t,i,O0);
                stratHModPrices[1].setEntry(t,i,O1);
            }
            double S1 = simPriceMatrix.getEntry(n,i);
            double nStockPrev = nStock;
            double nOptionPrev0 = nOption0;
            double nOptionPrev1 = nOption1;
            setVariable("currentPrice",S1);
            setVariable("maturity",0);
            hPricer0.setVariable("currentPrice",S1);
            hPricer0.setVariable("maturity",Th0-n*dt);
            hPricer1.setVariable("currentPrice",S1);
            hPricer1.setVariable("maturity",Th1-n*dt);
            double O01 = hPricer0.calcPrice("Closed Form");
            double O11 = hPricer1.calcPrice("Closed Form");
            modPrice = option.calcPayoff(S1);
            nStock = 0;
            nOption0 = 0;
            nOption1 = 0;
            cash = cash*riskFreeRateFactor+nStockPrev*S1+nOptionPrev0*O01+nOptionPrev1*O11;
            value = cash-modPrice;
            stratCashMatrix.setEntry(n,i,cash);
            stratNStockMatrix.setEntry(n,i,nStock);
            stratModPriceMatrix.setEntry(n,i,modPrice);
            stratModValueMatrix.setEntry(n,i,value);
            stratGrkDelta.setEntry(n,i,0);
            stratGrkGamma.setEntry(n,i,0);
            stratGrkVega.setEntry(n,i,0);
            stratGrkRho.setEntry(n,i,0);
            stratGrkTheta.setEntry(n,i,r*cash);
            stratNOptions[0].setEntry(n,i,nOption0);
            stratNOptions[1].setEntry(n,i,nOption1);
            stratHModPrices[0].setEntry(n,i,O01);
            stratHModPrices[1].setEntry(n,i,O11);
            if(GUI) cout << "[" << getCurrentTime() << "] simulation " <<
                i+1 << "/" << numSim << " completes" << endl;
        }
    }else if(strategy=="vol-delta-gamma-theta"){
        double hDelta0, hGamma0, hVega0, hRho0, hTheta0;
        double hDelta1, hGamma1, hVega1, hRho1, hTheta1;
        for(int i=0; i<2; i++){
            stratNOptions.push_back(matrix(n+1,numSim));
            stratHModPrices.push_back(matrix(n+1,numSim));
        }
        Option hOption0 = hOptions[0], hOption1 = hOptions[1];
        Pricer hPricer0(hOption0,market), hPricer1(hOption1,market);
        double Th0 = hPricer0.getVariable("maturity");
        double Th1 = hPricer1.getVariable("maturity");
        double O00 = hPricer0.calcPrice("Closed Form");
        double O10 = hPricer1.calcPrice("Closed Form");
        int idxK, idxK0, idxK1, idxT, idxT0, idxT1;
        if(!flatImpVolSurface){
            idxK = (log(getVariable("strike"))-impVolSurfaceSet[0]).apply(abs).minIdx()[1];
            idxK0 = (log(hPricer0.getVariable("strike"))-impVolSurfaceSet[0]).apply(abs).minIdx()[1];
            idxK1 = (log(hPricer1.getVariable("strike"))-impVolSurfaceSet[0]).apply(abs).minIdx()[1];
        }
        for(int i=0; i<numSim; i++){
            setVariable("currentPrice",S0);
            setVariable("maturity",T);
            hPricer0.setVariable("currentPrice",S0);
            hPricer0.setVariable("maturity",Th0);
            hPricer1.setVariable("currentPrice",S0);
            hPricer1.setVariable("maturity",Th1);
            if(!flatImpVolSurface){
                idxT = (getVariable("maturity")-impVolSurfaceSet[1]).apply(abs).minIdx()[1];
                idxT0 = (hPricer0.getVariable("maturity")-impVolSurfaceSet[1]).apply(abs).minIdx()[1];
                idxT1 = (hPricer1.getVariable("maturity")-impVolSurfaceSet[1]).apply(abs).minIdx()[1];
                setVariable("volatility",impVolSurfaceSet[2].getEntry(idxT,idxK));
                hPricer0.setVariable("volatility",impVolSurfaceSet[2].getEntry(idxT0,idxK0));
                hPricer1.setVariable("volatility",impVolSurfaceSet[2].getEntry(idxT1,idxK1));
            }
            double modPrice = calcPrice("Closed Form");
            double tmpM[2][2]
                = {{hPricer0.calcGreek("Theta"),hPricer1.calcGreek("Theta")},
                    hPricer0.calcGreek("Gamma"),hPricer1.calcGreek("Gamma")};
            double tmpV[2] = {calcGreek("Theta"),calcGreek("Gamma")};
            matrix nOptions = matrix(tmpM).inverse().dot(matrix(tmpV).transpose());
            double nOption0 = nOptions.getEntry(0,0),
                   nOption1 = nOptions.getEntry(1,0);
            double nStock = calcGreek("Delta")
                -nOption0*hPricer0.calcGreek("Delta")
                -nOption1*hPricer1.calcGreek("Delta");
            double cash = modPrice;
            cash -= nStock*S0+nOption0*O00+nOption1*O10;
            double value = cash+nStock*S0+nOption0*O00+nOption1*O10-modPrice;
            stratCashMatrix.setEntry(0,i,cash);
            stratNStockMatrix.setEntry(0,i,nStock);
            stratModPriceMatrix.setEntry(0,i,modPrice);
            stratModValueMatrix.setEntry(0,i,value);
            Delta = calcGreek("Delta"); hDelta0 = hPricer0.calcGreek("Delta"); hDelta1 = hPricer1.calcGreek("Delta");
            Gamma = calcGreek("Gamma"); hGamma0 = hPricer0.calcGreek("Gamma"); hGamma1 = hPricer1.calcGreek("Gamma");
            Vega = calcGreek("Vega"); hVega0 = hPricer0.calcGreek("Vega"); hVega1 = hPricer1.calcGreek("Vega");
            Rho = calcGreek("Rho"); hRho0 = hPricer0.calcGreek("Rho"); hRho1 = hPricer1.calcGreek("Rho");
            Theta = calcGreek("Theta"); hTheta0 = hPricer0.calcGreek("Theta"); hTheta1 = hPricer1.calcGreek("Theta");
            stratGrkDelta.setEntry(0,i,nStock+nOption0*hDelta0+nOption1*hDelta1-Delta);
            stratGrkGamma.setEntry(0,i,nOption0*hGamma0+nOption1*hGamma1-Gamma);
            stratGrkVega.setEntry(0,i,nOption0*hVega0+nOption1*hVega1-Vega);
            stratGrkRho.setEntry(0,i,nOption0*hRho0+nOption1*hRho1-Rho);
            stratGrkTheta.setEntry(0,i,r*cash+q*nStock*S0);
            stratNOptions[0].setEntry(0,i,nOption0);
            stratNOptions[1].setEntry(0,i,nOption1);
            stratHModPrices[0].setEntry(0,i,O00);
            stratHModPrices[1].setEntry(0,i,O10);
            for(int t=1; t<n; t++){
                double S = simPriceMatrix.getEntry(t,i);
                double nStockPrev = nStock;
                double nOptionPrev0 = nOption0;
                double nOptionPrev1 = nOption1;
                setVariable("currentPrice",S);
                setVariable("maturity",T-t*dt);
                hPricer0.setVariable("currentPrice",S);
                hPricer0.setVariable("maturity",Th0-t*dt);
                hPricer1.setVariable("currentPrice",S);
                hPricer1.setVariable("maturity",Th1-t*dt);
                if(!flatImpVolSurface){
                    idxT = (getVariable("maturity")-impVolSurfaceSet[1]).apply(abs).minIdx()[1];
                    idxT0 = (hPricer0.getVariable("maturity")-impVolSurfaceSet[1]).apply(abs).minIdx()[1];
                    idxT1 = (hPricer1.getVariable("maturity")-impVolSurfaceSet[1]).apply(abs).minIdx()[1];
                    setVariable("volatility",impVolSurfaceSet[2].getEntry(idxT,idxK));
                    hPricer0.setVariable("volatility",impVolSurfaceSet[2].getEntry(idxT0,idxK0));
                    hPricer1.setVariable("volatility",impVolSurfaceSet[2].getEntry(idxT1,idxK1));
                }
                double O0 = hPricer0.calcPrice("Closed Form");
                double O1 = hPricer1.calcPrice("Closed Form");
                modPrice = calcPrice("Closed Form");
                if(t%hedgeFreq==0){
                    double tmpM[2][2]
                        = {{hPricer0.calcGreek("Theta"),hPricer1.calcGreek("Theta")},
                            hPricer0.calcGreek("Gamma"),hPricer1.calcGreek("Gamma")};
                    double tmpV[2] = {calcGreek("Theta"),calcGreek("Gamma")};
                    nOptions = matrix(tmpM).inverse().dot(matrix(tmpV).transpose());
                    nOption0 = nOptions.getEntry(0,0),
                    nOption1 = nOptions.getEntry(1,0);
                    nStock = calcGreek("Delta")
                        -nOption0*hPricer0.calcGreek("Delta")
                        -nOption1*hPricer1.calcGreek("Delta");
                    cash = cash*riskFreeRateFactor
                        +nStock*S*(dividendYieldFactor-1)
                        -(nStock-nStockPrev)*S
                        -(nOption0-nOptionPrev0)*O0
                        -(nOption1-nOptionPrev1)*O1;
                }else{
                    cash = cash*riskFreeRateFactor
                        +nStockPrev*S*(dividendYieldFactor-1);
                }
                value = cash+nStock*S+nOption0*O0+nOption1*O1-modPrice;
                stratCashMatrix.setEntry(t,i,cash);
                stratNStockMatrix.setEntry(t,i,nStock);
                stratModPriceMatrix.setEntry(t,i,modPrice);
                stratModValueMatrix.setEntry(t,i,value);
                Delta = calcGreek("Delta"); hDelta0 = hPricer0.calcGreek("Delta"); hDelta1 = hPricer1.calcGreek("Delta");
                Gamma = calcGreek("Gamma"); hGamma0 = hPricer0.calcGreek("Gamma"); hGamma1 = hPricer1.calcGreek("Gamma");
                Vega = calcGreek("Vega"); hVega0 = hPricer0.calcGreek("Vega"); hVega1 = hPricer1.calcGreek("Vega");
                Rho = calcGreek("Rho"); hRho0 = hPricer0.calcGreek("Rho"); hRho1 = hPricer1.calcGreek("Rho");
                Theta = calcGreek("Theta"); hTheta0 = hPricer0.calcGreek("Theta"); hTheta1 = hPricer1.calcGreek("Theta");
                stratGrkDelta.setEntry(t,i,nStock+nOption0*hDelta0+nOption1*hDelta1-Delta);
                stratGrkGamma.setEntry(t,i,nOption0*hGamma0+nOption1*hGamma1-Gamma);
                stratGrkVega.setEntry(t,i,nOption0*hVega0+nOption1*hVega1-Vega);
                stratGrkRho.setEntry(t,i,nOption0*hRho0+nOption1*hRho1-Rho);
                stratGrkTheta.setEntry(t,i,r*cash+q*nStock*S);
                stratNOptions[0].setEntry(t,i,nOption0);
                stratNOptions[1].setEntry(t,i,nOption1);
                stratHModPrices[0].setEntry(t,i,O0);
                stratHModPrices[1].setEntry(t,i,O1);
            }
            double S1 = simPriceMatrix.getEntry(n,i);
            double nStockPrev = nStock;
            double nOptionPrev0 = nOption0;
            double nOptionPrev1 = nOption1;
            setVariable("currentPrice",S1);
            setVariable("maturity",0);
            hPricer0.setVariable("currentPrice",S1);
            hPricer0.setVariable("maturity",Th0-n*dt);
            hPricer1.setVariable("currentPrice",S1);
            hPricer1.setVariable("maturity",Th1-n*dt);
            if(!flatImpVolSurface){
                idxT = (getVariable("maturity")-impVolSurfaceSet[1]).apply(abs).minIdx()[1];
                idxT0 = (hPricer0.getVariable("maturity")-impVolSurfaceSet[1]).apply(abs).minIdx()[1];
                idxT1 = (hPricer1.getVariable("maturity")-impVolSurfaceSet[1]).apply(abs).minIdx()[1];
                setVariable("volatility",impVolSurfaceSet[2].getEntry(idxT,idxK));
                hPricer0.setVariable("volatility",impVolSurfaceSet[2].getEntry(idxT0,idxK0));
                hPricer1.setVariable("volatility",impVolSurfaceSet[2].getEntry(idxT1,idxK1));
            }
            double O01 = hPricer0.calcPrice("Closed Form");
            double O11 = hPricer1.calcPrice("Closed Form");
            modPrice = option.calcPayoff(S1);
            nStock = 0;
            nOption0 = 0;
            nOption1 = 0;
            cash = cash*riskFreeRateFactor+nStockPrev*S1+nOptionPrev0*O01+nOptionPrev1*O11;
            value = cash-modPrice;
            stratCashMatrix.setEntry(n,i,cash);
            stratNStockMatrix.setEntry(n,i,nStock);
            stratModPriceMatrix.setEntry(n,i,modPrice);
            stratModValueMatrix.setEntry(n,i,value);
            stratGrkDelta.setEntry(n,i,0);
            stratGrkGamma.setEntry(n,i,0);
            stratGrkVega.setEntry(n,i,0);
            stratGrkRho.setEntry(n,i,0);
            stratGrkTheta.setEntry(n,i,r*cash);
            stratNOptions[0].setEntry(n,i,nOption0);
            stratNOptions[1].setEntry(n,i,nOption1);
            stratHModPrices[0].setEntry(n,i,O01);
            stratHModPrices[1].setEntry(n,i,O11);
            if(GUI) cout << "[" << getCurrentTime() << "] simulation " <<
                i+1 << "/" << numSim << " completes" << endl;
        }
    }
    vector<matrix> results{
        simPriceMatrix,
        stratCashMatrix,
        stratNStockMatrix,
        stratModPriceMatrix,
        stratModValueMatrix,
        stratGrkDelta,
        stratGrkGamma,
        stratGrkVega,
        stratGrkRho,
        stratGrkTheta
    };
    vector<vector<matrix>> hResults{
        stratNOptions,
        stratHModPrices
    };
    Backtest backtest(results,hResults);
    return backtest;
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
