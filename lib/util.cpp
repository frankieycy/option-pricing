#ifndef UTIL
#define UTIL
#include <iostream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <set>
using namespace std;

inline void seperator(int length=20){cout << string(length,'-') << endl;}
inline double uniformRand(double min=0, double max=1){return min+(max-min)*rand()/RAND_MAX;}
inline double normalRand(double mu=0, double sig=1){return sqrt(-2*log(uniformRand()))*cos(2*M_PI*uniformRand());}
inline double normalPDF(double x, double mu=0, double sig=1){return exp(-(x-mu)*(x-mu)/(2*sig*sig))/(sqrt(2*M_PI)*sig);}
inline double normalCDF(double x, double mu=0, double sig=1){return erfc(-M_SQRT1_2*x)/2;}
inline double stdNormalPDF(double x){return normalPDF(x);}
inline double stdNormalCDF(double x){return normalCDF(x);}

double normalRand_(double mu=0, double sig=1){
    double n = normalRand(mu,sig);
    while(!isfinite(n)) n = normalRand(mu,sig);
    return n;
}

double lognormalPDF(double x, double mu=0, double sig=1){
    double log_x = log(x);
    return exp(-(log_x-mu)*(log_x-mu)/(2*sig*sig))/(sqrt(2*M_PI)*sig*x);
}

double mathFunc(double x, string type, vector<double> vec){
    double fx = NAN;
    if(type=="const"){
        double a = vec[0];
        fx = a;
    }else if(type=="linear"){
        double a = vec[0],
               b = vec[1];
        fx = a+b*x;
    }else if(type=="quadratic"){
        double a = vec[0],
               b = vec[1],
               c = vec[2];
        fx = a+b*x+c*x*x;
    }else if(type=="exponential"){
        double a = vec[0],
               b = vec[1],
               c = vec[2];
        fx = a+b*exp(-x/c);
    }
    return fx;
}

template <class T1, class T2>
vector<T1> apply(const function<T1(T2)>& f, const vector<T2>& vec){
    vector<T1> v(vec.size());
    for(int k=0; k<v.size(); k++) v[k] = f(vec[k]);
    return v;
}

template <class T>
ostream& operator<<(ostream& out, const vector<T>& vec){
    // print elements of a vector
    if(vec.size()==0) out << "[]";
    else{
        out << "[";
        for(auto p=vec.begin(); p!=vec.end(); p++)
            if(is_same<T,string>::value) out << "\"" << *p << "\"" << ((p==vec.end()-1)?"]":",");
            else out << *p << ((p==vec.end()-1)?"]":",");
    }
    return out;
}

template <class T>
string to_string(T obj){
    ostringstream oss;
    oss << obj;
    return oss.str();
}

bool isDouble(string str, double& v) {
    char* end;
    v = strtod(str.c_str(),&end);
    if(end==str.c_str() || *end!='\0') return false;
    return true;
}

string getCurrentTime(){
    time_t t = time(0);
    char time[100];
    strftime(time,100,"%Y%m%d %T",localtime(&t));
    return time;
}

string joinStr(vector<string> vec, string join=","){
    string str = "";
    if(vec.size()>0){
        for(auto p=vec.begin(); p!=vec.end(); p++) str += *p+((p==vec.end()-1)?"":join);
    }
    return str;
}

// string printJSON(string jsonStr, string padding){}

#endif
