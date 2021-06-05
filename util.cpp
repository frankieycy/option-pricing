#ifndef UTIL
#define UTIL
#include <iostream>
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

double lognormalPDF(double x, double mu=0, double sig=1){
    double log_x = log(x);
    return exp(-(log_x-mu)*(log_x-mu)/(2*sig*sig))/(sqrt(2*M_PI)*sig*x);
}

template <class T>
ostream& operator<<(ostream& out, const vector<T>& vec){
    // print elements of a vector
    if(vec.size()==0) out << "[]";
    else{
        out << "[";
        for(auto p=vec.begin(); p!=vec.end(); p++) out << *p << ((p==vec.end()-1)?"]":",");
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

// string printJSON(string jsonStr, string padding){}

#endif
