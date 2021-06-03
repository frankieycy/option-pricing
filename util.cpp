#ifndef UTIL
#define UTIL
#include <iostream>
#include <sstream>
#include <iomanip>
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
inline double normalPDF(double x, double mu=0, double sig=1){return exp(-(x-mu)*(x-mu)/(2*sig*sig))/sqrt(2*M_PI)/sig;}
inline double normalCDF(double x, double mu=0, double sig=1){return erfc(-M_SQRT1_2*x)/2;}

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

// string printJSON(string jsonStr, string padding){}

#endif
