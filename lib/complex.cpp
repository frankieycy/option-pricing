// complex number library
#ifndef COMPLEX
#define COMPLEX
#include "util.cpp"
using namespace std;

class complex{
private:
    double x,y; // real part: x, imaginary part: y
public:
    /**** constructors ****/
    complex():x(0),y(0){} // default constructor
    complex(double x):x(x),y(0){} // construct from a real number
    complex(double x, double y):x(x),y(y){} // construct from a complex number
    complex(const complex& c):x(c.x),y(c.y){} // copy constructor
    /**** accessors ****/
    void print(bool useRect) const;
    void print(int decPlaces, bool useRect) const;
    bool isPure() const {return (x==0 || y==0);}
    bool isReal() const {return (y==0);}
    bool isImag() const {return (x==0);}
    double getReal() const {return x;}
    double getImag() const {return y;}
    complex conjugate() const {return complex(x,-y);}
    double norm2() const {return x*x+y*y;}
    double modulus() const {return sqrt(norm2());}
    double argument() const {return atan(y/x);}
    /**** mutators ****/
    void setx(double x){this->x=x;}
    void sety(double y){this->y=y;}
    complex normalize();
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const complex& c);
    friend bool operator==(const complex& c1, const complex& c2);
    friend bool operator!=(const complex& c1, const complex& c2);
    friend complex operator+(const complex& c1, const complex& c2);
    friend complex operator+(const complex& c, double a);
    friend complex operator+(double a, const complex& c);
    friend complex operator+=(complex& c1, const complex& c2);
    friend complex operator+=(complex& c1, double a);
    friend complex operator-(const complex& c1, const complex& c2);
    friend complex operator-(const complex& c, double a);
    friend complex operator-(double a, const complex& c);
    friend complex operator-=(complex& c1, const complex& c2);
    friend complex operator-=(complex& c1, double a);
    friend complex operator-(const complex& c);
    friend complex operator*(const complex& c1, const complex& c2);
    friend complex operator*(const complex& c, double a);
    friend complex operator*(double a, const complex& c);
    friend complex operator*=(complex& c1, const complex& c2);
    friend complex operator*=(complex& c1, double a);
    friend complex operator/(const complex& c1, const complex& c2);
    friend complex operator/(const complex& c, double a);
    friend complex operator/(double a, const complex& c);
    friend complex operator/=(complex& c1, const complex& c2);
    friend complex operator/=(complex& c1, double a);
};

// IMPORTANT: var name i & I is reserved!
const complex
i = complex(0,1),
I = complex(1,0);

/**** accessors ****/

void complex::print(bool useRect=true) const {
    if(useRect) cout << *this; // rectangular form
    else cout << modulus() << "*e^(i" << argument() << ")"; // polar form
}

void complex::print(int decPlaces, bool useRect=true) const {
    // format decimal places
    if(useRect) cout << setprecision(decPlaces) << fixed << *this; // rectangular form
    else cout << setprecision(decPlaces) << fixed << modulus() << "*e^(i" << argument() << ")"; // polar form
}

/**** mutators ****/

complex complex::normalize(){
    // normalize complex number (unit modulus)
    double r = modulus();
    x /= r;
    y /= r;
    return *this;
}

/**** operators ****/

ostream& operator<<(ostream& out, const complex& c){
    if(c.isReal()) out << c.x;
    else if(c.isImag()) out << c.y << 'i';
    else if(c.y>0) out << c.x << '+' << c.y << 'i';
    else out << c.x << c.y << 'i';
    return out;
}

bool operator==(const complex& c1, const complex& c2){
    return (c1.x==c2.x) && (c1.y==c2.y);
}

bool operator!=(const complex& c1, const complex& c2){
    return !(c1==c2);
}

complex operator+(const complex& c1, const complex& c2){
    return complex(c1.x+c2.x,c1.y+c2.y);
}

complex operator+(const complex& c, double a){
    return complex(c.x+a,c.y);
}

complex operator+(double a, const complex& c){
    return c+a;
}

complex operator+=(complex& c1, const complex& c2){
    return (c1 = c1+c2);
}

complex operator+=(complex& c1, double a){
    return (c1 = c1+a);
}

complex operator-(const complex& c1, const complex& c2){
    return c1+(-c2);
}

complex operator-(const complex& c, double a){
    return c+(-a);
}

complex operator-(double a, const complex& c){
    return a+(-c);
}

complex operator-=(complex& c1, const complex& c2){
    return (c1 = c1-c2);
}

complex operator-=(complex& c1, double a){
    return (c1 = c1-a);
}

complex operator-(const complex& c){
    return complex(-c.x,-c.y);
}

complex operator*(const complex& c1, const complex& c2){
    return complex(c1.x*c2.x-c1.y*c2.y,c1.x*c2.y+c1.y*c2.x);
}

complex operator*(const complex& c, double a){
    return complex(c.x*a,c.y*a);
}

complex operator*(double a, const complex& c){
    return c*a;
}

complex operator*=(complex& c1, const complex& c2){
    return (c1 = c1*c2);
}

complex operator*=(complex& c1, double a){
    return (c1 = c1*a);
}

complex operator/(const complex& c1, const complex& c2){
    return c1*c2.conjugate()/c2.norm2();
}

complex operator/(const complex& c, double a){
    return c*(1/a);
}

complex operator/(double a, const complex& c){
    return complex(a,0)/c;
}

complex operator/=(complex& c1, const complex& c2){
    return (c1 = c1/c2);
}

complex operator/=(complex& c1, double a){
    return (c1 = c1/a);
}

#endif
