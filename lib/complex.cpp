// complx number library
#ifndef COMPLEX
#define COMPLEX
#include "util.cpp"
using namespace std;

class complx{
private:
    double x,y; // real part: x, imaginary part: y
public:
    /**** constructors ****/
    complx():x(0),y(0){} // default constructor
    complx(double x):x(x),y(0){} // construct from a real number
    complx(double x, double y):x(x),y(y){} // construct from a complx number
    complx(const complx& c):x(c.x),y(c.y){} // copy constructor
    /**** accessors ****/
    void print(bool useRect) const;
    void print(int decPlaces, bool useRect) const;
    bool isPure() const {return (x==0 || y==0);}
    bool isReal() const {return (y==0);}
    bool isImag() const {return (x==0);}
    double getReal() const {return x;}
    double getImag() const {return y;}
    complx conjugate() const {return complx(x,-y);}
    double norm2() const {return x*x+y*y;}
    double modulus() const {return sqrt(norm2());}
    double argument() const {return atan(y/x);}
    /**** mutators ****/
    void setx(double x){this->x=x;}
    void sety(double y){this->y=y;}
    complx normalize();
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const complx& c);
    friend bool operator==(const complx& c1, const complx& c2);
    friend bool operator!=(const complx& c1, const complx& c2);
    friend complx operator+(const complx& c1, const complx& c2);
    friend complx operator+(const complx& c, double a);
    friend complx operator+(double a, const complx& c);
    friend complx operator+=(complx& c1, const complx& c2);
    friend complx operator+=(complx& c1, double a);
    friend complx operator-(const complx& c1, const complx& c2);
    friend complx operator-(const complx& c, double a);
    friend complx operator-(double a, const complx& c);
    friend complx operator-=(complx& c1, const complx& c2);
    friend complx operator-=(complx& c1, double a);
    friend complx operator-(const complx& c);
    friend complx operator*(const complx& c1, const complx& c2);
    friend complx operator*(const complx& c, double a);
    friend complx operator*(double a, const complx& c);
    friend complx operator*=(complx& c1, const complx& c2);
    friend complx operator*=(complx& c1, double a);
    friend complx operator/(const complx& c1, const complx& c2);
    friend complx operator/(const complx& c, double a);
    friend complx operator/(double a, const complx& c);
    friend complx operator/=(complx& c1, const complx& c2);
    friend complx operator/=(complx& c1, double a);
    friend complx exp(complx c);
};

const complx i = complx(0,1);

/**** main ****/

complx exp(complx c){
    return exp(c.x)*complx(cos(c.y),sin(c.y));
}

/**** accessors ****/

void complx::print(bool useRect=true) const {
    if(useRect) cout << *this; // rectangular form
    else cout << modulus() << "*e^(i" << argument() << ")"; // polar form
}

void complx::print(int decPlaces, bool useRect=true) const {
    // format decimal places
    if(useRect) cout << setprecision(decPlaces) << fixed << *this; // rectangular form
    else cout << setprecision(decPlaces) << fixed << modulus() << "*e^(i" << argument() << ")"; // polar form
}

/**** mutators ****/

complx complx::normalize(){
    // normalize complx number (unit modulus)
    double r = modulus();
    x /= r;
    y /= r;
    return *this;
}

/**** operators ****/

ostream& operator<<(ostream& out, const complx& c){
    if(c.isReal()) out << c.x;
    else if(c.isImag()) out << c.y << 'i';
    else if(c.y>0) out << c.x << '+' << c.y << 'i';
    else out << c.x << c.y << 'i';
    return out;
}

bool operator==(const complx& c1, const complx& c2){
    return (c1.x==c2.x) && (c1.y==c2.y);
}

bool operator!=(const complx& c1, const complx& c2){
    return !(c1==c2);
}

complx operator+(const complx& c1, const complx& c2){
    return complx(c1.x+c2.x,c1.y+c2.y);
}

complx operator+(const complx& c, double a){
    return complx(c.x+a,c.y);
}

complx operator+(double a, const complx& c){
    return c+a;
}

complx operator+=(complx& c1, const complx& c2){
    return (c1 = c1+c2);
}

complx operator+=(complx& c1, double a){
    return (c1 = c1+a);
}

complx operator-(const complx& c1, const complx& c2){
    return c1+(-c2);
}

complx operator-(const complx& c, double a){
    return c+(-a);
}

complx operator-(double a, const complx& c){
    return a+(-c);
}

complx operator-=(complx& c1, const complx& c2){
    return (c1 = c1-c2);
}

complx operator-=(complx& c1, double a){
    return (c1 = c1-a);
}

complx operator-(const complx& c){
    return complx(-c.x,-c.y);
}

complx operator*(const complx& c1, const complx& c2){
    return complx(c1.x*c2.x-c1.y*c2.y,c1.x*c2.y+c1.y*c2.x);
}

complx operator*(const complx& c, double a){
    return complx(c.x*a,c.y*a);
}

complx operator*(double a, const complx& c){
    return c*a;
}

complx operator*=(complx& c1, const complx& c2){
    return (c1 = c1*c2);
}

complx operator*=(complx& c1, double a){
    return (c1 = c1*a);
}

complx operator/(const complx& c1, const complx& c2){
    return c1*c2.conjugate()/c2.norm2();
}

complx operator/(const complx& c, double a){
    return c*(1/a);
}

complx operator/(double a, const complx& c){
    return complx(a,0)/c;
}

complx operator/=(complx& c1, const complx& c2){
    return (c1 = c1/c2);
}

complx operator/=(complx& c1, double a){
    return (c1 = c1/a);
}

#endif
