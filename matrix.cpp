// matrix library
#ifndef MATRIX
#define MATRIX
#include "util.cpp"

template <class T=double> // bool int double
class matrix{
protected:
    int rows,cols;
    vector<vector<T>> m;
public:
    /**** constructors ****/
    matrix(); // default consructor
    matrix(const matrix<T>& M); // copy consructor
    matrix(int rows, int cols, T a=0);
    matrix(const vector<T>& v);
    matrix(const vector<vector<T>>& M);
    template <int _rows> matrix(const T (&v)[_rows]);
    template <int _rows, int _cols> matrix(const T (&M)[_rows][_cols]);
    /**** accessors ****/
    bool isEmpty() const;
    int getRows() const {return rows;}
    int getCols() const {return cols;}
    T getEntry(int row, int col) const {return m[row][col];}
    T getFirstEntry() const;
    T getLastEntry() const;
    matrix getRow(int row) const;
    matrix getCol(int col) const;
    matrix getFirstRow() const;
    matrix getLastRow() const;
    matrix getFirstCol() const;
    matrix getLastCol() const;
    matrix flatten() const;
    string print() const;
    string getAsJSONArray() const;
    /**** mutators ****/
    matrix setZero();
    matrix setZero(int rows, int cols);
    matrix setOne();
    matrix setOne(int rows, int cols);
    matrix setIdentity();
    matrix setIdentity(int rows, int cols);
    matrix setUniformRand(double min=0, double max=1);
    matrix setNormalRand(double mu=0, double sig=1);
    matrix setRow(int row, const matrix<T>& vec);
    matrix setCol(int col, const matrix<T>& vec);
    matrix setEntry(int row, int col, T a);
    /**** matrix operations ****/
    T getMax();
    T getMin();
    matrix maxWith(T a);
    matrix minWith(T a);
    double sum();
    double mean();
    matrix sum(int axis);
    matrix mean(int axis);
    matrix inverse(); // TO DO
    matrix transpose();
    matrix dot(const matrix<T>& M);
    /**** operators ****/
    template <class t> friend ostream& operator<<(ostream& out, const matrix<t>& M);
    template <class t> friend bool operator==(const matrix<t>& M1, const matrix<t>& M2);
    template <class t> friend bool operator!=(const matrix<t>& M1, const matrix<t>& M2);
    template <class t> friend matrix<t> operator+(const matrix<t>& M1, const matrix<t>& M2);
    template <class t1, class t2> friend matrix<t1> operator+(const matrix<t1>& M, const t2& a);
    template <class t1, class t2> friend matrix<t1> operator+(const t2& a, const matrix<t1>& M);
    template <class t> friend matrix<t> operator+=(matrix<t>& M1, const matrix<t>& M2);
    template <class t1, class t2> friend matrix<t1> operator+=(matrix<t1>& M, const t2& a);
    template <class t> friend matrix<t> operator-(const matrix<t>& M1, const matrix<t>& M2);
    template <class t1, class t2> friend matrix<t1> operator-(const matrix<t1>& M, const t2& a);
    template <class t1, class t2> friend matrix<t1> operator-(const t2& a, const matrix<t1>& M);
    template <class t> friend matrix<t> operator-=(matrix<t>& M1, const matrix<t>& M2);
    template <class t1, class t2> friend matrix<t1> operator-=(matrix<t1>& M, const t2& a);
    template <class t> friend matrix<t> operator-(const matrix<t>& M);
    template <class t> friend matrix<t> operator*(const matrix<t>& M1, const matrix<t>& M2);
    template <class t1, class t2> friend matrix<t1> operator*(const matrix<t1>& M, const t2& a);
    template <class t1, class t2> friend matrix<t1> operator*(const t2& a, const matrix<t1>& M);
    template <class t> friend matrix<t> operator*=(matrix<t>& M1, const matrix<t>& M2);
    template <class t1, class t2> friend matrix<t1> operator*=(matrix<t1>& M, const t2& a);
    template <class t1, class t2> friend matrix<t1> operator/(const matrix<t1>& M, const t2& a);
    template <class t1, class t2> friend matrix<t1> operator/=(matrix<t1>& M, const t2& a);
    template <class t1, class t2> friend matrix<t1> operator>(const matrix<t1>& M, const t2& a);
    template <class t1, class t2> friend matrix<t1> operator>=(const matrix<t1>& M, const t2& a);
    template <class t1, class t2> friend matrix<t1> operator<(const matrix<t1>& M, const t2& a);
    template <class t1, class t2> friend matrix<t1> operator<=(const matrix<t1>& M, const t2& a);
};

const matrix<double> NULL_VECTOR, NULL_MATRIX;

/**** constructors ****/

template <class T>
matrix<T>::matrix():rows(0),cols(0){}

template <class T>
matrix<T>::matrix(const matrix<T>& M):rows(M.rows),cols(M.cols),m(M.m){}

template <class T>
matrix<T>::matrix(int rows, int cols, T a):rows(rows),cols(cols){
    for(int row=0; row<rows; row++) m.push_back(vector<T>(cols,a));
}

template <class T>
matrix<T>::matrix(const vector<T>& v):rows(1),cols(v.size()){
    m.push_back(v);
}

// template <class T>
// matrix<T>::matrix(const vector<T>& v):rows(v.size()),cols(1){
//     for(int row=0; row<rows; row++) m.push_back(vector<T>({v[row]}));
// }

template <class T>
matrix<T>::matrix(const vector<vector<T>>& M):rows(M.size()),cols(M[0].size()){
    for(int row=0; row<rows; row++) assert(M[row].size()==cols);
    m = M;
}

template <class T>
template <int _rows>
matrix<T>::matrix(const T (&v)[_rows]):rows(1),cols(_rows){
    m.push_back(vector<T>(v));
}

// template <class T>
// template <int _rows>
// matrix<T>::matrix(const T (&v)[_rows]):rows(_rows),cols(1){
//     for(int row=0; row<rows; row++) m.push_back(vector<T>({v[row]}));
// }

template <class T>
template <int _rows, int _cols>
matrix<T>::matrix(const T (&M)[_rows][_cols]):rows(_rows),cols(_cols){
    for(int row=0; row<rows; row++){
        vector<T> v;
        for(int col=0; col<cols; col++) v.push_back(M[row][col]);
        m.push_back(v);
    }
}

/**** accessors ****/

template <class T>
bool matrix<T>::isEmpty() const {
    return m.empty();
}

template <class T>
T matrix<T>::getFirstEntry() const {
    assert(!isEmpty());
    return m[0][0];
}

template <class T>
T matrix<T>::getLastEntry() const {
    assert(!isEmpty());
    return m[rows-1][cols-1];
}

template <class T>
matrix<T> matrix<T>::getRow(int row) const {
    assert(row>=0 && row<rows);
    return matrix(m[row]);
}

template <class T>
matrix<T> matrix<T>::getCol(int col) const {
    assert(col>=0 && col<cols);
    vector<T> v;
    for(int row=0; row<rows; row++) v.push_back(m[row][col]);
    return matrix(v);
}

template <class T>
matrix<T> matrix<T>::getFirstRow() const {
    return getRow(0);
}

template <class T>
matrix<T> matrix<T>::getLastRow() const {
    return getRow(rows-1);
}

template <class T>
matrix<T> matrix<T>::getFirstCol() const {
    return getCol(0);
}

template <class T>
matrix<T> matrix<T>::getLastCol() const {
    return getCol(cols-1);
}

template <class T>
matrix<T> matrix<T>::flatten() const {
    vector<T> v;
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            v.push_back(m[row][col]);
    return matrix(v);
}

template <class T>
string matrix<T>::print() const {
    ostringstream oss;
    for(int row=0; row<rows; row++)
        oss << ((row==0)?"[":" ") << m[row] << ((row==rows-1)?"]":",") << ((row<rows-1)?"\n":"");
    return oss.str();
}

template <class T>
string matrix<T>::getAsJSONArray() const {
    ostringstream oss;
    for(int row=0; row<rows; row++)
        oss << ((row==0)?"[":"") << m[row] << ((row==rows-1)?"]":",");
    return oss.str();
}

/**** mutators ****/

template <class T>
matrix<T> matrix<T>::setZero(){
    *this = matrix<T>(rows,cols);
    return *this;
}

template <class T>
matrix<T> matrix<T>::setZero(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    *this = matrix<T>(rows,cols);
    return *this;
}

template <class T>
matrix<T> matrix<T>::setOne(){
    *this = matrix<T>(rows,cols)+1;
    return *this;
}

template <class T>
matrix<T> matrix<T>::setOne(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    *this = matrix<T>(rows,cols).setOne();
    return *this;
}

template <class T>
matrix<T> matrix<T>::setIdentity(){
    assert(rows==cols);
    setZero();
    for(int row=0; row<rows; row++) m[row][row] = 1;
    return *this;
}

template <class T>
matrix<T> matrix<T>::setIdentity(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    *this = matrix<T>(rows,cols).setIdentity();
    return *this;
}

template <class T>
matrix<T> matrix<T>::setUniformRand(double min, double max){
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            m[row][col] = uniformRand(min,max);
    return *this;
}

template <class T>
matrix<T> matrix<T>::setNormalRand(double mu, double sig){
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            m[row][col] = normalRand(mu,sig);
    return *this;
}

template <class T>
matrix<T> matrix<T>::setRow(int row, const matrix<T>& vec){
    assert(vec.rows==1);
    for(int col=0; col<vec.cols; col++) m[row][col] = vec.m[0][col];
    return *this;
}

template <class T>
matrix<T> matrix<T>::setCol(int col, const matrix<T>& vec){
    assert(vec.cols==1);
    for(int row=0; row<vec.cols; row++) m[row][col] = vec.m[row][0];
    return *this;
}

template <class T>
matrix<T> matrix<T>::setEntry(int row, int col, T a){
    assert(row>=0 && row<rows);
    assert(col>=0 && col<cols);
    m[row][col] = a;
    return *this;
}

/**** matrix operations ****/

template <class T>
T matrix<T>::getMax(){
    assert(!isEmpty());
    T a = m[0][0];
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            a = max(a, m[row][col]);
    return a;
}

template <class T>
T matrix<T>::getMin(){
    assert(!isEmpty());
    T a = m[0][0];
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            a = min(a, m[row][col]);
    return a;
}

template <class T>
matrix<T> matrix<T>::maxWith(T a){
    matrix<T> A(rows,cols);
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            A.m[row][col] = max(a, m[row][col]);
    return A;
}

template <class T>
matrix<T> matrix<T>::minWith(T a){
    matrix<T> A(rows,cols);
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            A.m[row][col] = min(a, m[row][col]);
    return A;
}

template <class T>
double matrix<T>::sum(){
    double a = 0;
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            a += m[row][col];
    return a;
}

template <class T>
double matrix<T>::mean(){
    return sum()/(rows*cols);
}

template <class T>
matrix<T> matrix<T>::sum(int axis){
    vector<T> v;
    switch(axis){
        case 1:
        for(int row=0; row<rows; row++) v.push_back(getRow(row).sum());
        case 2:
        for(int col=0; col<cols; col++) v.push_back(getCol(col).sum());
    }
    return matrix(v);
}

template <class T>
matrix<T> matrix<T>::mean(int axis){
    vector<T> v;
    switch(axis){
        case 1:
        for(int row=0; row<rows; row++) v.push_back(getRow(row).mean());
        case 2:
        for(int col=0; col<cols; col++) v.push_back(getCol(col).mean());
    }
    return matrix(v);
}

template <class T>
matrix<T> matrix<T>::transpose(){
    matrix<T> A(cols,rows);
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            A.m[col][row] = m[row][col];
    return A;
}

template <class T>
matrix<T> matrix<T>::dot(const matrix<T>& M){
    matrix<T> A(rows,M.cols);
    for(int row=0; row<rows; row++)
        for(int col=0; col<M.cols; col++){
            T dot = 0;
            for(int k=0; k<cols; k++) dot += m[row][k]*M.m[k][col];
            A.m[row][col] = dot;
        }
    return A;
}

/**** operators ****/

template <class T>
ostream& operator<<(ostream& out, const matrix<T>& M){
    out << M.getAsJSONArray();
    return out;
}

template <class T>
bool operator==(const matrix<T>& M1, const matrix<T>& M2){
    return (M1.m==M2.m);
}

template <class T>
bool operator!=(const matrix<T>& M1, const matrix<T>& M2){
    return !(M1==M2);
}

template <class T>
matrix<T> operator+(const matrix<T>& M1, const matrix<T>& M2){
    assert(M1.rows==M2.rows && M1.cols==M2.cols);
    matrix<T> A = M1;
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] += M2.m[row][col];
    return A;
}

template <class T1, class T2>
matrix<T1> operator+(const matrix<T1>& M, const T2& a){
    matrix<T1> A = M;
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] += static_cast<T1>(a);
    return A;
}

template <class T1, class T2>
matrix<T1> operator+(const T2& a, const matrix<T1>& M){
    return M+a;
}

template <class T>
matrix<T> operator+=(matrix<T>& M1, const matrix<T>& M2){
    return (M1 = M1+M2);
}

template <class T1, class T2>
matrix<T1> operator+=(matrix<T1>& M, const T2& a){
    return (M = M+a);
}

template <class T>
matrix<T> operator-(const matrix<T>& M1, const matrix<T>& M2){
    return M1+(-M2);
}

template <class T1, class T2>
matrix<T1> operator-(const matrix<T1>& M, const T2& a){
    return M+(-a);
}

template <class T1, class T2>
matrix<T1> operator-(const T2& a, const matrix<T1>& M){
    return a+(-M);
}

template <class T>
matrix<T> operator-=(matrix<T>& M1, const matrix<T>& M2){
    return (M1 = M1-M2);
}

template <class T1, class T2>
matrix<T1> operator-=(matrix<T1>& M, const T2& a){
    return (M = M-a);
}

template <class T>
matrix<T> operator-(const matrix<T>& M){
    return M*(-1);
}

template <class T>
matrix<T> operator*(const matrix<T>& M1, const matrix<T>& M2){
    assert(M1.rows==M2.rows && M1.cols==M2.cols);
    matrix<T> A(M1.rows,M1.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M1.m[row][col]*M2.m[row][col];
    return A;
}

template <class T1, class T2>
matrix<T1> operator*(const matrix<T1>& M, const T2& a){
    matrix<T1> A = M;
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] *= static_cast<T1>(a);
    return A;
}

template <class T1, class T2>
matrix<T1> operator*(const T2& a, const matrix<T1>& M){
    return M*a;
}

template <class T>
matrix<T> operator*=(matrix<T>& M1, const matrix<T>& M2){
    return (M1 = M1*M2);
}

template <class T1, class T2>
matrix<T1> operator*=(matrix<T1>& M, const T2& a){
    return (M = M*a);
}

template <class T1, class T2>
matrix<T1> operator/(const matrix<T1>& M, const T2& a){
    matrix<T1> A = M;
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] /= static_cast<T1>(a);
    return A;
}

template <class T1, class T2>
matrix<T1> operator/=(matrix<T1>& M, const T2& a){
    return (M = M/a);
}

template <class T1, class T2>
matrix<T1> operator>(const matrix<T1>& M, const T2& a){
    matrix<T1> A = M;
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = static_cast<T1>(M.m[row][col]>a);
    return A;
}

template <class T1, class T2>
matrix<T1> operator>=(const matrix<T1>& M, const T2& a){
    matrix<T1> A = M;
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = static_cast<T1>(M.m[row][col]>=a);
    return A;
}

template <class T1, class T2>
matrix<T1> operator<(const matrix<T1>& M, const T2& a){
    matrix<T1> A = M;
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = static_cast<T1>(M.m[row][col]<a);
    return A;
}

template <class T1, class T2>
matrix<T1> operator<=(const matrix<T1>& M, const T2& a){
    matrix<T1> A = M;
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = static_cast<T1>(M.m[row][col]<=a);
    return A;
}

#endif
