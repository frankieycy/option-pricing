// matrix library
#ifndef MATRIX
#define MATRIX
#include "util.cpp"

template <class T=double>
class matrix{
protected:
    int rows,cols;
    vector<vector<T>> m;
public:
    /**** constructors ****/
    matrix(); // default consructor
    matrix(const matrix<T>& M); // copy consructor
    matrix(int rows, int cols);
    matrix(const vector<T>& v);
    matrix(const vector<vector<T>>& M);
    template <int _rows> matrix(const T (&v)[_rows]);
    template <int _rows, int _cols> matrix(const T (&M)[_rows][_cols]);
    /**** accessors ****/
    int getRows() const {return rows;}
    int getCols() const {return cols;}
    vector<T> getRow(int row) const;
    vector<T> getCol(int col) const;
    vector<T> flatten() const;
    void print() const;
    /**** mutators ****/
    matrix setZero();
    matrix setZero(int rows, int cols);
    matrix setOne();
    matrix setOne(int rows, int cols);
    matrix setIdentity();
    matrix setIdentity(int rows, int cols);
    matrix setRandom(double min=0, double max=1);
    /**** matrix operations ****/
    matrix inverse(); // TO DO
    matrix transpose();
    /**** operators ****/
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
};

/**** constructors ****/

template <class T>
matrix<T>::matrix():rows(0),cols(0){}

template <class T>
matrix<T>::matrix(const matrix<T>& M):rows(M.rows),cols(M.cols),m(M.m){}

template <class T>
matrix<T>::matrix(int rows, int cols):rows(rows),cols(cols){
    for(int row=0; row<rows; row++) m.push_back(vector<T>(cols,0));
}

template <class T>
matrix<T>::matrix(const vector<T>& v):rows(v.size()),cols(1){
    for(int row=0; row<rows; row++) m.push_back(vector<T>({v[row]}));
}

template <class T>
matrix<T>::matrix(const vector<vector<T>>& M):rows(M.size()),cols(M[0].size()){
    for(int row=0; row<rows; row++) assert(M[row].size()==cols);
    m = M;
}

template <class T>
template <int _rows>
matrix<T>::matrix(const T (&v)[_rows]):rows(_rows),cols(1){
    for(int row=0; row<rows; row++) m.push_back(vector<T>({v[row]}));
}

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
vector<T> matrix<T>::getRow(int row) const {
    assert(row<rows);
    return m[row];
}

template <class T>
vector<T> matrix<T>::getCol(int col) const {
    assert(col<cols);
    vector<T> v;
    for(int row=0; row<rows; row++) v.push_back(m[row][col]);
    return v;
}

template <class T>
vector<T> matrix<T>::flatten() const {
    vector<T> v;
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            v.push_back(m[row][col]);
    return v;
}

template <class T>
void matrix<T>::print() const {
    for(int row=0; row<rows; row++) cout << m[row] << endl;
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
matrix<T> matrix<T>::setRandom(double min, double max){
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            m[row][col] = uniformRand(min,max);
    return *this;
}

/**** matrix operations ****/

template <class T>
matrix<T> matrix<T>::transpose(){
    matrix<T> A(cols,rows);
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            A.m[col][row] = m[row][col];
    return A;
}

/**** operators ****/

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
    assert(M1.cols==M2.rows);
    matrix<T> A(M1.rows,M2.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++){
            T dot = 0;
            for(int k=0; k<M1.cols; k++) dot += M1.m[row][k]*M2.m[k][col];
            A.m[row][col] = dot;
        }
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

#endif
