// matrix library
#ifndef MATRIX
#define MATRIX
#include "util.cpp"
#include <Eigen/Dense>
using namespace Eigen;
typedef Matrix<double,Dynamic,Dynamic,RowMajor> RowMatrixXd;

class matrix{
protected:
    int rows,cols;
    vector<vector<double>> m;
public:
    /**** constructors ****/
    matrix(); // default consructor
    matrix(const matrix& M); // copy consructor
    matrix(int rows, int cols, double a=0);
    matrix(int rows, int cols, string type, const vector<double>& params={});
    matrix(const vector<double>& v);
    matrix(const vector<vector<double>>& M);
    matrix(int rows, int cols, double *M);
    matrix(int rows, int cols, double **M);
    template <int _cols> matrix(const double (&v)[_cols]);
    template <int _rows, int _cols> matrix(const double (&M)[_rows][_cols]);
    /**** accessors ****/
    bool isEmpty() const;
    int getRows() const {return rows;}
    int getCols() const {return cols;}
    int getEntries() const {return rows*cols;}
    double getEntry(int row, int col) const {return m[row][col];}
    double getEntry(const vector<int>& idx) const {return m[idx[0]][idx[1]];}
    double getFirstEntry() const;
    double getLastEntry() const;
    matrix getRow(int row) const;
    vector<double> getRowVector(int row) const;
    matrix getCol(int col) const;
    vector<double> getColVector(int col) const;
    matrix submatrix(int row0=0, int row1=-1, int col0=0, int col1=-1) const;
    matrix submatrix(int i0, int i1, string mode) const;
    matrix getFirstRow() const;
    matrix getLastRow() const;
    matrix getFirstCol() const;
    matrix getLastCol() const;
    matrix flatten() const;
    string print() const;
    string getAsCsv() const;
    string getAsJson() const;
    void printToCsvFile(string file, string header="") const;
    void printToJsonFile(string file) const;
    /**** mutators ****/
    matrix setZero();
    matrix setZero(int rows, int cols);
    matrix setOne();
    matrix setOne(int rows, int cols);
    matrix setIdentity();
    matrix setIdentity(int rows);
    matrix setUniformRand(double min=0, double max=1);
    matrix setNormalRand(double mu=0, double sig=1);
    matrix setPoissonRand(double lambda=1);
    matrix setRow(int row, const matrix& vec);
    matrix setCol(int col, const matrix& vec);
    matrix setEntry(int row, int col, double a);
    matrix setSubmatrix(int row0, int row1, int col0, int col1, const matrix& M);
    matrix setDiags(const vector<double>& vec, const vector<int>& diags);
    matrix setRange(double x0, double x1, int n=-1, bool inc=false);
    /**** matrix operations ****/
    double trace() const;
    double getMax() const;
    double getMin() const;
    vector<int> maxIdx() const;
    vector<int> minIdx() const;
    vector<int> find(double a) const;
    vector<int> find(double a, string method) const;
    matrix maxWith(double a) const;
    matrix minWith(double a) const;
    double sum() const;
    double sum(const vector<double>& weights) const;
    double sum(const matrix& weights) const;
    double prod() const;
    double mean(string method="Arithmetic") const;
    double wmean(const vector<double>& weights, string method="Arithmetic") const;
    double var(int k=1) const;
    double stdev(int k=1) const;
    double cov(const matrix& M, int k=1) const;
    double cor(const matrix& M, int k=1) const;
    matrix sum(int axis) const;
    matrix mean(int axis, string method="Arithmetic") const;
    matrix inverse() const;
    matrix transpose() const;
    matrix T() const {return transpose();}
    matrix dot(const matrix& M) const;
    matrix apply(double (*f)(double)) const;
    matrix apply(const function<double(double)>& f) const;
    matrix sample(int n, bool replace=true) const;
    matrix concat(const matrix& M, int axis=0) const;
    matrix chol() const; // Choleskey
    /**** operators ****/
    friend ostream& operator<<(ostream& out, const matrix& M);
    friend bool operator==(const matrix& M1, const matrix& M2);
    friend bool operator!=(const matrix& M1, const matrix& M2);
    friend matrix operator+(const matrix& M1, const matrix& M2);
    friend matrix operator+(const matrix& M, double a);
    friend matrix operator+(double a, const matrix& M);
    friend matrix operator+=(matrix& M1, const matrix& M2);
    friend matrix operator+=(matrix& M, double a);
    friend matrix operator-(const matrix& M1, const matrix& M2);
    friend matrix operator-(const matrix& M, double a);
    friend matrix operator-(double a, const matrix& M);
    friend matrix operator-=(matrix& M1, const matrix& M2);
    friend matrix operator-=(matrix& M, double a);
    friend matrix operator-(const matrix& M);
    friend matrix operator*(const matrix& M1, const matrix& M2);
    friend matrix operator*(const matrix& M, double a);
    friend matrix operator*(double a, const matrix& M);
    friend matrix operator*=(matrix& M1, const matrix& M2);
    friend matrix operator*=(matrix& M, double a);
    friend matrix operator/(const matrix& M1, const matrix& M2);
    friend matrix operator/(const matrix& M, double a);
    friend matrix operator/=(matrix& M, double a);
    friend matrix operator>(const matrix& M, double a);
    friend matrix operator>=(const matrix& M, double a);
    friend matrix operator<(const matrix& M, double a);
    friend matrix operator<=(const matrix& M, double a);
    friend matrix operator||(const matrix& M1, const matrix& M2);
    friend matrix operator&&(const matrix& M1, const matrix& M2);
    friend matrix max(const matrix& M1, const matrix& M2);
    friend matrix min(const matrix& M1, const matrix& M2);
};

const matrix NULL_VECTOR, NULL_MATRIX;

matrix exp(const matrix& M){
    return M.apply(exp);
}

matrix abs(const matrix& M){
    return M.apply(abs);
}

matrix sqrt(const matrix& M){
    return M.apply(sqrt);
}

matrix pow(double a, const matrix& M){
    auto f = [a](double x){return pow(a,x);};
    return M.apply(f);
}

matrix pow(const matrix& M, double a){
    auto f = [a](double x){return pow(x,a);};
    return M.apply(f);
}

double max(const matrix& M){
    return M.getMax();
}

matrix max(const matrix& M, double a){
    return M.maxWith(a);
}

matrix max(const matrix& M1, const matrix& M2){
    assert(M1.rows==M2.rows && M1.cols==M2.cols);
    matrix A = M1;
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = max(M1.m[row][col],M2.m[row][col]);
    return A;
}

double min(const matrix& M){
    return M.getMin();
}

matrix min(const matrix& M, double a){
    return M.minWith(a);
}

matrix min(const matrix& M1, const matrix& M2){
    assert(M1.rows==M2.rows && M1.cols==M2.cols);
    matrix A = M1;
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = min(M1.m[row][col],M2.m[row][col]);
    return A;
}

double interp(const vector<double>& x, const vector<matrix>& coord, const matrix& data, string method="closest"){
    int n = x.size();
    assert(n==coord.size());
    assert(n==1||n==2);
    vector<int> idx;
    if(method=="exact" || method=="closest"){
        if(n==1) idx = {0,coord[0].find(x[0],method)[1]};
        else idx = {coord[0].find(x[0],method)[1],coord[1].find(x[1],method)[1]};
        return data.getEntry(idx);
    }else if(method=="linear"){
        if(n==1){
            int m0 = coord[0].getCols();
            double x0 = x[0];
            for(int k=1; k<m0; k++){
                double c0 = coord[0].getEntry(0,k-1);
                double c1 = coord[0].getEntry(0,k);
                double d0 = data.getEntry(0,k-1);
                double d1 = data.getEntry(0,k);
                if(x0<=c1) return d0+(d1-d0)*(x0-c0)/(c1-c0);
            }
        }else{
            int m0 = coord[0].getCols();
            int m1 = coord[1].getCols();
            double x0 = x[0]; double x1 = x[1];
            for(int k=1; k<m0; k++){
                for(int l=1; l<m1; l++){
                    double c00 = coord[0].getEntry(0,k-1);
                    double c01 = coord[0].getEntry(0,k);
                    double c10 = coord[1].getEntry(0,l-1);
                    double c11 = coord[1].getEntry(0,l);
                    double d00 = data.getEntry(k-1,l-1);
                    double d10 = data.getEntry(k,l-1);
                    double d01 = data.getEntry(k-1,l);
                    double d11 = data.getEntry(k,l);
                    if(x0<=c01 && x1<=c11){
                        matrix c0({c01-x0,x0-c00});
                        matrix c1({c11-x1,x1-c10});
                        matrix d({{d00,d10},{d01,d11}});
                        return c0.dot(d).dot(c1).getEntry(0,0)/((c01-c00)*(c11-c10));
                    }
                }
            }
        }
    }
    return NAN;
}

/**** constructors ****/

matrix::matrix():rows(0),cols(0){}

matrix::matrix(const matrix& M):rows(M.rows),cols(M.cols),m(M.m){}

matrix::matrix(int rows, int cols, double a):rows(rows),cols(cols){
    for(int row=0; row<rows; row++) m.push_back(vector<double>(cols,a));
}

matrix::matrix(int rows, int cols, string type, const vector<double>& params):rows(rows),cols(cols){
    *this = matrix(rows,cols);
    if(type=="identity") (*this).setIdentity();
    else if(type=="uniform rand") (*this).setUniformRand(params[0],params[1]);
    else if(type=="normal rand") (*this).setNormalRand(params[0],params[1]);
    else if(type=="poisson rand") (*this).setPoissonRand(params[0]);
}

matrix::matrix(const vector<double>& v):rows(1),cols(v.size()){
    m.push_back(v);
}

// matrix::matrix(const vector<double>& v):rows(v.size()),cols(1){
//     for(int row=0; row<rows; row++) m.push_back(vector<double>({v[row]}));
// }

matrix::matrix(const vector<vector<double>>& M):rows(M.size()),cols(M[0].size()){
    for(int row=0; row<rows; row++) assert(M[row].size()==cols);
    m = M;
}

matrix::matrix(int rows, int cols, double *M):rows(rows),cols(cols){
    for(int row=0; row<rows; row++){
        vector<double> vec;
        vec.assign(M+row*cols,M+(row+1)*cols);
        m.push_back(vec);
    }
}

matrix::matrix(int rows, int cols, double **M):rows(rows),cols(cols){
    for(int row=0; row<rows; row++){
        vector<double> vec;
        vec.assign(M[row],M[row]+cols);
        m.push_back(vec);
    }
}

template <int _cols>
matrix::matrix(const double (&v)[_cols]):rows(1),cols(_cols){
    vector<double> vec;
    for(int col=0; col<cols; col++) vec.push_back(v[col]);
    m.push_back(vec);
}

// template <int _rows>
// matrix::matrix(const double (&v)[_rows]):rows(_rows),cols(1){
//     for(int row=0; row<rows; row++) m.push_back(vector<double>({v[row]}));
// }

template <int _rows, int _cols>
matrix::matrix(const double (&M)[_rows][_cols]):rows(_rows),cols(_cols){
    for(int row=0; row<rows; row++){
        vector<double> vec;
        for(int col=0; col<cols; col++) vec.push_back(M[row][col]);
        m.push_back(vec);
    }
}

/**** accessors ****/

bool matrix::isEmpty() const {
    return m.empty();
}

double matrix::getFirstEntry() const {
    assert(!isEmpty());
    return m[0][0];
}

double matrix::getLastEntry() const {
    assert(!isEmpty());
    return m[rows-1][cols-1];
}

matrix matrix::getRow(int row) const {
    return matrix(getRowVector(row));
}

vector<double> matrix::getRowVector(int row) const {
    assert(row>=0 && row<rows);
    return m[row];
}

matrix matrix::getCol(int col) const {
    return matrix(getColVector(col));
}

vector<double> matrix::getColVector(int col) const {
    assert(col>=0 && col<cols);
    vector<double> v;
    for(int row=0; row<rows; row++) v.push_back(m[row][col]);
    return v;
}

matrix matrix::submatrix(int row0, int row1, int col0, int col1) const {
    if(row1<0) row1 += rows+1;
    if(col1<0) col1 += cols+1;
    matrix A(row1-row0,col1-col0);
    for(int row=row0; row<row1; row++)
        for(int col=col0; col<col1; col++)
            A.m[row-row0][col-col0] = m[row][col];
    return A;
}

matrix matrix::submatrix(int i0, int i1, string mode) const {
    matrix A;
    if(mode=="row") A = submatrix(i0,i1,0,-1);
    else if(mode=="col") A = submatrix(0,-1,i0,i1);
    return A;
}

matrix matrix::getFirstRow() const {
    return getRow(0);
}

matrix matrix::getLastRow() const {
    return getRow(rows-1);
}

matrix matrix::getFirstCol() const {
    return getCol(0);
}

matrix matrix::getLastCol() const {
    return getCol(cols-1);
}

matrix matrix::flatten() const {
    vector<double> v;
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            v.push_back(m[row][col]);
    return matrix(v);
}

string matrix::print() const {
    ostringstream oss;
    for(int row=0; row<rows; row++)
        oss << ((row==0)?"[":" ") << m[row] << ((row==rows-1)?"]":",") << ((row<rows-1)?"\n":"");
    return oss.str();
}

string matrix::getAsCsv() const {
    ostringstream oss;
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            oss << m[row][col] << ((col<cols-1)?",":((row<rows-1)?"\n":""));
    return oss.str();
}

string matrix::getAsJson() const {
    ostringstream oss;
    if(rows==0) oss << "[]";
    else{
        for(int row=0; row<rows; row++)
            oss << ((row==0)?"[":"") << m[row] << ((row==rows-1)?"]":",");
    }
    return oss.str();
}

void matrix::printToCsvFile(string file, string header) const {
    ofstream f;
    f.open(file);
    if(!header.empty()) f << header << endl;
    f << getAsCsv();
    f.close();
}

void matrix::printToJsonFile(string file) const {
    ofstream f;
    f.open(file);
    f << getAsJson();
    f.close();
}

/**** mutators ****/

matrix matrix::setZero(){
    *this = matrix(rows,cols);
    return *this;
}

matrix matrix::setZero(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    *this = matrix(rows,cols);
    return *this;
}

matrix matrix::setOne(){
    *this = matrix(rows,cols)+1;
    return *this;
}

matrix matrix::setOne(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    *this = matrix(rows,cols).setOne();
    return *this;
}

matrix matrix::setIdentity(){
    assert(rows==cols);
    setZero();
    for(int row=0; row<rows; row++) m[row][row] = 1;
    return *this;
}

matrix matrix::setIdentity(int rows){
    this->rows = rows;
    this->cols = rows;
    *this = matrix(rows,rows).setIdentity();
    return *this;
}

matrix matrix::setUniformRand(double min, double max){
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            m[row][col] = uniformRand(min,max);
    return *this;
}

matrix matrix::setNormalRand(double mu, double sig){
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            m[row][col] = normalRand_(mu,sig);
    return *this;
}

matrix matrix::setPoissonRand(double lambda){
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            m[row][col] = poissonRand(lambda);
    return *this;
}

matrix matrix::setRow(int row, const matrix& vec){
    assert(vec.rows==1 && cols==vec.cols);
    m[row] = vec.m[0];
    return *this;
}

matrix matrix::setCol(int col, const matrix& vec){
    assert(vec.rows==1 && rows==vec.cols);
    for(int row=0; row<vec.cols; row++) m[row][col] = vec.m[0][row];
    return *this;
}

matrix matrix::setSubmatrix(int row0, int row1, int col0, int col1, const matrix& M){
    if(row1<0) row1 += rows+1;
    if(col1<0) col1 += cols+1;
    for(int row=row0; row<row1; row++)
        for(int col=col0; col<col1; col++)
            m[row][col] = M.m[row-row0][col-col0];
    return *this;
}

matrix matrix::setEntry(int row, int col, double a){
    assert(row>=0 && row<rows);
    assert(col>=0 && col<cols);
    m[row][col] = a;
    return *this;
}

matrix matrix::setDiags(const vector<double>& vec, const vector<int>& diags){
    assert(rows==cols);
    assert(vec.size()==diags.size());
    for(int row=0; row<rows; row++)
        for(int i=0; i<vec.size(); i++){
            int col = row+diags[i];
            col = min(max(col,0),cols-1);
            m[row][col] = vec[i];
        }
    return *this;
}

matrix matrix::setRange(double x0, double x1, int n, bool inc){
    if(n<0) n = x1-x0;
    double dx = (x1-x0)/n;
    if(inc) n += 1;
    *this = matrix(1,n);
    for(int i=0; i<n; i++) (*this).setEntry(0,i,x0+i*dx);
    return *this;
}

/**** matrix operations ****/

double matrix::trace() const {
    assert(rows==cols);
    double a = 0;
    for(int row=0; row<rows; row++)
        a += m[row][row];
    return a;
}

double matrix::getMax() const {
    assert(!isEmpty());
    double a = m[0][0];
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            a = max(a, m[row][col]);
    return a;
}

double matrix::getMin() const {
    assert(!isEmpty());
    double a = m[0][0];
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            a = min(a, m[row][col]);
    return a;
}

vector<int> matrix::maxIdx() const {
    assert(!isEmpty());
    double a = m[0][0];
    vector<int> idx{0,0};
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            if(m[row][col]>a){
                a = m[row][col];
                idx = {row,col};
            }
    return idx;
}

vector<int> matrix::minIdx() const {
    assert(!isEmpty());
    double a = m[0][0];
    vector<int> idx{0,0};
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            if(m[row][col]<a){
                a = m[row][col];
                idx = {row,col};
            }
    return idx;
}

vector<int> matrix::find(double a) const {
    assert(!isEmpty());
    vector<int> idx;
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            if(m[row][col]==a){
                idx = {row,col};
                break;
            }
    return idx;
}

vector<int> matrix::find(double a, string method) const {
    if(method=="exact") return find(a);
    else if(method=="closest") return abs(*this-a).minIdx();
    return {};
}

matrix matrix::maxWith(double a) const {
    matrix A(rows,cols);
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            A.m[row][col] = max(a, m[row][col]);
    return A;
}

matrix matrix::minWith(double a) const {
    matrix A(rows,cols);
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            A.m[row][col] = min(a, m[row][col]);
    return A;
}

double matrix::sum() const {
    double a = 0;
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            a += m[row][col];
    return a;
}

double matrix::sum(const vector<double>& weights) const {
    assert(rows*cols==weights.size());
    double a = 0;
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            a += m[row][col]*weights[row*cols+col];
    return a;
}

double matrix::sum(const matrix& weights) const {
    assert(rows==weights.rows && cols==weights.cols);
    double a = 0;
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            a += m[row][col]*weights.m[row][col];
    return a;
}

double matrix::prod() const {
    double a = 1;
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            a *= m[row][col];
    return a;
}

double matrix::mean(string method) const {
    if(method=="Arithmetic") return sum()/(rows*cols);
    else if(method=="Geometric") return exp((*this).apply(log).mean());
    else return 0;
}

double matrix::wmean(const vector<double>& weights, string method) const {
    double weightSum = 0;
    for(auto w:weights) weightSum += w;
    if(method=="Arithmetic") return sum(weights)/(weightSum);
    else if(method=="Geometric") return exp((*this).apply(log).wmean(weights));
    else return 0;
}

double matrix::var(int k) const {
    double mu = mean();
    return (((*this)-mu)*((*this)-mu)).sum()/(rows*cols-k);
}

double matrix::stdev(int k) const {
    return sqrt(var(k));
}

double matrix::cov(const matrix& M, int k) const {
    double mu0 = mean();
    double mu1 = M.mean();
    return (((*this)-mu0)*(M-mu1)).sum()/(rows*cols-k);
}

double matrix::cor(const matrix& M, int k) const {
    return cov(M,k)/(stdev(k)*M.stdev(k));
}

matrix matrix::sum(int axis) const {
    vector<double> v;
    switch(axis){
        case 1:
            for(int row=0; row<rows; row++) v.push_back(getRow(row).sum()); break;
        case 2:
            for(int col=0; col<cols; col++) v.push_back(getCol(col).sum()); break;
    }
    return matrix(v);
}

matrix matrix::mean(int axis, string method) const {
    vector<double> v;
    switch(axis){
        case 1:
            for(int row=0; row<rows; row++) v.push_back(getRow(row).mean(method)); break;
        case 2:
            for(int col=0; col<cols; col++) v.push_back(getCol(col).mean(method)); break;
    }
    return matrix(v);
}

matrix matrix::matrix::inverse() const {
    assert(rows==cols);
    int n = rows;
    MatrixXd a(n,n);
    for(int i=0; i<n; i++) a.row(i) = VectorXd::Map(&m[i][0],m[i].size());
    a = a.inverse();
    double b[n][n];
    Map<RowMatrixXd>(&b[0][0],n,n) = a;
    matrix A(n,n);
    vector<vector<double>> m_;
    for(int i=0; i<n; i++) m_.push_back(vector<double>(b[i],b[i]+n));
    A.m = m_;
    return A;
}

// matrix matrix::inverse() const {
//     assert(rows==cols);
//     int n = rows;
//     double c[n];
//     matrix A, S[n+1], I, tmp;
//     I.setIdentity(n);
//     S[0].setIdentity(n);
//     for(int k=1; k<n+1; k++){
//         tmp = (*this).dot(S[k-1]);
//         c[n-k] = -tmp.trace()/k;
//         S[k] = tmp+c[n-k]*I;
//     }
//     A = -1/c[0]*S[n-1];
//     return A;
// }

matrix matrix::transpose() const {
    matrix A(cols,rows);
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            A.m[col][row] = m[row][col];
    return A;
}

matrix matrix::dot(const matrix& M) const {
    assert(cols==M.rows);
    matrix A(rows,M.cols);
    for(int row=0; row<rows; row++)
        for(int col=0; col<M.cols; col++){
            double dot = 0;
            for(int k=0; k<cols; k++) dot += m[row][k]*M.m[k][col];
            A.m[row][col] = dot;
        }
    return A;
}

matrix matrix::apply(double (*f)(double)) const {
    matrix A(rows,cols);
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            A.m[row][col] = f(m[row][col]);
    return A;
}

matrix matrix::apply(const function<double(double)>& f) const {
    matrix A(rows,cols);
    for(int row=0; row<rows; row++)
        for(int col=0; col<cols; col++)
            A.m[row][col] = f(m[row][col]);
    return A;
}

matrix matrix::sample(int n, bool replace) const {
    matrix A(1,n);
    if(replace){
        int i0,i1;
        for(int i=0; i<n; i++){
            i0 = floor(uniformRand(0,rows));
            i1 = floor(uniformRand(0,cols));
            A.setEntry(0,i,m[i0][i1]);
        }
    }else{
        // TO DO
    }
    return A;
}

matrix matrix::concat(const matrix& M, int axis) const {
    matrix A;
    switch(axis){
        case 1:
            assert(cols==M.cols);
            A = matrix(rows+M.rows,cols);
            A.setSubmatrix(0,rows,0,cols,*this);
            A.setSubmatrix(rows,-1,0,cols,M);
            break;
        case 2:
            assert(rows==M.rows);
            A = matrix(rows,cols+M.cols);
            A.setSubmatrix(0,rows,0,cols,*this);
            A.setSubmatrix(0,rows,cols,-1,M);
            break;
    }
    return A;
}

matrix matrix::chol() const {
    assert(rows==cols);
    int n = rows;
    MatrixXd a(n,n);
    for(int i=0; i<n; i++) a.row(i) = VectorXd::Map(&m[i][0],m[i].size());
    a = a.llt().matrixL();
    double b[n][n];
    Map<RowMatrixXd>(&b[0][0],n,n) = a;
    matrix A(n,n);
    vector<vector<double>> m_;
    for(int i=0; i<n; i++) m_.push_back(vector<double>(b[i],b[i]+n));
    A.m = m_;
    return A;
}

/**** operators ****/

ostream& operator<<(ostream& out, const matrix& M){
    out << M.getAsJson();
    return out;
}

bool operator==(const matrix& M1, const matrix& M2){
    return (M1.m==M2.m);
}

bool operator!=(const matrix& M1, const matrix& M2){
    return !(M1==M2);
}

matrix operator+(const matrix& M1, const matrix& M2){
    assert(M1.rows==M2.rows && M1.cols==M2.cols);
    matrix A(M1.rows,M1.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M1.m[row][col]+M2.m[row][col];
    return A;
}

matrix operator+(const matrix& M, double a){
    matrix A(M.rows,M.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M.m[row][col]+a;
    return A;
}

matrix operator+(double a, const matrix& M){
    return M+a;
}

matrix operator+=(matrix& M1, const matrix& M2){
    return (M1 = M1+M2);
}

matrix operator+=(matrix& M, double a){
    return (M = M+a);
}

matrix operator-(const matrix& M1, const matrix& M2){
    return M1+(-M2);
}

matrix operator-(const matrix& M, double a){
    return M+(-a);
}

matrix operator-(double a, const matrix& M){
    return a+(-M);
}

matrix operator-=(matrix& M1, const matrix& M2){
    return (M1 = M1-M2);
}

matrix operator-=(matrix& M, double a){
    return (M = M-a);
}

matrix operator-(const matrix& M){
    return M*(-1);
}

matrix operator*(const matrix& M1, const matrix& M2){
    assert(M1.rows==M2.rows && M1.cols==M2.cols);
    matrix A(M1.rows,M1.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M1.m[row][col]*M2.m[row][col];
    return A;
}

matrix operator*(const matrix& M, double a){
    matrix A(M.rows,M.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M.m[row][col]*a;
    return A;
}

matrix operator*(double a, const matrix& M){
    return M*a;
}

matrix operator*=(matrix& M1, const matrix& M2){
    return (M1 = M1*M2);
}

matrix operator*=(matrix& M, double a){
    return (M = M*a);
}

matrix operator/(const matrix& M1, const matrix& M2){
    assert(M1.rows==M2.rows && M1.cols==M2.cols);
    matrix A(M1.rows,M1.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M1.m[row][col]/M2.m[row][col];
    return A;
}

matrix operator/(const matrix& M, double a){
    matrix A(M.rows,M.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M.m[row][col]/a;
    return A;
}

matrix operator/=(matrix& M, double a){
    return (M = M/a);
}

matrix operator>(const matrix& M, double a){
    matrix A(M.rows,M.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M.m[row][col]>a;
    return A;
}

matrix operator>=(const matrix& M, double a){
    matrix A(M.rows,M.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M.m[row][col]>=a;
    return A;
}

matrix operator<(const matrix& M, double a){
    matrix A(M.rows,M.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M.m[row][col]<a;
    return A;
}

matrix operator<=(const matrix& M, double a){
    matrix A(M.rows,M.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M.m[row][col]<=a;
    return A;
}

matrix operator||(const matrix& M1, const matrix& M2){
    assert(M1.rows==M2.rows && M1.cols==M2.cols);
    matrix A(M1.rows,M1.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M1.m[row][col]||M2.m[row][col];
    return A;
}

matrix operator&&(const matrix& M1, const matrix& M2){
    assert(M1.rows==M2.rows && M1.cols==M2.cols);
    matrix A(M1.rows,M1.cols);
    for(int row=0; row<A.rows; row++)
        for(int col=0; col<A.cols; col++)
            A.m[row][col] = M1.m[row][col]&&M2.m[row][col];
    return A;
}

#endif
