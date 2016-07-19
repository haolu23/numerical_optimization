#ifndef  INCLUDED_line_search
#define  INCLUDED_line_search

#include	"functions.h"
#include <cassert>
#include <cmath>
#include <armadillo>

template<typename T, typename T2>
class LineSearch {
    public:
        virtual double step(const T &x, const T &p) = 0;
        virtual ~LineSearch() {};
};


template<typename T, typename T2>
class Backtracking: public LineSearch<T, T2> {
public:
    Backtracking(Function<T, T2> &func, double a, double r, double cc):f(func), alpha(a), rho(r), c(cc) {
        assert(alpha > 0 && rho > 0 && rho < 1 && c > 0 && c < 1);
    }
    double step(const T &x, const T &p) {
        while (f(x + alpha * p) > f(x) + c*alpha*(f.grad(x).t()*p)) {
            alpha *= rho;
        }
        return alpha;
    }
private:
    Function<T, T2> &f;
    double alpha;
    double rho;
    double c;
};

template<typename T, typename T2>
class Wolfe: public LineSearch<T, T2> {
    public:
        Wolfe(Function<T, T2> &ff, double a, double cf, double c):f(ff), alpha(a), c1(cf), c2(c){
            assert(c1>0 && c1 < c2 && c2 < 1);
        }
        double step(const T &x, const T &p) {
            double alphap = 0;
            double alphai = alpha / 2;
            while (true) {
                if(f(x+alphai*p) > f(x)+c1*alphai*arma::dot(f.grad(x), p)
                        || (alphap != 0 && f(x+alphai*p) > f(x+alphap*p))) {
                    return zoom(x, p, alphap, alphai);
                }
                double phialpha = arma::dot(f.grad(x+alphai*p), p);
                if (std::abs(phialpha) <= -c2*arma::dot(f.grad(x), p)) {
                    return alphai;
                }
                if (phialpha >= 0) {
                    alphai = zoom(x, p, alphai, alphap);
                }
                alphap = alphai;
                alphai = (alphai + alpha) / 2;
            }
        }
    private:
        Function<T, T2> &f;
        double alpha;
        double c1;
        double c2;

        double zoom(const T &x, const T &p, double alphalow, double alphahigh) {
            double alpha;
            while (true) {
                alpha = (alphalow + alphahigh) / 2;
                if (f(x+alpha*p) > f(x)+c1*alpha*arma::dot(f.grad(x),p) ||
                        f(x+alpha*p) >= f(x+alphalow*x)) {
                    alphahigh = alpha;
                } else {
                    double dphi = arma::dot(f.grad(x+alpha*p), p);
                    if (std::abs(dphi) <= -c2 * arma::dot(f.grad(x), p)) {
                        return alpha;
                    }
                    if (dphi * (alphahigh - alphalow) >= 0) {
                        alphahigh = alphalow;
                    }
                    alphalow = alpha;
                }
            }
        }
};

#endif   /* ----- #ifndef INCLUDED_line_search----- */
