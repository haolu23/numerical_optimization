#ifndef  INCLUDED_bfgs
#define  INCLUDED_bfgs
#include "functions.h"
#include "line_search.h"
#include "solver.h"

class BFGS: public Solver {
    public:
        BFGS(Function<arma::vec, arma::mat> &f, LineSearch<arma::vec, arma::mat> &s, const arma::vec x0):
            d_f(f), searcher(s), x(x0), 
            hessian(arma::mat(x0.n_rows, x0.n_rows, arma::fill::eye)),
            residual(1) {}
        void next();
        double getResidual() const {return residual;}
        const arma::vec& getX() const {return x;}

    private:
        Function<arma::vec, arma::mat> &d_f;
        LineSearch<arma::vec, arma::mat> &searcher;
        arma::vec x;
        arma::mat hessian;
        double residual;
};

#endif   /* ----- #ifndef INCLUDED_bfgs----- */
