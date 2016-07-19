#ifndef NEWTON_CG
#define NEWTON_CG
#include <armadillo>
#include "functions.h"
#include "solver.h"
#include "line_search.h"

class NewtonCG: public Solver {
    public:
        NewtonCG(Function<arma::vec, arma::mat> &f, LineSearch<arma::vec, arma::mat> &s,
                const arma::vec &x0):
            d_f(f), searcher(s), x(x0), residual(1.0) {}
        void next();
        double getResidual() const {return residual;}
        const arma::vec& getX() const {return x;}

    private:
        Function<arma::vec, arma::mat> &d_f;
        LineSearch<arma::vec, arma::mat> &searcher;
        arma::vec x;
        double residual;
};

#endif
