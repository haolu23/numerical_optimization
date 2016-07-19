
#ifndef  INCLUDED_cg_steihaug
#define  INCLUDED_cg_steihaug
#include "solver.h"
#include <cassert>
class CGSteihaug:public GradientSolver {
    public:
        CGSteihaug(const arma::mat &a, const arma::vec &b, const arma::vec &x0, double bound):
            a(a), b(b), x(x0), d_bound(bound), residual(1.0) {assert(d_bound > 0)};
        void next();
        double getResidual() const {return residual;}
        const arma::vec& getDirection() const {return direction;}
    private:
        arma::mat a;
        const arma::vec &b;
        arma::vec x;
        double d_bound;
        double residual;
        arma::vec direction;
};
#endif   /* ----- #ifndef INCLUDED_cg_steihaug----- */
