#ifndef CONJ_GRAD
#define CONJ_GRAD
#include <armadillo>
#include "observer.h"
#include <cstddef>
#include "solver.h"

/**
 * @brief linear conjugate gradient decend method
 *
 * @param a A matrix of Ax = b
 * @param b b matrix
 * @param x initial point
 * @param xout solution point
 */
class ConjGrad: public GradientSolver {
    public:
        ConjGrad(const arma::mat &a, const arma::vec &b, const arma::vec x0):
             a(a), b(b), x(x0), residual(1.0) { 
                r = a*x - b;
                direction = -r;
            }

        void next();
        double getResidual() const {return residual;}
        const arma::vec& getDirection() const {return direction;}
        const arma::vec& getX() const {return x;}

    private:
        const arma::mat &a;
        const arma::vec &b;
        arma::vec x;
        double residual;
        arma::vec r;
        arma::vec direction;
};

int conj_grad(const arma::mat &a, const arma::vec &b, arma::vec x, arma::vec *xout, Observer<double> *p=nullptr);
#endif
