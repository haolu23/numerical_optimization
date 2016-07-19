#include "newton_cg.h"
#include <functional>
#include "conj_grad.h"

void NewtonCG::next() {
    arma::mat grad2 = d_f.grad2(x);
    arma::vec p = -d_f.grad(x);
    double tolerance = std::min(0.5, arma::norm(-p, 2));
    // create a new CG solver
    ConjGrad grad(d_f.grad2(x), -d_f.grad(x), arma::zeros(x.size()));

    int i = 0;
    while(true) {
        if (arma::dot(grad.getDirection(), grad2*grad.getDirection()) < 0) {
            if (i != 0) {
                p = grad.getX();
            }
            break;
        }
        grad.next();
        if (grad.getResidual() < tolerance) {
            p = grad.getX();
            break;
        }
        i++;
    }
    // line search
    double alpha = searcher.step(x, p);
    x = x + alpha * p;
    residual = norm(d_f.grad(x), 2);
}
