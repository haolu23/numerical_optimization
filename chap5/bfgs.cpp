#include "bfgs.h"

void BFGS::next() {
    arma::vec p = -hessian*(d_f.grad(x));
    double alpha = searcher.step(x, p);
    arma::vec s = alpha*p;
    arma::vec y = d_f.grad(x+s) - d_f.grad(x);
    double rho = 1/arma::dot(y, s);
    arma::mat w = arma::eye<arma::mat>(x.n_rows, x.n_rows) - rho*y*s.t();
    hessian = w.t()*hessian*w + rho*s*s.t();
    x = x + s;
    residual = norm(d_f.grad(x), 2);
}
