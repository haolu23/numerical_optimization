#include "conj_grad.h"
#include <iostream>
#include <cstddef>

void ConjGrad::next() {
    arma::vec &p = direction;
    double alpha = dot(r, r)/(arma::dot(p, a*p));
    x = x + alpha * p;
    arma::vec rk = r + alpha * (a * p);
    double beta = dot(rk, rk)/arma::dot(r, r);
    p = -rk + beta * p;
    r = rk;
    residual = arma::norm(r, 2);
}

int conj_grad(const arma::mat &a, const arma::vec &b, arma::vec x, arma::vec *xout, Observer<double> *po) {
	arma::vec r = a*x - b;
	arma::vec p = -r;
    double total_error = arma::norm(r, 2);
    int num_loops = 0;
	while (total_error > 1e-6) {
        num_loops++;
		double alpha = dot(r, r)/(arma::dot(p, a*p));
		x = x + alpha * p;
		arma::vec rk = r + alpha * (a * p);
		double beta = dot(rk, rk)/arma::dot(r, r);
		p = -rk + beta * p;
		r = rk;
        // 1/2 xAx - bx
        double phix = 0.5*arma::dot(x, a*x) - arma::dot(b, x);
        if (po != nullptr) {
            po->update("phix", phix);
            po->update("error", total_error);
        }
        total_error = arma::norm(r, 2);
        //std::cout << total_error << std::endl;
	}
	*xout = x;
    return num_loops;
}
