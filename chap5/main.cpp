#include "chol.h"
#include "conj_grad.h"
#include "special_mat.h"
#include "observer.h"
//#include "config.h"
#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <typeinfo>
#include "functions.h"
#include "newton_cg.h"
#include "line_search.h"
#include "bfgs.h"
#include "trust_region.h"


std::ostream& operator<<(std::ostream& os, const std::vector<double>& c) {
    for (auto i = c.begin(); i != c.end(); ++i) {
        os << *i << ",";
    }
    os << std::endl;
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::map<std::string, std::vector<double> >& c) {
    for (auto i = c.begin(); i != c.end(); ++i) {
        os << i->first << ": " << i->second;
    }
    return os;
}

void test_algorithm(Solver &s, int max_iter=5e6) {
    int niter = 0;
    std::list<arma::vec> q;
    while (s.getResidual() > 1e-4 && niter < max_iter) {
        s.next();
        niter++;
        if (q.size() > 10) {
            q.pop_front();
        }
        q.push_back(s.getX());
    }
    if (niter >= max_iter) {
        std::cout  << "Solver " << typeid(s).name()
            << " failed to find solution" << std::endl;
    } else {
        std::cout << "Solver " << typeid(s).name() 
            << " converged after " << niter << " iterations." << std::endl;
    }
    for (auto i = std::begin(q); i != std::end(q); ++i) {
        std::cout << (*i).t() ;
    }
}

int main(int argc, char *argv[]) {
	const static int SIZE = 10;
    arma::mat a(SIZE, SIZE, arma::fill::randu);

    arma::mat l(SIZE, SIZE, arma::fill::eye);
    arma::vec d(SIZE);
    arma::mat tmp = a.t()*a;
	chol(tmp, &l, &d);
	//	std::cout << chol(tmp) << std::endl;

	std::cout<<l<<std::endl;
	std::cout<<d<<std::endl;

	std::cout << l * diagmat(sqrt(d)) << std::endl;

	chol(tmp, &l, &d, 1, 0.5);
	//std::cout<<l<<std::endl;
	//std::cout<<d<<std::endl;

    /*      std::cout << h << std::endl;
    conj_grad(tmp, b, x, &xout);
    std::cout << xout << std::endl;
    */

    // test Hibert matrix conjugrate gradient decent method
    int sizes[] = {5, 8, 12, 20};
    for (int i = 0; i < sizeof(sizes)/sizeof(int); ++i) {
        arma::vec b(sizes[i], arma::fill::ones);
        arma::mat h = hilbert(sizes[i]);
        arma::vec x(sizes[i], arma::fill::zeros);
        arma::vec xout(sizes[i], arma::fill::zeros);
        AccumulateObserver<double> obs;
        int j = conj_grad(h, b, x, &xout, &obs);
        std::cout << j << std::endl;
        std::cout << obs.get_vals() << std::endl;
        //std::cout << xout << std::endl;
    }
    

    // test matrices with clustered distribution of eigenvalues
    std::vector<int> clusters = {1, 2, 4, 6};
    for (auto j = clusters.begin(); j != clusters.end(); ++j) {
        arma::vec b(*j * SIZE, arma::fill::randu);
        double amplifier = 10;
        for (int i = 0; i < *j; ++i) {
            b(arma::span(i*SIZE, (i+1)*SIZE-1)) = b(arma::span(i*SIZE, (i+1)*SIZE-1)) * amplifier;
            amplifier *= 2;
        }
        arma::vec x(*j * SIZE, arma::fill::zeros);
        arma::vec xout(*j * SIZE, arma::fill::zeros);
        AccumulateObserver<double> obs;
        //std::cout << b << std::endl;
        arma::mat a = random_mat(b);
        conj_grad(a, arma::vec(*j * SIZE, arma::fill::ones), x, &xout, &obs);
        std::cout << obs.get_vals() << std::endl;
    } 

    Rosenbrock<arma::vec, arma::mat> f;
    std::cout << "Rosenbrock function valuation & gradient at (1,2)" << std::endl;
    std::cout << f(arma::vec({1, 2})) << std::endl;
    std::cout << f.grad(arma::vec({1, 2}));

    arma::vec startx = arma::vec({-1.2,1});

    Wolfe<arma::vec, arma::mat> w = Wolfe<arma::vec, arma::mat>(f, 1.0, 1e-4, 0.9);

    NewtonCG ncg(f, w, startx);
    test_algorithm(ncg);

    BFGS b = BFGS(f, w, startx);
    test_algorithm(b);

    TrustRegion t = TrustRegion(f, startx, 0.2, 1e-4);
    test_algorithm(t);
}
