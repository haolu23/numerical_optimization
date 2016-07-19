#ifndef CHOL
#define CHOL
#include <armadillo>
void chol(const arma::mat &A, arma::mat *l, arma::vec *d); 
void chol(const arma::mat &A, arma::mat *l, arma::vec *d, double beta, double delta);
#endif
