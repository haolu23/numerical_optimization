#ifndef SPECIAL_MAT
#define SPECIAL_MAT
#include <armadillo>

arma::mat hilbert(int size);

arma::mat random_mat(const arma::vec &eigenvals) ;
#endif
