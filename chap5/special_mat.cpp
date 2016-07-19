#include "special_mat.h"

arma::mat hilbert(int size) {
    arma::mat m(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            m(i, j) = 1.0/(i+j+1);
            //m(j, i) = 1.0/(i+j-1);
        }
    }
    return m;
}

arma::mat random_mat(const arma::vec &eigenvals) {
    arma::mat a(eigenvals.n_elem, eigenvals.n_elem, arma::fill::randu);
    return a.t() * diagmat(eigenvals) * a;
}
