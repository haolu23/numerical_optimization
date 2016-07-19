#include <cassert>
#include <iostream>
#include <algorithm>
#include "chol.h"

using namespace arma;
void chol(const mat &A, mat *l, vec *d) {
	assert(A.n_rows == A.n_cols);
	for (int j = 0; j < A.n_rows; ++j) {
		double cjj = A(j, j);
		for (int k = 0; k < j; ++k) {
			cjj -= (*d)(k) * ((*l)(j, k)) * ((*l)(j, k));
		}
		(*d)(j) = cjj;
		for (int i = j+1; i < A.n_rows; ++i) {
			double cij = A(i, j);
			for (int k = 0; k < j; ++k) {
				cij -= (*d)(k)* ((*l)(i, k)) * ((*l)(j, k));
			}
			(*l)(i, j) = cij / (*d)(j);
		}
	}
}

void chol(const mat &A, mat *l, vec *d, double beta, double delta) {
	assert(A.n_rows == A.n_cols);
	for (int j = 0; j < A.n_rows; ++j) {
		double cjj = A(j, j);
		for (int k = 0; k < j; ++k) {
			cjj -= (*d)(k) * ((*l)(j, k)) * ((*l)(j, k));
		}
		double theta = 0;
		vec cij(A.n_rows, fill::zeros);
		for (int i = j+1; i < A.n_rows; ++i) {
			cij(i) = A(i, j);
			for (int k = 0; k < j; ++k) {
				cij(i) -= (*d)(k)* ((*l)(i, k)) * ((*l)(j, k));
			}
			theta = max(abs(cij));
			(*d)(j) = std::max(std::abs(cjj), std::max(theta*theta/beta/beta, delta));
		}
		for (int i = j+1; i < A.n_rows; ++i) {
			(*l)(i, j) = cij(i) / (*d)(j);
		}

	}
}
