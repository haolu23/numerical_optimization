#include "cg_steihauh.h"
#include <cmath>

namespace {
// calculate m_k(z + tau*d), only the part that depends on tau
double mk(double dfd, double dbz, double dbd, double tau) {
    return dbd*tau*tau/2 + tau*(dfd+dbz);
}

// find the range of tau so that ||z + tau*d|| <= dk
void trust_tau_range(const arma::vec &zj, const arma::vec &dj, double dk, double *p_lower, double *p_upper) {
    // calculate range of tau
    double zd = arma::dot(zj, dj);
    double zz = arma::dot(zj, zj);
    double dd = arma::dot(dj, dj);
    double lower = 0, upper = 0;
    // ||p_k|| < dk
    double range = ((zd*zd/dd/dd)+dk*dk-zz)/dd;
    if (range < 0) {
        // no tau can be found
        return 0;
    }
    if (zd < 0) {
        *p_lower = sqrt(range) + zd/dd;
        *p_upper = sqrt(range) - zd/dd;
    } else {
        *p_lower = sqrt(range) - zd/dd;
        *p_upper = sqrt(range) + zd/dd;
    }
}

// minimize m_k
double trust_region_search(double dfd, double dbz, double dbd, double lower, double upper) {
    double mintau = -(dfd + dbz)/dbd;
    if (dbd > 0 && mintau >= lower && mintau <= upper) {
        return mintau;
    } else {
        // m_k only has minimum value
        if (mk(dfd, dbz, dbd, lower) < mk(dfd, dbz, dbd, upper)) {
            return lower;
        } else {
            return upper;
        }
    }
} 
}

void CGSteihaug::next() {
    arma::vec b = -d_f.grad(x);
    arma::vec r = -b;
    double tolerance = std::min(0.5, arma::norm(-b, 2));

    if (arma::norm(d) < tolerance) {
        break;
    }
    double dbd = arma::dot(b, a*b);
    double dfd = arma::dot(f.grad(x), b);
    double dbz = arma::dot(b, a*x);
    double lower = 0;
    double upper = 0;
    trust_tau_range(x, b, d_bound, &lower, &upper);
    if (dbd < 0) {
        direction = x + trust_region_search(dfd, dbd, dbz, lower, upper)*d;
    }
    double alpha = dot(r, r)/(arma::dot(b, a*b));
    x = x + alpha * b;
    if (arma::norm(x) >= d_bound) {
        if (lower > 0) {
            direction = x + lower * d; 
        } else if (upper > 0) {
            direction = x + upper * d;
        }
    }
    arma::vec rk = r + alpha * (a * b);
}
