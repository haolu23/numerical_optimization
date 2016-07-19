#include "trust_region.h"
#include <cmath>

namespace{
arma::vec dogleg(Function<arma::vec, arma::mat> &f, const arma::vec &x, double region_radius) {
    arma::vec newton_direction = -arma::solve(f.grad2(x), f.grad(x));
    if (arma::norm(newton_direction, 2) <= region_radius) {
        return newton_direction;
    } else {
        arma::vec descent = f.grad(x);
        arma::mat bmat = f.grad2(x);
        arma::vec e = arma::eig_sym(bmat);
        auto tmp = arma::dot(descent, f.grad2(x)*descent);
        if (tmp <= 0 || arma::any(e <= 0)) {
            return -region_radius/arma::norm(descent, 2)*descent;
        }

        arma::vec steep_descent = -arma::dot(descent, descent)/tmp*descent;

        if (arma::norm(steep_descent, 2) > region_radius) {
            return region_radius/arma::norm(steep_descent, 2)*steep_descent;
        }
        double a = arma::norm(steep_descent - newton_direction, 2);
        a *= a;
        double b = 2*arma::dot(steep_descent, newton_direction - steep_descent);
        double c = arma::dot(steep_descent, steep_descent) - region_radius*region_radius;

        double solution = (-b + std::sqrt(b*b - 4*a*c)) / 2 / a;
        if (solution > 1) {
            solution =  (-b - std::sqrt(b*b - 4*a*c)) / 2 / a;
        }
        if (!steep_descent.is_finite() || !newton_direction.is_finite() ||
                b*b < 4*a*c || solution < 0 || solution > 1) {
            std::cout << f.grad2(x) << std::endl;
            std::cout << arma::dot(descent, f.grad2(x)*descent) << std::endl;
            std::cout << descent;
            throw std::range_error("No solution found in the dogleg algorithm.");
        }
        return steep_descent + (solution) * (newton_direction - steep_descent);
    }
}
}

void TrustRegion::next() {
    // solve approximate quadratic problem
    auto p = dogleg(d_f, x, region_radius);
    double reduction_ratio = -(d_f(x) - d_f(x+p)) / 
        (arma::dot(d_f.grad(x), p) + arma::dot(p, d_f.grad2(x)*p)/2);
    if (reduction_ratio < 0.25) {
        region_radius = region_radius * 0.25;
    } else if (reduction_ratio > 0.75 && std::abs(norm(p, 2)-region_radius)<1e-5) {
        region_radius = std::min(2*region_radius, d_max_region_radius);
    } 
    if (reduction_ratio > d_eta) {
        x = x + p;
    }
    residual = arma::norm(d_f.grad(x), 2);
}

void trust_region(Function<arma::vec, arma::mat> &f, const arma::vec &x0, arma::vec *xout, 
        double max_region_radius, double eta) {
    assert(eta > 0 && eta < 0.25 && xout != nullptr);

    double region_radius = max_region_radius / 2;
    arma::vec x = x0;
    while (arma::norm(f.grad(x), 2) > 1e-6) {
        // solve approximate quadratic problem
        auto p = dogleg(f, x, region_radius);
        double reduction_ratio = -(f(x) - f(x+p)) / (arma::dot(f.grad(x), p) + arma::dot(p, f.grad2(x)*p)/2);
        if (reduction_ratio < 0.25) {
            region_radius = region_radius * 0.25;
        } else if (reduction_ratio > 0.75 && norm(p, 2) == region_radius) {
            region_radius = std::min(2*region_radius, max_region_radius);
        } 
        if (reduction_ratio > eta) {
            x = x + p;
        }
    }
    *xout = x;
}
