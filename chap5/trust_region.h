#ifndef  INCLUDED_trust_region
#define  INCLUDED_trust_region
#include <armadillo>
#include "functions.h"
#include "solver.h"
#include <cassert>

class TrustRegion: public Solver {
    public:
        TrustRegion(Function<arma::vec, arma::mat> &f, const arma::vec &x0, double eta, 
                double max_region_radius):
            d_f(f), x(x0), d_eta(eta),  d_max_region_radius(max_region_radius), 
            region_radius(max_region_radius/2), residual(1.0){
                assert(eta > 0 && eta < 0.25 && max_region_radius > 0);
            }
        void next();
        double getResidual() const {return residual;}
        const arma::vec& getX() const {return x;}

    private:
        Function<arma::vec, arma::mat> &d_f;
        arma::vec x;
        double d_eta;
        double d_max_region_radius;
        double region_radius;
        double residual;
};
void trust_region(Function<arma::vec, arma::mat> &f, const arma::vec &x, arma::vec *xout, 
        double max_region_radius, double eta); 
#endif   /* ----- #ifndef INCLUDED_trust_region----- */
