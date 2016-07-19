#ifndef SOLVER
#define SOLVER

#include <armadillo>

class Solver {
    public:
        virtual void next() = 0;
        virtual double getResidual() const = 0;
        virtual const arma::vec& getX() const = 0;
        virtual ~Solver() {};
}; 

class GradientSolver: public Solver {
    public:
        virtual const arma::vec& getDirection() const = 0;
};
#endif
