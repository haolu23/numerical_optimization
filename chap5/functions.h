#ifndef  INCLUDED_functions
#define  INCLUDED_functions

#include	<cmath>
template <typename T, typename T2>
class Function {
    public:
        virtual double operator() (const T&) = 0;
        virtual T grad(const T&);
        virtual T2 grad2(const T&);
};

template <typename T, typename T2>
class Rosenbrock: public Function<T, T2> {
    public:
        double operator() (const T& x) {
            return 100*pow(x[1] - x[0]*x[0], 2) + pow(1-x[0], 2);
        }

        T grad(const T& x) {
            return T({-400*x[0]*(x[1]-x[0]*x[0])-2+2*x[0], 200*(x[1]-x[0]*x[0])});
        }

        T2 grad2(const T&x) {
            return T2(
                    { {-400*(x[1]-x[0]*x[0])-2+800*x[0]*x[1],
                     -400*x[0]} ,
                     { -400*x[0], 200}
                    }
                    );
        }
};
#endif   /* ----- #ifndef INCLUDED_functions----- */
