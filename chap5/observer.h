#ifndef OBSERVER_H
#define OBSERVER_H
#include <vector>
#include <string>
#include <map>

template <typename T>
class Observer {
    public:
        virtual void update(const std::string&, const T&)=0;
};

template <typename T>
class AccumulateObserver: public Observer<T> {
    public:
        void update(const std::string& what, const T &t) {
            d_vals[what].push_back(t);
        }
        const std::map<std::string, std::vector<T> >& get_vals() {return d_vals;}
    private:
        std::map<std::string, std::vector<T> > d_vals;
};
#endif
