
#ifndef RATDATA_H
#define RATDATA_H
#include <RcppArmadillo.h>

class RatData{
    public:
        RatData(const Rcpp::S4 & ratdata){
              
              rat = Rcpp::as<std::string>(ratdata.slot("rat"));
              Paths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
              Hybrid1 = Rcpp::as<arma::mat>(ratdata.slot("hybridModel1"));
              Hybrid2 = Rcpp::as<arma::mat>(ratdata.slot("hybridModel2"));
              Hybrid3 = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
              Hybrid4 = Rcpp::as<arma::mat>(ratdata.slot("hybridModel4"));
              Turns = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
              
        }

        arma::mat getPaths() const{
            return Paths;
        }

        arma::mat getHybrid1() const {
            return Hybrid1;
        }

        arma::mat getHybrid2() const {
            return Hybrid2;
        }

        arma::mat getHybrid3() const {
            return Hybrid3;
        }

        arma::mat getHybrid4() const {
            return Hybrid4;
        }

        arma::mat getTurns() const {
            return Turns;
        }

        std::string getRat() const {
            return rat;
        }


    private:
        std::string rat;
        arma::mat Paths;
        arma::mat Hybrid1;
        arma::mat Hybrid2;
        arma::mat Hybrid3;
        arma::mat Hybrid4;
        arma::mat Turns;    

};

#endif