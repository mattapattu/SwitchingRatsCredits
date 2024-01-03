
#ifndef RATDATA_H
#define RATDATA_H
#include <RcppArmadillo.h>

class RatData{
    public:

        RatData() {}

        RatData(const Rcpp::S4 & ratdata){
              
              rat = Rcpp::as<std::string>(ratdata.slot("rat"));
              Paths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
              Hybrid1 = Rcpp::as<arma::mat>(ratdata.slot("hybridModel1"));
              Hybrid2 = Rcpp::as<arma::mat>(ratdata.slot("hybridModel2"));
              Hybrid3 = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
              Hybrid4 = Rcpp::as<arma::mat>(ratdata.slot("hybridModel4"));
              Turns = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
              sim = false; 
        }

        RatData(arma::mat Paths_, arma::mat Hybrid3_, std::string rat_, bool sim_, std::vector<std::string> generatorStrategies_): Paths(Paths_), 
        Hybrid3(Hybrid3_), rat(rat_), sim(sim_), generatorStrategies(generatorStrategies_) {}

       

        arma::mat getPaths() const{
            return Paths;
        }

        void setPaths(arma::mat Paths_) {
            Paths = Paths_;
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

        void setHybrid3(arma::mat Hybrid3_) {
            Hybrid3 = Hybrid3_;
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

        bool getSim() const {
            return sim;
        }

        void setGeneratorStrategies(std::vector<std::string> genStrategies)
        {
            generatorStrategies = genStrategies;   
        }

        std::vector<std::string> getGeneratorStrategies()
        {
            return generatorStrategies;
        }




    private:
        std::string rat;
        arma::mat Paths;
        arma::mat Hybrid1;
        arma::mat Hybrid2;
        arma::mat Hybrid3;
        arma::mat Hybrid4;
        arma::mat Turns;  
        bool sim;  
        std::vector<std::string> generatorStrategies;

};

#endif