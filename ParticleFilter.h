#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <cstdlib>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problems/schwefel.hpp>
#include "Strategy.h"





using namespace Rcpp;
using namespace pagmo;

class ParticleFilter {
public:
  ParticleFilter();
  // Constructor
  ParticleFilter(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_,  
  const MazeGraph& Optimal_Hybrid3_):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_) {}

  ParticleFilter(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_,  
  const MazeGraph& Optimal_Hybrid3_, std::vector<double> v, int particleId_, double weight_):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_), particleId(particleId_), weight(weight_) {

    double alpha_aca_subOptimal = v[4];
    double gamma_aca_subOptimal = v[5];

    double alpha_aca_optimal = v[4];
    double gamma_aca_optimal = v[5];

    //DRL params
    double alpha_drl_subOptimal = v[6];
    double beta_drl_subOptimal = v[7];
    double lambda_drl_subOptimal = v[8];
    
    double alpha_drl_optimal = v[6];
    double beta_drl_optimal = v[7];
    double lambda_drl_optimal = v[8];
    double phi = v[9];

    
    int n1 = static_cast<int>(std::floor(v[0]));
    int n2 = static_cast<int>(std::floor(v[1]));
    int n3 = static_cast<int>(std::floor(v[2]));
    int n4 = static_cast<int>(std::floor(v[3]));
       
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, 0, 0, 0, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, 0, 0, 0, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, 0, 0, 0, true);

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> suboptimalRewardfuncs =  getRewardFunctions(ratdata, *aca2_Suboptimal_Hybrid3, phi);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> optimalRewardfuncs =  getRewardFunctions(ratdata, *aca2_Optimal_Hybrid3,phi);

    aca2_Suboptimal_Hybrid3->setRewardsS0(suboptimalRewardfuncs.first);
    drl_Suboptimal_Hybrid3->setRewardsS0(suboptimalRewardfuncs.first);

    aca2_Optimal_Hybrid3->setRewardsS0(optimalRewardfuncs.first);
    aca2_Optimal_Hybrid3->setRewardsS1(optimalRewardfuncs.second);

    drl_Optimal_Hybrid3->setRewardsS0(optimalRewardfuncs.first);
    drl_Optimal_Hybrid3->setRewardsS1(optimalRewardfuncs.second);  

    strategies.push_back(aca2_Suboptimal_Hybrid3);
    strategies.push_back(aca2_Optimal_Hybrid3);

    strategies.push_back(drl_Suboptimal_Hybrid3);
    strategies.push_back(drl_Optimal_Hybrid3);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    for(int ses=0; ses<sessions; ses++)
    {
        chosenStrategy.push_back(-1);
    }



  }

  ParticleFilter::ParticleFilter(const ParticleFilter& other) :
    ratdata(other.ratdata),
    Suboptimal_Hybrid3(other.Suboptimal_Hybrid3),
    Optimal_Hybrid3(other.Optimal_Hybrid3),
    strategies(other.strategies),
    chosenStrategy(other.chosenStrategy),
    weight(other.weight),
    particleId(other.particleId) {
    // Any additional member variables that need to be copied should be added here
}



  // Destructor
  ~PagmoProb() {}

  // Fitness function
  vector_double fitness(const vector_double& v) const;

  // Bounds function
  std::pair<vector_double, vector_double> get_bounds() const;

  //CRP
  std::vector<double> getCrpPrior(int ses)
  {
    if(ses > 0)
    {
        std::vector<int> n(4, 0);

        for (int i = 0; i < ses; i++) {
            n[chosenStrategy[i]]++;
        }
          
        std::vector<double> q(4, 0);

        for (int k = 0; k < 4; k++) {
            if(n[k] > 0)
            {
                q[k] = n[k] / (ses - 1 + alpha_crp);
            }else{
                q[k] = alpha_crp / (ses - 1 + alpha_crp);
            }
            
        }

        return(q);

    }
    else{
        return {0.25,0.25,0.25,0.25};
    }

  }

  int sample_crp(vector<double> q) {

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);
    double u = dis(gen);

    // Find the smallest index k such that the cumulative sum of q up to k is greater than or equal to u
    int k = 0;
    double s = 0;
    while (s < u) {
        k++;
        s += q[k - 1];
    }

    // Return k as the sample
    return k;

  }


  double getSesLikelihood(int strat, int ses)
  {
    ll_ses  = strategies[strat]->getTrajectoryLikelihood(ratdata, ses); 
    return(exp(ll_ses));
  }


  void addAssignment(int ses, int selectedStrat)
  {
    chosenStrategy[ses] = selectedStrat;
  }

  std::vector<int> getChosenStratgies()
  {
    return chosenStrategy;
  }


  

  // }


private:
  // Members
  const RatData& ratdata;
  const MazeGraph& Suboptimal_Hybrid3;
  const MazeGraph& Optimal_Hybrid3;
  //mutable std::vector<std::atomic<std::pair<double, std::vector<double>>>> indexedValues;
  // mutable std::mutex  vectorMutex;

  std::vector<std::shared_ptr<Strategy>> strategies;
  std::vector<int> chosenStrategy;
  double weight;
  int particleId;
  
  


  
};


std::vector<double> particle_filter(int N, RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3,  vector<double> v);
void print_vector(std::vector<double> v);


#endif