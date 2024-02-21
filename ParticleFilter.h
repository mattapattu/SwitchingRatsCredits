#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <cstdlib>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problems/schwefel.hpp>
#include "Strategy.h"
#include "InverseRL.h"
#include <algorithm>





using namespace Rcpp;
using namespace pagmo;

class ParticleFilter {
public:

  ParticleFilter(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_, const MazeGraph& Optimal_Hybrid3_, std::vector<double> v, int particleId_, double weight_):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_), particleId(particleId_), weight(weight_) {
    //std::cout << "Initializing particleId=" << particleId << std::endl;
    particleId = particleId_;
    double alpha_aca_subOptimal = v[0];
    double gamma_aca_subOptimal = v[1];

    double alpha_aca_optimal = v[0];
    double gamma_aca_optimal = v[1];

    //DRL params
    double alpha_drl_subOptimal = v[2];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[3];
    
    double alpha_drl_optimal = v[2];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[3];
    alpha_crp = v[4];
    initCrpProbs = {0.25, 0.25, 0.25, 0.25};

       
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, 0, 0, 0, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, 0, 0, 0, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, 0, 0, 0, true);

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

  

    // ParticleFilter(const ParticleFilter& other) :
    //     ratdata(other.ratdata),
    //     Suboptimal_Hybrid3(other.Suboptimal_Hybrid3),
    //     Optimal_Hybrid3(other.Optimal_Hybrid3),
    //     weight(other.weight),
    //     particleId(other.particleId),
    //     alpha_crp(other.alpha_crp) {
    //     // Deep copy the vectors
    //     for (const auto& strategy : other.strategies) {
    //         strategies.push_back(std::make_shared<Strategy>(*strategy));
    //     }
    //     chosenStrategy = other.chosenStrategy; // Simple assignment
    // }





  // Destructor
  ~ParticleFilter() {}

 //CRP
  std::vector<double> crpPrior(int ses)
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
                q[k] = n[k] / (ses + alpha_crp);
            }else{
                q[k] = alpha_crp / (ses + alpha_crp);
                int zeroCount = std::count(n.begin(), n.end(), 0);
                q[k] = q[k]/zeroCount;

            }
            
        }

        double sum = std::accumulate(q.begin(), q.end(), 0.0);

        // Set a tolerance for floating-point comparison
        double tolerance = 1e-6;

        // Check if the sum is approximately equal to 1 within the tolerance
        if (std::abs(sum - 1.0) < tolerance) {
            //std::cout << "The sum is approximately equal to 1." << std::endl;
        } else {
            
            // std::cout << "crp_prior sum is not equal to 1. Check" << std::endl;

            // std::cout << "ses=" << ", crp_priors n = ";
            // for (auto const& i : n)
            //     std::cout << i << ", ";
            // std::cout << "\n" ;

            // std::cout << "ses=" << ", crp_priors q = ";
            // for (auto const& i : q)
            //     std::cout << i << ", ";
            // std::cout << "\n" ;

        }


        return(q);

    }
    else{
        return initCrpProbs;
    }

  }



//   std::vector<double> getCrpPrior(int ses)
//   {
//         std::vector<int> n(4, 0);

//         for (int i = 0; i < ses; i++) {
//             n[chosenStrategy[i]]++;
//         }
          
//         std::vector<double> q(4, 0);

//         // If all n == 0
//         if(n[0]==0 && n[1] == 0 && n[2] == 0 && n[3] == 0)
//         {
//             q={0.25,0.25,0.25,0.25};
//         }else{
//             // If any subopt n > 0
//             if(n[0] > 0 && n[1]==0)
//             {
//                 q[0] = n[0] / (ses + alpha_crp);
//                 q[1] = 0;

//                 if(n[2] > 0 && n[3] == 0)
//                 {
//                     q[2] = n[2] / (ses + alpha_crp);
//                     q[3] = 0;
//                 }else if(n[2] == 0 && n[3] > 0)
//                 {
//                     q[2] = 0;
//                     q[3] = n[3] / (ses + alpha_crp);
//                 }else if(n[2]==0 && n[3] == 0)
//                 {
//                     q[2] = alpha_crp / (2*(ses + alpha_crp));
//                     q[3] = alpha_crp / (2*(ses + alpha_crp));
//                 }
//             }
//             else if (n[0] == 0 && n[1] > 0)
//             {
//                 q[0] = 0;
//                 q[1] = n[1] / (ses + alpha_crp);

//                 if(n[2] > 0 && n[3] == 0)
//                 {
//                     q[2] = n[2] / (ses + alpha_crp);
//                     q[3] = 0;
//                 }else if(n[2] == 0 && n[3] > 0)
//                 {
//                     q[2] = 0;
//                     q[3] = n[3] / (ses + alpha_crp);
//                 }else if(n[2]==0 && n[3] == 0)
//                 {
//                     q[2] = alpha_crp / (2*(ses + alpha_crp));
//                     q[3] = alpha_crp / (2*(ses + alpha_crp));
//                 }
//             }

        
            
//             if(n[0] == 0 && n[1]==0 && n[2] == 0 && n[3] > 0)
//             {
//                 q[0] = 0;
//                 q[1] = 0;

//                 q[2] = 0;
//                 q[3] = n[3] / (ses + alpha_crp);

//             }else if(n[0] == 0 && n[1]==0 && n[2] > 0 && n[3] == 0)
//             {
//                 q[0] = 0;
//                 q[1] = 0;

//                 q[2] = n[2] / (ses + alpha_crp);
//                 q[3] = 0;
//             }

//             double sum = std::accumulate(q.begin(), q.end(), 0.0);

//             for (int i = 0; i < q.size(); i++) {
//                 q[i] /= sum;
//             }

//             if (sum == 0) {
                    
//                 throw std::runtime_error("Error crp prior vec is zero");
//             }
//         }
//         return(q);
//   }


  

  int sample_crp(std::vector<double> q) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
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
    double ll_ses  = strategies[strat]->getTrajectoryLikelihood(ratdata, ses); 
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

 std::vector<std::shared_ptr<Strategy>> getStrategies()
 {
    return strategies;
 }


  int getParticleId()
  {
    return particleId;
  }

  void setStrategies(std::vector<std::shared_ptr<Strategy>> strategies_)
  {
    for(int i=0; i<strategies_.size();i++)
    {
        strategies[i]->setStateS0Credits(strategies_[i]->getS0Credits());
        if(strategies[i]->getOptimal())
        {
            strategies[i]->setStateS1Credits(strategies_[i]->getS1Credits());
        }

        
    }
  }

  void setChosenStrategies(std::vector<int> chosenStrategy_) 
  {
    chosenStrategy = chosenStrategy_;
  }

  void setCrpPriors(std::vector<double> crpPrior)
  {
    crpPriors.push_back(crpPrior);
  }

  std::vector<std::vector<double>> getCrpPriors()
  {
    return(crpPriors);
  }

  void setLikelihood(double lik)
  {
    likelihoods.push_back(lik);
  }

  std::vector<double> getLikelihoods()
  {
    return likelihoods;
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
  double alpha_crp;
  std::vector<std::vector<double>> crpPriors;
  std::vector<double> likelihoods;
  std::vector<double> initCrpProbs;
      
};


std::pair<std::vector<std::vector<double>>, double> particle_filter(int N, const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3,  std::vector<double> v);
void print_vector(std::vector<double> v);
double M_step(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, int N, std::pair<std::vector<std::vector<double>>,std::vector<std::vector<std::vector<double>>>> smoothed_w, std::vector<double> params);
std::vector<double> EM(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, int N);


#endif